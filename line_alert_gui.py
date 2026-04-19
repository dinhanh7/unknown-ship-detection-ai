from __future__ import annotations

import math
import subprocess
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Tuple

import cv2
import torch
from PIL import Image, ImageTk


@dataclass
class Track:
    track_id: int
    cls_id: int
    center: Tuple[float, float]
    last_side: float
    missed: int = 0
    alerted: bool = False


def parse_point_list(text: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in text.split(',')]
    if len(parts) != 4:
        raise ValueError('Line must have 4 comma-separated values: x1,y1,x2,y2')
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def line_side(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def center_of_box(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _orientation(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    return (
        min(a[0], b[0]) - 1e-6 <= c[0] <= max(a[0], b[0]) + 1e-6
        and min(a[1], b[1]) - 1e-6 <= c[1] <= max(a[1], b[1]) + 1e-6
    )


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True

    if abs(o1) < 1e-6 and _on_segment(p1, p2, q1):
        return True
    if abs(o2) < 1e-6 and _on_segment(p1, p2, q2):
        return True
    if abs(o3) < 1e-6 and _on_segment(q1, q2, p1):
        return True
    if abs(o4) < 1e-6 and _on_segment(q1, q2, p2):
        return True

    return False


def point_in_rect(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    return x1 <= px <= x2 and y1 <= py <= y2


def box_touches_line(
    xyxy: Tuple[float, float, float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> bool:
    x1, y1, x2, y2 = xyxy
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    if point_in_rect(p1[0], p1[1], x1, y1, x2, y2) or point_in_rect(p2[0], p2[1], x1, y1, x2, y2):
        return True

    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]
    for a, b in edges:
        if segments_intersect(p1, p2, a, b):
            return True
    return False


def match_detections_to_tracks(
    tracks: Dict[int, Track],
    detections: List[Tuple[int, Tuple[float, float, float, float], float]],
    max_match_distance: float,
) -> Dict[int, int]:
    unmatched_tracks = set(tracks.keys())
    unmatched_dets = set(range(len(detections)))
    matches: Dict[int, int] = {}

    candidates: List[Tuple[float, int, int]] = []
    for det_idx, (cls_id, xyxy, _) in enumerate(detections):
        c_det = center_of_box(xyxy)
        for track_id, tr in tracks.items():
            if tr.cls_id != cls_id:
                continue
            d = distance(c_det, tr.center)
            if d <= max_match_distance:
                candidates.append((d, det_idx, track_id))

    candidates.sort(key=lambda x: x[0])
    for _, det_idx, track_id in candidates:
        if det_idx in unmatched_dets and track_id in unmatched_tracks:
            matches[det_idx] = track_id
            unmatched_dets.remove(det_idx)
            unmatched_tracks.remove(track_id)

    return matches


class LineAlertApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title('Unknown Ship Alert - Tkinter')

        self.repo_root = Path(__file__).resolve().parent
        self.model = None
        self.model_names = None
        self.target_cls_id = None

        self.cap = None
        self.writer = None
        self.running = False
        self.photo = None

        self.frame_index = 0
        self.last_alert_frame = -10**9
        self.fps = 25.0
        self.frame_w = 0
        self.frame_h = 0

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

        self.pick_line_mode = False
        self.pick_points_px: List[Tuple[int, int]] = []
        self.display_w = 0
        self.display_h = 0
        self.target_touching_prev = False
        self.tts_process = None

        self._build_ui()

    def _build_ui(self) -> None:
        control = ttk.Frame(self.root, padding=8)
        control.pack(side=tk.TOP, fill=tk.X)

        self.weights_var = tk.StringVar(value='runs/train/tau_yolov5n_live/weights/best.pt')
        self.source_var = tk.StringVar(value='data/origin_video.mp4')
        self.output_var = tk.StringVar(value='runs/detect/origin_video_line_alert_gui.mp4')
        self.alert_audio_var = tk.StringVar(value='runs/voice_alert.mp3')
        self.device_var = tk.StringVar(value='cuda:0')
        self.line_var = tk.StringVar(value='0.56,0.49,0.95,0.57')
        self.target_class_var = tk.StringVar(value='tau_la')
        self.conf_var = tk.DoubleVar(value=0.25)
        self.iou_var = tk.DoubleVar(value=0.45)
        self.imgsz_var = tk.IntVar(value=640)
        self.cooldown_var = tk.IntVar(value=1)
        self.max_missed_var = tk.IntVar(value=15)
        self.max_match_dist_var = tk.DoubleVar(value=90.0)

        row0 = ttk.Frame(control)
        row0.pack(fill=tk.X, pady=2)
        ttk.Label(row0, text='Weights').pack(side=tk.LEFT)
        ttk.Entry(row0, textvariable=self.weights_var, width=62).pack(side=tk.LEFT, padx=4)
        ttk.Button(row0, text='Browse', command=self._browse_weights).pack(side=tk.LEFT)

        row1 = ttk.Frame(control)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text='Video').pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.source_var, width=62).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text='Browse', command=self._browse_source).pack(side=tk.LEFT)

        row2 = ttk.Frame(control)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text='Save').pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.output_var, width=62).pack(side=tk.LEFT, padx=9)

        row2b = ttk.Frame(control)
        row2b.pack(fill=tk.X, pady=2)
        ttk.Label(row2b, text='Alert audio').pack(side=tk.LEFT)
        ttk.Entry(row2b, textvariable=self.alert_audio_var, width=62).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2b, text='Browse', command=self._browse_alert_audio).pack(side=tk.LEFT)

        row3 = ttk.Frame(control)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text='Line x1,y1,x2,y2').pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.line_var, width=20).pack(side=tk.LEFT, padx=4)
        ttk.Label(row3, text='Target class').pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(row3, textvariable=self.target_class_var, width=10).pack(side=tk.LEFT)
        ttk.Label(row3, text='Device').pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(row3, textvariable=self.device_var, width=10).pack(side=tk.LEFT)

        row4 = ttk.Frame(control)
        row4.pack(fill=tk.X, pady=2)
        ttk.Label(row4, text='conf').pack(side=tk.LEFT)
        ttk.Entry(row4, textvariable=self.conf_var, width=6).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(row4, text='iou').pack(side=tk.LEFT)
        ttk.Entry(row4, textvariable=self.iou_var, width=6).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(row4, text='imgsz').pack(side=tk.LEFT)
        ttk.Entry(row4, textvariable=self.imgsz_var, width=6).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(row4, text='cooldown(frames)').pack(side=tk.LEFT)
        ttk.Entry(row4, textvariable=self.cooldown_var, width=6).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(row4, text='max missed').pack(side=tk.LEFT)
        ttk.Entry(row4, textvariable=self.max_missed_var, width=6).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(row4, text='max match dist').pack(side=tk.LEFT)
        ttk.Entry(row4, textvariable=self.max_match_dist_var, width=7).pack(side=tk.LEFT, padx=(2, 10))

        row5 = ttk.Frame(control)
        row5.pack(fill=tk.X, pady=(6, 2))
        ttk.Button(row5, text='Start Inference', command=self.start).pack(side=tk.LEFT)
        ttk.Button(row5, text='Stop', command=self.stop).pack(side=tk.LEFT, padx=6)
        ttk.Button(row5, text='Pick 2 Points', command=self.enable_pick_line_mode).pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value='Ready')
        ttk.Label(row5, textvariable=self.status_var).pack(side=tk.LEFT, padx=10)

        self.alert_var = tk.StringVar(value='No alert')
        self.alert_label = tk.Label(
            self.root,
            textvariable=self.alert_var,
            bg='#203040',
            fg='white',
            font=('TkDefaultFont', 12, 'bold'),
            padx=10,
            pady=6,
        )
        self.alert_label.pack(fill=tk.X)

        self.video_label = tk.Label(self.root, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.video_label.bind('<Button-1>', self._on_video_click)

        log_frame = ttk.Frame(self.root, padding=(8, 4, 8, 8))
        log_frame.pack(fill=tk.BOTH)
        ttk.Label(log_frame, text='Alert log').pack(anchor='w')
        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    def _browse_weights(self) -> None:
        path = filedialog.askopenfilename(title='Select weights', filetypes=[('PyTorch Weights', '*.pt'), ('All files', '*.*')])
        if path:
            self.weights_var.set(path)

    def _browse_source(self) -> None:
        path = filedialog.askopenfilename(title='Select source video', filetypes=[('Video files', '*.mp4 *.avi *.mov *.mkv'), ('All files', '*.*')])
        if path:
            self.source_var.set(path)

    def _browse_alert_audio(self) -> None:
        path = filedialog.askopenfilename(title='Select alert audio', filetypes=[('Audio files', '*.mp3 *.wav *.ogg *.m4a'), ('All files', '*.*')])
        if path:
            self.alert_audio_var.set(path)

    def _resolve_path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.repo_root / p)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def _log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + '\n')
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _trigger_alert(self, frame_index: int) -> None:
        now = time.strftime('%H:%M:%S')
        message = f'[{now}] ALERT: tau_la touched red line at frame {frame_index}'
        self._log(message)
        self.alert_var.set('ALERT: co tau la di qua')
        self.alert_label.configure(bg='#b00020', fg='white')
        self._start_alert_audio()

    def _clear_alert(self) -> None:
        if self.running:
            self.alert_var.set('Monitoring...')
        else:
            self.alert_var.set('No alert')
        self.alert_label.configure(bg='#203040', fg='white')

    def _stop_tts(self) -> None:
        if self.tts_process is not None and self.tts_process.poll() is None:
            try:
                self.tts_process.terminate()
            except Exception:
                pass
        self.tts_process = None

    def _speak_vietnamese_alert(self) -> None:
        self._stop_tts()
        commands = [
            ['spd-say', '-l', 'vi', 'có tàu lạ đi qua'],
            ['espeak-ng', '-v', 'vi', 'có tàu lạ đi qua'],
            ['espeak', '-v', 'vi', 'có tàu lạ đi qua'],
        ]
        for cmd in commands:
            try:
                self.tts_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                continue
        self.root.bell()

    def _start_alert_audio(self) -> None:
        if self.tts_process is not None and self.tts_process.poll() is None:
            return

        audio_path = self._resolve_path(self.alert_audio_var.get())
        if audio_path.exists():
            loop_commands = [
                ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', '-loop', '0', str(audio_path)],
                ['mpg123', '--loop', '-1', str(audio_path)],
                ['mpv', '--no-video', '--loop-file=inf', str(audio_path)],
                ['mplayer', '-loop', '0', str(audio_path)],
            ]
            for cmd in loop_commands:
                try:
                    self.tts_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except Exception:
                    continue

        # Fallback if no player/audio file is available.
        self._speak_vietnamese_alert()

    def enable_pick_line_mode(self) -> None:
        self.pick_line_mode = True
        self.pick_points_px = []
        self._set_status('Pick mode: click 2 points on video frame')

    def _on_video_click(self, event: tk.Event) -> None:
        if not self.pick_line_mode or self.frame_w <= 0 or self.frame_h <= 0:
            return
        if self.display_w <= 0 or self.display_h <= 0:
            return

        widget_w = self.video_label.winfo_width()
        widget_h = self.video_label.winfo_height()
        offset_x = max((widget_w - self.display_w) // 2, 0)
        offset_y = max((widget_h - self.display_h) // 2, 0)
        px = event.x - offset_x
        py = event.y - offset_y

        if px < 0 or py < 0 or px >= self.display_w or py >= self.display_h:
            return

        self.pick_points_px.append((int(px), int(py)))
        if len(self.pick_points_px) == 1:
            self._set_status('Pick mode: first point set, click second point')
            return

        (x1p, y1p), (x2p, y2p) = self.pick_points_px
        x1_src = x1p * (self.frame_w / float(self.display_w))
        y1_src = y1p * (self.frame_h / float(self.display_h))
        x2_src = x2p * (self.frame_w / float(self.display_w))
        y2_src = y2p * (self.frame_h / float(self.display_h))

        x1n = x1_src / float(self.frame_w)
        y1n = y1_src / float(self.frame_h)
        x2n = x2_src / float(self.frame_w)
        y2n = y2_src / float(self.frame_h)
        self.line_var.set(f'{x1n:.4f},{y1n:.4f},{x2n:.4f},{y2n:.4f}')

        self.x1n, self.y1n, self.x2n, self.y2n = x1n, y1n, x2n, y2n
        self.x1 = int(self.x1n * self.frame_w)
        self.y1 = int(self.y1n * self.frame_h)
        self.x2 = int(self.x2n * self.frame_w)
        self.y2 = int(self.y2n * self.frame_h)

        self.pick_line_mode = False
        self.pick_points_px = []
        self._set_status('Line updated from clicks')

    def _load_model(self) -> None:
        weights = self._resolve_path(self.weights_var.get())
        if not weights.exists():
            raise FileNotFoundError(f'Weights not found: {weights}')

        self._set_status('Loading model...')
        model = torch.hub.load(str(self.repo_root / 'yolov5'), 'custom', path=str(weights), source='local')
        model.conf = float(self.conf_var.get())
        model.iou = float(self.iou_var.get())
        model.max_det = 1000
        model.to(self.device_var.get())

        names = model.names
        if isinstance(names, dict):
            name_to_id = {v: k for k, v in names.items()}
        else:
            name_to_id = {v: i for i, v in enumerate(names)}

        target_name = self.target_class_var.get().strip()
        if target_name not in name_to_id:
            raise ValueError(f"Target class '{target_name}' not in model names: {list(name_to_id.keys())}")

        self.model = model
        self.model_names = names
        self.target_cls_id = int(name_to_id[target_name])

    def start(self) -> None:
        if self.running:
            return

        try:
            self._load_model()

            source = self._resolve_path(self.source_var.get())
            if not source.exists():
                raise FileNotFoundError(f'Video source not found: {source}')

            self.cap = cv2.VideoCapture(str(source))
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open video: {source}')

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_w = width
            self.frame_h = height
            fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                fps = 25.0
            self.fps = fps

            self.x1n, self.y1n, self.x2n, self.y2n = parse_point_list(self.line_var.get())
            self.x1 = int(self.x1n * width)
            self.y1 = int(self.y1n * height)
            self.x2 = int(self.x2n * width)
            self.y2 = int(self.y2n * height)

            out_path = self._resolve_path(self.output_var.get())
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height),
            )

            self.frame_index = 0
            self.last_alert_frame = -10**9
            self.tracks = {}
            self.next_track_id = 1
            self.target_touching_prev = False
            self.running = True
            self.alert_var.set('Monitoring...')
            self._set_status('Running inference...')
            self._loop()
        except Exception as exc:
            messagebox.showerror('Start error', str(exc))
            self.stop()

    def stop(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self._stop_tts()
        self._set_status('Stopped')
        self._clear_alert()

    def _loop(self) -> None:
        if not self.running:
            return
        if self.cap is None or self.model is None or self.target_cls_id is None:
            self.stop()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            self._set_status('Finished')
            return

        results = self.model(frame, size=int(self.imgsz_var.get()))
        preds = results.xyxy[0].detach().cpu().numpy()

        detections: List[Tuple[int, Tuple[float, float, float, float], float]] = []
        for row in preds:
            x1b, y1b, x2b, y2b, conf, cls_id = row.tolist()
            detections.append((int(cls_id), (x1b, y1b, x2b, y2b), float(conf)))

        matches = match_detections_to_tracks(self.tracks, detections, float(self.max_match_dist_var.get()))
        updated_tracks: Dict[int, Track] = {}

        matched_track_ids = set(matches.values())
        for tr_id, tr in self.tracks.items():
            if tr_id not in matched_track_ids:
                tr.missed += 1
                if tr.missed <= int(self.max_missed_var.get()):
                    updated_tracks[tr_id] = tr

        names = self.model_names
        target_touching_now = False

        for det_idx, (cls_id, xyxy, _) in enumerate(detections):
            c = center_of_box(xyxy)
            side = line_side(c[0], c[1], self.x1, self.y1, self.x2, self.y2)
            is_target = cls_id == self.target_cls_id
            touching = box_touches_line(xyxy, (self.x1, self.y1), (self.x2, self.y2))
            if is_target and touching:
                target_touching_now = True

            if det_idx in matches:
                tr_id = matches[det_idx]
                tr = self.tracks[tr_id]

                tr.center = c
                tr.last_side = side
                tr.cls_id = cls_id
                tr.missed = 0
                updated_tracks[tr_id] = tr
            else:
                tr = Track(track_id=self.next_track_id, cls_id=cls_id, center=c, last_side=side)
                updated_tracks[self.next_track_id] = tr
                self.next_track_id += 1

            color = (255, 128, 0) if cls_id == self.target_cls_id else (255, 0, 0)
            x1b, y1b, x2b, y2b = (int(v) for v in xyxy)
            cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), color, 1)
            label_name = names[cls_id] if isinstance(names, list) else names[int(cls_id)]
            cv2.putText(
                frame,
                label_name,
                (x1b, max(18, y1b - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        self.tracks = updated_tracks

        if target_touching_now and not self.target_touching_prev:
            self._trigger_alert(self.frame_index)
        elif not target_touching_now and self.target_touching_prev:
            self._stop_tts()
            self._clear_alert()
        self.target_touching_prev = target_touching_now

        cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255), 1)

        if target_touching_now:
            cv2.putText(
                frame,
                'ALERT: co tau la di qua',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        if self.writer is not None:
            self.writer.write(frame)

        widget_w = max(1, self.video_label.winfo_width())
        widget_h = max(1, self.video_label.winfo_height())
        scale = min(widget_w / float(self.frame_w), widget_h / float(self.frame_h))
        scale = max(scale, 1e-6)
        draw_w = max(1, int(self.frame_w * scale))
        draw_h = max(1, int(self.frame_h * scale))

        if draw_w != self.frame_w or draw_h != self.frame_h:
            display_frame = cv2.resize(frame, (draw_w, draw_h), interpolation=cv2.INTER_LINEAR)
        else:
            display_frame = frame

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.display_w, self.display_h = draw_w, draw_h
        self.photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=self.photo)

        self.frame_index += 1
        self.root.after(1, self._loop)

    def _on_close(self) -> None:
        self.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    root.geometry('1100x760')
    app = LineAlertApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
