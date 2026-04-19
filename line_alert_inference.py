from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch


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
    # 2D cross product sign: >0 one side, <0 other side, ~0 on line.
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def center_of_box(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def match_detections_to_tracks(
    tracks: Dict[int, Track],
    detections: List[Tuple[int, Tuple[float, float, float, float], float]],
    max_match_distance: float,
) -> Dict[int, int]:
    # Returns mapping: det_idx -> track_id
    unmatched_tracks = set(tracks.keys())
    unmatched_dets = set(range(len(detections)))
    matches: Dict[int, int] = {}

    # Greedy matching by nearest distance, class-consistent.
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


def play_alert_sound() -> None:
    # Terminal bell as fallback, plus optional Linux sound players.
    print('\a', end='', flush=True)
    for cmd in (['paplay', '/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga'], ['aplay', '/usr/share/sounds/alsa/Front_Center.wav']):
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            break
        except Exception:
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Inference video with red-line crossing alert: tau_la crossing triggers alert, tau_tvien ignored.'
    )
    parser.add_argument('--weights', type=Path, default=Path('runs/train/tau_yolov5n_live/weights/best.pt'))
    parser.add_argument('--source', type=Path, default=Path('data/origin_video.mp4'))
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--line', default='0.56,0.49,0.95,0.57', help='Red line in normalized coords x1,y1,x2,y2 (0..1)')
    parser.add_argument('--line-thickness', type=int, default=3)
    parser.add_argument('--save-path', type=Path, default=Path('runs/detect/origin_video_line_alert.mp4'))
    parser.add_argument('--view', action='store_true', help='Show live window')
    parser.add_argument('--alert-cooldown-frames', type=int, default=30)
    parser.add_argument('--max-track-missed', type=int, default=15)
    parser.add_argument('--max-match-distance', type=float, default=90.0)
    parser.add_argument('--target-class', default='tau_la', help='Class name that should trigger alert when crossing line')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    weights = args.weights if args.weights.is_absolute() else repo_root / args.weights
    source = args.source if args.source.is_absolute() else repo_root / args.source
    save_path = args.save_path if args.save_path.is_absolute() else repo_root / args.save_path

    if not weights.exists():
        raise FileNotFoundError(f'Weights not found: {weights}')
    if not source.exists():
        raise FileNotFoundError(f'Video source not found: {source}')

    device = args.device
    model = torch.hub.load(str(repo_root / 'yolov5'), 'custom', path=str(weights), source='local')
    model.conf = args.conf_thres
    model.iou = args.iou_thres
    model.max_det = 1000
    model.to(device)

    names = model.names
    if isinstance(names, dict):
        name_to_id = {v: k for k, v in names.items()}
    else:
        name_to_id = {v: i for i, v in enumerate(names)}

    if args.target_class not in name_to_id:
        raise ValueError(f"target class '{args.target_class}' not found in model names: {list(name_to_id.keys())}")
    target_cls_id = int(name_to_id[args.target_class])

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {source}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    x1n, y1n, x2n, y2n = parse_point_list(args.line)
    x1 = int(x1n * width)
    y1 = int(y1n * height)
    x2 = int(x2n * width)
    y2 = int(y2n * height)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    tracks: Dict[int, Track] = {}
    next_track_id = 1
    frame_index = 0
    last_alert_frame = -10**9

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, size=args.imgsz)
        preds = results.xyxy[0].detach().cpu().numpy()  # x1,y1,x2,y2,conf,cls

        detections: List[Tuple[int, Tuple[float, float, float, float], float]] = []
        for row in preds:
            x1b, y1b, x2b, y2b, conf, cls_id = row.tolist()
            detections.append((int(cls_id), (x1b, y1b, x2b, y2b), float(conf)))

        matches = match_detections_to_tracks(tracks, detections, args.max_match_distance)
        updated_tracks: Dict[int, Track] = {}

        matched_track_ids = set(matches.values())
        for tr_id, tr in tracks.items():
            if tr_id not in matched_track_ids:
                tr.missed += 1
                if tr.missed <= args.max_track_missed:
                    updated_tracks[tr_id] = tr

        for det_idx, (cls_id, xyxy, conf) in enumerate(detections):
            c = center_of_box(xyxy)
            side = line_side(c[0], c[1], x1, y1, x2, y2)

            if det_idx in matches:
                tr_id = matches[det_idx]
                tr = tracks[tr_id]
                crossed = (tr.last_side == 0) or (side == 0) or ((tr.last_side > 0) != (side > 0))

                if crossed and cls_id == target_cls_id and (frame_index - last_alert_frame) >= args.alert_cooldown_frames:
                    tr.alerted = True
                    last_alert_frame = frame_index
                    print(f'[ALERT] {args.target_class} crossed red line at frame {frame_index}')
                    play_alert_sound()

                tr.center = c
                tr.last_side = side
                tr.cls_id = cls_id
                tr.missed = 0
                updated_tracks[tr_id] = tr
            else:
                tr = Track(track_id=next_track_id, cls_id=cls_id, center=c, last_side=side)
                updated_tracks[next_track_id] = tr
                next_track_id += 1

            color = (255, 128, 0) if cls_id == target_cls_id else (255, 0, 0)
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

        tracks = updated_tracks

        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), args.line_thickness)
        cv2.putText(
            frame,
            'RED LINE',
            (min(x1, x2) + 10, min(y1, y2) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if (frame_index - last_alert_frame) <= int(fps):
            cv2.putText(
                frame,
                'ALERT: tau_la crossed line',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        writer.write(frame)

        if args.view:
            cv2.imshow('line_alert_inference', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_index += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f'Done. Output video saved at: {save_path}')


if __name__ == '__main__':
    main()
