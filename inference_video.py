from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_latest_best(repo_root: Path) -> Path:
    candidates = sorted(
        repo_root.glob('runs/train/**/weights/best.pt'),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError('Khong tim thay runs/train/**/weights/best.pt')
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Chay inference YOLOv5 cho video voi model da train.'
    )
    parser.add_argument(
        '--weights',
        type=Path,
        default=None,
        help='Duong dan den file weights .pt. Mac dinh: best.pt moi nhat trong runs/train/**/weights/',
    )
    parser.add_argument(
        '--source',
        type=Path,
        default=Path('data/origin_video.mp4'),
        help='Video dau vao. Mac dinh: data/origin_video.mp4',
    )
    parser.add_argument('--imgsz', type=int, default=640, help='Kich thuoc anh suy luan')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Nguong confidence')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='Nguong IoU cho NMS')
    parser.add_argument('--device', default='0', help="Thiet bi, vd: '0' hoac 'cpu'")
    parser.add_argument('--project', default='runs/detect', help='Thu muc output')
    parser.add_argument('--name', default='origin_video_infer', help='Ten run output')
    parser.add_argument('--line-thickness', type=int, default=1, help='Do day khung bbox')
    parser.add_argument(
        '--show-conf',
        action='store_true',
        help='Hien thi confidence tren nhan (mac dinh an de nhan gon hon)',
    )
    parser.add_argument('--save-txt', action='store_true', help='Luu nhan du doan ra file txt')
    parser.add_argument('--save-conf', action='store_true', help='Luu confidence trong file txt')
    parser.add_argument('--exist-ok', action='store_true', help='Cho phep ghi de thu muc output')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    detect_py = repo_root / 'yolov5' / 'detect.py'

    if not detect_py.exists():
        raise FileNotFoundError(f'Khong tim thay file: {detect_py}')

    source_path = args.source if args.source.is_absolute() else (repo_root / args.source)
    if not source_path.exists():
        raise FileNotFoundError(f'Khong tim thay video dau vao: {source_path}')

    if args.weights is None:
        weights_path = find_latest_best(repo_root)
    else:
        weights_path = args.weights if args.weights.is_absolute() else (repo_root / args.weights)

    if not weights_path.exists():
        raise FileNotFoundError(f'Khong tim thay weights: {weights_path}')

    command = [
        sys.executable,
        str(detect_py),
        '--weights',
        str(weights_path),
        '--source',
        str(source_path),
        '--imgsz',
        str(args.imgsz),
        '--conf-thres',
        str(args.conf_thres),
        '--iou-thres',
        str(args.iou_thres),
        '--device',
        str(args.device),
        '--project',
        str(repo_root / args.project),
        '--name',
        args.name,
        '--line-thickness',
        str(args.line_thickness),
    ]

    # Hide confidence by default so label boxes are shorter and overlap less.
    if not args.show_conf:
        command.append('--hide-conf')

    if args.save_txt:
        command.append('--save-txt')
    if args.save_conf:
        command.append('--save-conf')
    if args.exist_ok:
        command.append('--exist-ok')

    print('Running command:')
    print(' '.join(command))
    subprocess.run(command, cwd=repo_root, check=True)

    output_dir = repo_root / args.project / args.name
    print(f'Inference xong. Ket qua tai: {output_dir}')


if __name__ == '__main__':
    main()
