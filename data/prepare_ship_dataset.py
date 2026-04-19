from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_NAMES = ['tau_tvien', 'tau_la']


def numeric_key(path: Path) -> tuple[int, str]:
    digits = ''.join(ch for ch in path.stem if ch.isdigit())
    return (int(digits) if digits else 0, path.stem)


def load_labels(label_path: Path) -> list[list[float]]:
    labels: list[list[float]] = []
    for line in label_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'Invalid label line in {label_path}: {line}')
        labels.append([float(value) for value in parts])
    return labels


def save_labels(label_path: Path, labels: list[list[float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cls, x, y, w, h in labels:
        lines.append(f'{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
    label_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def copy_sample(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path) -> None:
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)


def augment_color(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image.astype(np.float32)

    alpha = rng.uniform(0.85, 1.15)
    beta = rng.uniform(-18.0, 18.0)
    out = np.clip(out * alpha + beta, 0, 255)

    hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-8.0, 8.0)) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.85, 1.20), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.85, 1.20), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.random() < 0.45:
        kernel = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (kernel, kernel), 0)

    if rng.random() < 0.35:
        noise = rng.normalvariate(0.0, 6.0)
        gaussian = rng.normalvariate(0.0, 1.0)
        noise_map = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(noise, 6.0, out.shape)
        out = np.clip(out.astype(np.float32) + noise_map * gaussian, 0, 255).astype(np.uint8)

    if rng.random() < 0.35:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(55, 90)]
        success, buffer = cv2.imencode('.jpg', out, encode_param)
        if success:
            out = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    return out


def transform_labels(labels: list[list[float]], matrix: np.ndarray, width: int, height: int) -> list[list[float]]:
    transformed: list[list[float]] = []

    for cls, x, y, w, h in labels:
        x1 = (x - w / 2.0) * width
        y1 = (y - h / 2.0) * height
        x2 = (x + w / 2.0) * width
        y2 = (y + h / 2.0) * height

        corners = np.array(
            [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
            dtype=np.float32,
        )
        warped = corners @ matrix.T
        xs = warped[:, 0]
        ys = warped[:, 1]

        nx1 = float(np.clip(xs.min(), 0, width - 1))
        ny1 = float(np.clip(ys.min(), 0, height - 1))
        nx2 = float(np.clip(xs.max(), 0, width - 1))
        ny2 = float(np.clip(ys.max(), 0, height - 1))

        nw = nx2 - nx1
        nh = ny2 - ny1
        if nw < 2.0 or nh < 2.0:
            continue

        cx = (nx1 + nx2) / 2.0 / width
        cy = (ny1 + ny2) / 2.0 / height
        nw /= width
        nh /= height

        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            continue

        transformed.append([cls, cx, cy, nw, nh])

    return transformed


def augment_affine(image: np.ndarray, labels: list[list[float]], rng: random.Random) -> tuple[np.ndarray, list[list[float]]]:
    height, width = image.shape[:2]
    angle = rng.uniform(-4.0, 4.0)
    scale = rng.uniform(0.96, 1.04)
    tx = rng.uniform(-0.04, 0.04) * width
    ty = rng.uniform(-0.04, 0.04) * height

    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[:, 2] += [tx, ty]

    warped = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    new_labels = transform_labels(labels, matrix, width, height)
    return warped, new_labels


def prepare_dataset(dataset_root: Path, output_root: Path, train_ratio: float, augment_copies: int, seed: int) -> None:
    image_dir = dataset_root / 'image'
    label_dir = dataset_root / 'label'

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=numeric_key)
    labels = {p.stem: p for p in label_dir.glob('*.txt') if p.name != 'classes.txt'}

    missing = [p.stem for p in images if p.stem not in labels]
    if missing:
        raise FileNotFoundError(f'Missing labels for images: {missing[:10]}')

    split_index = max(1, min(len(images) - 1, int(len(images) * train_ratio)))
    train_images = images[:split_index]
    val_images = images[split_index:]

    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_root / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_root / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_root / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    for image_path in train_images:
        label_path = labels[image_path.stem]
        dst_img = output_root / 'images' / 'train' / image_path.name
        dst_lbl = output_root / 'labels' / 'train' / f'{image_path.stem}.txt'
        copy_sample(image_path, label_path, dst_img, dst_lbl)

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f'Unable to read image: {image_path}')
        original_labels = load_labels(label_path)

        for copy_index in range(augment_copies):
            if copy_index % 2 == 0:
                aug_image = augment_color(image, rng)
                aug_labels = original_labels
                suffix = f'augc{copy_index + 1}'
            else:
                aug_image, aug_labels = augment_affine(image, original_labels, rng)
                if not aug_labels:
                    aug_image = augment_color(image, rng)
                    aug_labels = original_labels
                suffix = f'augg{copy_index + 1}'

            aug_img_name = f'{image_path.stem}_{suffix}{image_path.suffix}'
            aug_lbl_name = f'{image_path.stem}_{suffix}.txt'
            cv2.imwrite(str(output_root / 'images' / 'train' / aug_img_name), aug_image)
            save_labels(output_root / 'labels' / 'train' / aug_lbl_name, aug_labels)

    for image_path in val_images:
        label_path = labels[image_path.stem]
        dst_img = output_root / 'images' / 'val' / image_path.name
        dst_lbl = output_root / 'labels' / 'val' / f'{image_path.stem}.txt'
        copy_sample(image_path, label_path, dst_img, dst_lbl)

    yaml_path = dataset_root / 'dataset.yaml'
    yaml_text = (
        f'path: {output_root.as_posix()}\n'
        'train: images/train\n'
        'val: images/val\n'
        'nc: 2\n'
        'names:\n'
        f'  0: {CLASS_NAMES[0]}\n'
        f'  1: {CLASS_NAMES[1]}\n'
    )
    yaml_path.write_text(yaml_text, encoding='utf-8')

    print(f'Prepared dataset at: {output_root}')
    print(f'Train images: {len(train_images)}')
    print(f'Validation images: {len(val_images)}')
    print(f'Augment copies per train image: {augment_copies}')
    print(f'YAML written to: {yaml_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare YOLO dataset with train/val split and offline augmentation.')
    parser.add_argument('--dataset-root', type=Path, default=Path('/home/abuntu/Documents/github_Temp/unknown-ship-detection-ai/dataset'))
    parser.add_argument('--output-root', type=Path, default=Path('/home/abuntu/Documents/github_Temp/unknown-ship-detection-ai/temp/yolo_ship_dataset'))
    parser.add_argument('--train-ratio', type=float, default=0.85)
    parser.add_argument('--augment-copies', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(args.dataset_root, args.output_root, args.train_ratio, args.augment_copies, args.seed)


if __name__ == '__main__':
    main()
