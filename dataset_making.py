#!/usr/bin/env python3
"""
Build label-based dataset from sharded image directories.

Source structure:
cvpr_dfd/
  shard_0/
    images/
    labels.csv
  ...
  shard_5/

Target structure:
dataset/
  dataset_0/
  dataset_1/
"""

import csv
import shutil
from pathlib import Path


SOURCE_ROOT = Path("cvpr_dfd")
DEST_ROOT = Path("dataset")

DATASET_0 = DEST_ROOT / "dataset_0"
DATASET_1 = DEST_ROOT / "dataset_1"

SHARD_NAMES = [f"shard_{i}" for i in range(6)]


def ensure_directories() -> None:
    DATASET_0.mkdir(parents=True, exist_ok=True)
    DATASET_1.mkdir(parents=True, exist_ok=True)


def resolve_destination(label: str) -> Path:
    if label == "0":
        return DATASET_0
    if label == "1":
        return DATASET_1
    raise ValueError(f"Invalid label encountered: {label}")


def copy_image_safe(src: Path, dst_dir: Path, shard_name: str) -> None:
    if not src.exists():
        print(f"[WARN] Missing image: {src}")
        return

    dst_path = dst_dir / src.name

    # Avoid overwriting files with same name from different shards
    if dst_path.exists():
        dst_path = dst_dir / f"{shard_name}_{src.name}"

    shutil.copy2(src, dst_path)


def process_shard(shard_name: str) -> None:
    shard_path = SOURCE_ROOT / shard_name
    images_dir = shard_path / "images"
    labels_csv = shard_path / "labels.csv"

    if not labels_csv.exists():
        print(f"[WARN] labels.csv not found in {shard_name}")
        return

    with labels_csv.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if len(row) < 2:
                continue

            image_name, label = row[0].strip(), row[1].strip()

            try:
                dest_dir = resolve_destination(label)
            except ValueError as exc:
                print(f"[WARN] {exc}")
                continue

            src_image = images_dir / image_name
            copy_image_safe(src_image, dest_dir, shard_name)


def main() -> None:
    ensure_directories()

    for shard_name in SHARD_NAMES:
        print(f"[INFO] Processing {shard_name}")
        process_shard(shard_name)

    print("[DONE] Dataset creation completed successfully.")


if __name__ == "__main__":
    main()
