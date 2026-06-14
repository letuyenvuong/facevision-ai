"""
Download và tổ chức Olivetti Faces (AT&T) vào experiments/dataset/.

Olivetti Faces là dataset công khai, sẵn trong sklearn:
  - 40 người × 10 ảnh = 400 ảnh frontal
  - 64×64 pixel grayscale
  - Không cần cài thêm, không cần internet (sklearn tải tự động ~1.4MB)

Kết quả:
  experiments/dataset/person_01/ ... person_40/
  Mỗi người: frontal_001.jpg ... frontal_010.jpg
  Cập nhật metadata.csv với các ảnh mới

Sử dụng:
  python experiments/download_olivetti.py
  python experiments/download_olivetti.py --n_persons 9   # chỉ lấy 9 người đầu
"""
from __future__ import annotations

import os
import sys
import csv
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
META_FILE   = DATASET_DIR / "metadata.csv"


def download_and_save(n_persons: int = 9):
    print("Downloading Olivetti Faces via sklearn (~1.4MB)...")
    data = fetch_olivetti_faces(shuffle=False)
    # data.images: (400, 64, 64) float32 [0,1]
    # data.target: (400,) — person index 0..39
    print(f"  {data.images.shape[0]} images, {len(set(data.target))} persons loaded")

    persons_to_use = min(n_persons, 40)
    print(f"  Saving {persons_to_use} persons as person_01..person_{persons_to_use:02d}")

    new_rows = []
    existing_files = _load_existing_files()

    for person_idx in range(persons_to_use):
        folder_name = f"person_{person_idx+1:02d}"
        person_dir  = DATASET_DIR / folder_name
        person_dir.mkdir(parents=True, exist_ok=True)

        imgs_for_person = data.images[data.target == person_idx]

        for i, img_f32 in enumerate(imgs_for_person, start=1):
            fname = f"frontal_{i:03d}.jpg"
            fpath = person_dir / fname

            # Lưu file nếu chưa tồn tại
            if not fpath.exists():
                img_u8 = (img_f32 * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(
                    cv2.resize(img_u8, (128, 128), interpolation=cv2.INTER_LINEAR),
                    cv2.COLOR_GRAY2BGR
                )
                cv2.imwrite(str(fpath), img_bgr)

            # Thêm vào metadata nếu chưa có (độc lập với việc file tồn tại)
            if (folder_name, fname) not in existing_files:
                new_rows.append({
                    "person_id":  folder_name,
                    "image_file": fname,
                    "session":    "1",
                    "pose":       "0deg",
                    "lighting":   "even",
                    "occlusion":  "none",
                    "camera":     "olivetti",
                    "resolution": "128x128",
                    "note":       "olivetti_public",
                })

        print(f"  {folder_name}: {len(imgs_for_person)} images saved")

    # Ghi vào metadata.csv
    if new_rows:
        need_header = not META_FILE.exists() or META_FILE.stat().st_size < 5
        with open(META_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "person_id", "image_file", "session",
                "pose", "lighting", "occlusion", "camera", "resolution", "note"
            ])
            if need_header:
                writer.writeheader()
            writer.writerows(new_rows)
        print(f"\nAppended {len(new_rows)} rows to metadata.csv")

    _print_summary()


def _load_existing_files():
    """Returns set of (person_id, image_file) tuples already in metadata."""
    if not META_FILE.exists():
        return set()
    with open(META_FILE, newline="", encoding="utf-8") as f:
        return {(row["person_id"], row["image_file"]) for row in csv.DictReader(f)}


def _print_summary():
    from collections import Counter
    if not META_FILE.exists():
        return
    with open(META_FILE, newline="", encoding="utf-8") as f:
        persons = Counter(row["person_id"] for row in csv.DictReader(f))
    total = sum(persons.values())
    print(f"\n=== Dataset Summary ===")
    print(f"Total: {total} images, {len(persons)} persons")
    for pid in sorted(persons):
        print(f"  {pid}: {persons[pid]} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Olivetti Faces to dataset")
    parser.add_argument("--n_persons", type=int, default=9,
                        help="Number of Olivetti persons to download (default: 9)")
    args = parser.parse_args()
    download_and_save(n_persons=args.n_persons)
