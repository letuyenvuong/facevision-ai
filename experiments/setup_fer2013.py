"""
Setup FER2013 dataset cho E3 — Emotion Recognition Evaluation.

Hướng dẫn tải FER2013 từ Kaggle:
  1. Vào: https://www.kaggle.com/datasets/msambare/fer2013
  2. Click "Download" → tải file zip (~63MB)
  3. Giải nén → bạn sẽ có thư mục chứa: train/ và test/
  4. Chạy script này với đường dẫn đến thư mục đó:

     python experiments/setup_fer2013.py --fer2013_dir "D:/Downloads/archive"

Script sẽ copy N ảnh mỗi lớp từ test/ sang:
  experiments/dataset/emotion/angry/
  experiments/dataset/emotion/happy/
  ... (7 lớp)

Không cần cài thêm gì. Chỉ dùng shutil + os.
"""
from __future__ import annotations

import os
import sys
import shutil
import argparse
from pathlib import Path
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

DATASET_DIR  = Path(__file__).resolve().parent / "dataset"
EMOTION_DIR  = DATASET_DIR / "emotion"


def setup(fer2013_dir: Path, n_per_class: int = 100, split: str = "test"):
    fer2013_dir = Path(fer2013_dir)

    # Tự động detect cấu trúc folder
    split_dir = fer2013_dir / split
    if not split_dir.exists():
        # Thử tìm trực tiếp trong fer2013_dir
        if (fer2013_dir / "angry").exists():
            split_dir = fer2013_dir
        else:
            # Scan đệ quy tìm folder chứa 'angry'
            for candidate in fer2013_dir.rglob("angry"):
                if candidate.is_dir():
                    split_dir = candidate.parent
                    break

    if not split_dir.exists():
        print(f"[ERROR] Không tìm thấy thư mục '{split}' trong: {fer2013_dir}")
        print("  Cấu trúc dự kiến:")
        print(f"    {fer2013_dir}/{split}/angry/*.jpg")
        print(f"    {fer2013_dir}/{split}/happy/*.jpg  ...")
        sys.exit(1)

    print(f"Source: {split_dir}")
    print(f"Target: {EMOTION_DIR}")
    print(f"Copying up to {n_per_class} images per class\n")

    total_copied = 0
    counts = Counter()

    for label in EMOTION_LABELS:
        src_dir = split_dir / label
        dst_dir = EMOTION_DIR / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Kể ảnh đã có
        existing = len(list(dst_dir.glob("*.jpg")) + list(dst_dir.glob("*.png")))

        if not src_dir.exists():
            print(f"  {label:10s}: thư mục không tồn tại trong source — bỏ qua")
            continue

        # Lấy danh sách ảnh nguồn (ưu tiên jpg rồi png)
        src_imgs = sorted(list(src_dir.glob("*.jpg")) +
                          list(src_dir.glob("*.png")) +
                          list(src_dir.glob("*.jpeg")))

        need = n_per_class - existing
        if need <= 0:
            print(f"  {label:10s}: đã có {existing} ảnh — bỏ qua")
            counts[label] = existing
            continue

        to_copy = src_imgs[:need]
        for i, src_path in enumerate(to_copy, start=existing + 1):
            dst_path = dst_dir / f"{label}_{i:04d}{src_path.suffix}"
            shutil.copy2(src_path, dst_path)

        copied = len(to_copy)
        total_copied += copied
        counts[label] = existing + copied
        print(f"  {label:10s}: {copied} ảnh copied ({existing + copied} total)")

    print(f"\nDone! {total_copied} ảnh copied.")
    print("\n=== Emotion Dataset Summary ===")
    for label in EMOTION_LABELS:
        bar = "#" * (counts[label] // 5)
        print(f"  {label:10s}: {counts[label]:4d}  {bar}")

    total = sum(counts.values())
    print(f"\nTotal: {total} ảnh, {len([l for l in EMOTION_LABELS if counts[l] > 0])} classes")
    if total > 0:
        print("\nSan sang chay: python experiments/run_E3_emotion.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup FER2013 for E3 experiment")
    parser.add_argument("--fer2013_dir", required=True,
                        help="Duong dan den thu muc FER2013 da giai nen (chua train/ va test/)")
    parser.add_argument("--n_per_class", type=int, default=100,
                        help="So anh moi lop (default: 100, tong ~700 anh)")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                        help="Dung split nao (default: test)")
    args = parser.parse_args()
    setup(Path(args.fer2013_dir), n_per_class=args.n_per_class, split=args.split)
