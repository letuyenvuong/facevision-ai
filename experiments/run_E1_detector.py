"""
E1 — Detector Comparison: Haar Cascade vs MTCNN
================================================
Đo Recall, latency (ms/frame), FPS của 2 detector trên từng điều kiện.

Yêu cầu:
  - Dataset ảnh trong experiments/dataset/<person_XX>/<condition>_NNN.jpg
  - opencv-python (luôn có)
  - mtcnn  (cần cài: pip install mtcnn tensorflow)

Chạy:
  python experiments/run_E1_detector.py

Output:
  experiments/results/E1_detector/E1_results.csv
  experiments/results/E1_detector/E1_recall_by_condition.png
  experiments/results/E1_detector/E1_latency_boxplot.png
"""
from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import time
import glob
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ── Thêm project root vào path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Cấu hình ────────────────────────────────────────────────────────────────
DATASET_DIR = PROJECT_ROOT / "experiments" / "dataset"
RESULT_DIR  = PROJECT_ROOT / "experiments" / "results" / "E1_detector"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Điều kiện ảnh (prefix tên file)
CONDITIONS = ["frontal", "pose15", "pose30", "dark", "occluded"]

MIN_FACE_SIZE = 50   # pixel

# ── Haar detector ────────────────────────────────────────────────────────────

def make_haar():
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(haar_path)
    if clf.empty():
        raise RuntimeError(f"Cannot load Haar cascade: {haar_path}")
    return clf


def detect_haar(clf, frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                  minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces] if len(faces) else []


# ── MTCNN detector ───────────────────────────────────────────────────────────

def make_mtcnn():
    try:
        from mtcnn import MTCNN
        return MTCNN()
    except ImportError:
        print("[E1] MTCNN không khả dụng — pip install mtcnn tensorflow")
        return None


def detect_mtcnn(det, frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
    if det is None:
        return []
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    detections = det.detect_faces(rgb)
    results = []
    for d in detections:
        if d["confidence"] < 0.85:
            continue
        bx, by, bw, bh = d["box"]
        x, y = max(0, bx * 2), max(0, by * 2)
        fw, fh = bw * 2, bh * 2
        if fw >= MIN_FACE_SIZE and fh >= MIN_FACE_SIZE:
            results.append((x, y, fw, fh))
    return results


# ── Tìm ảnh theo điều kiện ──────────────────────────────────────────────────

def _person_dirs() -> List[Path]:
    """Lấy tất cả thư mục con chứa ảnh (bao gồm TuyenVuong, person_XX, ...)."""
    exclude = {"emotion"}
    return sorted(
        p for p in DATASET_DIR.iterdir()
        if p.is_dir() and p.name not in exclude
    )

def load_image_paths() -> Dict[str, List[Path]]:
    """Trả về {condition: [image_path, ...]}"""
    grouped: Dict[str, List[Path]] = {c: [] for c in CONDITIONS}
    person_dirs = _person_dirs()
    if not person_dirs:
        print(f"[E1] Không tìm thấy thư mục person_XX trong {DATASET_DIR}")
        return grouped
    for pdir in person_dirs:
        for img_path in sorted(pdir.glob("*.jpg")) + sorted(pdir.glob("*.png")):
            stem = img_path.stem.lower()
            for cond in CONDITIONS:
                if stem.startswith(cond):
                    grouped[cond].append(img_path)
                    break
    return grouped


# ── Benchmark một detector ──────────────────────────────────────────────────

def benchmark(detect_fn, images: List[Path]) -> Tuple[int, int, List[float]]:
    """Returns (tp, fn, latencies_ms)"""
    tp, fn, latencies = 0, 0, []
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        t0 = time.perf_counter()
        detections = detect_fn(frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        if len(detections) > 0:
            tp += 1
        else:
            fn += 1
    return tp, fn, latencies


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("E1 — Detector Comparison: Haar vs MTCNN")
    print("=" * 60)

    image_map = load_image_paths()
    total_images = sum(len(v) for v in image_map.values())
    if total_images == 0:
        print(f"\n[E1] Chưa có ảnh dataset. Thu thập ảnh trước:\n"
              f"  {DATASET_DIR}/<person_XX>/<condition>_NNN.jpg\n"
              f"  Điều kiện: {CONDITIONS}\n")
        _generate_demo_data()
        return

    print(f"\nDataset: {total_images} ảnh / {len(_person_dirs())} người")
    for cond, paths in image_map.items():
        print(f"  {cond:10s}: {len(paths)} ảnh")

    haar = make_haar()
    mtcnn = make_mtcnn()
    if mtcnn is None:
        print("\n[E1] Chạy Haar-only (MTCNN bị bỏ qua vì chưa cài)")

    rows = []
    haar_all_lat, mtcnn_all_lat = [], []

    for cond in CONDITIONS:
        imgs = image_map[cond]
        if not imgs:
            continue

        # Haar
        h_tp, h_fn, h_lat = benchmark(lambda f: detect_haar(haar, f), imgs)
        haar_all_lat.extend(h_lat)
        h_recall = h_tp / max(1, h_tp + h_fn)
        h_fps = 1000 / max(1, np.mean(h_lat)) if h_lat else 0

        # MTCNN
        if mtcnn is not None:
            m_tp, m_fn, m_lat = benchmark(lambda f: detect_mtcnn(mtcnn, f), imgs)
            mtcnn_all_lat.extend(m_lat)
            m_recall = m_tp / max(1, m_tp + m_fn)
            m_fps = 1000 / max(1, np.mean(m_lat)) if m_lat else 0
        else:
            m_tp, m_fn, m_lat = 0, 0, []
            m_recall, m_fps = float("nan"), float("nan")

        print(f"\n  [{cond}]")
        print(f"    Haar  — Recall={h_recall:.3f}, lat={np.mean(h_lat):.1f}ms, FPS={h_fps:.1f}")
        if mtcnn is not None:
            print(f"    MTCNN — Recall={m_recall:.3f}, lat={np.mean(m_lat):.1f}ms, FPS={m_fps:.1f}")

        rows.append({
            "condition": cond,
            "n_images": len(imgs),
            "haar_tp": h_tp, "haar_fn": h_fn,
            "haar_recall": round(h_recall, 4),
            "haar_lat_mean_ms": round(float(np.mean(h_lat)) if h_lat else 0, 2),
            "haar_fps": round(h_fps, 1),
            "mtcnn_tp": m_tp, "mtcnn_fn": m_fn,
            "mtcnn_recall": round(m_recall, 4) if not np.isnan(m_recall) else "",
            "mtcnn_lat_mean_ms": round(float(np.mean(m_lat)) if m_lat else 0, 2),
            "mtcnn_fps": round(m_fps, 1) if not np.isnan(m_fps) else "",
        })

    # ── Xuất CSV ────────────────────────────────────────────────────────────
    csv_path = RESULT_DIR / "E1_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {csv_path}")

    # ── Vẽ biểu đồ ──────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _plot_recall(rows, mtcnn is not None)
        _plot_latency(haar_all_lat, mtcnn_all_lat, mtcnn is not None)
        print("Plots saved to", RESULT_DIR)
    except ImportError:
        print("[E1] matplotlib chưa cài — bỏ qua vẽ biểu đồ")

    print("\nE1 XONG.")


def _plot_recall(rows, has_mtcnn: bool):
    import matplotlib.pyplot as plt

    conditions = [r["condition"] for r in rows]
    haar_recall = [float(r["haar_recall"]) for r in rows]
    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2 if has_mtcnn else x, haar_recall, width,
                   label="Haar Cascade", color="#4c72b0")
    if has_mtcnn:
        mtcnn_recall = [float(r["mtcnn_recall"]) if r["mtcnn_recall"] != "" else 0
                        for r in rows]
        ax.bar(x + width/2, mtcnn_recall, width, label="MTCNN", color="#dd8452")

    ax.set_xlabel("Điều kiện")
    ax.set_ylabel("Recall")
    ax.set_title("E1 — Recall theo điều kiện: Haar vs MTCNN")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "E1_recall_by_condition.png", dpi=150)
    plt.close()


def _plot_latency(haar_lat: List[float], mtcnn_lat: List[float], has_mtcnn: bool):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [haar_lat]
    labels = ["Haar Cascade"]
    if has_mtcnn and mtcnn_lat:
        data.append(mtcnn_lat)
        labels.append("MTCNN")

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    colors = ["#4c72b0", "#dd8452"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency (ms/frame)")
    ax.set_title("E1 — Latency Boxplot: Haar vs MTCNN")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "E1_latency_boxplot.png", dpi=150)
    plt.close()


def _generate_demo_data():
    """Tạo dữ liệu mô phỏng khi chưa có dataset thực, để xem format output."""
    import random
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("[E1] Tạo demo output với số liệu mô phỏng...")
    random.seed(42)
    rows = []
    haar_lat, mtcnn_lat = [], []
    for cond in CONDITIONS:
        h_rec = random.uniform(0.70, 0.98) if cond == "frontal" else random.uniform(0.45, 0.85)
        m_rec = min(1.0, h_rec + random.uniform(0.05, 0.15))
        h_ms  = random.uniform(8, 18)
        m_ms  = random.uniform(60, 120)
        n     = 30
        h_lat_cond = [random.gauss(h_ms, h_ms * 0.15) for _ in range(n)]
        m_lat_cond = [random.gauss(m_ms, m_ms * 0.15) for _ in range(n)]
        haar_lat.extend(h_lat_cond)
        mtcnn_lat.extend(m_lat_cond)
        rows.append({
            "condition": cond, "n_images": n,
            "haar_tp": int(h_rec * n), "haar_fn": n - int(h_rec * n),
            "haar_recall": round(h_rec, 4),
            "haar_lat_mean_ms": round(h_ms, 2), "haar_fps": round(1000/h_ms, 1),
            "mtcnn_tp": int(m_rec * n), "mtcnn_fn": n - int(m_rec * n),
            "mtcnn_recall": round(m_rec, 4),
            "mtcnn_lat_mean_ms": round(m_ms, 2), "mtcnn_fps": round(1000/m_ms, 1),
        })

    csv_path = RESULT_DIR / "E1_results_DEMO.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _plot_recall(rows, True)
    _plot_latency(haar_lat, mtcnn_lat, True)
    print(f"  Demo files saved to {RESULT_DIR}/")
    print("  (Thay bằng ảnh thật sau khi có dataset)")


if __name__ == "__main__":
    run()
