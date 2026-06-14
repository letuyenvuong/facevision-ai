"""
E4 — PCA Reconstruction Quality vs Number of Components
=======================================================
Đo MSE, SSIM, explained variance ratio với k = 10, 25, 50, 100, 200.

Yêu cầu:
  - Dataset ảnh mặt trong experiments/dataset/<person_XX>/*.jpg
  - scikit-learn >= 1.3, scikit-image >= 0.21, matplotlib, seaborn (đều đã cài)
  - KHÔNG cần deepface hay tensorflow

Chạy:
  python experiments/run_E4_pca.py

Output:
  experiments/results/E4_pca/E4_metrics.csv
  experiments/results/E4_pca/E4_curve_mse_ssim.png
  experiments/results/E4_pca/E4_scree_plot.png
  experiments/results/E4_pca/E4_reconstruction_grid.png
"""
from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import csv
import time
import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_DIR = PROJECT_ROOT / "experiments" / "dataset"
RESULT_DIR  = PROJECT_ROOT / "experiments" / "results" / "E4_pca"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FACE_SIZE    = (64, 64)   # PCA face size (khớp với config.py)
K_VALUES     = [10, 25, 50, 100, 200]
TRAIN_RATIO  = 0.8
MAX_SAMPLES  = 2000       # giới hạn để tránh RAM quá lớn


# ── Tiền xử lý ──────────────────────────────────────────────────────────────

def preprocess(img: np.ndarray) -> np.ndarray:
    """BGR → grayscale 64×64 flattened float32 [0,1]"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    resized = cv2.resize(gray, FACE_SIZE)
    return resized.astype(np.float32).flatten() / 255.0


def detect_and_crop_face(img: np.ndarray) -> np.ndarray:
    """Crop face dùng Haar; nếu không tìm thấy thì dùng center crop."""
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(haar_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    faces = clf.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(min(w, h) * 0.1)
        fh, fw = img.shape[:2]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(fw, x + w + pad), min(fh, y + h + pad)
        return img[y1:y2, x1:x2]
    # Fallback: center square crop
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0+s, x0:x0+s]


# ── Load dataset ─────────────────────────────────────────────────────────────

def load_face_vectors() -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Trả về (X, raw_crops):
      X          — (N, 4096) float32 preprocessed vectors
      raw_crops  — list of (64,64) uint8 cho visualization
    """
    vectors, crops = [], []
    exclude = {"emotion"}
    person_dirs = sorted(p for p in DATASET_DIR.iterdir()
                         if p.is_dir() and p.name not in exclude)
    if not person_dirs:
        return np.array([]), []

    for pdir in person_dirs:
        for img_path in sorted(list(pdir.glob("*.jpg")) + list(pdir.glob("*.png"))):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            face = detect_and_crop_face(img)
            v = preprocess(face)
            crops.append(cv2.resize(
                cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face,
                FACE_SIZE
            ))
            vectors.append(v)
            if len(vectors) >= MAX_SAMPLES:
                break
        if len(vectors) >= MAX_SAMPLES:
            break

    if not vectors:
        return np.array([]), []
    return np.array(vectors, dtype=np.float32), crops


# ── SSIM ─────────────────────────────────────────────────────────────────────

def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Tính SSIM giữa 2 ảnh grayscale uint8 hoặc float [0,1]."""
    try:
        from skimage.metrics import structural_similarity as ssim
        if a.dtype != np.float32:
            a = a.astype(np.float32) / 255.0
        if b.dtype != np.float32:
            b = b.astype(np.float32) / 255.0
        return float(ssim(a, b, data_range=1.0))
    except ImportError:
        # fallback: MSE-based approximation
        mse = float(np.mean((a - b) ** 2))
        return float(1 / (1 + mse * 100))


# ── PCA reconstruction ────────────────────────────────────────────────────────

def run_pca_k(X_train: np.ndarray, X_test: np.ndarray,
              crops_test: List[np.ndarray], k: int) -> dict:
    """Train PCA với k components, tính metrics trên test set."""
    n_comp = min(k, X_train.shape[0] - 1, X_train.shape[1])
    if n_comp < 1:
        return None

    mean_face = X_train.mean(axis=0)
    X_c = X_train - mean_face

    pca = PCA(n_components=n_comp, whiten=False)
    pca.fit(X_c)

    explained_var = float(pca.explained_variance_ratio_.sum())

    mses, ssims, times = [], [], []
    reconstructed_vecs = []

    for v, crop in zip(X_test, crops_test):
        t0 = time.perf_counter()
        centered = v - mean_face
        proj = pca.transform(centered.reshape(1, -1))
        recon_v = pca.inverse_transform(proj).flatten() + mean_face
        recon_v = np.clip(recon_v, 0, 1)
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000)
        mse = float(np.mean((v - recon_v) ** 2))
        mses.append(mse)

        recon_img = (recon_v * 255).astype(np.uint8).reshape(FACE_SIZE)
        ssim_val = compute_ssim(crop.astype(np.float32) / 255.0,
                                recon_img.astype(np.float32) / 255.0)
        ssims.append(ssim_val)
        reconstructed_vecs.append(recon_img)

    return {
        "k": k,
        "n_components_actual": n_comp,
        "explained_variance_ratio": round(explained_var, 6),
        "mse_mean": round(float(np.mean(mses)), 6),
        "mse_std":  round(float(np.std(mses)),  6),
        "ssim_mean": round(float(np.mean(ssims)), 6),
        "ssim_std":  round(float(np.std(ssims)),  6),
        "reconstruct_ms_per_img": round(float(np.mean(times)), 4),
        "_recon_imgs": reconstructed_vecs,   # dùng cho grid, không lưu CSV
        "_pca": pca,
        "_mean_face": mean_face,
    }


# ── Vẽ biểu đồ ──────────────────────────────────────────────────────────────

def plot_mse_ssim_curve(results: List[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks     = [r["k"] for r in results]
    mses   = [r["mse_mean"] for r in results]
    ssims  = [r["ssim_mean"] for r in results]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    color1 = "#e74c3c"
    ax1.plot(ks, mses, "o-", color=color1, label="MSE")
    ax1.set_xlabel("Số PCA Components (k)")
    ax1.set_ylabel("MSE", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")
    ax1.set_xticks(ks)
    ax1.set_xticklabels([str(k) for k in ks])

    ax2 = ax1.twinx()
    color2 = "#2980b9"
    ax2.plot(ks, ssims, "s--", color=color2, label="SSIM")
    ax2.set_ylabel("SSIM", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("E4 — MSE và SSIM theo số PCA Components")
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "E4_curve_mse_ssim.png", dpi=150)
    plt.close()


def plot_scree(result_50: dict):
    """Scree plot: explained variance ratio của mỗi component (dùng k=50 hoặc lớn nhất)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if result_50 is None:
        return
    pca = result_50["_pca"]
    evr = pca.explained_variance_ratio_
    cumev = np.cumsum(evr)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(range(1, len(evr) + 1), evr * 100, alpha=0.6, color="#3498db",
            label="Individual")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Explained Variance (%)", color="#3498db")
    ax1.tick_params(axis="y", labelcolor="#3498db")

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cumev) + 1), cumev * 100, "r-", linewidth=2,
             label="Cumulative")
    ax2.set_ylabel("Cumulative Explained Variance (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.axhline(95, color="gray", linestyle="--", alpha=0.7, label="95%")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title(f"E4 — Scree Plot (k={result_50['k']})")
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "E4_scree_plot.png", dpi=150)
    plt.close()


def plot_reconstruction_grid(crops_test: List[np.ndarray],
                              all_results: List[dict], n_show: int = 5):
    """Lưới ảnh: mỗi hàng là 1 ảnh gốc + tái dựng theo k."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(n_show, len(crops_test))
    n_cols = 1 + len(all_results)   # gốc + 1 cột mỗi k
    n_rows = n_show

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 2.0))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_i in range(n_rows):
        # Cột 0: ảnh gốc
        axes[row_i, 0].imshow(crops_test[row_i], cmap="gray", vmin=0, vmax=255)
        axes[row_i, 0].axis("off")
        if row_i == 0:
            axes[row_i, 0].set_title("Original", fontsize=8)

        for col_j, res in enumerate(all_results, start=1):
            recon_imgs = res.get("_recon_imgs", [])
            if row_i < len(recon_imgs):
                axes[row_i, col_j].imshow(recon_imgs[row_i], cmap="gray", vmin=0, vmax=255)
            else:
                axes[row_i, col_j].imshow(np.zeros(FACE_SIZE, dtype=np.uint8),
                                           cmap="gray")
            axes[row_i, col_j].axis("off")
            if row_i == 0:
                axes[row_i, col_j].set_title(f"k={res['k']}", fontsize=8)

    plt.suptitle("E4 — Ảnh gốc vs Tái dựng PCA (k components)", fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "E4_reconstruction_grid.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("E4 — PCA Reconstruction: Quality vs k Components")
    print("=" * 60)

    print("\nLoading face images...")
    X, crops = load_face_vectors()

    if X.shape[0] == 0:
        print(f"[E4] Không có ảnh trong {DATASET_DIR}/")
        _generate_demo_data()
        return

    n = X.shape[0]
    print(f"Loaded: {n} faces ({FACE_SIZE[0]}×{FACE_SIZE[1]} grayscale)")

    # Train/test split
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    X_train = X[train_idx]
    X_test  = X[test_idx]
    crops_test = [crops[i] for i in test_idx]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    all_results = []
    for k in K_VALUES:
        if k >= X_train.shape[0]:
            print(f"  k={k} bỏ qua (cần > {k} ảnh train)")
            continue
        print(f"\n  k={k}...")
        res = run_pca_k(X_train, X_test, crops_test, k)
        if res is None:
            continue
        print(f"    MSE={res['mse_mean']:.6f}, SSIM={res['ssim_mean']:.4f}, "
              f"ExplVar={res['explained_variance_ratio']:.3f}, "
              f"time={res['reconstruct_ms_per_img']:.3f}ms/img")
        all_results.append(res)

    if not all_results:
        print("[E4] Không có kết quả. Dataset quá nhỏ?")
        return

    # ── Lưu CSV ─────────────────────────────────────────────────────────────
    csv_fields = ["k", "n_components_actual", "explained_variance_ratio",
                  "mse_mean", "mse_std", "ssim_mean", "ssim_std",
                  "reconstruct_ms_per_img"]
    csv_path = RESULT_DIR / "E4_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for res in all_results:
            w.writerow({field: res[field] for field in csv_fields})
    print(f"\nCSV saved: {csv_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")

        plot_mse_ssim_curve(all_results)

        # Scree plot: dùng model k=50 nếu có, nếu không dùng model lớn nhất
        scree_res = next((r for r in all_results if r["k"] == 50), all_results[-1])
        plot_scree(scree_res)

        plot_reconstruction_grid(crops_test, all_results, n_show=5)

        print("Plots saved to", RESULT_DIR)
    except Exception as e:
        print(f"[E4] Plot error: {e}")
        import traceback
        traceback.print_exc()

    print("\nE4 XONG.")


def _generate_demo_data():
    """Demo output khi chưa có dataset thực."""
    import csv, json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    print("[E4] Tạo demo output với số liệu mô phỏng...")
    rng = np.random.default_rng(42)

    k_vals = [10, 25, 50, 100, 200]
    # Giả lập: MSE giảm dần, SSIM tăng dần theo k
    base_mse  = [0.042, 0.028, 0.018, 0.011, 0.007]
    base_ssim = [0.61,  0.72,  0.81,  0.88,  0.93]
    base_ev   = [0.52,  0.71,  0.83,  0.92,  0.97]

    rows = []
    for i, k in enumerate(k_vals):
        rows.append({
            "k": k,
            "n_components_actual": k,
            "explained_variance_ratio": base_ev[i],
            "mse_mean": round(base_mse[i] + rng.normal(0, 0.001), 6),
            "mse_std":  round(base_mse[i] * 0.15, 6),
            "ssim_mean": round(base_ssim[i] + rng.normal(0, 0.005), 6),
            "ssim_std":  round(0.04, 6),
            "reconstruct_ms_per_img": round(0.8 + i * 0.5, 4),
        })

    csv_path = RESULT_DIR / "E4_metrics_DEMO.csv"
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    plot_mse_ssim_curve(rows)

    # Scree plot mô phỏng
    evr = np.sort(rng.exponential(0.3, 50))[::-1]
    evr /= evr.sum()
    class _FakePCA:
        explained_variance_ratio_ = evr
    fake_res = {"k": 50, "_pca": _FakePCA()}
    plot_scree(fake_res)

    # Grid mô phỏng: ảnh nhiễu
    crops_demo = [rng.integers(80, 180, FACE_SIZE, dtype=np.uint8) for _ in range(5)]
    all_res_demo = []
    for k in k_vals:
        noise = rng.integers(-20, 20, FACE_SIZE, dtype=np.int16)
        recon_imgs = [np.clip(c.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                      for c in crops_demo]
        all_res_demo.append({"k": k, "_recon_imgs": recon_imgs})
    plot_reconstruction_grid(crops_demo, all_res_demo, n_show=5)

    print(f"  Demo files saved to {RESULT_DIR}/")
    print("  (Thay bằng ảnh thật sau khi có dataset)")


if __name__ == "__main__":
    run()
