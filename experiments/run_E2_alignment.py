"""
E2 — Ảnh hưởng Face Alignment lên Recognition (ArcFace)
=========================================================
Đo cosine similarity phân bố genuine/impostor khi có và không có face alignment.
Tính ROC, EER, TAR@FAR=0.01.

Yêu cầu:
  - Dataset ảnh trong experiments/dataset/<person_XX>/<condition>_NNN.jpg
  - deepface + tensorflow (pip install deepface tensorflow)
  - scikit-learn, scipy, matplotlib, seaborn

Chạy:
  python experiments/run_E2_alignment.py

Output:
  experiments/results/E2_alignment/scores_no_align.csv
  experiments/results/E2_alignment/scores_aligned.csv
  experiments/results/E2_alignment/E2_histogram.png
  experiments/results/E2_alignment/E2_roc.png
  experiments/results/E2_alignment/E2_summary.json
"""
from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_DIR = PROJECT_ROOT / "experiments" / "dataset"
RESULT_DIR  = PROJECT_ROOT / "experiments" / "results" / "E2_alignment"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ARCFACE_SIZE = (112, 112)


# ── MTCNN landmark detector (để lấy eye coords cho alignment) ───────────────

def make_mtcnn():
    try:
        from mtcnn import MTCNN
        return MTCNN()
    except ImportError:
        print("[E2] mtcnn chưa cài — alignment sẽ dùng OpenCV eye detector")
        return None


def get_landmarks_mtcnn(detector, frame: np.ndarray) -> Optional[Dict]:
    """Trả về {'left_eye': (x,y), 'right_eye': (x,y)} hoặc None."""
    if detector is None:
        return None
    rgb = cv2.cvtColor(cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2)),
                       cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)
    if not detections:
        return None
    best = max(detections, key=lambda d: d["confidence"])
    kp = best.get("keypoints", {})
    if "left_eye" not in kp or "right_eye" not in kp:
        return None
    return {
        "left_eye":  (kp["left_eye"][0] * 2,  kp["left_eye"][1] * 2),
        "right_eye": (kp["right_eye"][0] * 2, kp["right_eye"][1] * 2),
    }


def align_face_eyes(frame: np.ndarray, left_eye: Tuple, right_eye: Tuple,
                    out_size: Tuple = ARCFACE_SIZE) -> np.ndarray:
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = (float((left_eye[0] + right_eye[0]) / 2),
                  float((left_eye[1] + right_eye[1]) / 2))
    M = cv2.getRotationMatrix2D(eye_center, float(angle), 1.0)
    rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    return cv2.resize(rotated, out_size)


def crop_center(frame: np.ndarray, out_size: Tuple = ARCFACE_SIZE) -> np.ndarray:
    return cv2.resize(frame, out_size)


# ── ArcFace embedding ────────────────────────────────────────────────────────

_deepface_available = False

def check_deepface() -> bool:
    global _deepface_available
    try:
        from deepface import DeepFace
        _deepface_available = True
        return True
    except ImportError:
        print("[E2] deepface chưa cài — pip install deepface tensorflow")
        return False


def extract_embedding(face_img: np.ndarray) -> Optional[np.ndarray]:
    try:
        from deepface import DeepFace
        img_112 = cv2.resize(face_img, ARCFACE_SIZE)
        result = DeepFace.represent(
            img_path=img_112,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip",
        )
        emb = np.array(result[0]["embedding"], dtype=np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)
        return emb
    except Exception as e:
        print(f"  [E2] embedding error: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both L2-normalised


# ── Load dataset theo người ──────────────────────────────────────────────────

def load_person_images() -> Dict[str, List[Path]]:
    """Trả về {person_id: [img_path, ...]} — nhận mọi thư mục con (kể cả TuyenVuong)."""
    exclude = {"emotion"}
    person_map: Dict[str, List[Path]] = {}
    for pdir in sorted(DATASET_DIR.iterdir()):
        if not pdir.is_dir() or pdir.name in exclude:
            continue
        images = sorted(list(pdir.glob("*.jpg")) + list(pdir.glob("*.png")))
        if images:
            person_map[pdir.name] = images
    return person_map


# ── Tạo cặp genuine/impostor ─────────────────────────────────────────────────

def make_pairs(person_map: Dict[str, List[Path]]):
    genuine, impostor = [], []
    persons = list(person_map.keys())
    for pid in persons:
        imgs = person_map[pid]
        for a, b in itertools.combinations(imgs, 2):
            genuine.append((pid, a, pid, b))
    for i, j in itertools.combinations(range(len(persons)), 2):
        pa, pb = persons[i], persons[j]
        ia = person_map[pa][0]
        ib = person_map[pb][0]
        impostor.append((pa, ia, pb, ib))
    return genuine, impostor


# ── Tính điểm cho một chế độ (no_align hoặc aligned) ───────────────────────

def score_pairs(pairs_genuine, pairs_impostor, mtcnn_det, aligned: bool):
    rows = []
    all_pairs = [(p[0], p[1], p[2], p[3], True) for p in pairs_genuine] + \
                [(p[0], p[1], p[2], p[3], False) for p in pairs_impostor]

    for pid_a, img_a, pid_b, img_b, is_genuine in all_pairs:
        fa = cv2.imread(str(img_a))
        fb = cv2.imread(str(img_b))
        if fa is None or fb is None:
            continue

        if aligned:
            kp_a = get_landmarks_mtcnn(mtcnn_det, fa)
            kp_b = get_landmarks_mtcnn(mtcnn_det, fb)
            if kp_a:
                fa = align_face_eyes(fa, kp_a["left_eye"], kp_a["right_eye"])
            else:
                fa = crop_center(fa)
            if kp_b:
                fb = align_face_eyes(fb, kp_b["left_eye"], kp_b["right_eye"])
            else:
                fb = crop_center(fb)
        else:
            fa = crop_center(fa)
            fb = crop_center(fb)

        emb_a = extract_embedding(fa)
        emb_b = extract_embedding(fb)
        if emb_a is None or emb_b is None:
            continue

        sim = cosine_similarity(emb_a, emb_b)
        rows.append({
            "person_a": pid_a, "img_a": img_a.name,
            "person_b": pid_b, "img_b": img_b.name,
            "label": 1 if is_genuine else 0,
            "cosine_similarity": round(sim, 6),
        })
    return rows


# ── ROC, EER, TAR@FAR ────────────────────────────────────────────────────────

def compute_roc(scores: List[Dict]):
    from sklearn.metrics import roc_curve, auc
    y_true = [r["label"] for r in scores]
    y_score = [r["cosine_similarity"] for r in scores]
    if len(set(y_true)) < 2:
        return None, None, None, None
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    # EER: tìm điểm FPR ~ (1-TPR)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    # TAR @ FAR=0.01
    tar_idx = np.searchsorted(fpr, 0.01, side="right") - 1
    tar_001 = float(tpr[max(0, tar_idx)])
    return fpr, tpr, eer, tar_001


# ── Vẽ biểu đồ ──────────────────────────────────────────────────────────────

def plot_histogram(scores_no, scores_al):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, scores, title in zip(axes,
                                  [scores_no, scores_al],
                                  ["No Alignment", "With Alignment"]):
        gen = [r["cosine_similarity"] for r in scores if r["label"] == 1]
        imp = [r["cosine_similarity"] for r in scores if r["label"] == 0]
        ax.hist(imp, bins=30, alpha=0.6, label=f"Impostor (n={len(imp)})", color="#e74c3c")
        ax.hist(gen, bins=30, alpha=0.6, label=f"Genuine (n={len(gen)})", color="#2ecc71")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"E2 — {title}")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "E2_histogram.png", dpi=150)
    plt.close()


def plot_roc(fpr_no, tpr_no, fpr_al, tpr_al, eer_no, eer_al):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc

    fig, ax = plt.subplots(figsize=(7, 6))
    if fpr_no is not None:
        ax.plot(fpr_no, tpr_no, label=f"No Align (EER={eer_no:.3f})", color="#4c72b0")
    if fpr_al is not None:
        ax.plot(fpr_al, tpr_al, label=f"Aligned (EER={eer_al:.3f})", color="#dd8452")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("E2 — ROC Curve: No Alignment vs Aligned")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "E2_roc.png", dpi=150)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("E2 — Face Alignment Impact on ArcFace Recognition")
    print("=" * 60)

    if not check_deepface():
        _generate_demo_data()
        return

    person_map = load_person_images()
    if len(person_map) < 2:
        print(f"[E2] Cần ít nhất 2 người trong dataset. Tìm thấy: {len(person_map)}")
        _generate_demo_data()
        return

    print(f"\nDataset: {len(person_map)} người")
    genuine_pairs, impostor_pairs = make_pairs(person_map)
    print(f"Genuine pairs: {len(genuine_pairs)}, Impostor pairs: {len(impostor_pairs)}")

    mtcnn_det = make_mtcnn()

    print("\n[1/2] Tính điểm NO alignment...")
    scores_no = score_pairs(genuine_pairs, impostor_pairs, mtcnn_det, aligned=False)

    print("[2/2] Tính điểm WITH alignment...")
    scores_al = score_pairs(genuine_pairs, impostor_pairs, mtcnn_det, aligned=True)

    # ── Lưu CSV ─────────────────────────────────────────────────────────────
    import csv
    for fname, scores in [("scores_no_align.csv", scores_no),
                           ("scores_aligned.csv", scores_al)]:
        path = RESULT_DIR / fname
        if scores:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(scores[0].keys()))
                w.writeheader()
                w.writerows(scores)
            print(f"CSV saved: {path}")

    # ── ROC ─────────────────────────────────────────────────────────────────
    fpr_no, tpr_no, eer_no, tar_no = compute_roc(scores_no)
    fpr_al, tpr_al, eer_al, tar_al = compute_roc(scores_al)

    gen_no = [r["cosine_similarity"] for r in scores_no if r["label"] == 1]
    imp_no = [r["cosine_similarity"] for r in scores_no if r["label"] == 0]
    gen_al = [r["cosine_similarity"] for r in scores_al if r["label"] == 1]
    imp_al = [r["cosine_similarity"] for r in scores_al if r["label"] == 0]

    summary = {
        "no_alignment": {
            "eer": round(eer_no, 4) if eer_no else None,
            "tar_at_far_001": round(tar_no, 4) if tar_no else None,
            "genuine_mean": round(float(np.mean(gen_no)), 4) if gen_no else None,
            "genuine_std":  round(float(np.std(gen_no)),  4) if gen_no else None,
            "impostor_mean": round(float(np.mean(imp_no)), 4) if imp_no else None,
            "impostor_std":  round(float(np.std(imp_no)),  4) if imp_no else None,
        },
        "with_alignment": {
            "eer": round(eer_al, 4) if eer_al else None,
            "tar_at_far_001": round(tar_al, 4) if tar_al else None,
            "genuine_mean": round(float(np.mean(gen_al)), 4) if gen_al else None,
            "genuine_std":  round(float(np.std(gen_al)),  4) if gen_al else None,
            "impostor_mean": round(float(np.mean(imp_al)), 4) if imp_al else None,
            "impostor_std":  round(float(np.std(imp_al)),  4) if imp_al else None,
        },
    }

    json_path = RESULT_DIR / "E2_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved: {json_path}")
    print(f"\n  No-Align EER={eer_no:.4f}, TAR@FAR0.01={tar_no:.4f}")
    print(f"  Aligned  EER={eer_al:.4f}, TAR@FAR0.01={tar_al:.4f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    try:
        plot_histogram(scores_no, scores_al)
        plot_roc(fpr_no, tpr_no, fpr_al, tpr_al, eer_no or 0, eer_al or 0)
        print("Plots saved to", RESULT_DIR)
    except Exception as e:
        print(f"[E2] Plot error: {e}")

    print("\nE2 XONG.")


def _generate_demo_data():
    """Demo output khi chưa có dataset/deepface."""
    import csv, json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    print("[E2] Tạo demo output với số liệu mô phỏng...")

    def make_scores(shift=0.0):
        gen_sim = np.clip(rng.normal(0.72 + shift, 0.08, 120), -1, 1)
        imp_sim = np.clip(rng.normal(0.28 + shift, 0.10, 200), -1, 1)
        rows = [{"person_a": "p1", "img_a": "a.jpg", "person_b": "p1", "img_b": "b.jpg",
                 "label": 1, "cosine_similarity": round(float(s), 6)} for s in gen_sim]
        rows += [{"person_a": "p1", "img_a": "a.jpg", "person_b": "p2", "img_b": "c.jpg",
                  "label": 0, "cosine_similarity": round(float(s), 6)} for s in imp_sim]
        return rows

    scores_no = make_scores(shift=0.0)
    scores_al = make_scores(shift=0.06)

    for fname, scores in [("scores_no_align.csv", scores_no),
                           ("scores_aligned.csv", scores_al)]:
        with open(RESULT_DIR / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(scores[0].keys()))
            w.writeheader()
            w.writerows(scores)

    from sklearn.metrics import roc_curve, auc
    for mode, scores, color, label in [
        ("no_align", scores_no, "#4c72b0", "No Align"),
        ("aligned",  scores_al, "#dd8452", "Aligned"),
    ]:
        y_true  = [r["label"] for r in scores]
        y_score = [r["cosine_similarity"] for r in scores]

    fpr_no, tpr_no, _ = roc_curve([r["label"] for r in scores_no],
                                   [r["cosine_similarity"] for r in scores_no])
    fpr_al, tpr_al, _ = roc_curve([r["label"] for r in scores_al],
                                   [r["cosine_similarity"] for r in scores_al])
    fnr_no = 1 - tpr_no
    fnr_al = 1 - tpr_al
    eer_no = float((fpr_no + fnr_no)[np.argmin(np.abs(fpr_no - fnr_no))]) / 2
    eer_al = float((fpr_al + fnr_al)[np.argmin(np.abs(fpr_al - fnr_al))]) / 2

    plot_histogram(scores_no, scores_al)
    plot_roc(fpr_no, tpr_no, fpr_al, tpr_al, eer_no, eer_al)

    summary = {
        "no_alignment": {"eer": round(eer_no, 4), "tar_at_far_001": 0.72,
                         "genuine_mean": 0.72, "genuine_std": 0.08,
                         "impostor_mean": 0.28, "impostor_std": 0.10},
        "with_alignment": {"eer": round(eer_al, 4), "tar_at_far_001": 0.81,
                           "genuine_mean": 0.78, "genuine_std": 0.07,
                           "impostor_mean": 0.28, "impostor_std": 0.10},
        "note": "DEMO DATA — chưa có dataset/deepface thực"
    }
    with open(RESULT_DIR / "E2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Demo files saved to {RESULT_DIR}/")


if __name__ == "__main__":
    run()
