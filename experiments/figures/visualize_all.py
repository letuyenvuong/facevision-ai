"""
Trực quan hóa toàn bộ kết quả thực nghiệm E1–E4.
Output: experiments/figures/dashboard_E1.png ... dashboard_E4.png
        experiments/figures/dashboard_full.png  (tổng hợp 1 trang)
"""
from __future__ import annotations
import sys, json, csv, os
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ── Paths ───────────────────────────────────────────────────────────────────
EXPERIMENTS = Path(__file__).resolve().parent.parent   # .../experiments/
ROOT    = EXPERIMENTS.parent                           # .../facevision-ai/
RES     = EXPERIMENTS / "results"
FIG_DIR = EXPERIMENTS / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
TEAL   = "#00e5cc"
CORAL  = "#ff6b6b"
GOLD   = "#ffd166"
BLUE   = "#4e9af1"
PURPLE = "#b48ead"
BG     = "#0d0f14"
PANEL  = "#1a1d27"
TEXT   = "#e0e0e0"
GRID   = "#2a2d3a"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "text.color":       TEXT,
    "grid.color":       GRID,
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "legend.facecolor": PANEL,
    "legend.edgecolor": GRID,
})


# ════════════════════════════════════════════════════════════════════════════
# E1 — Detector Comparison
# ════════════════════════════════════════════════════════════════════════════
def load_e1():
    rows = []
    with open(RES / "E1_detector" / "E1_results.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v.replace(".", "", 1).isdigit() else v)
                         for k, v in r.items()})
    return rows

def plot_e1(ax_recall, ax_fps):
    data = load_e1()
    conds   = [r["condition"] for r in data]
    haar_r  = [r["haar_recall"]  * 100 for r in data]
    mtcnn_r = [r["mtcnn_recall"] * 100 for r in data]
    haar_fps  = [r["haar_fps"]  for r in data]
    mtcnn_fps = [r["mtcnn_fps"] for r in data]

    x = np.arange(len(conds))
    w = 0.35
    cond_labels = ["Frontal", "Pose 15°", "Pose 30°", "Dark", "Occluded"]

    # Recall grouped bar
    b1 = ax_recall.bar(x - w/2, haar_r,  w, label="Haar",  color=CORAL,  alpha=0.88)
    b2 = ax_recall.bar(x + w/2, mtcnn_r, w, label="MTCNN", color=TEAL,   alpha=0.88)
    ax_recall.set_xticks(x); ax_recall.set_xticklabels(cond_labels)
    ax_recall.set_ylabel("Recall (%)"); ax_recall.set_ylim(0, 115)
    ax_recall.set_title("E1 — Recall theo điều kiện: Haar vs MTCNN")
    ax_recall.legend(); ax_recall.yaxis.grid(True)
    ax_recall.axhline(100, color=GRID, lw=0.8, ls="--")
    for bar in b1: ax_recall.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                                   f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, color=TEXT)
    for bar in b2: ax_recall.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                                   f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, color=TEXT)

    # FPS grouped bar
    b3 = ax_fps.bar(x - w/2, haar_fps,  w, label="Haar",  color=CORAL,  alpha=0.88)
    b4 = ax_fps.bar(x + w/2, mtcnn_fps, w, label="MTCNN", color=TEAL,   alpha=0.88)
    ax_fps.set_xticks(x); ax_fps.set_xticklabels(cond_labels)
    ax_fps.set_ylabel("FPS"); ax_fps.set_title("E1 — Tốc độ xử lý (FPS)")
    ax_fps.legend(); ax_fps.yaxis.grid(True)
    for bar in b3: ax_fps.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
                                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, color=TEXT)
    for bar in b4: ax_fps.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
                                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, color=TEXT)


# ════════════════════════════════════════════════════════════════════════════
# E2 — Face Alignment
# ════════════════════════════════════════════════════════════════════════════
def load_e2():
    with open(RES / "E2_alignment" / "E2_summary.json", encoding="utf-8") as f:
        return json.load(f)

def load_e2_scores():
    def _read(fname):
        scores, labels = [], []
        with open(RES / "E2_alignment" / fname, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                scores.append(float(r["cosine_similarity"]))
                labels.append(int(r["label"]))
        return np.array(scores), np.array(labels)
    return _read("scores_no_align.csv"), _read("scores_aligned.csv")

def plot_e2(ax_bar, ax_dist):
    summary = load_e2()
    (sc_na, lb_na), (sc_al, lb_al) = load_e2_scores()

    # Bar chart: EER + TAR so sánh
    metrics = ["EER (%)", "TAR@FAR=0.01 (%)"]
    no_al   = [summary["no_alignment"]["eer"]*100,
               summary["no_alignment"]["tar_at_far_001"]*100]
    with_al = [summary["with_alignment"]["eer"]*100,
               summary["with_alignment"]["tar_at_far_001"]*100]

    x = np.arange(2); w = 0.35
    b1 = ax_bar.bar(x - w/2, no_al,   w, label="Không alignment", color=CORAL, alpha=0.88)
    b2 = ax_bar.bar(x + w/2, with_al, w, label="Có alignment",    color=TEAL,  alpha=0.88)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(metrics)
    ax_bar.set_title("E2 — Tác động của Face Alignment lên ArcFace")
    ax_bar.set_ylabel("Giá trị (%)"); ax_bar.yaxis.grid(True)
    ax_bar.legend()
    for bar in list(b1) + list(b2):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9, color=TEXT)

    # Histogram cosine similarity (aligned version)
    gen  = sc_al[lb_al == 1]
    imp  = sc_al[lb_al == 0]
    bins = np.linspace(-0.1, 1.05, 50)
    ax_dist.hist(imp, bins=bins, alpha=0.65, color=CORAL,  label=f"Impostor (n={len(imp)})")
    ax_dist.hist(gen, bins=bins, alpha=0.65, color=TEAL,   label=f"Genuine  (n={len(gen)})")
    ax_dist.set_xlabel("Cosine Similarity"); ax_dist.set_ylabel("Số lượng cặp")
    ax_dist.set_title("E2 — Phân bố cosine similarity (Có alignment)")
    ax_dist.legend(); ax_dist.yaxis.grid(True)

    gm = summary["with_alignment"]["genuine_mean"]
    im = summary["with_alignment"]["impostor_mean"]
    ax_dist.axvline(gm, color=TEAL,  ls="--", lw=1.5, label=f"μ_genuine={gm:.3f}")
    ax_dist.axvline(im, color=CORAL, ls="--", lw=1.5, label=f"μ_impostor={im:.3f}")
    ax_dist.axvline(0.40, color=GOLD, ls=":", lw=1.8, label="Threshold=0.40")
    ax_dist.legend(fontsize=9)


# ════════════════════════════════════════════════════════════════════════════
# E3 — Emotion FER
# ════════════════════════════════════════════════════════════════════════════
def load_e3():
    with open(RES / "E3_emotion" / "E3_results.json", encoding="utf-8") as f:
        return json.load(f)

def plot_e3(ax_f1, ax_switch):
    data = load_e3()
    labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    f1_vals = [data["per_class_f1"].get(l, 0) for l in labels]
    colors  = [CORAL if v < 0.5 else (GOLD if v < 0.65 else TEAL) for v in f1_vals]

    # Per-class F1 bar chart
    bars = ax_f1.barh(labels, f1_vals, color=colors, alpha=0.88)
    ax_f1.axvline(data["macro_f1"], color=GOLD, ls="--", lw=1.5,
                  label=f"Macro-F1 = {data['macro_f1']:.3f}")
    ax_f1.axvline(data["accuracy"], color=PURPLE, ls=":", lw=1.5,
                  label=f"Accuracy = {data['accuracy']:.3f}")
    ax_f1.set_xlabel("F1 Score"); ax_f1.set_xlim(0, 1.05)
    ax_f1.set_title("E3 — F1 Score từng lớp cảm xúc (FER2013)")
    ax_f1.legend(fontsize=9); ax_f1.xaxis.grid(True)
    for bar, v in zip(bars, f1_vals):
        ax_f1.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{v:.3f}", va="center", fontsize=9, color=TEXT)

    # Switch rate comparison
    cats   = ["Raw\n(không smooth)", "Smooth\n(window=5)"]
    rates  = [data["switch_rate_raw"]*100, data["switch_rate_smooth"]*100]
    bcols  = [CORAL, TEAL]
    bb = ax_switch.bar(cats, rates, color=bcols, alpha=0.88, width=0.45)
    ax_switch.set_ylabel("Switch Rate (%)"); ax_switch.set_ylim(0, 80)
    ax_switch.set_title("E3 — Switch Rate: Raw vs Temporal Smoothing")
    ax_switch.yaxis.grid(True)
    for bar, v in zip(bb, rates):
        ax_switch.text(bar.get_x()+bar.get_width()/2, v+1.5,
                       f"{v:.1f}%", ha="center", va="bottom", fontsize=11,
                       fontweight="bold", color=TEXT)
    pct = (1 - data["switch_rate_smooth"] / data["switch_rate_raw"]) * 100
    ax_switch.text(0.5, 0.88, f"Giảm {pct:.0f}%", ha="center",
                   fontsize=12, color=GOLD, fontweight="bold",
                   transform=ax_switch.transAxes)


# ════════════════════════════════════════════════════════════════════════════
# E4 — PCA Reconstruction
# ════════════════════════════════════════════════════════════════════════════
def load_e4():
    rows = []
    with open(RES / "E4_pca" / "E4_metrics.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows

def plot_e4(ax_curve, ax_var):
    data = load_e4()
    ks    = [int(r["k"]) for r in data]
    mse   = [r["mse_mean"]              for r in data]
    ssim  = [r["ssim_mean"]             for r in data]
    expv  = [r["explained_variance_ratio"]*100 for r in data]

    # MSE + SSIM dual axis
    ax2 = ax_curve.twinx()
    l1, = ax_curve.plot(ks, mse,  "o-", color=CORAL, lw=2.2, ms=8, label="MSE")
    l2, = ax2.plot(     ks, ssim, "s--", color=TEAL,  lw=2.2, ms=8, label="SSIM")
    ax_curve.set_xlabel("k (số principal components)")
    ax_curve.set_ylabel("MSE",  color=CORAL)
    ax2.set_ylabel("SSIM", color=TEAL)
    ax_curve.tick_params(axis="y", labelcolor=CORAL)
    ax2.tick_params(axis="y",      labelcolor=TEAL)
    ax_curve.set_title("E4 — Chất lượng tái dựng PCA vs k components")
    ax_curve.set_xticks(ks)
    ax_curve.xaxis.grid(True)
    for xi, yi in zip(ks, mse):
        ax_curve.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                          xytext=(0, 10), ha="center", fontsize=9, color=CORAL)
    for xi, yi in zip(ks, ssim):
        ax2.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=9, color=TEAL)
    lines = [l1, l2]; labels = [l.get_label() for l in lines]
    ax_curve.legend(lines, labels, loc="center right")

    # Explained variance bar
    bcols = [TEAL if v >= 98 else (GOLD if v >= 93 else BLUE) for v in expv]
    bb = ax_var.bar([str(k) for k in ks], expv, color=bcols, alpha=0.88, width=0.45)
    ax_var.set_xlabel("k"); ax_var.set_ylabel("Explained Variance (%)")
    ax_var.set_ylim(70, 103); ax_var.yaxis.grid(True)
    ax_var.set_title("E4 — Cumulative Explained Variance")
    ax_var.axhline(95, color=GOLD, ls="--", lw=1.3, label="95% threshold")
    ax_var.legend(fontsize=9)
    for bar, v in zip(bb, expv):
        ax_var.text(bar.get_x()+bar.get_width()/2, v+0.4,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=TEXT)


# ════════════════════════════════════════════════════════════════════════════
# Dashboard tổng hợp
# ════════════════════════════════════════════════════════════════════════════
def build_dashboard():
    fig = plt.figure(figsize=(22, 20), facecolor=BG)
    fig.suptitle("FaceVision AI — Kết Quả Thực Nghiệm E1–E4",
                 fontsize=18, fontweight="bold", color=TEAL, y=0.98)

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.52, wspace=0.38,
                           top=0.94, bottom=0.04, left=0.07, right=0.96)

    ax_e1_recall = fig.add_subplot(gs[0, 0])
    ax_e1_fps    = fig.add_subplot(gs[0, 1])
    ax_e2_bar    = fig.add_subplot(gs[1, 0])
    ax_e2_dist   = fig.add_subplot(gs[1, 1])
    ax_e3_f1     = fig.add_subplot(gs[2, 0])
    ax_e3_sw     = fig.add_subplot(gs[2, 1])
    ax_e4_curve  = fig.add_subplot(gs[3, 0])
    ax_e4_var    = fig.add_subplot(gs[3, 1])

    plot_e1(ax_e1_recall, ax_e1_fps)
    plot_e2(ax_e2_bar,    ax_e2_dist)
    plot_e3(ax_e3_f1,     ax_e3_sw)
    plot_e4(ax_e4_curve,  ax_e4_var)

    out = FIG_DIR / "dashboard_full.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ════════════════════════════════════════════════════════════════════════════
# Individual charts per experiment
# ════════════════════════════════════════════════════════════════════════════
def build_individual():
    for tag, fn, size in [
        ("E1", lambda f: plot_e1(f.add_subplot(1,2,1), f.add_subplot(1,2,2)), (16,6)),
        ("E2", lambda f: plot_e2(f.add_subplot(1,2,1), f.add_subplot(1,2,2)), (16,6)),
        ("E3", lambda f: plot_e3(f.add_subplot(1,2,1), f.add_subplot(1,2,2)), (16,6)),
        ("E4", lambda f: plot_e4(f.add_subplot(1,2,1), f.add_subplot(1,2,2)), (16,6)),
    ]:
        fig = plt.figure(figsize=size, facecolor=BG)
        fig.suptitle(f"FaceVision AI — Thực nghiệm {tag}",
                     fontsize=14, color=TEAL, fontweight="bold")
        fn(fig)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out = FIG_DIR / f"chart_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating visualizations...")
    build_individual()
    build_dashboard()
    print("\nDone! Files in experiments/figures/")
