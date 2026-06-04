"""
Face quality scoring and data augmentation.

Quality metrics used to filter bad frames before saving embeddings:
- Sharpness : Laplacian variance (blurry face = bad embedding)
- Brightness: distance from ideal mean (too dark/bright = feature loss)
- Contrast  : std deviation (low contrast = flat face, hard to distinguish)
- Size      : face pixel area (too small = poor detail)
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List


# ── Quality scoring ─────────────────────────────────────────────────

def score_quality(face_bgr: np.ndarray) -> Dict[str, float]:
    """Return quality metrics [0-1] for a face crop. Higher = better."""
    gray = (cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            if len(face_bgr.shape) == 3 else face_bgr)

    # Sharpness: Laplacian variance — sharp face ≥ 100, blurry < 30
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = min(lap_var / 150.0, 1.0)

    # Brightness: ideal mean ≈ 110-140 (not too dark, not overexposed)
    mean_b = float(gray.mean())
    brightness = max(0.0, 1.0 - abs(mean_b - 125.0) / 100.0)

    # Contrast: std dev — good face ≥ 35
    contrast = min(float(gray.std()) / 55.0, 1.0)

    # Size: at least 80×80 pixels for reliable embedding
    h, w = face_bgr.shape[:2]
    size = min((h * w) / (80.0 * 80.0), 1.0)

    overall = 0.40 * sharpness + 0.25 * brightness + 0.25 * contrast + 0.10 * size

    return {
        "sharpness":  round(sharpness,  3),
        "brightness": round(brightness, 3),
        "contrast":   round(contrast,   3),
        "size":       round(size,       3),
        "overall":    round(overall,    3),
    }


def quality_label(score: float) -> str:
    if score >= 0.70: return "excellent"
    if score >= 0.50: return "good"
    if score >= 0.35: return "fair"
    return "poor"


# ── Data augmentation ────────────────────────────────────────────────

def augment_face(face_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Generate 5 augmented variants from one face crop.
    Covers: mirror, brightness ±, slight rotation ±.
    Returns list including the original as first element.
    """
    h, w = face_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    variants: List[np.ndarray] = [face_bgr]

    # Mirror — simulates opposite pose angle
    variants.append(cv2.flip(face_bgr, 1))

    # Brighter (+20 %) — indoor vs outdoor lighting
    variants.append(cv2.convertScaleAbs(face_bgr, alpha=1.20, beta=15))

    # Darker  (-20 %) — shadow / backlight
    variants.append(cv2.convertScaleAbs(face_bgr, alpha=0.80, beta=-15))

    # Slight clockwise rotation (+6°)
    M = cv2.getRotationMatrix2D((cx, cy), 6, 1.0)
    variants.append(cv2.warpAffine(face_bgr, M, (w, h),
                                    borderMode=cv2.BORDER_REPLICATE))

    # Slight counter-clockwise rotation (-6°)
    M = cv2.getRotationMatrix2D((cx, cy), -6, 1.0)
    variants.append(cv2.warpAffine(face_bgr, M, (w, h),
                                    borderMode=cv2.BORDER_REPLICATE))

    return variants   # 6 total (original + 5 augmented)


# ── Diversity check ──────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2]. Values near 0 = nearly identical embeddings."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(1.0 - np.dot(a, b))


def is_diverse_enough(new_emb: np.ndarray,
                      existing: List[np.ndarray],
                      min_dist: float = 0.08) -> bool:
    """
    Return True if new_emb is sufficiently different from all existing embeddings.
    Prevents capturing the same expression/pose repeatedly.
    """
    if not existing:
        return True
    for e in existing:
        if cosine_distance(new_emb, e) < min_dist:
            return False
    return True
