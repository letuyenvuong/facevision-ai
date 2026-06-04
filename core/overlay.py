import cv2
import numpy as np
from typing import List, Optional, Tuple

from config import EMOTION_COLORS, DEFAULT_COLOR
from utils.face_db import FaceRegion, EmotionResult


def _get_emotion_color(emotion: str) -> Tuple[int, int, int]:
    return EMOTION_COLORS.get(emotion.lower(), DEFAULT_COLOR)


def draw_face_box(
    frame: np.ndarray,
    region: FaceRegion,
    name: str,
    confidence: float,
    emotion: Optional[EmotionResult] = None,
) -> None:
    x, y, w, h = region.x, region.y, region.w, region.h
    color = _get_emotion_color(emotion.dominant if emotion else "neutral")

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Identity badge — only show % when actually recognised
    label = name if name == "Unknown" else f"{name} ({confidence:.0%})"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    bg_y1 = max(0, y - th - baseline - 6)
    cv2.rectangle(frame, (x, bg_y1), (x + tw + 6, y), color, -1)
    cv2.putText(frame, label, (x + 3, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    if emotion:
        _draw_emotion_bar(frame, x, y + h + 4, w, emotion)


def _draw_emotion_bar(
    frame: np.ndarray,
    x: int, y: int, width: int,
    emotion: EmotionResult,
) -> None:
    top_emotions = sorted(emotion.scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    bar_h = 12
    gap = 2
    for i, (emo, score) in enumerate(top_emotions):
        bar_y = y + i * (bar_h + gap)
        bar_w = int(width * score)
        color = _get_emotion_color(emo)
        cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"{emo[:3]} {score:.0%}", (x + bar_w + 3, bar_y + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    label = f"FPS: {fps:.1f}"
    cv2.putText(frame, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2, cv2.LINE_AA)


def draw_face_count(frame: np.ndarray, count: int) -> None:
    label = f"Faces: {count}"
    cv2.putText(frame, label, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)


def draw_reconstruction_thumbnail(
    frame: np.ndarray,
    region: FaceRegion,
    recon: np.ndarray,
    size: int = 48,
) -> None:
    thumb = cv2.resize(recon, (size, size))
    if len(thumb.shape) == 2:
        thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    x2 = region.x + region.w
    y1 = region.y
    if x2 + size < frame.shape[1] and y1 + size < frame.shape[0]:
        frame[y1:y1 + size, x2:x2 + size] = thumb
