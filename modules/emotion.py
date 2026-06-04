from __future__ import annotations

import threading
import tensorflow  # must be imported before deepface on Windows
from collections import deque
from typing import Dict, Deque

import numpy as np

from utils.face_db import EmotionResult
from utils.logger import get_logger

logger = get_logger("emotion")

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SMOOTH_WINDOW = 2


class EmotionTracker:
    def __init__(self, window: int = SMOOTH_WINDOW):
        self._history: Deque[Dict[str, float]] = deque(maxlen=window)

    def update(self, scores: Dict[str, float]) -> Dict[str, float]:
        self._history.append(scores)
        return {
            label: sum(h.get(label, 0.0) for h in self._history) / len(self._history)
            for label in EMOTION_LABELS
        }


class EmotionAnalyzer:
    def __init__(self):
        self._trackers: Dict[int, EmotionTracker] = {}
        # Emotion Keras model is NOT thread-safe — serialize all inference calls
        self._infer_lock = threading.Lock()
        logger.info("EmotionAnalyzer ready")

    def analyze(self, face_crop: np.ndarray, face_id: int = 0) -> EmotionResult:
        h, w = face_crop.shape[:2]
        if h < 20 or w < 20:
            return EmotionResult()

        try:
            from deepface import DeepFace
            # Acquire lock before calling into TF/Keras
            with self._infer_lock:
                result = DeepFace.analyze(
                    img_path=face_crop,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="skip",
                    silent=True,
                )
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            return EmotionResult()

        if isinstance(result, list):
            result = result[0]

        raw: Dict[str, float] = {
            k.lower(): v / 100.0
            for k, v in result.get("emotion", {}).items()
        }

        if face_id not in self._trackers:
            self._trackers[face_id] = EmotionTracker()
        smoothed = self._trackers[face_id].update(raw)

        dominant = max(smoothed, key=smoothed.get, default="neutral")
        return EmotionResult(
            dominant=dominant,
            scores=smoothed,
            confidence=smoothed.get(dominant, 0.0),
        )
