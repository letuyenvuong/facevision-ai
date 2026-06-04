from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import List, Optional, Tuple

import tensorflow  # must be imported before deepface on Windows
import cv2
import numpy as np

from config import RECOGNITION_MODEL, RECOGNITION_THRESHOLD, UNKNOWN_LABEL
from utils.face_db import FaceDatabase
from utils.logger import get_logger

logger = get_logger("recognition")

_ARCFACE_SIZE = (112, 112)
_QUALITY_MIN  = 0.30   # reject frames below this overall quality score


class FaceRecognizer:
    def __init__(self):
        self.model_name = RECOGNITION_MODEL
        self.threshold  = RECOGNITION_THRESHOLD
        self.db         = FaceDatabase()
        # ArcFace Keras model is NOT thread-safe — one lock per instance
        self._infer_lock = threading.Lock()
        logger.info(f"FaceRecognizer ready — model={self.model_name}, "
                    f"threshold={self.threshold}")

    # ── Embedding extraction ─────────────────────────────────────────

    def extract_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        from deepface import DeepFace

        h, w = face_crop.shape[:2]
        if h < 20 or w < 20:
            raise ValueError(f"Face crop too small: {w}x{h}")

        face_112 = cv2.resize(face_crop, _ARCFACE_SIZE)

        with self._infer_lock:   # serialize TF/Keras calls
            result = DeepFace.represent(
                img_path=face_112,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend="skip",
            )
        return np.array(result[0]["embedding"], dtype=np.float32)

    # ── Single-shot registration (from file) ─────────────────────────

    def register(self, name: str, image_path: str, detector=None) -> str:
        """Register from an image file path. Detects face then augments."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        face_crop = self._detect_and_crop(img, detector=detector)
        n = self.register_from_crop(name, face_crop, augment=True)
        logger.info(f"register(): saved {n} embeddings for '{name}'")
        return f"{n} embeddings saved"

    def _detect_and_crop(self, img: np.ndarray, detector=None) -> np.ndarray:
        """Detect largest face and return crop; fall back to full image."""
        try:
            if detector is None:
                from modules.detection import FaceDetector
                detector = FaceDetector(backend="opencv")   # Haar — thread-safe
            regions = detector.detect(img)
            if regions:
                best = max(regions, key=lambda r: r.w * r.h)
                from utils.image_utils import crop_face
                crop = crop_face(img, best.x, best.y, best.w, best.h, padding=0.1)
                if crop is not None and crop.size > 0:
                    logger.info(f"  face {best.w}x{best.h} conf={best.confidence:.2f}")
                    return crop
        except Exception as e:
            logger.warning(f"Detection during registration failed: {e}")
        logger.warning("No face detected — using full image as fallback")
        return img

    # ── Crop-based registration (used by auto-capture) ───────────────

    def register_from_crop(
        self,
        name: str,
        face_crop: np.ndarray,
        augment: bool = True,
    ) -> int:
        """
        Register directly from a face crop (no file needed).
        Applies quality check then generates augmented variants.
        Returns number of embeddings saved.
        """
        from utils.face_quality import score_quality, augment_face

        quality = score_quality(face_crop)
        if quality["overall"] < _QUALITY_MIN:
            raise ValueError(
                f"Face quality too low ({quality['overall']:.0%}): "
                f"sharpness={quality['sharpness']:.0%}, "
                f"brightness={quality['brightness']:.0%}"
            )

        crops = augment_face(face_crop) if augment else [face_crop]

        saved = 0
        for i, crop in enumerate(crops):
            try:
                embedding = self.extract_embedding(crop)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.db.save(name, embedding, f"{ts}_{i:02d}")
                saved += 1
            except Exception as e:
                logger.warning(f"  aug[{i}] embedding failed: {e}")

        logger.info(f"register_from_crop('{name}'): {saved}/{len(crops)} embeddings saved")
        return saved

    def register_many(
        self,
        name: str,
        face_crops: List[np.ndarray],
        augment: bool = True,
    ) -> int:
        """Register from a list of diverse face crops. Returns total saved."""
        total = 0
        for crop in face_crops:
            try:
                total += self.register_from_crop(name, crop, augment=augment)
            except Exception as e:
                logger.warning(f"  crop rejected: {e}")
        logger.info(f"register_many('{name}'): {total} total embeddings from "
                    f"{len(face_crops)} crops")
        return total

    # ── Identification ───────────────────────────────────────────────

    def identify(self, face_crop: np.ndarray) -> Tuple[str, float]:
        try:
            embedding = self.extract_embedding(face_crop)
        except Exception as e:
            logger.warning(f"Embedding extraction failed: {e}")
            return UNKNOWN_LABEL, 1.0
        return self.db.find_closest(embedding, self.threshold)

    def reload_database(self) -> None:
        self.db.reload()
