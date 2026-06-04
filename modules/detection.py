from __future__ import annotations

import threading
from typing import List, Optional

import cv2
import numpy as np

from config import DETECTION_BACKEND, MIN_FACE_SIZE
from utils.face_db import FaceRegion
from utils.image_utils import resize_for_detection
from utils.logger import get_logger

logger = get_logger("detection")

_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceDetector:
    def __init__(self, backend: str = DETECTION_BACKEND):
        self.backend = backend
        self._mtcnn = None
        self._haar: Optional[cv2.CascadeClassifier] = None
        # Serialize all calls — MTCNN is not thread-safe
        self._lock = threading.Lock()
        self._init_backend()

    def _init_backend(self) -> None:
        if self.backend == "mtcnn":
            try:
                from mtcnn import MTCNN
                self._mtcnn = MTCNN()
                logger.info("MTCNN detector initialized")
            except Exception as e:
                logger.warning(f"MTCNN unavailable ({e}), falling back to Haar")
                self.backend = "opencv"
        if self.backend in ("opencv", "haar") or self._mtcnn is None:
            self._haar = cv2.CascadeClassifier(_HAAR_PATH)
            if self._haar.empty():
                raise RuntimeError(f"Cannot load Haar cascade from {_HAAR_PATH}")
            logger.info("Haar cascade detector initialized")

    def detect(self, frame: np.ndarray) -> List[FaceRegion]:
        with self._lock:
            if self._mtcnn is not None:
                try:
                    return self._detect_mtcnn(frame)
                except Exception as e:
                    logger.warning(f"MTCNN error: {e}, falling back to Haar")
            return self._detect_haar(frame)

    def _detect_mtcnn(self, frame: np.ndarray) -> List[FaceRegion]:
        small = resize_for_detection(frame, scale=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        detections = self._mtcnn.detect_faces(rgb)
        results = []
        for d in detections:
            if d["confidence"] < 0.85:
                continue
            bx, by, bw, bh = d["box"]
            x = max(0, bx * 2)
            y = max(0, by * 2)
            w, h = bw * 2, bh * 2
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue
            kp = d.get("keypoints", {})
            landmarks = {
                "left_eye":  (kp["left_eye"][0]  * 2, kp["left_eye"][1]  * 2) if "left_eye"  in kp else None,
                "right_eye": (kp["right_eye"][0] * 2, kp["right_eye"][1] * 2) if "right_eye" in kp else None,
                "nose":      (kp["nose"][0]       * 2, kp["nose"][1]       * 2) if "nose"      in kp else None,
            }
            results.append(FaceRegion(x=x, y=y, w=w, h=h,
                                      confidence=d["confidence"], landmarks=landmarks))
        return results

    def _detect_haar(self, frame: np.ndarray) -> List[FaceRegion]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        )
        results = []
        if len(faces):
            for (x, y, w, h) in faces:
                results.append(FaceRegion(x=int(x), y=int(y), w=int(w), h=int(h), confidence=0.9))
        return results
