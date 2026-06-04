"""
Pipeline architecture:
  Main thread  → detect faces every frame (fast)
               → push frame+regions to inference queue (non-blocking, drop if busy)
               → match current regions with cached results by face_id
               → draw overlays → return annotated frame

  Inference thread (background)
               → pull from queue → recognition + emotion + reconstruction
               → update result cache (thread-safe)

This ensures the video stream is never blocked by slow DeepFace inference.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import UNKNOWN_LABEL
from utils.face_db import FaceRegion, EmotionResult
from utils.logger import get_logger

logger = get_logger("pipeline")

_IOU_THRESHOLD = 0.25   # minimum IoU to consider two regions as the same face


def _iou(a: FaceRegion, b: FaceRegion) -> float:
    x1 = max(a.x, b.x);  y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w);  y2 = min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


class FacePipeline:
    def __init__(self):
        self._frame_count = 0
        self._fps_counter = 0
        self._last_fps_time = time.monotonic()
        self.current_fps = 0.0
        self._current_face_count = 0

        self._detector = None
        self._recognizer = None
        self._emotion_analyzer = None
        self._reconstructor = None

        # Face ID tracking
        self._face_id_counter = 0
        self._prev_tracked: List[Tuple[FaceRegion, int]] = []  # (region, face_id)

        # Background inference
        self._infer_queue: queue.Queue = queue.Queue(maxsize=1)
        self._result_cache: List[dict] = []
        self._result_lock = threading.Lock()

        self._load_modules()

        self._infer_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="inference"
        )
        self._infer_thread.start()

    # ------------------------------------------------------------------
    # Module loading
    # ------------------------------------------------------------------

    def _load_modules(self) -> None:
        specs = [
            ("_detector",         "modules.detection",      "FaceDetector"),
            ("_recognizer",       "modules.recognition",    "FaceRecognizer"),
            ("_emotion_analyzer", "modules.emotion",        "EmotionAnalyzer"),
            ("_reconstructor",    "modules.reconstruction", "FaceReconstructor"),
        ]
        import importlib
        for attr, mod_name, cls_name in specs:
            try:
                mod = importlib.import_module(mod_name)
                setattr(self, attr, getattr(mod, cls_name)())
                logger.info(f"{cls_name} loaded")
            except Exception as e:
                logger.error(f"{cls_name} failed to load: {e}")

    # ------------------------------------------------------------------
    # Main entry point — called from Flask stream thread
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        self._frame_count += 1
        self._update_fps()

        # 1. Detect faces (fast — every frame)
        regions: List[FaceRegion] = []
        if self._detector:
            try:
                regions = self._detector.detect(frame)
            except Exception as e:
                logger.warning(f"Detection error: {e}")

        self._current_face_count = len(regions)

        # 2. Assign stable face IDs via IoU tracking
        face_ids = self._assign_ids(regions)

        # 3. Push to inference thread (drop if busy — non-blocking)
        try:
            self._infer_queue.put_nowait((frame.copy(), regions[:], face_ids[:]))
        except queue.Full:
            pass

        # 4. Merge live detection regions with latest cached inference results
        with self._result_lock:
            cache = list(self._result_cache)

        active = self._merge(regions, face_ids, cache)

        # 5. Draw overlays
        annotated = frame.copy()
        from core.overlay import (
            draw_face_box, draw_fps, draw_face_count,
            draw_reconstruction_thumbnail,
        )
        for r in active:
            draw_face_box(
                annotated, r["region"],
                r["name"], 1.0 - r["distance"],
                r.get("emotion"),
            )
            if r.get("reconstruction") is not None:
                draw_reconstruction_thumbnail(annotated, r["region"], r["reconstruction"])

        draw_fps(annotated, self.current_fps)
        draw_face_count(annotated, len(active))

        return annotated, active

    # ------------------------------------------------------------------
    # Face ID tracker (IoU-based)
    # ------------------------------------------------------------------

    def _assign_ids(self, regions: List[FaceRegion]) -> List[int]:
        ids = []
        used = set()

        for region in regions:
            best_iou = _IOU_THRESHOLD
            best_j = -1
            for j, (prev_r, _) in enumerate(self._prev_tracked):
                if j in used:
                    continue
                iou = _iou(region, prev_r)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0:
                fid = self._prev_tracked[best_j][1]
                used.add(best_j)
            else:
                fid = self._face_id_counter
                self._face_id_counter += 1

            ids.append(fid)

        self._prev_tracked = list(zip(regions, ids))
        return ids

    # ------------------------------------------------------------------
    # Merge live regions with cached inference (by face_id)
    # ------------------------------------------------------------------

    def _merge(
        self,
        regions: List[FaceRegion],
        face_ids: List[int],
        cache: List[dict],
    ) -> List[dict]:
        by_id: Dict[int, dict] = {r["face_id"]: r for r in cache if "face_id" in r}
        results = []
        for region, fid in zip(regions, face_ids):
            if fid in by_id:
                c = by_id[fid]
                results.append({
                    "region":         region,          # current bbox (tracks correctly)
                    "face_id":        fid,
                    "name":           c.get("name", UNKNOWN_LABEL),
                    "distance":       c.get("distance", 1.0),
                    "emotion":        c.get("emotion", EmotionResult()),
                    "reconstruction": c.get("reconstruction"),
                })
            else:
                results.append({
                    "region":         region,
                    "face_id":        fid,
                    "name":           UNKNOWN_LABEL,
                    "distance":       1.0,
                    "emotion":        EmotionResult(),
                    "reconstruction": None,
                })
        return results

    # ------------------------------------------------------------------
    # Background inference thread
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        from utils.image_utils import crop_face

        while True:
            try:
                frame, regions, face_ids = self._infer_queue.get()
            except Exception:
                continue

            new_cache = []
            for region, fid in zip(regions, face_ids):
                entry: dict = {
                    "face_id":        fid,
                    "region":         region,
                    "name":           UNKNOWN_LABEL,
                    "distance":       1.0,
                    "emotion":        EmotionResult(),
                    "reconstruction": None,
                }

                # Recognition: use padded crop (0.2) for better alignment
                crop_recog = crop_face(frame, region.x, region.y, region.w, region.h, padding=0.2)
                # Emotion: tight crop (0.05) — less background noise
                crop_emo   = crop_face(frame, region.x, region.y, region.w, region.h, padding=0.05)

                if crop_recog is None or crop_recog.size == 0:
                    new_cache.append(entry)
                    continue

                if self._recognizer:
                    try:
                        name, dist = self._recognizer.identify(crop_recog)
                        entry["name"] = name
                        entry["distance"] = dist
                    except Exception as e:
                        logger.warning(f"Recognition: {e}")

                if self._emotion_analyzer and crop_emo is not None and crop_emo.size > 0:
                    try:
                        entry["emotion"] = self._emotion_analyzer.analyze(crop_emo, face_id=fid)
                    except Exception as e:
                        logger.warning(f"Emotion: {e}")

                if self._reconstructor and self._reconstructor.is_trained():
                    try:
                        entry["reconstruction"] = self._reconstructor.reconstruct(crop_recog)
                    except Exception as e:
                        logger.warning(f"Reconstruction: {e}")

                new_cache.append(entry)

            with self._result_lock:
                self._result_cache = new_cache

    # ------------------------------------------------------------------
    # FPS counter
    # ------------------------------------------------------------------

    def _update_fps(self) -> None:
        self._fps_counter += 1
        now = time.monotonic()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self.current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._last_fps_time = now

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        with self._result_lock:
            cache = list(self._result_cache)

        emotions = [r["emotion"].dominant for r in cache if r.get("emotion")]
        dominant_emotion = max(set(emotions), key=emotions.count) if emotions else None

        return {
            "fps":             round(self.current_fps, 1),
            "frame_count":     self._frame_count,
            "face_count":      self._current_face_count,
            "dominant_emotion": dominant_emotion,
            "modules": {
                "detection":     self._detector is not None,
                "recognition":   self._recognizer is not None,
                "emotion":       self._emotion_analyzer is not None,
                "reconstruction":self._reconstructor is not None,
            },
        }
