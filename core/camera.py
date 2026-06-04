import threading
import queue
import time
from typing import Optional, Union

import cv2
import numpy as np

from config import (
    CAMERA_SOURCE, FRAME_WIDTH, FRAME_HEIGHT, FPS_LIMIT,
    RTSP_RECONNECT_DELAY, RTSP_MAX_RECONNECT_DELAY,
)
from utils.logger import get_logger

logger = get_logger("camera")


def _source_type(source: Union[int, str]) -> str:
    if isinstance(source, int):
        return "webcam"
    s = str(source).lower()
    if s.startswith("rtsp://") or s.startswith("rtsps://"):
        return "rtsp"
    return "file"


class CameraStream:
    def __init__(
        self,
        source: Union[int, str] = CAMERA_SOURCE,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        fps_limit: int = FPS_LIMIT,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps_limit = fps_limit
        self.source_type = _source_type(source)

        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._latest_frame: Optional[np.ndarray] = None   # non-destructive snapshot
        self._latest_lock = threading.Lock()
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "CameraStream":
        self._cap = self._open()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Camera started [{self.source_type}]: {self.source}")
        return self

    def read(self) -> Optional[np.ndarray]:
        """Pop one frame from queue (blocking, used by stream generator)."""
        try:
            return self._frame_queue.get(timeout=0.15)
        except queue.Empty:
            return None

    def get_latest(self) -> Optional[np.ndarray]:
        """Return the most recent frame WITHOUT consuming from queue.
        Safe to call from any thread while stream is running."""
        with self._latest_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._release()
        logger.info("Camera stopped")

    def info(self) -> dict:
        res = "unknown"
        if self._cap:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w and h:
                res = f"{w}×{h}"
        return {
            "source":     str(self.source),
            "type":       self.source_type,
            "running":    self._running,
            "resolution": res,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> cv2.VideoCapture:
        if self.source_type == "rtsp":
            # Use FFMPEG backend for RTSP; TCP transport avoids UDP packet loss
            url = str(self.source)
            if "?" not in url and "&" not in url:
                # append transport hint if not already set
                pass
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 1-frame buffer → minimal latency
        else:
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps_limit)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source}")
        return cap

    def _release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def _capture_loop(self) -> None:
        interval = 1.0 / self.fps_limit
        reconnect_delay = RTSP_RECONNECT_DELAY

        while self._running:
            t0 = time.monotonic()
            ret, frame = self._cap.read()

            if not ret:
                if self.source_type in ("rtsp", "file") and self._running:
                    logger.warning(
                        f"Stream read failed — reconnecting in {reconnect_delay}s "
                        f"[{self.source_type}]"
                    )
                    self._release()
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, RTSP_MAX_RECONNECT_DELAY)
                    try:
                        self._cap = self._open()
                        reconnect_delay = RTSP_RECONNECT_DELAY  # reset on success
                        logger.info("Stream reconnected")
                    except Exception as e:
                        logger.error(f"Reconnect failed: {e}")
                    continue
                else:
                    logger.warning("Webcam read failed — stopping")
                    break

            # Reset backoff counter on successful read
            reconnect_delay = RTSP_RECONNECT_DELAY

            # Drop stale frame if queue is full (keep only latest)
            # Always keep latest frame for non-destructive access
            with self._latest_lock:
                self._latest_frame = frame.copy()

            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)

            # Throttle to fps_limit (mainly for webcam / file; RTSP has its own rate)
            elapsed = time.monotonic() - t0
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

        self._running = False
