import cv2
import numpy as np
from typing import Tuple, Optional


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_for_detection(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def normalize_face(face: np.ndarray, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    face_resized = cv2.resize(face, size)
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY) if len(face_resized.shape) == 3 else face_resized
    return face_gray.astype(np.float32) / 255.0


def crop_face(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    padding: float = 0.2
) -> Optional[np.ndarray]:
    fh, fw = frame.shape[:2]
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def align_face(
    frame: np.ndarray,
    left_eye: Tuple[int, int],
    right_eye: Tuple[int, int],
    output_size: Tuple[int, int] = (112, 112)
) -> np.ndarray:
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = (
        (left_eye[0] + right_eye[0]) // 2,
        (left_eye[1] + right_eye[1]) // 2,
    )
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    return cv2.resize(rotated, output_size)


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
