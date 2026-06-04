from __future__ import annotations

import os
import pickle
from typing import Optional, List

import cv2
import numpy as np

from config import PCA_COMPONENTS, FACE_SIZE, PCA_MODEL_PATH, EIGENFACES_DIR
from utils.logger import get_logger

logger = get_logger("reconstruction")


class FaceReconstructor:
    def __init__(self):
        self._pca = None
        self._mean_face: Optional[np.ndarray] = None
        self._trained = False
        self._load_model()

    def _load_model(self) -> None:
        if os.path.exists(PCA_MODEL_PATH):
            try:
                with open(PCA_MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                self._pca = data["pca"]
                self._mean_face = data["mean_face"]
                self._trained = True
                logger.info(f"PCA model loaded from {PCA_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load PCA model: {e}")
        else:
            logger.info("No PCA model found — reconstruction disabled until trained")

    def is_trained(self) -> bool:
        return self._trained

    def train(self, face_images: List[np.ndarray], n_components: int = PCA_COMPONENTS) -> None:
        from sklearn.decomposition import PCA
        vectors = []
        for img in face_images:
            v = self._preprocess(img)
            if v is not None:
                vectors.append(v)
        if len(vectors) < n_components:
            logger.warning(f"Not enough samples ({len(vectors)}) for {n_components} components")
            n_components = max(1, len(vectors) - 1)
        X = np.array(vectors)
        self._mean_face = X.mean(axis=0)
        X_centered = X - self._mean_face
        pca = PCA(n_components=n_components, whiten=False)
        pca.fit(X_centered)
        self._pca = pca
        self._trained = True
        self._save_model()
        logger.info(f"PCA trained: {n_components} components, {len(vectors)} samples")

    def _save_model(self) -> None:
        os.makedirs(EIGENFACES_DIR, exist_ok=True)
        with open(PCA_MODEL_PATH, "wb") as f:
            pickle.dump({"pca": self._pca, "mean_face": self._mean_face}, f)
        logger.info(f"PCA model saved to {PCA_MODEL_PATH}")

    def _preprocess(self, face: np.ndarray) -> Optional[np.ndarray]:
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
            resized = cv2.resize(gray, FACE_SIZE)
            return resized.astype(np.float32).flatten() / 255.0
        except Exception:
            return None

    def reconstruct(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        if not self._trained:
            return None
        v = self._preprocess(face_crop)
        if v is None:
            return None
        centered = v - self._mean_face
        projected = self._pca.transform(centered.reshape(1, -1))
        reconstructed = self._pca.inverse_transform(projected).flatten() + self._mean_face
        reconstructed = np.clip(reconstructed, 0, 1)
        img = (reconstructed * 255).astype(np.uint8).reshape(FACE_SIZE)
        return img

    def get_reconstruction_error(self, face_crop: np.ndarray) -> float:
        if not self._trained:
            return 0.0
        v = self._preprocess(face_crop)
        if v is None:
            return 0.0
        centered = v - self._mean_face
        projected = self._pca.transform(centered.reshape(1, -1))
        reconstructed = self._pca.inverse_transform(projected).flatten() + self._mean_face
        return float(np.mean((v - reconstructed) ** 2))
