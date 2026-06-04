from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import EMBEDDINGS_DIR
from utils.logger import get_logger

logger = get_logger("face_db")

# Strip timestamp + optional sequence suffix: _YYYYMMDD_HHMMSS[_NN]
_TS_RE = re.compile(r"_\d{8}_\d{6}(?:_\d{2,3})?$")


@dataclass
class FaceRegion:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None


@dataclass
class EmotionResult:
    dominant: str = "neutral"
    scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class FaceRecord:
    name: str
    embedding: np.ndarray
    source_file: str = ""


class FaceDatabase:
    def __init__(self, embeddings_dir: str = EMBEDDINGS_DIR):
        self.embeddings_dir = embeddings_dir
        self._records: List[FaceRecord] = []
        self.load()

    # ── Load / Save ──────────────────────────────────────────────────

    def load(self) -> None:
        self._records = []
        for fp in sorted(glob.glob(os.path.join(self.embeddings_dir, "*.npy"))):
            try:
                embedding = np.load(fp)
                stem = os.path.splitext(os.path.basename(fp))[0]
                name = _TS_RE.sub("", stem) or stem
                self._records.append(FaceRecord(name=name, embedding=embedding,
                                                source_file=fp))
            except Exception as e:
                logger.warning(f"Failed to load {fp}: {e}")
        logger.info(f"Loaded {len(self._records)} embeddings "
                    f"({len(self.get_names())} person(s))")

    def reload(self) -> None:
        self.load()

    def save(self, name: str, embedding: np.ndarray, timestamp: str) -> str:
        os.makedirs(self.embeddings_dir, exist_ok=True)
        filepath = os.path.join(self.embeddings_dir, f"{name}_{timestamp}.npy")
        np.save(filepath, embedding)
        self._records.append(FaceRecord(name=name, embedding=embedding,
                                        source_file=filepath))
        return filepath

    def delete(self, name: str) -> int:
        deleted = 0
        remaining = []
        for r in self._records:
            if r.name == name:
                try:
                    os.remove(r.source_file)
                    deleted += 1
                except OSError as e:
                    logger.warning(f"Could not delete {r.source_file}: {e}")
            else:
                remaining.append(r)
        self._records = remaining
        logger.info(f"Deleted {deleted} embedding(s) for '{name}'")
        return deleted

    # ── Queries ──────────────────────────────────────────────────────

    def get_all(self) -> List[FaceRecord]:
        return list(self._records)

    def get_names(self) -> List[str]:
        return sorted({r.name for r in self._records})

    def get_stats(self) -> Dict[str, int]:
        """Return {name: embedding_count} for all registered persons."""
        stats: Dict[str, int] = {}
        for r in self._records:
            stats[r.name] = stats.get(r.name, 0) + 1
        return stats

    # ── Matching ─────────────────────────────────────────────────────

    def find_closest(self, query: np.ndarray, threshold: float) -> Tuple[str, float]:
        """
        Per-person minimum cosine distance matching.

        Groups all embeddings by person, computes the best (minimum) distance
        for each person, then picks the globally best person.  This prevents
        a person with many embeddings having an unfair advantage over others.
        """
        if not self._records:
            return "Unknown", 1.0

        q_norm = query / (np.linalg.norm(query) + 1e-10)

        # Build per-person list of normalised embeddings
        per_person: Dict[str, List[np.ndarray]] = {}
        for record in self._records:
            e_norm = record.embedding / (np.linalg.norm(record.embedding) + 1e-10)
            per_person.setdefault(record.name, []).append(e_norm)

        # Best person = minimum cosine distance across their embeddings
        best_name = "Unknown"
        best_dist = float("inf")
        for name, embs in per_person.items():
            # Vectorised: compute dot products for all embeddings at once
            dots = np.array([float(np.dot(q_norm, e)) for e in embs])
            min_dist = float(1.0 - dots.max())          # max dot = min dist
            if min_dist < best_dist:
                best_dist = min_dist
                best_name = name

        if best_dist > threshold:
            return "Unknown", best_dist
        return best_name, best_dist
