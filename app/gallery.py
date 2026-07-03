"""
app/gallery.py — In-memory match index + alert confirmation logic.

Gallery         All enrolled embeddings as one numpy matrix (rebuilt from the
                DB whenever persons/photos/assignments change). Matching one
                probe against thousands of enrolled photos is a single matmul —
                sub-millisecond; no ANN index needed at this scale.

ConfirmTracker  The K-of-N + cooldown state machine that turns raw per-frame
                matches into at most one alert per sighting:
                  • a person must match ≥ confirm_hits times within
                    confirm_window_s on the SAME camera before an alert fires
                    (single-frame flickers can never alert), and
                  • after firing, that (camera, person) pair is silenced for
                    cooldown_s.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from settings import settings


@dataclass
class Match:
    person_id: int
    similarity: float


class Gallery:
    def __init__(self, db, engine):
        self._db = db
        self._engine = engine
        self._lock = threading.Lock()
        self._matrix = np.zeros((0, 1), np.float32)   # (n_photos, dim)
        self._person_ids: list[int] = []              # row → person_id
        self._by_camera: dict[str, set[int]] = {}     # camera → allowed persons
        self.reload()

    def reload(self) -> None:
        """Rebuild the index from the DB (call after any enrollment change)."""
        if self._engine is None:
            return
        pairs = self._db.all_embeddings(self._engine.name, self._engine.dim)
        by_cam = self._db.assignments_by_camera()
        with self._lock:
            if pairs:
                self._matrix = np.stack([e for _, e in pairs]).astype(np.float32)
                self._person_ids = [pid for pid, _ in pairs]
            else:
                self._matrix = np.zeros((0, self._engine.dim), np.float32)
                self._person_ids = []
            self._by_camera = by_cam

    def cameras_watched(self) -> set[str]:
        with self._lock:
            return set(self._by_camera.keys())

    def size(self) -> dict:
        with self._lock:
            return {"photos_indexed": len(self._person_ids),
                    "persons_indexed": len(set(self._person_ids)),
                    "cameras_watched": sorted(self._by_camera.keys())}

    def match(self, embedding: np.ndarray, camera_id: str) -> Optional[Match]:
        """Best match for this probe among persons ASSIGNED to this camera."""
        threshold = self._engine.match_threshold
        with self._lock:
            allowed = self._by_camera.get(camera_id)
            if not allowed or self._matrix.shape[0] == 0:
                return None
            sims = self._matrix @ embedding.astype(np.float32)
            # max over each person's photos, restricted to this camera's list
            best_pid, best_sim = -1, -1.0
            for i, pid in enumerate(self._person_ids):
                if pid in allowed and sims[i] > best_sim:
                    best_sim, best_pid = float(sims[i]), pid
        if best_pid != -1 and best_sim >= threshold:
            return Match(person_id=best_pid, similarity=best_sim)
        return None


class ConfirmTracker:
    """K-of-N temporal confirmation + per-(camera, person) cooldown."""

    def __init__(self, hits: int | None = None, window_s: float | None = None,
                 cooldown_s: float | None = None):
        self.hits = settings.confirm_hits if hits is None else hits
        self.window = settings.confirm_window_s if window_s is None else window_s
        self.cooldown = settings.cooldown_s if cooldown_s is None else cooldown_s
        self._lock = threading.Lock()
        self._recent: dict[tuple[str, int], deque] = {}
        self._last_alert: dict[tuple[str, int], float] = {}

    def register(self, camera_id: str, person_id: int, ts: float) -> bool:
        """Record one raw match; True when the alert should fire NOW."""
        key = (camera_id, person_id)
        with self._lock:
            if ts - self._last_alert.get(key, -1e12) < self.cooldown:
                return False
            dq = self._recent.setdefault(key, deque())
            dq.append(ts)
            while dq and ts - dq[0] > self.window:
                dq.popleft()
            if len(dq) >= self.hits:
                self._last_alert[key] = ts
                dq.clear()
                return True
            return False
