"""
tests/conftest.py — shared fixtures for the blacklist test-suite.

Offline + deterministic: a FakeEngine stands in for the neural nets so every
production code path (enrollment gates, gallery matching, K-of-N confirmation,
cooldown, sighting persistence, API) is exercised without model weights.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Isolate all data (db/photos/crops) into a temp dir BEFORE settings import.
_TMP = tempfile.mkdtemp(prefix="bl-test-")
os.environ.setdefault("BL_DATA_DIR", _TMP)
os.environ.setdefault("CAMERAS_YAML", str(Path(_TMP) / "cameras.yaml"))

from settings import settings  # noqa: E402
from app.engine import DetectedFace  # noqa: E402

settings.ensure_dirs()

# A camera registry for tests.
Path(settings.cameras_yaml).write_text("""
cameras:
  - id: cam_door
    name: Door
    enabled: true
    source: "rtsp://example/door"
  - id: cam_till
    name: Till
    enabled: true
    source: "rtsp://example/till"
""")


class FakeEngine:
    """Deterministic engine: 'recognises' a face by the mean colour of the
    frame — same colour → same embedding. dim=8, backend name 'fake'."""
    name = "fake"
    dim = 8
    device = "cpu"
    match_threshold = 0.8

    def __init__(self):
        self.next_faces = None   # override per-call if set

    def _embed_from_colour(self, img) -> np.ndarray:
        # Mean-centred channel means → distinct colours give near-orthogonal
        # (even negative) similarities; grey/neutral frames land far from all
        # enrolled colours instead of spuriously close.
        means = img.mean(axis=(0, 1)).astype(np.float32)      # (b, g, r)
        c = means - float(means.mean())
        v = np.array([c[0], c[1], c[2], c[0], c[1], c[2], 1.0, 0.0], np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def analyze(self, img_bgr):
        if self.next_faces is not None:
            faces, self.next_faces = self.next_faces, None
            return faces
        h, w = img_bgr.shape[:2]
        if img_bgr.max() == 0:          # black frame → nobody in view
            return []
        return [DetectedFace(
            bbox=(w * 0.25, h * 0.1, w * 0.75, h * 0.9),
            score=0.95,
            embedding=self._embed_from_colour(img_bgr),
        )]


def solid_frame(bgr, h=480, w=640):
    """A solid-colour frame with noise (passes the blur gate)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bgr
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


@pytest.fixture()
def fake_engine():
    return FakeEngine()


@pytest.fixture()
def db(tmp_path):
    from app.db import BlacklistDB
    d = BlacklistDB(str(tmp_path / "test.db"))
    yield d
    d.close()


@pytest.fixture()
def state(db, fake_engine):
    """Full service state wired around the fake engine (no camera threads)."""
    from app.gallery import Gallery, ConfirmTracker
    from app.capture import FrameProcessor

    gallery = Gallery(db, fake_engine)
    confirm = ConfirmTracker(hits=3, window_s=5.0, cooldown_s=300.0)
    alerts: list = []
    processor = FrameProcessor(fake_engine, gallery, confirm, db,
                               on_alert=alerts.append)

    class S:
        pass
    s = S()
    s.db, s.engine, s.gallery, s.manager = db, fake_engine, gallery, None
    s.processor, s.confirm, s.alerts = processor, confirm, alerts
    return s
