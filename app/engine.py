"""
app/engine.py — Face detection + embedding engine.

Backend is resolved dynamically (FACE_BACKEND=auto|insightface|opencv):

  insightface — SCRFD detector + ArcFace 512-d embeddings (buffalo_l pack).
                Best accuracy; uses GPU automatically when onnxruntime exposes
                a CUDA provider, otherwise CPU.
  opencv      — YuNet detector + SFace 128-d embeddings (tiny ONNX models
                baked into the image). CPU-friendly fallback; zero heavy deps.

Both return the same DetectedFace shape, so gallery matching and quality gates
are backend-agnostic. Per-backend default match thresholds live here.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from settings import settings

log = logging.getLogger(__name__)

# Per-backend cosine-similarity defaults (used when MATCH_THRESHOLD is unset).
DEFAULT_THRESHOLDS = {"insightface": 0.50, "opencv": 0.40}


@dataclass
class DetectedFace:
    bbox: tuple                  # (x1, y1, x2, y2) float pixel coords
    score: float                 # detector confidence
    embedding: Optional[np.ndarray]   # L2-normalised, or None if embed failed


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)


def cuda_available() -> bool:
    # onnxruntime-gpu reports CUDAExecutionProvider as *compiled in* even with
    # no usable GPU, and initialising it then hard-crashes the process. The
    # only reliable in-container signal is a GPU DEVICE NODE (/dev/nvidia*),
    # which the NVIDIA runtime injects iff the container was given a GPU.
    # (/proc/driver/nvidia exists in ALL containers on a GPU host — never use it.)
    import glob
    if not glob.glob("/dev/nvidia[0-9]*"):
        return False
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


# ── InsightFace backend (SCRFD + ArcFace) ─────────────────────────────────────
class _InsightFaceBackend:
    name = "insightface"

    def __init__(self):
        from insightface.app import FaceAnalysis
        use_cuda = cuda_available()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if use_cuda else ["CPUExecutionProvider"])
        self.device = "cuda" if use_cuda else "cpu"
        self._app = FaceAnalysis(
            name="buffalo_l",
            root=settings.faceweights_dir,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=0 if use_cuda else -1,
                          det_size=(settings.det_size, settings.det_size))
        self.dim = 512

    def analyze(self, img_bgr: np.ndarray) -> List[DetectedFace]:
        out = []
        for f in self._app.get(img_bgr):
            emb = getattr(f, "normed_embedding", None)
            out.append(DetectedFace(
                bbox=tuple(float(v) for v in f.bbox),
                score=float(f.det_score),
                embedding=_l2(emb) if emb is not None else None,
            ))
        return out


# ── OpenCV backend (YuNet + SFace) ────────────────────────────────────────────
class _OpenCVBackend:
    name = "opencv"

    def __init__(self):
        wdir = Path(settings.faceweights_dir) / "opencv"
        det_path = wdir / "yunet.onnx"
        rec_path = wdir / "sface.onnx"
        if not det_path.exists() or not rec_path.exists():
            raise RuntimeError(f"OpenCV face models missing under {wdir}")
        self.device = "cpu"
        self._det = cv2.FaceDetectorYN.create(
            str(det_path), "", (settings.det_size, settings.det_size),
            score_threshold=0.5)
        self._rec = cv2.FaceRecognizerSF.create(str(rec_path), "")
        self.dim = 128

    def analyze(self, img_bgr: np.ndarray) -> List[DetectedFace]:
        h, w = img_bgr.shape[:2]
        self._det.setInputSize((w, h))
        _, rows = self._det.detect(img_bgr)
        out: List[DetectedFace] = []
        if rows is None:
            return out
        for row in rows:
            x, y, bw, bh = row[:4]
            score = float(row[-1])
            emb = None
            try:
                aligned = self._rec.alignCrop(img_bgr, row)
                emb = _l2(self._rec.feature(aligned).flatten())
            except Exception:  # noqa: BLE001 — degenerate crop
                pass
            out.append(DetectedFace(
                bbox=(float(x), float(y), float(x + bw), float(y + bh)),
                score=score, embedding=emb))
        return out


# ── Resolver ──────────────────────────────────────────────────────────────────
class FaceEngine:
    """Resolves the best available backend; thread-safe analyze()."""

    def __init__(self, backend: str | None = None):
        requested = (backend or settings.face_backend).lower()
        order = {
            "auto": ["insightface", "opencv"],
            "insightface": ["insightface"],
            "opencv": ["opencv"],
        }.get(requested, ["opencv"])

        self.backend = None
        for name in order:
            try:
                self.backend = (_InsightFaceBackend() if name == "insightface"
                                else _OpenCVBackend())
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("Face backend '%s' unavailable: %s", name, exc)
        if self.backend is None:
            raise RuntimeError(
                "No face backend available (insightface not installed and "
                "OpenCV zoo models missing). Rebuild the image with network "
                "access, or mount models into FACEWEIGHTS.")

        self.name = self.backend.name
        self.dim = self.backend.dim
        self.device = self.backend.device
        self._lock = threading.Lock()   # serialise inference across threads
        log.info("Face engine: backend=%s dim=%d device=%s",
                 self.name, self.dim, self.device)

    @property
    def match_threshold(self) -> float:
        if settings.match_threshold > 0:
            return settings.match_threshold
        return DEFAULT_THRESHOLDS.get(self.name, 0.5)

    def analyze(self, img_bgr: np.ndarray) -> List[DetectedFace]:
        with self._lock:
            return self.backend.analyze(img_bgr)


def build_engine() -> Optional[FaceEngine]:
    """Non-fatal factory: the API must come up even if no models are present."""
    try:
        return FaceEngine()
    except Exception as exc:  # noqa: BLE001
        log.error("Face engine disabled: %s", exc)
        return None
