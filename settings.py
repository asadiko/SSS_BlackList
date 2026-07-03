"""
settings.py — Blacklist microservice configuration.

Everything is overridable via environment variables (UPPER_CASE, optionally
prefixed BL_ where noted). Zero third-party dependencies so the test-suite can
import it in a bare venv. Nothing is hardcoded at call-sites — read from here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).parent.resolve()


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # ── Paths / storage ──────────────────────────────────────────────────────
    data_dir: str = field(default_factory=lambda: _env_str("BL_DATA_DIR", "/data"))
    cameras_yaml: str = field(default_factory=lambda: _env_str("CAMERAS_YAML", "/app/cameras.yaml"))
    # Baked model weights (populated at image build; falls back gracefully)
    faceweights_dir: str = field(default_factory=lambda: _env_str("FACEWEIGHTS", "/opt/faceweights"))

    # ── Face engine ──────────────────────────────────────────────────────────
    # auto → insightface (SCRFD+ArcFace) → opencv (YuNet+SFace). GPU is used
    # automatically when onnxruntime reports a CUDA provider; otherwise CPU.
    face_backend: str = field(default_factory=lambda: _env_str("FACE_BACKEND", "auto"))
    det_size: int = field(default_factory=lambda: _env_int("DET_SIZE", 640))

    # ── Matching / false-positive controls ───────────────────────────────────
    # Cosine-similarity threshold. 0 → per-backend default (insightface 0.50,
    # opencv-sface 0.40). Raise for fewer FPs, lower for fewer misses.
    match_threshold: float = field(default_factory=lambda: _env_float("MATCH_THRESHOLD", 0.0))
    confirm_hits: int = field(default_factory=lambda: _env_int("CONFIRM_HITS", 3))
    confirm_window_s: float = field(default_factory=lambda: _env_float("CONFIRM_WINDOW_S", 5.0))
    cooldown_s: float = field(default_factory=lambda: _env_float("COOLDOWN_S", 300.0))

    # ── Quality gates ────────────────────────────────────────────────────────
    # Enrollment (reference photos) — strict:
    enroll_min_face_px: int = field(default_factory=lambda: _env_int("ENROLL_MIN_FACE_PX", 112))
    enroll_min_det_score: float = field(default_factory=lambda: _env_float("ENROLL_MIN_DET_SCORE", 0.65))
    enroll_min_blur: float = field(default_factory=lambda: _env_float("ENROLL_MIN_BLUR", 60.0))
    # Runtime (camera frames) — permissive enough for CCTV, strict enough to
    # never score a smeared 30px face:
    runtime_min_face_px: int = field(default_factory=lambda: _env_int("RUNTIME_MIN_FACE_PX", 80))
    runtime_min_det_score: float = field(default_factory=lambda: _env_float("RUNTIME_MIN_DET_SCORE", 0.55))
    runtime_min_blur: float = field(default_factory=lambda: _env_float("RUNTIME_MIN_BLUR", 25.0))

    # ── Capture ──────────────────────────────────────────────────────────────
    face_fps: float = field(default_factory=lambda: _env_float("FACE_FPS", 3.0))
    reconcile_s: float = field(default_factory=lambda: _env_float("RECONCILE_S", 10.0))

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = field(default_factory=lambda: _env_str("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8100))
    # Optional shared-secret auth for the REST API (empty → auth disabled).
    api_key: str = field(default_factory=lambda: _env_str("BL_API_KEY", ""))

    # ── Alert forwarding (Go backend) ────────────────────────────────────────
    backend_url: str = field(default_factory=lambda: _env_str("BACKEND_URL", ""))
    backend_token: str = field(default_factory=lambda: _env_str("BACKEND_SERVICE_TOKEN", ""))

    # ── Derived ──────────────────────────────────────────────────────────────
    @property
    def db_path(self) -> str:
        return _env_str("BL_DB_PATH", str(Path(self.data_dir) / "blacklist.db"))

    @property
    def photos_dir(self) -> str:
        return str(Path(self.data_dir) / "photos")

    @property
    def crops_dir(self) -> str:
        return str(Path(self.data_dir) / "crops")

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.photos_dir, self.crops_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
