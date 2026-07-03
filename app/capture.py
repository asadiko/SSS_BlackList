"""
app/capture.py — Camera capture workers + the per-frame recognition path.

FrameProcessor  detect → quality gate → gallery match → K-of-N confirm →
                alert (save crop, insert sighting, forward to backend).
                Pure logic, injectable clock/engine — used directly by tests.

CameraWorker    One thread per WATCHED camera (a camera with ≥1 assigned,
                active person). Reads the shared cameras.yaml for the source,
                samples frames at FACE_FPS (reading every frame so RTSP buffers
                never back up), reconnects with capped exponential backoff.

CameraManager   Reconciles workers against DB assignments every RECONCILE_S:
                assignments added via the API start a worker automatically;
                removing the last person from a camera stops its worker.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import yaml

from settings import settings
from app.quality import check_runtime, face_crop
from app.sink import forward_alert

log = logging.getLogger(__name__)

_STREAM_SCHEMES = ("rtsp://", "rtmp://", "rtmps://", "http://", "https://", "udp://", "tcp://")


# ── camera registry (shared cameras.yaml) ─────────────────────────────────────
def load_camera_sources(path: str | None = None) -> dict[str, dict]:
    """camera_id → {source, name, enabled} from the shared cameras.yaml."""
    p = Path(path or settings.cameras_yaml)
    if not p.exists():
        log.warning("cameras.yaml not found at %s", p)
        return {}
    cfg = yaml.safe_load(p.read_text()) or {}
    out = {}
    for c in cfg.get("cameras", []):
        src = c.get("source")
        if isinstance(src, str) and src.isdigit():
            src = int(src)                        # webcam device index
        out[str(c.get("id"))] = {
            "source": src,
            "name": c.get("name", c.get("id")),
            "enabled": c.get("enabled", True) is not False,
        }
    return out


# ── per-frame recognition path ────────────────────────────────────────────────
class FrameProcessor:
    def __init__(self, engine, gallery, confirm, db,
                 on_alert: Optional[Callable[[dict], None]] = None):
        self.engine = engine
        self.gallery = gallery
        self.confirm = confirm
        self.db = db
        self.on_alert = on_alert

    def process(self, camera_id: str, frame_bgr: np.ndarray, ts: float) -> list[dict]:
        """Run one frame; returns the alerts fired (usually empty)."""
        fired: list[dict] = []
        if self.engine is None:
            return fired
        for face in self.engine.analyze(frame_bgr):
            if not check_runtime(frame_bgr, face):
                continue
            m = self.gallery.match(face.embedding, camera_id)
            if m is None:
                continue
            if not self.confirm.register(camera_id, m.person_id, ts):
                continue
            fired.append(self._fire(camera_id, m, face, frame_bgr, ts))
        return fired

    def _fire(self, camera_id: str, m, face, frame_bgr, ts: float) -> dict:
        crop_rel = None
        try:
            crop = face_crop(frame_bgr, face)
            if crop.size:
                Path(settings.crops_dir).mkdir(parents=True, exist_ok=True)
                fname = f"{camera_id}_p{m.person_id}_{uuid.uuid4().hex[:8]}.jpg"
                fpath = str(Path(settings.crops_dir) / fname)
                cv2.imwrite(fpath, crop)
                os.chmod(fpath, 0o644)
                crop_rel = f"crops/{fname}"
        except Exception as exc:  # noqa: BLE001
            log.error("crop save failed: %s", exc)

        sid = self.db.insert_sighting(m.person_id, camera_id, ts,
                                      m.similarity, crop_rel)
        person = self.db.get_person(m.person_id) or {}
        event = {
            "sighting_id": sid,
            "camera_id": camera_id,
            "person_id": m.person_id,
            "person_name": person.get("name", f"#{m.person_id}"),
            "similarity": round(m.similarity, 4),
            "ts": round(ts, 2),
        }
        log.warning("[%s] BLACKLIST MATCH person=%s sim=%.3f (sighting %d)",
                    camera_id, event["person_name"], m.similarity, sid)
        forward_alert(event)
        if self.on_alert:
            try:
                self.on_alert(event)
            except Exception as exc:  # noqa: BLE001
                log.error("on_alert callback failed: %s", exc)
        return event


# ── capture worker ────────────────────────────────────────────────────────────
def _is_stream(source) -> bool:
    return isinstance(source, str) and source.lower().startswith(_STREAM_SCHEMES)


def _open_capture(source):
    if isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://")):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:  # noqa: BLE001
        pass
    return cap


class CameraWorker(threading.Thread):
    def __init__(self, camera_id: str, source, processor: FrameProcessor):
        super().__init__(daemon=True, name=f"bl-{camera_id}")
        self.camera_id = camera_id
        self.source = source
        self.processor = processor
        self.stop_event = threading.Event()
        self.frames_processed = 0

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        interval = 1.0 / max(settings.face_fps, 0.1)
        stream = _is_stream(self.source)
        backoff = 1.0
        t0 = time.monotonic()

        while not self.stop_event.is_set():
            cap = _open_capture(self.source)
            if not cap.isOpened():
                log.warning("[%s] cannot open %s — retry in %.0fs",
                            self.camera_id, self.source, backoff)
                if not stream and not isinstance(self.source, int):
                    break                    # missing local file: give up
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 30.0)
                continue

            backoff = 1.0
            log.info("[%s] watching (face_fps=%.1f source=%s)",
                     self.camera_id, settings.face_fps, self.source)
            last_proc = 0.0
            while not self.stop_event.is_set():
                ok, frame = cap.read()      # read EVERY frame (keeps RTSP live)
                if not ok:
                    break
                now = time.monotonic()
                if now - last_proc < interval:
                    continue                 # sample at FACE_FPS
                last_proc = now
                try:
                    self.processor.process(self.camera_id, frame, now - t0)
                    self.frames_processed += 1
                except Exception as exc:  # noqa: BLE001
                    log.exception("[%s] frame error: %s", self.camera_id, exc)
            cap.release()

            if self.stop_event.is_set():
                break
            if not stream and not isinstance(self.source, int):
                # local file: loop it (demo/replay semantics)
                continue
            log.warning("[%s] stream dropped — reconnecting in %.0fs",
                        self.camera_id, backoff)
            self.stop_event.wait(backoff)
            backoff = min(backoff * 2, 30.0)

        log.info("[%s] worker stopped.", self.camera_id)


class CameraManager:
    """Start/stop workers so exactly the ASSIGNED+ENABLED cameras are watched."""

    def __init__(self, db, processor: FrameProcessor):
        self.db = db
        self.processor = processor
        self.workers: dict[str, CameraWorker] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def wanted_cameras(self) -> dict[str, dict]:
        sources = load_camera_sources()
        assigned = self.db.assignments_by_camera()
        wanted = {}
        for cam_id, persons in assigned.items():
            info = sources.get(cam_id)
            if not persons:
                continue
            if info is None:
                log.warning("camera '%s' assigned but not in cameras.yaml", cam_id)
                continue
            if not info["enabled"] or info["source"] in (None, ""):
                continue
            wanted[cam_id] = info
        return wanted

    def reconcile(self) -> None:
        wanted = self.wanted_cameras()
        # stop workers no longer wanted (or whose source changed)
        for cam_id in list(self.workers):
            w = self.workers[cam_id]
            if cam_id not in wanted or wanted[cam_id]["source"] != w.source or not w.is_alive():
                w.stop()
                self.workers.pop(cam_id)
        # start missing workers
        for cam_id, info in wanted.items():
            if cam_id not in self.workers:
                w = CameraWorker(cam_id, info["source"], self.processor)
                self.workers[cam_id] = w
                w.start()

    def start(self) -> None:
        def loop():
            while not self._stop.is_set():
                try:
                    self.reconcile()
                except Exception as exc:  # noqa: BLE001
                    log.exception("reconcile failed: %s", exc)
                self._stop.wait(settings.reconcile_s)
        self._thread = threading.Thread(target=loop, daemon=True, name="bl-manager")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        for w in self.workers.values():
            w.stop()

    def status(self) -> dict:
        return {cam: {"alive": w.is_alive(), "frames": w.frames_processed}
                for cam, w in self.workers.items()}
