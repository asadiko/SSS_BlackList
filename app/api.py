"""
app/api.py — REST API + admin page for the blacklist service.

Endpoints (all JSON unless noted):
  GET    /health                      liveness + engine/backend info
  GET    /stats                       DB counts, index size, worker status
  GET    /cameras                     cameras from the shared cameras.yaml
  POST   /persons                     {name, notes} → create person
  GET    /persons                     list (with photos + camera assignments)
  GET    /persons/{id}                detail
  DELETE /persons/{id}                deactivate (drops from match index)
  POST   /persons/{id}/photos         multipart photo upload (quality-gated)
  DELETE /persons/{id}/photos/{pid}   remove a reference photo
  PUT    /persons/{id}/cameras        {camera_ids: [...]}
  GET    /sightings?status=&limit=    alert history
  POST   /sightings/{id}/verify       {status: confirmed|false_match}

Static:
  GET /          admin single-page UI
  GET /media/…   stored photos + alert crops

Auth: when BL_API_KEY is set, every /API route requires X-API-Key. The admin
page prompts for the key and sends it. /health and static stay open.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from settings import settings
from app.capture import load_camera_sources

log = logging.getLogger(__name__)

_STATIC = Path(__file__).parent.parent / "static"


class PersonIn(BaseModel):
    name: str
    notes: str = ""


class CamerasIn(BaseModel):
    camera_ids: list[str]


class VerifyIn(BaseModel):
    status: str  # confirmed | false_match


def create_app(state) -> FastAPI:
    """state: object with .db .engine .gallery .manager (manager may be None)."""
    app = FastAPI(title="SSS Blacklist Service", version="1.0")

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(401, "invalid or missing X-API-Key")

    auth = [Depends(require_key)]

    # ── health / stats ───────────────────────────────────────────────────────
    @app.get("/health")
    def health():
        eng = state.engine
        return {
            "status": "ok",
            "engine": None if eng is None else {
                "backend": eng.name, "device": eng.device,
                "dim": eng.dim, "threshold": eng.match_threshold,
            },
        }

    @app.get("/stats", dependencies=auth)
    def stats():
        return {
            **state.db.stats(),
            "index": state.gallery.size() if state.gallery else {},
            "workers": state.manager.status() if state.manager else {},
            "confirm": {"hits": settings.confirm_hits,
                        "window_s": settings.confirm_window_s,
                        "cooldown_s": settings.cooldown_s},
        }

    # ── cameras ──────────────────────────────────────────────────────────────
    @app.get("/cameras", dependencies=auth)
    def cameras():
        return [{"id": cid, **info} for cid, info in load_camera_sources().items()]

    # ── persons ──────────────────────────────────────────────────────────────
    @app.post("/persons", dependencies=auth, status_code=201)
    def create_person(body: PersonIn):
        if not body.name.strip():
            raise HTTPException(422, "name is required")
        pid = state.db.create_person(body.name.strip(), body.notes.strip())
        return state.db.get_person(pid)

    @app.get("/persons", dependencies=auth)
    def list_persons():
        return state.db.list_persons()

    @app.get("/persons/{person_id}", dependencies=auth)
    def get_person(person_id: int):
        p = state.db.get_person(person_id)
        if not p:
            raise HTTPException(404, "person not found")
        p["photos"] = state.db.list_photos(person_id)
        p["cameras"] = state.db.cameras_for_person(person_id)
        return p

    @app.delete("/persons/{person_id}", dependencies=auth)
    def deactivate_person(person_id: int):
        if not state.db.deactivate_person(person_id):
            raise HTTPException(404, "person not found")
        state.gallery.reload()
        return {"ok": True}

    # ── photos (enrollment) ──────────────────────────────────────────────────
    @app.post("/persons/{person_id}/photos", dependencies=auth, status_code=201)
    async def upload_photo(person_id: int, file: UploadFile = File(...)):
        if not state.db.get_person(person_id):
            raise HTTPException(404, "person not found")
        if state.engine is None:
            raise HTTPException(503, "face engine unavailable — cannot enroll")

        raw = await file.read()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(422, "not a decodable image")

        from app.quality import check_enrollment
        faces = state.engine.analyze(img)
        q = check_enrollment(img, faces)
        if not q.ok:
            raise HTTPException(422, f"photo rejected: {q.reason}")

        Path(settings.photos_dir).mkdir(parents=True, exist_ok=True)
        fname = f"p{person_id}_{uuid.uuid4().hex[:10]}.jpg"
        fpath = str(Path(settings.photos_dir) / fname)
        cv2.imwrite(fpath, img)
        os.chmod(fpath, 0o644)

        photo_id = state.db.add_photo(person_id, f"photos/{fname}",
                                      q.face.embedding, state.engine.name, q.blur)
        state.gallery.reload()
        return {"photo_id": photo_id, "file_path": f"photos/{fname}",
                "quality": round(q.blur, 1), "backend": state.engine.name}

    @app.delete("/persons/{person_id}/photos/{photo_id}", dependencies=auth)
    def delete_photo(person_id: int, photo_id: int):
        if not state.db.delete_photo(photo_id):
            raise HTTPException(404, "photo not found")
        state.gallery.reload()
        return {"ok": True}

    # ── camera assignments ───────────────────────────────────────────────────
    @app.put("/persons/{person_id}/cameras", dependencies=auth)
    def set_cameras(person_id: int, body: CamerasIn):
        if not state.db.get_person(person_id):
            raise HTTPException(404, "person not found")
        known = set(load_camera_sources().keys())
        unknown = [c for c in body.camera_ids if c not in known]
        if unknown:
            raise HTTPException(422, f"unknown camera ids: {unknown}")
        state.db.set_cameras(person_id, body.camera_ids)
        state.gallery.reload()
        if state.manager:
            state.manager.reconcile()      # start/stop workers immediately
        return {"ok": True, "cameras": body.camera_ids}

    # ── sightings ────────────────────────────────────────────────────────────
    @app.get("/sightings", dependencies=auth)
    def sightings(status: str | None = None, limit: int = 100):
        return state.db.list_sightings(status, min(limit, 500))

    @app.post("/sightings/{sighting_id}/verify", dependencies=auth)
    def verify(sighting_id: int, body: VerifyIn):
        if body.status not in ("confirmed", "false_match"):
            raise HTTPException(422, "status must be confirmed|false_match")
        if not state.db.verify_sighting(sighting_id, body.status):
            raise HTTPException(404, "sighting not found")
        return {"ok": True}

    # ── static: admin page + stored media ────────────────────────────────────
    if Path(settings.data_dir).is_dir():
        app.mount("/media", StaticFiles(directory=settings.data_dir), name="media")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(_STATIC / "index.html")

    return app
