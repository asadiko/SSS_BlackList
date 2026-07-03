"""
main.py — Blacklist microservice entrypoint.

Boots: SQLite DB → face engine (auto GPU/CPU, insightface→opencv fallback) →
match gallery → camera workers (only for cameras with assigned persons) →
FastAPI admin/REST server.

The service is fully self-contained: local DB + photo storage under BL_DATA_DIR,
camera sources from the shared cameras.yaml, optional alert forwarding to the
operator backend via BACKEND_URL.
"""
from __future__ import annotations

import logging

from settings import settings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("blacklist")


class ServiceState:
    """Shared singletons handed to the API layer."""
    def __init__(self, db, engine, gallery, manager):
        self.db = db
        self.engine = engine
        self.gallery = gallery
        self.manager = manager


def build_state() -> ServiceState:
    settings.ensure_dirs()

    from app.db import BlacklistDB
    from app.engine import build_engine
    from app.gallery import Gallery, ConfirmTracker
    from app.capture import CameraManager, FrameProcessor

    db = BlacklistDB(settings.db_path)
    engine = build_engine()          # None → API up, enrollment disabled
    gallery = Gallery(db, engine) if engine else None
    manager = None
    if engine and gallery:
        confirm = ConfirmTracker()
        processor = FrameProcessor(engine, gallery, confirm, db)
        manager = CameraManager(db, processor)
        manager.start()

    log.info("Blacklist service ready. engine=%s db=%s data=%s",
             engine.name if engine else "none", settings.db_path, settings.data_dir)
    return ServiceState(db, engine, gallery, manager)


def main() -> None:
    state = build_state()

    from app.api import create_app
    import uvicorn

    app = create_app(state)
    log.info("Admin/API on http://%s:%d  (docs at /docs)",
             settings.api_host, settings.api_port)
    try:
        uvicorn.run(app, host=settings.api_host, port=settings.api_port,
                    log_level="info")
    finally:
        if state.manager:
            state.manager.stop()
        state.db.close()


if __name__ == "__main__":
    main()
