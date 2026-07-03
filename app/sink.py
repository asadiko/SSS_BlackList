"""
app/sink.py — Alert forwarding to the operator backend.

POSTs a blacklist-match event to the Go backend's ingest endpoint so it reaches
the operator UI in real time. Pure stdlib, fire-and-forget on a daemon thread,
and every failure is swallowed — a backend hiccup must never stall a camera
worker. No-op unless BACKEND_URL is configured.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.request

from settings import settings

log = logging.getLogger(__name__)


def forward_alert(event: dict) -> None:
    if not settings.backend_url:
        return
    threading.Thread(target=_post, args=(dict(event),), daemon=True).start()


def _post(event: dict) -> None:
    url = settings.backend_url.rstrip("/") + "/internal/events"
    payload = {
        "event_type": "blacklist_match",
        **event,
    }
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {settings.backend_token}"}
                   if settings.backend_token else {}),
            }, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
    except Exception as exc:  # noqa: BLE001
        log.warning("backend forward failed (ignored): %s", exc)
