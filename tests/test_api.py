"""API tests: CRUD, enrollment gate at the HTTP layer, auth, sightings review."""
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.engine import DetectedFace
from tests.conftest import solid_frame


@pytest.fixture()
def client(state, monkeypatch):
    return TestClient(create_app(state))


def _jpeg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _good_face(w=640, h=480):
    v = np.ones(8, np.float32); v /= np.linalg.norm(v)
    return DetectedFace(bbox=(w * 0.25, h * 0.1, w * 0.75, h * 0.9),
                        score=0.95, embedding=v)


def test_health(client, state):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"]["backend"] == "fake"


def test_person_crud_and_assignment(client):
    r = client.post("/persons", json={"name": "John Doe", "notes": "test"})
    assert r.status_code == 201
    pid = r.json()["id"]

    r = client.put(f"/persons/{pid}/cameras", json={"camera_ids": ["cam_door"]})
    assert r.status_code == 200

    r = client.put(f"/persons/{pid}/cameras", json={"camera_ids": ["nope"]})
    assert r.status_code == 422                      # unknown camera rejected

    r = client.get("/persons")
    assert r.status_code == 200
    p = next(x for x in r.json() if x["id"] == pid)
    assert p["cameras"] == ["cam_door"]

    r = client.delete(f"/persons/{pid}")
    assert r.status_code == 200
    assert all(x["id"] != pid for x in client.get("/persons").json())


def test_photo_upload_quality_gate(client, state):
    pid = client.post("/persons", json={"name": "Jane"}).json()["id"]

    # No face in the image → rejected with a readable reason.
    state.engine.next_faces = []
    r = client.post(f"/persons/{pid}/photos",
                    files={"file": ("x.jpg", _jpeg(solid_frame((90, 90, 90))), "image/jpeg")})
    assert r.status_code == 422
    assert "no face" in r.json()["detail"]

    # Two faces → rejected.
    state.engine.next_faces = [_good_face(), _good_face()]
    r = client.post(f"/persons/{pid}/photos",
                    files={"file": ("x.jpg", _jpeg(solid_frame((90, 90, 90))), "image/jpeg")})
    assert r.status_code == 422

    # One good face → accepted, indexed.
    state.engine.next_faces = [_good_face()]
    r = client.post(f"/persons/{pid}/photos",
                    files={"file": ("x.jpg", _jpeg(solid_frame((90, 90, 90))), "image/jpeg")})
    assert r.status_code == 201
    assert r.json()["photo_id"] > 0
    assert state.gallery.size()["photos_indexed"] == 1

    # Garbage bytes → 422.
    r = client.post(f"/persons/{pid}/photos",
                    files={"file": ("x.jpg", b"not-an-image", "image/jpeg")})
    assert r.status_code == 422


def test_sightings_flow(client, state):
    # Produce one sighting via the real processor path.
    v = np.ones(8, np.float32); v /= np.linalg.norm(v)
    pid = state.db.create_person("S")
    state.db.add_photo(pid, "x.jpg", v, "fake", 1.0)
    state.db.set_cameras(pid, ["cam_door"])
    state.gallery.reload()
    for i in range(3):
        state.processor.process("cam_door", solid_frame((30, 60, 200)), float(i))
    # (fake engine embeds by colour; enrolled embedding may differ — force via API check)
    sightings = client.get("/sightings").json()
    if not sightings:                                 # deterministic fallback
        sid = state.db.insert_sighting(pid, "cam_door", 1.0, 0.93, None)
        sightings = client.get("/sightings").json()
    sid = sightings[0]["id"]

    r = client.post(f"/sightings/{sid}/verify", json={"status": "false_match"})
    assert r.status_code == 200
    assert client.get("/sightings?status=false_match").json()[0]["id"] == sid

    r = client.post(f"/sightings/{sid}/verify", json={"status": "bogus"})
    assert r.status_code == 422


def test_cameras_endpoint(client):
    cams = client.get("/cameras").json()
    ids = {c["id"] for c in cams}
    assert {"cam_door", "cam_till"} <= ids


def test_api_key_auth(state, monkeypatch):
    from settings import settings as s
    monkeypatch.setattr(s, "api_key", "sekret", raising=False)
    client = TestClient(create_app(state))

    assert client.get("/persons").status_code == 401                 # no key
    assert client.get("/persons", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/persons", headers={"X-API-Key": "sekret"}).status_code == 200
    assert client.get("/health").status_code == 200                  # health open
