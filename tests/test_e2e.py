"""End-to-end (no video, no models): enrolled person walks past the camera →
exactly one warning; a different person / unassigned camera → silence."""
import numpy as np

from tests.conftest import solid_frame

RED = (30, 60, 200)     # "the suspect"
BLUE = (200, 60, 30)    # somebody else


def _enroll(state, name, colour, cameras):
    frame = solid_frame(colour)
    emb = state.engine._embed_from_colour(frame)
    pid = state.db.create_person(name)
    state.db.add_photo(pid, f"photos/{name}.jpg", emb, state.engine.name, 100.0)
    state.db.set_cameras(pid, cameras)
    state.gallery.reload()
    return pid


def test_blacklisted_person_triggers_single_alert(state):
    pid = _enroll(state, "Suspect", RED, ["cam_door"])

    # Suspect visible for 3 seconds at 3 fps on the assigned camera.
    fps = 3.0
    for i in range(9):
        state.processor.process("cam_door", solid_frame(RED), i / fps)

    assert len(state.alerts) == 1, f"expected exactly one alert, got {state.alerts}"
    a = state.alerts[0]
    assert a["person_id"] == pid
    assert a["camera_id"] == "cam_door"
    assert a["similarity"] >= state.engine.match_threshold

    # Sighting persisted with a saved face crop.
    sightings = state.db.list_sightings()
    assert len(sightings) == 1
    assert sightings[0]["status"] == "pending"
    assert sightings[0]["crop_path"]


def test_unknown_person_never_alerts(state):
    _enroll(state, "Suspect", RED, ["cam_door"])
    for i in range(20):
        state.processor.process("cam_door", solid_frame(BLUE), i / 3.0)
    assert state.alerts == []
    assert state.db.list_sightings() == []


def test_assigned_camera_only(state):
    _enroll(state, "Suspect", RED, ["cam_door"])
    # Suspect appears on a camera they are NOT assigned to → must stay silent.
    for i in range(20):
        state.processor.process("cam_till", solid_frame(RED), i / 3.0)
    assert state.alerts == []


def test_single_frame_flicker_never_alerts(state):
    """The K-of-N rule: an isolated one-frame match cannot fire."""
    _enroll(state, "Suspect", RED, ["cam_door"])
    state.processor.process("cam_door", solid_frame(RED), 0.0)     # 1 hit
    for i in range(1, 10):
        state.processor.process("cam_door", solid_frame((0, 0, 0)), float(i))  # empty
    assert state.alerts == []


def test_cooldown_one_alert_per_sighting(state):
    _enroll(state, "Suspect", RED, ["cam_door"])
    # Suspect loiters for 60 seconds — still only ONE alert (cooldown 300s).
    for i in range(180):
        state.processor.process("cam_door", solid_frame(RED), i / 3.0)
    assert len(state.alerts) == 1


def test_empty_frames_are_cheap_noops(state):
    _enroll(state, "Suspect", RED, ["cam_door"])
    black = np.zeros((480, 640, 3), np.uint8)
    for i in range(30):
        fired = state.processor.process("cam_door", black, i / 3.0)
        assert fired == []
    assert state.alerts == []
