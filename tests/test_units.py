"""Unit tests: quality gates, gallery matching, K-of-N confirm + cooldown, DB."""
import numpy as np

from app.engine import DetectedFace
from app.gallery import ConfirmTracker
from app.quality import blur_score, check_enrollment, check_runtime
from tests.conftest import solid_frame


def _face(x1=100, y1=100, x2=300, y2=340, score=0.95, emb=True, dim=8):
    e = None
    if emb:
        v = np.ones(dim, np.float32)
        e = v / np.linalg.norm(v)
    return DetectedFace(bbox=(x1, y1, x2, y2), score=score, embedding=e)


# ── quality gates ─────────────────────────────────────────────────────────────
def test_enroll_rejects_no_face():
    q = check_enrollment(solid_frame((100, 100, 100)), [])
    assert not q.ok and "no face" in q.reason


def test_enroll_rejects_multiple_faces():
    q = check_enrollment(solid_frame((100, 100, 100)), [_face(), _face(x1=400, x2=600)])
    assert not q.ok and "faces" in q.reason


def test_enroll_rejects_small_face():
    q = check_enrollment(solid_frame((100, 100, 100)), [_face(x2=150, y2=150)])  # 50px
    assert not q.ok and "small" in q.reason


def test_enroll_rejects_low_confidence():
    q = check_enrollment(solid_frame((100, 100, 100)), [_face(score=0.3)])
    assert not q.ok and "confidence" in q.reason


def test_enroll_rejects_blurry():
    flat = np.full((480, 640, 3), 128, np.uint8)   # zero texture → blur ~0
    q = check_enrollment(flat, [_face()])
    assert not q.ok and "blur" in q.reason.lower()


def test_enroll_accepts_good_photo():
    q = check_enrollment(solid_frame((100, 100, 100)), [_face()])
    assert q.ok and q.face is not None and q.blur > 0


def test_runtime_gate():
    img = solid_frame((100, 100, 100))
    assert check_runtime(img, _face())
    assert not check_runtime(img, _face(emb=False))          # no embedding
    assert not check_runtime(img, _face(score=0.2))          # low det score
    assert not check_runtime(img, _face(x2=150, y2=150))     # tiny face
    flat = np.full((480, 640, 3), 128, np.uint8)
    assert not check_runtime(flat, _face())                  # blurred


def test_blur_score_orders_correctly():
    flat = np.full((200, 200, 3), 128, np.uint8)
    noisy = solid_frame((128, 128, 128), 200, 200)
    assert blur_score(noisy) > blur_score(flat)


# ── K-of-N confirmation + cooldown ────────────────────────────────────────────
def test_confirm_requires_k_hits():
    c = ConfirmTracker(hits=3, window_s=5.0, cooldown_s=300.0)
    assert not c.register("cam", 1, 0.0)
    assert not c.register("cam", 1, 1.0)
    assert c.register("cam", 1, 2.0)          # 3rd hit within window → fire


def test_confirm_window_expiry():
    c = ConfirmTracker(hits=3, window_s=5.0, cooldown_s=300.0)
    assert not c.register("cam", 1, 0.0)
    assert not c.register("cam", 1, 1.0)
    # 3rd hit arrives too late — first two dropped out of the window
    assert not c.register("cam", 1, 10.0)


def test_confirm_cooldown_suppresses_refire():
    c = ConfirmTracker(hits=1, window_s=5.0, cooldown_s=60.0)
    assert c.register("cam", 1, 0.0)
    assert not c.register("cam", 1, 10.0)      # inside cooldown
    assert c.register("cam", 1, 61.0)          # cooldown elapsed


def test_confirm_is_per_camera_and_person():
    c = ConfirmTracker(hits=1, window_s=5.0, cooldown_s=300.0)
    assert c.register("cam_a", 1, 0.0)
    assert c.register("cam_b", 1, 0.0)         # other camera unaffected
    assert c.register("cam_a", 2, 0.0)         # other person unaffected


# ── gallery matching ──────────────────────────────────────────────────────────
def test_gallery_matches_only_assigned_camera(db, fake_engine):
    from app.gallery import Gallery
    v = np.ones(8, np.float32); v /= np.linalg.norm(v)
    pid = db.create_person("Suspect A")
    db.add_photo(pid, "photos/x.jpg", v, "fake", 100.0)
    db.set_cameras(pid, ["cam_door"])
    g = Gallery(db, fake_engine)

    assert g.match(v, "cam_door") is not None            # assigned → match
    assert g.match(v, "cam_till") is None                # NOT assigned → never


def test_gallery_threshold_blocks_weak_match(db, fake_engine):
    from app.gallery import Gallery
    v = np.zeros(8, np.float32); v[0] = 1.0
    probe = np.zeros(8, np.float32); probe[1] = 1.0      # orthogonal
    pid = db.create_person("Suspect B")
    db.add_photo(pid, "photos/x.jpg", v, "fake", 100.0)
    db.set_cameras(pid, ["cam_door"])
    g = Gallery(db, fake_engine)
    assert g.match(probe, "cam_door") is None


def test_gallery_multi_photo_takes_best(db, fake_engine):
    from app.gallery import Gallery
    a = np.zeros(8, np.float32); a[0] = 1.0
    b = np.zeros(8, np.float32); b[1] = 1.0
    pid = db.create_person("Suspect C")
    db.add_photo(pid, "p1.jpg", a, "fake", 1.0)
    db.add_photo(pid, "p2.jpg", b, "fake", 1.0)          # second look
    db.set_cameras(pid, ["cam_door"])
    g = Gallery(db, fake_engine)
    m = g.match(b, "cam_door")                            # matches photo 2
    assert m is not None and m.similarity > 0.99


def test_gallery_deactivated_person_drops_out(db, fake_engine):
    from app.gallery import Gallery
    v = np.ones(8, np.float32); v /= np.linalg.norm(v)
    pid = db.create_person("Suspect D")
    db.add_photo(pid, "x.jpg", v, "fake", 1.0)
    db.set_cameras(pid, ["cam_door"])
    g = Gallery(db, fake_engine)
    assert g.match(v, "cam_door") is not None
    db.deactivate_person(pid)
    g.reload()
    assert g.match(v, "cam_door") is None


def test_gallery_skips_stale_backend_embeddings(db, fake_engine):
    from app.gallery import Gallery
    v512 = np.ones(512, np.float32); v512 /= np.linalg.norm(v512)
    pid = db.create_person("Suspect E")
    db.add_photo(pid, "x.jpg", v512, "insightface", 1.0)  # wrong backend/dim
    db.set_cameras(pid, ["cam_door"])
    g = Gallery(db, fake_engine)                          # fake: dim=8
    assert g.size()["photos_indexed"] == 0                # stale rows excluded
