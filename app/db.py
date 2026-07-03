"""
app/db.py — Local SQLite store for the blacklist service.

Holds persons, their reference photos (with face embeddings), person↔camera
assignments and sighting (alert) history. SQLite in WAL mode is plenty for
this workload (writes are rare: enrollments + alerts); the whole service stays
self-contained with zero external DB dependency, per the microservice brief.

Thread-safe: one connection guarded by an RLock (API threads + camera workers).
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    notes      TEXT NOT NULL DEFAULT '',
    active     INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS photos (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    file_path     TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    backend       TEXT NOT NULL,
    quality       REAL NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assignments (
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    camera_id TEXT NOT NULL,
    PRIMARY KEY (person_id, camera_id)
);
CREATE TABLE IF NOT EXISTS sightings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER NOT NULL REFERENCES persons(id),
    camera_id  TEXT NOT NULL,
    ts         REAL NOT NULL,
    wall_time  TEXT NOT NULL,
    similarity REAL NOT NULL,
    crop_path  TEXT,
    status     TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_sightings_status ON sightings(status);
CREATE INDEX IF NOT EXISTS idx_photos_person ON photos(person_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class BlacklistDB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── persons ──────────────────────────────────────────────────────────────
    def create_person(self, name: str, notes: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO persons(name, notes, created_at) VALUES (?,?,?)",
                (name, notes, _now()))
            self._conn.commit()
            return int(cur.lastrowid)

    def deactivate_person(self, person_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE persons SET active=0 WHERE id=?", (person_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def get_person(self, person_id: int) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM persons WHERE id=?", (person_id,)).fetchone()
            return dict(row) if row else None

    def list_persons(self, include_inactive: bool = False) -> list[dict]:
        q = "SELECT * FROM persons" + ("" if include_inactive else " WHERE active=1")
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY id").fetchall()
            persons = [dict(r) for r in rows]
            for p in persons:
                p["photos"] = self.list_photos(p["id"])
                p["cameras"] = self.cameras_for_person(p["id"])
            return persons

    # ── photos / embeddings ──────────────────────────────────────────────────
    def add_photo(self, person_id: int, file_path: str,
                  embedding: np.ndarray, backend: str, quality: float) -> int:
        emb = np.asarray(embedding, dtype=np.float32)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO photos(person_id, file_path, embedding, embedding_dim,"
                " backend, quality, created_at) VALUES (?,?,?,?,?,?,?)",
                (person_id, file_path, emb.tobytes(), emb.shape[0], backend,
                 float(quality), _now()))
            self._conn.commit()
            return int(cur.lastrowid)

    def list_photos(self, person_id: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, person_id, file_path, embedding_dim, backend, quality,"
                " created_at FROM photos WHERE person_id=? ORDER BY id",
                (person_id,)).fetchall()
            return [dict(r) for r in rows]

    def delete_photo(self, photo_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM photos WHERE id=?", (photo_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def all_embeddings(self, backend: str, dim: int) -> list[tuple[int, np.ndarray]]:
        """(person_id, embedding) for every photo of every ACTIVE person whose
        embedding matches the running backend/dim (stale ones are skipped)."""
        with self._lock:
            rows = self._conn.execute("""
                SELECT ph.person_id, ph.embedding FROM photos ph
                JOIN persons p ON p.id = ph.person_id
                WHERE p.active=1 AND ph.backend=? AND ph.embedding_dim=?
            """, (backend, dim)).fetchall()
        return [(int(r["person_id"]),
                 np.frombuffer(r["embedding"], dtype=np.float32).copy())
                for r in rows]

    # ── assignments ──────────────────────────────────────────────────────────
    def set_cameras(self, person_id: int, camera_ids: list[str]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM assignments WHERE person_id=?", (person_id,))
            self._conn.executemany(
                "INSERT OR IGNORE INTO assignments(person_id, camera_id) VALUES (?,?)",
                [(person_id, c) for c in camera_ids])
            self._conn.commit()

    def cameras_for_person(self, person_id: int) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT camera_id FROM assignments WHERE person_id=? ORDER BY camera_id",
                (person_id,)).fetchall()
            return [r["camera_id"] for r in rows]

    def assignments_by_camera(self) -> dict[str, set[int]]:
        """camera_id → set of ACTIVE person_ids assigned to it."""
        with self._lock:
            rows = self._conn.execute("""
                SELECT a.camera_id, a.person_id FROM assignments a
                JOIN persons p ON p.id = a.person_id WHERE p.active=1
            """).fetchall()
        out: dict[str, set[int]] = {}
        for r in rows:
            out.setdefault(r["camera_id"], set()).add(int(r["person_id"]))
        return out

    # ── sightings ────────────────────────────────────────────────────────────
    def insert_sighting(self, person_id: int, camera_id: str, ts: float,
                        similarity: float, crop_path: Optional[str]) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sightings(person_id, camera_id, ts, wall_time,"
                " similarity, crop_path) VALUES (?,?,?,?,?,?)",
                (person_id, camera_id, ts, _now(), float(similarity), crop_path))
            self._conn.commit()
            return int(cur.lastrowid)

    def verify_sighting(self, sighting_id: int, status: str) -> bool:
        assert status in ("confirmed", "false_match")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sightings SET status=? WHERE id=?", (status, sighting_id))
            self._conn.commit()
            return cur.rowcount > 0

    def list_sightings(self, status: Optional[str] = None, limit: int = 100) -> list[dict]:
        with self._lock:
            if status:
                rows = self._conn.execute("""
                    SELECT s.*, p.name AS person_name FROM sightings s
                    JOIN persons p ON p.id = s.person_id
                    WHERE s.status=? ORDER BY s.id DESC LIMIT ?""",
                    (status, limit)).fetchall()
            else:
                rows = self._conn.execute("""
                    SELECT s.*, p.name AS person_name FROM sightings s
                    JOIN persons p ON p.id = s.person_id
                    ORDER BY s.id DESC LIMIT ?""", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self._lock:
            n_p = self._conn.execute("SELECT COUNT(*) c FROM persons WHERE active=1").fetchone()["c"]
            n_ph = self._conn.execute("SELECT COUNT(*) c FROM photos").fetchone()["c"]
            n_s = self._conn.execute("SELECT COUNT(*) c FROM sightings").fetchone()["c"]
        return {"persons": n_p, "photos": n_ph, "sightings": n_s}
