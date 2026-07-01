"""
SQLite-backed storage for the face dataset.

Replaces the old "labeled folders" layout
    data/face_db/<name>/embedding.npy, angle_1.jpg, ...
    data/processed/<name>/*.jpg
with a single portable database file:
    data/face_dataset.db

Schema:
    persons     -- one row per registered person
    embeddings  -- one row per captured angle's 512-D InsightFace embedding
    images      -- one row per stored image (raw capture OR background-
                   removed "processed" version), stored as JPEG bytes (BLOB)

Every other module (face_registrar, image_preprocessor, face_recognizer,
app_dashboard) goes through this module instead of touching the filesystem
directly -- this is the single source of truth for "who is registered" and
"what does their data look like".

Migrating existing folder-based data: run migrate_to_sqlite.py once.
"""

import os
import sqlite3
import threading

import cv2
import numpy as np

import config

DB_PATH = getattr(config, "FACE_DB_PATH", os.path.join(config.DATA_DIR, "face_dataset.db"))

# sqlite3 connections aren't shared across threads; a short-lived connection
# per call plus a module-level lock keeps writes/reads safe across the
# registration Flask app, the dashboard Flask app, and every AIPipeline
# thread that calls load_all_embeddings().
_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    angle_index INTEGER NOT NULL,
    vector      BLOB NOT NULL,      -- 512 float32 values, packed via .tobytes()
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('raw', 'processed')),
    angle_index INTEGER NOT NULL,
    jpeg_data   BLOB NOT NULL,      -- cv2.imencode('.jpg', img)[1].tobytes()
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);
CREATE INDEX IF NOT EXISTS idx_images_person_kind ON images(person_id, kind);
"""


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create the database file + tables if they don't exist yet. Safe to
    call every time an app starts (registration app, dashboard app, or the
    migration script)."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _lock, _connect() as conn:
        conn.executescript(_SCHEMA)
    print(f"[FaceDatabase] Ready at {DB_PATH}")


# ---------------------------------------------------------------------------
# Person helpers
# ---------------------------------------------------------------------------
def _get_or_create_person(conn, name):
    row = conn.execute("SELECT id FROM persons WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO persons (name) VALUES (?)", (name,))
    return cur.lastrowid


def person_exists(name):
    with _lock, _connect() as conn:
        row = conn.execute("SELECT 1 FROM persons WHERE name = ?", (name,)).fetchone()
        return row is not None


def list_people():
    """Every registered person's name, sorted -- replaces
    os.listdir(config.FACE_DB_DIR)."""
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT name FROM persons ORDER BY name").fetchall()
        return [r["name"] for r in rows]


def delete_person(name):
    """Remove a person and (via ON DELETE CASCADE) all their embeddings and
    images. Replaces shutil.rmtree() on face_db/<name> and processed/<name>.
    Returns True if a row was actually deleted."""
    with _lock, _connect() as conn:
        cur = conn.execute("DELETE FROM persons WHERE name = ?", (name,))
        conn.commit()
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Registration writes (called by registration/face_registrar.py)
# ---------------------------------------------------------------------------
def save_face_data(name, embedding_list, image_list):
    """
    Persist a person's embeddings + raw source images. Re-registering the
    same name REPLACES their previous raw data (matches the old behavior of
    overwriting person_dir on disk).
    """
    with _lock, _connect() as conn:
        person_id = _get_or_create_person(conn, name)
        conn.execute("DELETE FROM embeddings WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM images WHERE person_id = ? AND kind = 'raw'", (person_id,))

        for idx, (embedding, img_bgr) in enumerate(zip(embedding_list, image_list)):
            vector = np.asarray(embedding, dtype=np.float32).tobytes()
            conn.execute(
                "INSERT INTO embeddings (person_id, angle_index, vector) VALUES (?, ?, ?)",
                (person_id, idx, vector),
            )
            ok, buf = cv2.imencode(".jpg", img_bgr)
            if ok:
                conn.execute(
                    "INSERT INTO images (person_id, kind, angle_index, jpeg_data) "
                    "VALUES (?, 'raw', ?, ?)",
                    (person_id, idx, buf.tobytes()),
                )
        conn.commit()

    print(f"[FaceDatabase] Saved {len(image_list)} image(s)/embedding(s) for '{name}'")


def save_processed_images(name, image_list):
    """
    Store the background-removed/enhanced versions produced by
    image_preprocessor.py. Replaces writing to data/processed/<name>/*.jpg.
    """
    with _lock, _connect() as conn:
        person_id = _get_or_create_person(conn, name)
        conn.execute("DELETE FROM images WHERE person_id = ? AND kind = 'processed'", (person_id,))
        for idx, img_bgr in enumerate(image_list):
            ok, buf = cv2.imencode(".jpg", img_bgr)
            if ok:
                conn.execute(
                    "INSERT INTO images (person_id, kind, angle_index, jpeg_data) "
                    "VALUES (?, 'processed', ?, ?)",
                    (person_id, idx, buf.tobytes()),
                )
        conn.commit()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------
def get_raw_images(name):
    """Return this person's raw registered images as a list of BGR numpy
    arrays, in angle order -- used by
    image_preprocessor.process_person_background() instead of reading
    data/face_db/<name>/*.jpg from disk."""
    with _lock, _connect() as conn:
        row = conn.execute("SELECT id FROM persons WHERE name = ?", (name,)).fetchone()
        if row is None:
            return []
        rows = conn.execute(
            "SELECT jpeg_data FROM images WHERE person_id = ? AND kind = 'raw' "
            "ORDER BY angle_index",
            (row["id"],),
        ).fetchall()

    images = []
    for r in rows:
        arr = np.frombuffer(r["jpeg_data"], np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    return images


def get_processed_images(name):
    """Return this person's processed (background-removed) images, in
    angle order. Handy if the dashboard ever wants to show a person's
    registered photo."""
    with _lock, _connect() as conn:
        row = conn.execute("SELECT id FROM persons WHERE name = ?", (name,)).fetchone()
        if row is None:
            return []
        rows = conn.execute(
            "SELECT jpeg_data FROM images WHERE person_id = ? AND kind = 'processed' "
            "ORDER BY angle_index",
            (row["id"],),
        ).fetchall()

    images = []
    for r in rows:
        arr = np.frombuffer(r["jpeg_data"], np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    return images


def load_all_embeddings():
    """
    Return {name: np.ndarray of shape (N, 512)} for every registered person
    -- drop-in replacement for the old "scan data/face_db/*/embedding.npy"
    logic in FaceRecognizer.load_database().
    """
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT p.name AS name, e.vector AS vector "
            "FROM embeddings e JOIN persons p ON p.id = e.person_id "
            "ORDER BY p.name, e.angle_index"
        ).fetchall()

    database = {}
    for r in rows:
        vec = np.frombuffer(r["vector"], dtype=np.float32)
        database.setdefault(r["name"], []).append(vec)

    return {name: np.stack(vectors) for name, vectors in database.items()}


# Ensure the DB/tables exist as soon as this module is imported, same as
# how face_registrar.py / face_recognizer.py load their models at import
# time.
init_db()