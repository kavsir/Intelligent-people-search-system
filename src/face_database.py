# face_database.py
"""
SQLite-backed storage for the face dataset.

Replaces the old "labeled folders" layout
    data/face_db/<name>/embedding.npy, angle_1.jpg, ...
    data/processed/<name>/*.jpg
with a single portable database file:
    data/face_dataset.db

Schema:
    persons       -- one row per registered person
    embeddings    -- one row per captured angle's 512-D InsightFace embedding
    images        -- one row per stored image (raw capture OR background-
                     removed "processed" version), stored as JPEG bytes (BLOB)
    behavior_log  -- time-series snapshots of every registered person's
                     cumulative behavior counters (Đứng, Di chuyển, Nhảy,
                     Giơ tay, Nằm, squat, hít đất), written every 5s by
                     operation/behavior_manager.py. NOT reset between rows
                     -- each row is "the running total as of `logged_at`",
                     so the dashboard/report can plot activity over time.
    theo_doi      -- tracking snapshots: mỗi lần danh sách người đăng ký thay đổi
                     (thêm/xóa/di chuyển phòng), chụp ảnh mỗi phòng có người
                     và lưu kèm thông tin người + phòng + thời gian.

Every other module (face_registrar, image_preprocessor, face_recognizer,
app_dashboard) goes through this module instead of touching the filesystem
directly -- this is the single source of truth for "who is registered" and
"what does their data look like".

Migrating existing folder-based data: run migrate_to_sqlite.py once.
"""

import os
import sqlite3
import threading
import time

import cv2
import numpy as np

import config

DB_PATH = getattr(config, "FACE_DB_PATH", os.path.join(config.DATA_DIR, "face_dataset.db"))

# sqlite3 connections aren't shared across threads; a short-lived connection
# per call plus a module-level lock keeps writes/reads safe across the
# registration Flask app, the dashboard Flask app, and every AIPipeline
# thread that calls load_all_embeddings() / log_behavior_snapshot().
_lock = threading.Lock()

# Columns tracked in behavior_log, in display order. Keep this in sync with
# operation/behavior_manager.py's BEHAVIORS tuple.
BEHAVIOR_COLUMNS = ("stand", "move", "jump", "raise_hand", "lie", "squat", "pushup")

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

CREATE TABLE IF NOT EXISTS behavior_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name      TEXT NOT NULL,
    stand_count      INTEGER NOT NULL DEFAULT 0,
    move_count       INTEGER NOT NULL DEFAULT 0,
    jump_count       INTEGER NOT NULL DEFAULT 0,
    raise_hand_count INTEGER NOT NULL DEFAULT 0,
    lie_count        INTEGER NOT NULL DEFAULT 0,
    squat_count      INTEGER NOT NULL DEFAULT 0,
    pushup_count     INTEGER NOT NULL DEFAULT 0,
    logged_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS theo_doi (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name TEXT NOT NULL,
    room_name   TEXT NOT NULL,
    image_data  BLOB NOT NULL,      -- JPEG bytes từ camera của phòng đó
    captured_at TEXT NOT NULL       -- thời gian thực chụp
);

CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);
CREATE INDEX IF NOT EXISTS idx_images_person_kind ON images(person_id, kind);
CREATE INDEX IF NOT EXISTS idx_behavior_log_person_time
    ON behavior_log(person_name, logged_at);
CREATE INDEX IF NOT EXISTS idx_theo_doi_person ON theo_doi(person_name);
CREATE INDEX IF NOT EXISTS idx_theo_doi_time ON theo_doi(captured_at);
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
    Returns True if a row was actually deleted.

    Note: behavior_log rows are kept (they're a historical activity report,
    not "current registration state"), matched only by person_name -- call
    delete_behavior_history(name) too if you want a full wipe."""
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


# ---------------------------------------------------------------------------
# Behavior recognition log (Đứng / Di chuyển / Nhảy / Giơ tay / Nằm / squat /
# hít đất) -- written by operation/behavior_manager.py every 5s.
# ---------------------------------------------------------------------------
def log_behavior_snapshot(counts_by_person):
    """
    Insert ONE row per person into behavior_log, timestamped with the
    current real time (SQLite's own localtime clock, so `logged_at` is a
    real wall-clock time, not simulation time).

    counts_by_person: {
        "alice": {"stand": 12, "move": 3, "jump": 1, "raise_hand": 2,
                   "lie": 0, "squat": 8, "pushup": 0},
        ...
    }
    Missing keys default to 0. Called every BEHAVIOR_SNAPSHOT_INTERVAL_SEC
    seconds by BehaviorManager's background thread -- counts are the
    person's RUNNING TOTAL as of that moment (not reset each snapshot), so
    plotting logged_at vs each *_count column gives an activity-over-time
    curve for the technical report.
    """
    if not counts_by_person:
        return

    now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    with _lock, _connect() as conn:
        for name, counts in counts_by_person.items():
            conn.execute(
                "INSERT INTO behavior_log "
                "(person_name, stand_count, move_count, jump_count, "
                " raise_hand_count, lie_count, squat_count, pushup_count, logged_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    name,
                    counts.get("stand", 0),
                    counts.get("move", 0),
                    counts.get("jump", 0),
                    counts.get("raise_hand", 0),
                    counts.get("lie", 0),
                    counts.get("squat", 0),
                    counts.get("pushup", 0),
                    now_str,
                ),
            )
        conn.commit()


def get_latest_behavior_counts():
    """
    Return {name: {"stand":..,...,"logged_at":..}} using each person's most
    recent behavior_log row -- what the dashboard's behavior table shows
    right now.
    """
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT b.* FROM behavior_log b "
            "JOIN (SELECT person_name, MAX(id) AS max_id FROM behavior_log "
            "      GROUP BY person_name) latest "
            "ON b.person_name = latest.person_name AND b.id = latest.max_id"
        ).fetchall()

    result = {}
    for r in rows:
        result[r["person_name"]] = {
            "stand": r["stand_count"],
            "move": r["move_count"],
            "jump": r["jump_count"],
            "raise_hand": r["raise_hand_count"],
            "lie": r["lie_count"],
            "squat": r["squat_count"],
            "pushup": r["pushup_count"],
            "logged_at": r["logged_at"],
        }
    return result


def get_behavior_history(name, limit=500):
    """Time-ordered snapshots for one person -- e.g. for an activity chart
    on the dashboard's per-person detail view."""
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT stand_count, move_count, jump_count, raise_hand_count, "
            "       lie_count, squat_count, pushup_count, logged_at "
            "FROM behavior_log WHERE person_name = ? "
            "ORDER BY id DESC LIMIT ?",
            (name, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def delete_behavior_history(name):
    """Wipe a person's behavior_log rows -- call alongside delete_person()
    if you want a full reset, not just an unregister."""
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM behavior_log WHERE person_name = ?", (name,))
        conn.commit()


# ---------------------------------------------------------------------------
# Theo dõi (tracking) snapshots
# ---------------------------------------------------------------------------
# Mỗi lần danh sách người đăng ký thay đổi (thêm/xóa/di chuyển phòng),
# chụp ảnh từng phòng có người và lưu vào bảng theo_doi.
# Quy tắc:
#   - Thêm người → chụp
#   - Giảm người (từ N→N-1, N-1→N-2, ...) → chụp mỗi lần giảm
#   - Giảm về 0 (không còn ai) → KHÔNG chụp
#   - Người di chuyển phòng (danh sách tên không đổi nhưng phòng thay đổi) → chụp
# ---------------------------------------------------------------------------

def save_tracking_snapshot(entries, image_by_room, timestamp=None):
    """
    Save a tracking snapshot: for each person currently detected in a room,
    store a row with that room's camera frame as JPEG.

    entries: [{"name": "alice", "room_name": "Phòng 1"}, ...]
    image_by_room: {"Phòng 1": jpeg_bytes, "Phòng 2": jpeg_bytes, ...}
                   map từ room_name → JPEG bytes của frame camera phòng đó
    timestamp: str (định dạng "YYYY-MM-DD HH:MM:SS"), mặc định dùng thời gian hiện tại
    """
    if not entries:
        return
    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    with _lock, _connect() as conn:
        for entry in entries:
            room_name = entry["room_name"]
            jpeg_data = image_by_room.get(room_name)
            if jpeg_data is None:
                # Không có ảnh cho phòng này, bỏ qua người này
                continue
            conn.execute(
                "INSERT INTO theo_doi (person_name, room_name, image_data, captured_at) "
                "VALUES (?, ?, ?, ?)",
                (entry["name"], room_name, jpeg_data, timestamp),
            )
        conn.commit()

    print(f"[FaceDatabase] Saved tracking snapshot: {len(entries)} person(s), {len(image_by_room)} room(s)")


def get_latest_tracking():
    """
    Return the most recent tracking snapshot for each person.
    Returns: {name: {"room_name": str, "image_data": bytes|None, "captured_at": str|None}}
    Nếu người chưa từng được theo dõi, sẽ không có key trong dict trả về.
    """
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT t.* FROM theo_doi t "
            "JOIN (SELECT person_name, MAX(id) AS max_id FROM theo_doi "
            "      GROUP BY person_name) latest "
            "ON t.person_name = latest.person_name AND t.id = latest.max_id"
        ).fetchall()

    result = {}
    for r in rows:
        result[r["person_name"]] = {
            "room_name": r["room_name"],
            "image_data": bytes(r["image_data"]) if r["image_data"] else None,
            "captured_at": r["captured_at"],
        }
    return result


def get_tracking_history(name, limit=100):
    """
    Return all tracking snapshots for one person, newest first.
    Handy for a "history" view on the people page.
    """
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT room_name, image_data, captured_at "
            "FROM theo_doi WHERE person_name = ? "
            "ORDER BY id DESC LIMIT ?",
            (name, limit),
        ).fetchall()
    return [
        {
            "room_name": r["room_name"],
            "image_data": bytes(r["image_data"]) if r["image_data"] else None,
            "captured_at": r["captured_at"],
        }
        for r in rows
    ]


def delete_tracking_history(name):
    """Wipe a person's theo_doi rows -- call alongside delete_person()
    if you want a full reset."""
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM theo_doi WHERE person_name = ?", (name,))
        conn.commit()


# Ensure the DB/tables exist as soon as this module is imported, same as
# how face_registrar.py / face_recognizer.py load their models at import
# time.
init_db()