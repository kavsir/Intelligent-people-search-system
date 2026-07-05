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
    body_recognition -- long-term, clothing-invariant body-shape profile
                     (skeleton ratios only) per registered person, refined
                     every time their body is measured in any room. See
                     operation/body_features.py.
    db_meta       -- small key/value table. Currently holds
                     'face_db_version', bumped every time save_face_data()
                     or delete_person() changes who's registered, so every
                     running FaceRecognizer (possibly in a different OS
                     process) can hot-reload without an app restart.

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

-- Small key/value table -- see get_face_db_version() / _bump_face_db_version().
CREATE TABLE IF NOT EXISTS db_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Long-term, clothing-invariant body-shape profile per registered person.
-- ONE row per person, continuously refined (exponential moving average)
-- every time their body is measured in ANY room. Every column is a
-- dimensionless RATIO between skeletal segment lengths (see
-- operation/body_features.py) -- never an absolute pixel length, never a
-- clothing color/texture feature -- so the profile stays valid regardless
-- of distance-from-camera or what the person is wearing that day.
CREATE TABLE IF NOT EXISTS body_recognition (
    person_name           TEXT PRIMARY KEY,
    sample_count          INTEGER NOT NULL DEFAULT 0,
    shoulder_hip_ratio    REAL,   -- shoulder width / hip width
    torso_leg_ratio       REAL,   -- torso length / (thigh+shin) length
    thigh_shin_ratio      REAL,   -- thigh length / shin length
    shoulder_torso_ratio  REAL,   -- shoulder width / torso length
    body_aspect_ratio     REAL,   -- bbox height/width -- crude thin/stocky proxy, noisiest column
    updated_at            TEXT NOT NULL DEFAULT (datetime('now'))
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
    # WAL lets one process write while another reads without "database is
    # locked" errors -- important here because app_registration.py (port
    # 5000) and app_dashboard.py (port 5001) are two SEPARATE OS processes
    # hitting this same file concurrently.
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _bump_face_db_version(conn):
    """Increment the face_db_version counter inside an already-open
    connection (call BEFORE conn.commit()). Every FaceRecognizer instance
    -- in app_dashboard.py, app_operation.py, or any other process reading
    this same SQLite file -- polls get_face_db_version() and calls
    load_database() again the moment it sees this counter change. This is
    what lets a newly-registered person (or a deletion) become recognizable
    immediately, without restarting app_dashboard.py."""
    conn.execute(
        "INSERT INTO db_meta (key, value) VALUES ('face_db_version', '1') "
        "ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)"
    )


def get_face_db_version():
    """Current face_db_version counter (0 if nobody has ever registered
    yet). Cheap single-row SELECT -- safe to poll every few seconds from
    every AIPipeline thread."""
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT value FROM db_meta WHERE key = 'face_db_version'"
        ).fetchone()
        return int(row["value"]) if row else 0


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
        if cur.rowcount > 0:
            _bump_face_db_version(conn)
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
        _bump_face_db_version(conn)
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


# ---------------------------------------------------------------------------
# Body recognition (long-term, clothing-invariant body-shape profile)
# ---------------------------------------------------------------------------
# One row per person, smoothed with an exponential moving average so a
# single noisy pose reading never overwrites the profile outright -- see
# operation/body_features.py for how the ratios themselves are computed.
_BODY_EMA_ALPHA = 0.15
_BODY_FEATURE_COLUMNS = (
    "shoulder_hip_ratio",
    "torso_leg_ratio",
    "thigh_shin_ratio",
    "shoulder_torso_ratio",
    "body_aspect_ratio",
)


def update_body_profile(name, features):
    """
    Fold one fresh body-feature reading into `name`'s running-average
    profile. `features` is a dict with the 5 keys in
    _BODY_FEATURE_COLUMNS (see operation/body_features.py). Missing keys
    are left untouched. Creates the row on the person's first sighting.
    """
    now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM body_recognition WHERE person_name = ?", (name,)
        ).fetchone()

        if row is None:
            conn.execute(
                "INSERT INTO body_recognition "
                "(person_name, sample_count, shoulder_hip_ratio, torso_leg_ratio, "
                " thigh_shin_ratio, shoulder_torso_ratio, body_aspect_ratio, updated_at) "
                "VALUES (?, 1, ?, ?, ?, ?, ?, ?)",
                (
                    name,
                    features.get("shoulder_hip_ratio"),
                    features.get("torso_leg_ratio"),
                    features.get("thigh_shin_ratio"),
                    features.get("shoulder_torso_ratio"),
                    features.get("body_aspect_ratio"),
                    now_str,
                ),
            )
        else:
            def ema(col):
                old, new = row[col], features.get(col)
                if new is None:
                    return old
                return new if old is None else (1 - _BODY_EMA_ALPHA) * old + _BODY_EMA_ALPHA * new

            conn.execute(
                "UPDATE body_recognition SET sample_count = sample_count + 1, "
                "shoulder_hip_ratio = ?, torso_leg_ratio = ?, thigh_shin_ratio = ?, "
                "shoulder_torso_ratio = ?, body_aspect_ratio = ?, updated_at = ? "
                "WHERE person_name = ?",
                (
                    ema("shoulder_hip_ratio"),
                    ema("torso_leg_ratio"),
                    ema("thigh_shin_ratio"),
                    ema("shoulder_torso_ratio"),
                    ema("body_aspect_ratio"),
                    now_str,
                    name,
                ),
            )
        conn.commit()


def get_body_profile(name):
    """Return this person's current body profile dict, or None if they've
    never been measured yet."""
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM body_recognition WHERE person_name = ?", (name,)
        ).fetchone()
    return dict(row) if row else None


def get_all_body_profiles():
    """Every registered person's current body profile -- {name: {...}}."""
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM body_recognition").fetchall()
    return {r["person_name"]: dict(r) for r in rows}


# Weights used by feature_similarity() below -- body_aspect_ratio is the
# noisiest of the 5 columns (pose/arm-position sensitive), so it counts
# least toward the final score.
BODY_FEATURE_WEIGHTS = {
    "shoulder_hip_ratio": 0.25,
    "torso_leg_ratio": 0.30,
    "thigh_shin_ratio": 0.20,
    "shoulder_torso_ratio": 0.20,
    "body_aspect_ratio": 0.05,
}


def feature_similarity(a, b):
    """
    Weighted similarity in [0, 1] between two body-feature dicts (same 5
    keys as body_recognition's columns -- either a fresh reading from
    operation/body_features.py or a stored profile row). 1.0 = identical
    ratios, drops toward 0 as they diverge. Each ratio is compared as a
    RELATIVE difference (not absolute), since different people naturally
    cluster around different baseline ratios.
    """
    weighted_score = 0.0
    total_weight = 0.0
    for key, weight in BODY_FEATURE_WEIGHTS.items():
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            continue
        denom = max(abs(va), abs(vb), 1e-6)
        rel_diff = abs(va - vb) / denom
        score = max(0.0, 1.0 - rel_diff)
        weighted_score += weight * score
        total_weight += weight
    return (weighted_score / total_weight) if total_weight > 0 else 0.0


def match_body_profile(features, candidate_names=None, min_sample_count=5):
    """
    Compare one fresh body-feature reading against stored profiles.

    candidate_names: if given, only these people are considered (a
        TARGETED re-check -- e.g. "is this probably the person we just
        lost track of"). If None, every profile with enough history is
        considered (a COLD match against everyone registered).
    min_sample_count: a profile needs at least this many real sightings
        before it's trusted enough to match against -- a 1-sample profile
        is too noisy to use as a fingerprint.

    Returns (best_name, similarity), or (None, 0.0) if nobody qualifies.
    Does NOT apply an acceptance threshold itself -- callers decide what
    similarity counts as "confident enough" (body shape alone is a much
    weaker biometric than face recognition, so this should always be a
    higher, stricter bar -- see config.BODY_MATCH_MIN_SIMILARITY*).
    """
    profiles = get_all_body_profiles()
    if candidate_names is not None:
        profiles = {n: p for n, p in profiles.items() if n in candidate_names}

    best_name, best_score = None, -1.0
    for name, profile in profiles.items():
        if profile.get("sample_count", 0) < min_sample_count:
            continue
        score = feature_similarity(features, profile)
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is None:
        return None, 0.0
    return best_name, best_score


def delete_body_profile(name):
    """Wipe a person's body_recognition row -- call alongside
    delete_person() if you want a full reset."""
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM body_recognition WHERE person_name = ?", (name,))
        conn.commit()


# Ensure the DB/tables exist as soon as this module is imported, same as
# how face_registrar.py / face_recognizer.py load their models at import
# time.
init_db()