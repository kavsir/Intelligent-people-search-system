"""
Shared configuration for both the registration (Flask) app and the
operation (camera + AI recognition) app.
"""

import os

# ---------------------------------------------------------------------------
# Device selection (NVIDIA GPU if available, otherwise CPU)
# ---------------------------------------------------------------------------
# USE_GPU=True means "prefer GPU". Every model loader still falls back to
# CPU automatically if no usable GPU/CUDA setup is found at runtime, so
# this never crashes the app on a machine without a GPU.
USE_GPU = True


def get_torch_device():
    """
    Return "cuda" if PyTorch can see a usable NVIDIA GPU, otherwise "cpu".
    Used by Ultralytics/YOLO, which is built on PyTorch.
    """
    if not USE_GPU:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_insightface_ctx_id():
    """
    Return the ctx_id expected by InsightFace's FaceAnalysis.prepare():
    a GPU index (0, 1, ...) to use CUDA, or -1 to force CPU.

    InsightFace uses onnxruntime under the hood. If onnxruntime-gpu (with a
    working CUDA setup) isn't installed, onnxruntime silently falls back to
    its CPU provider on its own -- but checking torch.cuda.is_available()
    first lets us pick the GPU ctx_id only when a GPU genuinely looks usable,
    and avoids the "Specified provider 'CUDAExecutionProvider' is not in
    available provider names" warning when there's clearly no GPU at all.
    """
    if not USE_GPU:
        return -1
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return -1


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# src/config.py -> project root is one level up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Legacy "labeled folders" dataset layout. No longer written to -- kept
# only so migrate_to_sqlite.py can read the old data once and import it
# into FACE_DB_PATH below. Safe to delete these folders after migrating.
FACE_DB_DIR = os.path.join(DATA_DIR, "face_db")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Current dataset storage: a single SQLite file holding every registered
# person's embeddings + raw/processed images. See face_database.py.
FACE_DB_PATH = os.path.join(DATA_DIR, "face_dataset.db")

# ---------------------------------------------------------------------------
# Door servo ESP32 (Dev Module) -- WebSocket
# ---------------------------------------------------------------------------
# ONE physical ESP32 Dev Module drives BOTH door servos (Phòng 1 + Phòng 2)
# and connects IN to this server as a single WebSocket *client* (see
# operation/door_ws_server.py and esp32_servo.ino). Each door/room is
# addressed by the same id as its entry in CAMERAS (e.g. "cam1", "cam2"),
# multiplexed over that one connection -- we don't need to know the
# ESP32's IP, we just bind and listen here.
# ---------------------------------------------------------------------------
# Cross-app links
# ---------------------------------------------------------------------------
# Port app_registration.py listens on (see its `app.run(..., port=...)`).
# app_dashboard.py exposes this via /api/config so dashboard.html can build
# a working "Đăng ký khuôn mặt" link regardless of which host/IP the
# dashboard is being viewed from.
REGISTRATION_APP_PORT = 5000

DOOR_WS_HOST = "0.0.0.0"   # interface the door WebSocket server binds to
DOOR_WS_PORT = 8765        # must match `ws_port` in esp32_servo.ino

# Door lockdown model (see app_dashboard.py's _lockdown_all_doors):
#   - With nobody registered detected anywhere, every door button works
#     normally -- open/close on request, no auto behavior.
#   - The instant a registered person is detected in ANY room, every door
#     in the system is force-closed automatically. Doors never reopen by
#     themselves afterwards -- opening always requires a manual dashboard
#     button press, regardless of who's present.
# There is no "close after N seconds of absence" timer anymore: closing is
# triggered by presence being detected, not by absence.

CAMERAS = [
    {
        "id": "cam1",
        "room_name": "Phong 1",
        "url": "http://10.153.15.178/stream",
    },
    {
        "id": "cam2",
        "room_name": "Phong 2",
        "url": "http://10.153.15.227/stream",
    },
]

# Kept for backward compatibility with any code (e.g. registration app)
# that still imports a single CAMERA_URL directly.
CAMERA_URL = CAMERAS[0]["url"]

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
TARGET_FPS = 30

# How long (seconds) CameraReader's watchdog waits with NO new frame,
# despite the connection still looking "open", before it force-releases
# the stream and reconnects. Fixes the case where the ESP32-CAM's stream
# dies mid-connection (Wi-Fi drop, camera reboot, momentary wrong IP...)
# and cv2.VideoCapture.read() ends up blocked forever instead of actually
# returning an error -- without this, the camera never automatically
# recovers even after its signal comes back. See operation/camera_reader.py.
CAMERA_STALL_TIMEOUT_SEC = 5.0

# ---------------------------------------------------------------------------
# YOLO / face detection
# ---------------------------------------------------------------------------
YOLO_FACE_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n-face.pt")
YOLO_PERSON_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")

# Minimum confidence required to accept a detected face/person box.
AI_CONF_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Face recognition (InsightFace)
# ---------------------------------------------------------------------------
# Minimum cosine similarity (normed embeddings -> plain dot product) for a
# detected face to be considered a match for a registered person. Lower this
# (e.g. 0.4) if recognition feels too strict; raise it if strangers are
# getting matched to a registered name.
FACE_RECOGNITION_THRESHOLD = 0.35

# How many AI-pipeline steps to wait between identity re-checks while a face
# is already locked and tracked by CSRT. Identity doesn't need to be
# re-verified every frame -- this just confirms we're still tracking the
# right person.
IDENTITY_RECHECK_INTERVAL = 10

# How often (seconds) each AIPipeline thread checks face_database's
# face_db_version counter to see if app_registration.py (a SEPARATE
# process) just registered/deleted someone, and hot-reloads its embeddings
# if so. This is what removes the "must restart app_dashboard.py after
# registering someone" requirement. Cheap (one small SELECT) -- 1-3s is a
# good default; lower it if you want new registrations picked up faster.
FACE_DB_RELOAD_CHECK_INTERVAL_SEC = 2.0

# ---------------------------------------------------------------------------
# Body recognition (long-term, clothing-invariant body-shape profile)
# ---------------------------------------------------------------------------
# How often (seconds), PER PERSON, each AIPipeline re-measures a
# registered face's body-shape ratios (see operation/body_features.py).
# Runs for EVERY registered face seen this step, not just the locked
# target -- an extra YOLO-person pass + a mediapipe Pose call per person
# is real CPU cost on top of the existing pipeline, so this is throttled
# rather than run every frame. Raise it (e.g. 5.0) if CPU is tight with
# multiple rooms running at once.
BODY_PROFILE_UPDATE_INTERVAL_SEC = 2.0

# ---------------------------------------------------------------------------
# Body-shape fallback identification (used when face recognition can't see
# a face at all -- turned away, too far, bad angle). See ai_pipeline.py's
# _find_identity_by_body_shape(). Deliberately a SEPARATE, much higher bar
# than face recognition's own FACE_RECOGNITION_THRESHOLD, since body shape
# alone is a far weaker biometric signal than a face embedding.
# ---------------------------------------------------------------------------
# Targeted re-check: "is this probably the specific person we JUST lost
# track of" -- a lower bar is acceptable because context already narrows
# it down to one candidate.
BODY_MATCH_MIN_SIMILARITY = 0.85
# Cold match: "does this unrecognized body belong to ANY registered
# person" -- stricter, since there's no context narrowing the candidates.
BODY_MATCH_MIN_SIMILARITY_COLD = 0.92
# A profile needs at least this many real sightings before it's trusted
# enough to be matched against at all.
BODY_MATCH_MIN_SAMPLES = 5

# ---------------------------------------------------------------------------
# Target-loss safety behavior
# ---------------------------------------------------------------------------
# Seconds of consecutive "no target" frames tolerated before the FSM
# actually declares the target lost. Prevents 1-2 frame detection hiccups
# from flapping the state back and forth.
LOST_GRACE_PERIOD_SEC = 1.0

# ---------------------------------------------------------------------------
# CPU load / latency tuning
# ---------------------------------------------------------------------------
# Running multiple AIPipeline threads (one per camera) on CPU means they
# compete for the same cores -- YOLO-face/YOLO-person/InsightFace are all
# heavy enough that running 2+ at once noticeably slows each one down.
# These knobs reduce wasted CPU time without changing input resolution or
# detection accuracy.

# In SEARCHING, how many pipeline steps to skip between YOLO-face calls.
# 0 = run every step (old behavior). 2 means: run on step 0, skip steps 1
# and 2, run again on step 3, etc. -- roughly a 1/3 reduction in CPU spent
# on face detection while still scanning often enough that a newly-arrived
# registered person is found within a fraction of a second, not noticeably
# slower from a user's point of view.
SEARCHING_SKIP_FRAMES = 2

# Fixed delay (seconds) at the end of every pipeline step, applied only
# when the step itself was fast. If a step already took a while (because
# the CPU was busy with the other camera's thread), this sleep is skipped
# entirely instead of stacking on top of an already-slow frame. This
# replaces a flat time.sleep(0.01) that fired unconditionally even when
# the CPU had no time to spare.
STEP_SLEEP_SEC = 0.01
STEP_SLEEP_SKIP_IF_STEP_TOOK_LONGER_THAN_SEC = 0.03


# ---------------------------------------------------------------------------
# Behavior recognition (Đứng / Di chuyển / Nhảy / Giơ tay / Nằm)
# ---------------------------------------------------------------------------
BEHAVIOR_SNAPSHOT_INTERVAL_SEC = 5
# ---------------------------------------------------------------------------
# Floor plan & inferred presence
# ---------------------------------------------------------------------------
# How long (seconds) a "inferred presence" highlight stays on a no-cam room
# before being automatically cleared if no camera confirms the target.
# Can also be cleared manually via the dashboard Reset button.
# Set to 0 to disable auto-clear (keep until cam sees target again).
INFERRED_PRESENCE_TIMEOUT_SEC = 60

# Floor plan room definitions.
# Each room has:
#   id         – must match a CAMERAS[*]["id"] if the room has a camera,
#                or any unique string for cam-less rooms.
#   name       – display label on the floor map.
#   cam_id     – set to the matching camera id if this room has a cam,
#                or None for cam-less rooms.
#   neighbors  – list of room ids directly reachable from this room.
#                Used by the inference engine to decide which no-cam rooms
#                to highlight when a target disappears from a cam room.
FLOOR_PLAN = [
    {
        "id": "p1",
        "name": "Phòng 1",
        "cam_id": "cam1",
        "neighbors": ["p2"],
    },
    {
        "id": "p2",
        "name": "Phòng 2",
        "cam_id": "cam2",
        "neighbors": ["p1", "p3", "p4", "p5"],
    },
    {
        "id": "p3",
        "name": "Phòng 3",
        "cam_id": None,
        "neighbors": ["p2"],
    },
    {
        "id": "p4",
        "name": "Phòng 4",
        "cam_id": None,
        "neighbors": ["p2"],
    },
    {
        "id": "p5",
        "name": "Phòng 5",
        "cam_id": None,
        "neighbors": ["p2"],
    },
]