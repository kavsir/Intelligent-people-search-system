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

FACE_DB_DIR = os.path.join(DATA_DIR, "face_db")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# ---------------------------------------------------------------------------
# Cameras (ESP32-CAM MJPEG streams)
# ---------------------------------------------------------------------------
# Each ESP32-CAM streams MJPEG directly to this machine (no longer routed
# through the ESP32-S3, which now only acts as a coordination gateway over
# HTTP/MQTT to tell each ESP32-CAM when to stream/snapshot/sleep).
#
# Cameras are fixed-position (no pan/tilt servo). Each one only detects and
# recognizes whoever is in its own room; there is no mechanism that moves
# the camera to follow a person.
CAMERAS = [
    {
        "id": "cam1",
        "room_name": "Phong 1",
        "url": "http://192.168.0.10/stream",
    },
    {
        "id": "cam2",
        "room_name": "Phong 2",
        "url": "http://192.168.0.11/stream",
    },
]

# Kept for backward compatibility with any code (e.g. registration app)
# that still imports a single CAMERA_URL directly.
CAMERA_URL = CAMERAS[0]["url"]

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
TARGET_FPS = 30

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
FACE_RECOGNITION_THRESHOLD = 0.5

# How many AI-pipeline steps to wait between identity re-checks while a face
# is already locked and tracked by CSRT. Identity doesn't need to be
# re-verified every frame -- this just confirms we're still tracking the
# right person.
IDENTITY_RECHECK_INTERVAL = 10

# ---------------------------------------------------------------------------
# Target-loss safety behavior
# ---------------------------------------------------------------------------
# Seconds of consecutive "no target" frames tolerated before the FSM
# actually declares the target lost. Prevents 1-2 frame detection hiccups
# from flapping the state back and forth.
LOST_GRACE_PERIOD_SEC = 1.0

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