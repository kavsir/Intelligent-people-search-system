"""
Shared configuration for both the registration (Flask) app and the
operation (camera + AI + servo) app.
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
# Camera (ESP32-S3-CAM MJPEG stream)
# ---------------------------------------------------------------------------
# Change to "http://192.168.0.10/stream" or "http://192.168.0.10:81/stream"
# depending on your firmware.
CAMERA_URL = "http://192.168.0.10/stream"

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
# from flapping the state (and the servo) back and forth.
LOST_GRACE_PERIOD_SEC = 1.0

# ---------------------------------------------------------------------------
# Servo / serial
# ---------------------------------------------------------------------------
SERVO_PORT = "COM5"          # e.g. "COM5" on Windows, "/dev/ttyUSB0" on Linux
SERVO_BAUDRATE = 115200

# Ignore target offsets smaller than this many pixels (keeps the servo from
# hunting/jittering when the target is already close enough to center).
SERVO_DEADZONE_PX = 15

# Maximum degrees the pan/tilt angle is allowed to change in a single
# control step, regardless of what the PID output says. Prevents sudden
# jerky motion, e.g. right after switching tracking targets.
SERVO_MAX_DEGREE_STEP = 3.0