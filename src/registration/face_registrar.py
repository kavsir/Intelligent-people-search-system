"""
Face registration utilities.

- Face embeddings are produced by InsightFace (buffalo_sc), which gives a
  proper 512-D ArcFace embedding suitable for face verification/recognition.
- Face landmarks (used only to draw a preview grid on the registration page)
  are produced by a YOLOv8-face model, since it is fast and good enough for
  a visual overlay -- it is NOT used for recognition.
"""

import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# InsightFace: used for the actual face embedding (registration + recognition)
# Prefer GPU (ctx_id >= 0) when a usable NVIDIA/CUDA setup is detected,
# otherwise fall back to CPU (ctx_id=-1). If GPU init fails for any reason
# (missing onnxruntime-gpu, driver mismatch, etc.) we retry on CPU instead
# of crashing the app.
_face_app = FaceAnalysis(name="buffalo_sc")
_ctx_id = config.get_insightface_ctx_id()
try:
    _face_app.prepare(ctx_id=_ctx_id, det_size=(640, 640))
    print(f"[FaceRegistrar] InsightFace running on {'GPU' if _ctx_id >= 0 else 'CPU'}.")
except Exception as exc:
    if _ctx_id != -1:
        print(f"[FaceRegistrar] GPU init failed ({exc}); falling back to CPU.")
        _face_app.prepare(ctx_id=-1, det_size=(640, 640))
    else:
        raise

# YOLOv8-face: used only for the landmark preview overlay on the web UI.
# Ultralytics/PyTorch picks the device per-call (see get_face_landmarks),
# so loading the model itself doesn't need a device argument.
try:
    _yolo_face_model = YOLO(config.YOLO_FACE_MODEL_PATH)
except Exception:
    _yolo_face_model = YOLO(config.YOLO_PERSON_MODEL_PATH)

_device = config.get_torch_device()
print(f"[FaceRegistrar] YOLO landmark model running on {_device.upper()}.")


def _select_best_face(faces):
    """Pick the largest/most confident face among InsightFace detections."""
    if not faces:
        return None

    best_face = None
    best_score = -1.0

    for face in faces:
        x1, y1, x2, y2 = face.bbox
        area = (x2 - x1) * (y2 - y1)
        conf = float(getattr(face, "det_score", 0.5))
        score = area * conf

        if score > best_score:
            best_score = score
            best_face = face

    return best_face


def get_face_embedding(image_bgr):
    """Return the 512-D normalized embedding of the best face, or None."""
    faces = _face_app.get(image_bgr)
    face = _select_best_face(faces)
    if face is None:
        return None
    return face.normed_embedding


def get_face_landmarks(image_bgr):
    """
    Return a set of 2D points used purely for the preview overlay on the
    registration page. Not used for recognition.
    """
    if image_bgr is None:
        return None

    results = _yolo_face_model(image_bgr, verbose=False, device=_device)

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        best_idx = 0
        best_score = -1.0

        for i, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item() if box.conf is not None else 0.5
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            score = area * conf

            if score > best_score:
                best_score = score
                best_idx = i

        # If the model has real keypoints (YOLOv8-face), use them.
        if (
            hasattr(result, "keypoints")
            and result.keypoints is not None
            and len(result.keypoints) > 0
        ):
            kp = result.keypoints.xy[best_idx].cpu().numpy()
            if len(kp) > 0:
                return kp

        # Fallback: synthesize a few placeholder points from the bbox so the
        # UI still has something to draw.
        box = result.boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = box[:4]
        return np.array(
            [
                [(x1 + x2) / 2 - 20, (y1 + y2) / 2 - 20],
                [(x1 + x2) / 2 + 20, (y1 + y2) / 2 - 20],
                [(x1 + x2) / 2, (y1 + y2) / 2],
                [(x1 + x2) / 2 - 15, (y1 + y2) / 2 + 20],
                [(x1 + x2) / 2 + 15, (y1 + y2) / 2 + 20],
            ]
        )

    return None


def save_face_data(name, embedding_list, image_list):
    """
    Persist a person's embeddings and source images under:
        data/face_db/<name>/embedding.npy
        data/face_db/<name>/angle_1.jpg, angle_2.jpg, ...
    """
    person_dir = os.path.join(config.FACE_DB_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    embeddings = np.array(embedding_list)
    np.save(os.path.join(person_dir, "embedding.npy"), embeddings)

    for index, img_bgr in enumerate(image_list):
        img_name = (
            f"angle_{index + 1}.jpg" if len(image_list) > 1 else "profile.jpg"
        )
        cv2.imwrite(os.path.join(person_dir, img_name), img_bgr)

    print(f"[REGISTER] Saved {len(image_list)} image(s) for '{name}' -> {person_dir}")
    return person_dir
