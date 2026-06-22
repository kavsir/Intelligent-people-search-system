"""
Background removal + image enhancement pipeline applied to registered
face images before they are used downstream.

Pipeline per image:
    1. Detect the closest/largest face with YOLO.
    2. Crop around that face with extra padding (so hair/shoulders survive).
    3. Remove the background (rembg) and replace it with solid black.
    4. Denoise + sharpen the result.

Input:  data/face_db/<name>/*.jpg
Output: data/processed/<name>/*.jpg
"""

import io
import os
import sys

import cv2
import numpy as np
from PIL import Image
from rembg import remove
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ---------------------------------------------------------------------------
# Load YOLO (face model, falling back to the general person model)
# ---------------------------------------------------------------------------
try:
    _yolo_model = YOLO(config.YOLO_FACE_MODEL_PATH)
except Exception:
    print("[PREPROCESS] Face model not found, falling back to person model.")
    _yolo_model = YOLO(config.YOLO_PERSON_MODEL_PATH)

_device = config.get_torch_device()
print(f"[PREPROCESS] YOLO running on {_device.upper()}.")


def enhance_image_quality(img):
    """Denoise then sharpen an image."""
    denoised = cv2.fastNlMeansDenoisingColored(
        img, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
    )

    blur = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)

    return sharpened


def remove_background(img):
    """Remove the background and replace it with solid black."""
    success, buffer = cv2.imencode(".png", img)
    if not success:
        return img

    output_data = remove(buffer.tobytes())
    img_rgba = Image.open(io.BytesIO(output_data)).convert("RGBA")

    background = Image.new("RGB", img_rgba.size, (0, 0, 0))
    background.paste(img_rgba, mask=img_rgba.split()[3])

    return cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)


def _select_best_face_box(img):
    """Return the bbox (x1, y1, x2, y2) of the largest/most confident face."""
    h, w = img.shape[:2]
    results = _yolo_model(img, verbose=False, device=_device)

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

        box = result.boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, box[:4])

        padding_x = int((x2 - x1) * 0.5)
        padding_y = int((y2 - y1) * 0.8)

        nx1 = max(0, x1 - padding_x)
        ny1 = max(0, y1 - padding_y)
        nx2 = min(w, x2 + padding_x)
        ny2 = min(h, y2 + padding_y)

        return nx1, ny1, nx2, ny2

    return None


def process_background_pipeline(img):
    """Crop to the most prominent face (if any), then remove the background."""
    box = _select_best_face_box(img)
    region = img if box is None else img[box[1]:box[3], box[0]:box[2]]
    return remove_background(region)


def process_person_background(name):
    """
    Process every image registered for `name` under data/face_db/<name>
    and write the result to data/processed/<name>.
    """
    source_dir = os.path.join(config.FACE_DB_DIR, name)
    target_dir = os.path.join(config.PROCESSED_DIR, name)

    if not os.path.exists(source_dir):
        print(f"[PREPROCESS] No source directory for '{name}'.")
        return False

    os.makedirs(target_dir, exist_ok=True)

    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [
        f for f in os.listdir(source_dir) if f.lower().endswith(valid_extensions)
    ]

    for file_name in image_files:
        img_path = os.path.join(source_dir, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        segmented_img = process_background_pipeline(img)
        final_img = enhance_image_quality(segmented_img)

        cv2.imwrite(os.path.join(target_dir, file_name), final_img)

    print(f"[PREPROCESS] Finished processing '{name}'.")
    return True
