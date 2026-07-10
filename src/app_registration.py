"""
Flask app for face registration.

Routes:
    GET  /                       -> registration page (upload tab + webcam tab)
    POST /get_landmarks           -> preview-only landmark detection for the UI
    POST /register_from_image     -> register a single uploaded image
    POST /register_final          -> register a person from 5 webcam angles
    POST /register_body           -> register upper body shape (front + back)
"""

import base64
import os
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import face_database
from registration.face_registrar import (
    get_face_embedding,
    get_face_landmarks,
    save_face_data,
)
from registration.image_preprocessor import process_person_background
from operation.body_features import BodyFeatureExtractor, upper_body_box

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "registration", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "registration", "static"),
)

# Models for Body Registration
_yolo_body_model = YOLO(config.YOLO_PERSON_MODEL_PATH)
_body_extractor = BodyFeatureExtractor()
_device = config.get_torch_device()


def _decode_base64_image(image_data):
    """Decode a 'data:image/...;base64,...' string into a BGR numpy image."""
    if "," not in image_data:
        return None
    _, encoded = image_data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    return render_template("register.html")


@app.route("/get_landmarks", methods=["POST"])
def get_landmarks():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    if not image_data:
        return jsonify({"status": "error", "message": "Missing image data"})

    try:
        img = _decode_base64_image(image_data)
    except (ValueError, base64.binascii.Error):
        return jsonify({"status": "error", "message": "Invalid image data"})

    if img is None:
        return jsonify({"status": "error", "message": "Could not decode image"})

    landmarks = get_face_landmarks(img)
    if landmarks is None:
        return jsonify({"status": "error", "message": "No valid face found"})

    return jsonify({"status": "success", "landmarks": landmarks.tolist()})


@app.route("/register_from_image", methods=["POST"])
def register_from_image():
    """Register a person from a single uploaded image."""
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    name = data.get("name", "").strip()

    if not name:
        return jsonify({"status": "error", "message": "Missing person name"})
    if not image_data:
        return jsonify({"status": "error", "message": "Missing image data"})

    try:
        img = _decode_base64_image(image_data)
    except (ValueError, base64.binascii.Error):
        return jsonify({"status": "error", "message": "Invalid image data"})

    if img is None:
        return jsonify({"status": "error", "message": "Could not decode image"})

    embedding = get_face_embedding(img)
    if embedding is None:
        return jsonify({"status": "error", "message": "No face detected by the model"})

    save_face_data(name, [embedding], [img])
    process_person_background(name)

    return jsonify(
        {"status": "success", "message": f"Registered and preprocessed data for '{name}'"}
    )


@app.route("/register_final", methods=["POST"])
def register_final():
    """Register a person from 5 webcam angle captures."""
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    images = data.get("images", [])

    if not name:
        return jsonify({"status": "error", "message": "Missing person name"})
    if not images or len(images) < 5:
        return jsonify({"status": "error", "message": "Not enough captured angles (need 5)"})

    embeddings = []
    valid_images = []

    for image_data in images:
        try:
            img = _decode_base64_image(image_data)
        except (ValueError, base64.binascii.Error):
            continue

        if img is None:
            continue

        embedding = get_face_embedding(img)
        if embedding is not None:
            embeddings.append(embedding)
            valid_images.append(img)

    if not embeddings:
        return jsonify({"status": "error", "message": "Could not extract any valid face features"})

    save_face_data(name, embeddings, valid_images)
    process_person_background(name)

    return jsonify(
        {
            "status": "success",
            "message": f"Finished processing all captured angles for '{name}'.",
        }
    )


@app.route("/register_body", methods=["POST"])
def register_body():
    """Register upper body shape from 2 images (Front and Back)."""
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    images = data.get("images", [])  # Expecting exactly 2: Front, Back

    if not name:
        return jsonify({"status": "error", "message": "Missing person name"})
    
    # Ensure face is registered first
    if not face_database.person_exists(name):
        return jsonify({"status": "error", "message": f"Person '{name}' not found. Please register face first."})
        
    if len(images) < 2:
        return jsonify({"status": "error", "message": "Need 2 images (Front and Back)"})

    saved_count = 0
    for i, image_data in enumerate(images):
        try:
            img = _decode_base64_image(image_data)
        except (ValueError, base64.binascii.Error):
            continue

        if img is None:
            continue

        frame_h = img.shape[0]
        results = _yolo_body_model(img, verbose=False, device=_device, classes=[0])
        
        if not results or not results[0].boxes:
            label = "phía trước" if i == 0 else "phía sau"
            return jsonify({"status": "error", "message": f"Không thấy người trong ảnh {label}. Vui lòng đứng vào khung hình."})

        # Get largest person box
        best_box = None
        best_area = 0
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if area > best_area:
                best_area = area
                best_box = xyxy

                # Lấy TOÀN BỘ cơ thể thay vì cắt nửa thân trên.
        # Khi lùi xa để chụp, cắt nửa thân trên dễ làm mất điểm mốc hông -> gây lỗi.
        # Việc lấy toàn thân giúp giữ 100% các điểm xương cần thiết.
        crop = img[max(0, best_box[1]):best_box[3], max(0, best_box[0]):best_box[2]]
        
        if crop.size == 0:
            continue

        features = _body_extractor.extract(crop)

        if features is not None:
            face_database.update_body_profile(name, features)
            saved_count += 1
        else:
            label = "phía trước" if i == 0 else "phía sau"
            # Thông báo lỗi cụ thể thay vì lỗi chung chung
            return jsonify({"status": "error", "message": f"Không trích xuất được điểm mốc cơ thể ở ảnh {label}. Vui lòng đảm bảo đủ ánh sáng và lùi xa hơn chút nữa."})

    if saved_count == 0:
        return jsonify({"status": "error", "message": "Không thể trích xuất đặc trưng cơ thể."})

    return jsonify({"status": "success", "message": f"Lưu hồ sơ cơ thể cho '{name}' thành công ({saved_count}/2 mẫu)!"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)