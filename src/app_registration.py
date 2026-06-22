"""
Flask app for face registration.

Routes:
    GET  /                  -> registration page (upload tab + webcam tab)
    POST /get_landmarks      -> preview-only landmark detection for the UI
    POST /register_from_image -> register a single uploaded image
    POST /register_final     -> register a person from 5 webcam angles
"""

import base64
import os
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from registration.face_registrar import (
    get_face_embedding,
    get_face_landmarks,
    save_face_data,
)
from registration.image_preprocessor import process_person_background

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "registration", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "registration", "static"),
)


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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
