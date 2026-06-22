import os
import io
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from ultralytics import YOLO

# ==============================
# Load YOLO
# ==============================
try:
    yolo_model = YOLO("yolov8n-face.pt")
except:
    yolo_model = YOLO("yolov8n.pt")


# ==============================
# Tăng chất lượng ảnh
# ==============================
def enhance_image_quality(img):

    # 1. Khử nhiễu
    denoised = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=3,
        hColor=3,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 2. Sharpen
    blur = cv2.GaussianBlur(
        denoised,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(
        denoised,
        1.5,
        blur,
        -0.5,
        0
    )

    return sharpened


# ==============================
# Tách nền bằng rembg
# ==============================
def remove_background_rembg(img):

    success, buffer = cv2.imencode(".png", img)

    if not success:
        return img

    output_data = remove(buffer.tobytes())

    img_rgba = Image.open(
        io.BytesIO(output_data)
    ).convert("RGBA")

    background = Image.new(
        "RGB",
        img_rgba.size,
        (0, 0, 0)
    )

    background.paste(
        img_rgba,
        mask=img_rgba.split()[3]
    )

    result = cv2.cvtColor(
        np.array(background),
        cv2.COLOR_RGB2BGR
    )

    return result


# ==============================
# Chọn người gần camera nhất
# ==============================
def process_background_pipeline(img):

    h, w = img.shape[:2]

    results = yolo_model(
        img,
        verbose=False
    )

    selected_region = img

    for result in results:

        if result.boxes is None:
            continue

        if len(result.boxes) == 0:
            continue

        best_idx = 0
        best_score = -1

        for i, box in enumerate(result.boxes):

            xyxy = box.xyxy[0].cpu().numpy()

            conf = (
                box.conf[0].cpu().item()
                if box.conf is not None
                else 0.5
            )

            area = (
                (xyxy[2] - xyxy[0]) *
                (xyxy[3] - xyxy[1])
            )

            score = area * conf

            if score > best_score:
                best_score = score
                best_idx = i

        box = result.boxes.xyxy[
            best_idx
        ].cpu().numpy()

        x1, y1, x2, y2 = map(
            int,
            box[:4]
        )

        padding_x = int(
            (x2 - x1) * 0.5
        )

        padding_y = int(
            (y2 - y1) * 0.8
        )

        nx1 = max(
            0,
            x1 - padding_x
        )

        ny1 = max(
            0,
            y1 - padding_y
        )

        nx2 = min(
            w,
            x2 + padding_x
        )

        ny2 = min(
            h,
            y2 + padding_y
        )

        selected_region = img[
            ny1:ny2,
            nx1:nx2
        ]

        break

    result = remove_background_rembg(
        selected_region
    )

    return result


# ==============================
# Xử lý toàn bộ ảnh
# ==============================
def process_person_background(name):

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    source_dir = os.path.join(
        BASE_DIR,
        "face_db",
        name
    )

    target_dir = os.path.join(
        BASE_DIR,
        "tien_su_ly",
        name
    )

    if not os.path.exists(source_dir):
        return False

    os.makedirs(
        target_dir,
        exist_ok=True
    )

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png"
    )

    image_files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith(valid_extensions)
    ]

    for file_name in image_files:

        img_path = os.path.join(
            source_dir,
            file_name
        )

        img = cv2.imread(img_path)

        if img is None:
            continue

        segmented_img = process_background_pipeline(
            img
        )

        final_img = enhance_image_quality(
            segmented_img
        )

        output_path = os.path.join(
            target_dir,
            file_name
        )

        cv2.imwrite(
            output_path,
            final_img
        )

    print(
        f"[PREPROCESS] Hoàn tất xử lý: {name}"
    )

    return True