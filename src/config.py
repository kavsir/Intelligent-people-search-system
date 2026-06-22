
import os

# Đường dẫn luồng stream MJPEG từ ESP32-S3-CAM
# Bạn có thể đổi thành "http://192.168.0.10/stream" hoặc "http://192.168.0.10:81/stream" nếu cần
CAMERA_URL = "http://10.145.30.117/stream"

# Cấu hình kích thước khung hình (Nếu muốn giảm tải cho AI ở bước sau)
# Thông thường ESP32-S3-CAM để độ phân giải QVGA (320x240) hoặc VGA (640x480)
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
TARGET_FPS = 30

YOLO_MODEL_PATH = "yolov8n-face.pt"  # Thư viện Ultralytics sẽ tự tải về nếu chưa có sẵn
AI_CONF_THRESHOLD = 0.5              # Ngưỡng tin cậy (confidence) để chấp nhận mặt