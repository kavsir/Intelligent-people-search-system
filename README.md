# 🎯 AIoT Face Tracking & Behavior Recognition System

Hệ thống AIoT nhận diện theo khuôn mặt người đã đăng ký, kết hợp nhận diện hành vi realtime bằng ESP32-CAM và ESP32 làm gateway 

---

## 📋 Tính năng chính

- **Face Registration** — Đăng ký khuôn mặt người
- **Face Recognition** — Nhận diện người đã đăng ký bằng InsightFace buffalo_sc
- **Face Tracking** — Bám theo khuôn mặt mục tiêu 
- **Behavior Recognition** — Nhận diện hành vi: đứng, di chuyển, nhảy, dơ 2 tay, nằm
- **servo Control** — Điều khiển servo SG90 để đóng mở cửa tự động và thủ công qua giao thức websoket
- **Dashboard Realtime** — Hiển thị video, trạng thái, log sự kiện

---

## 🔧 Phần cứng yêu cầu

| Thiết bị | Vai trò |
|---|---|
| ESP32-CAM | Camera, stream MJPEG qua WiFi |
| ESP32  | Nhận lệnh Serial, điều khiển servo |
| SG90 x2 | Servo đóng mở cửa |
| Nguồn 5V/2A (riêng) | Cấp điện cho 2 servo sg90 |
| PC Windows | Xử lý AI, dashboard |

---

## 🤖 Mô hình AI sử dụng

| Mô hình | Vai trò |
|---|---|
| YOLOv8n-face | Detect khuôn mặt trong frame |
| InsightFace buffalo_sc | Nhận diện danh tính (embedding 512D) |
| CSRT Tracker | Bám mục tiêu khi mất mặt (quay ngang/lưng) |
| YOLOv8n (person) | Fallback bám toàn thân khi tracker thất bại |
| MediaPipe Pose | Detect skeleton 33 keypoints, nhận diện hành vi |
| Kalman Filter | Làm mượt tọa độ (u, v) |
| PID Controller | Điều khiển servo mượt, không rung |

---

## 📁 Cấu trúc dự án

```text
INTELLIGENT-PEOPLE-SEARCH-SYSTEM/
│
├── camera_esp32CAM1/                  # Code ESP32-CAM số 1
│   └── camera_esp32CAM1.ino
│
├── camera_esp32CAM2/                  # Code ESP32-CAM số 2
│   └── camera_esp32CAM2.ino
│
├── esp32_servo/                       # ESP32-S3 điều khiển Servo
│   └── esp32_servo.ino
│
├── data/
│   └── face_dataset.db                # CSDL khuôn mặt đã đăng ký
│
├── logs/                              # Nhật ký hoạt động hệ thống
│
├── models/
│   ├── yolov8n.pt                     # YOLO Person Detection
│   └── yolov8n-face.pt                # YOLO Face Detection
│
├── src/
│   │
│   ├── operation/                     # Hệ thống vận hành chính
│   │   │
│   │   ├── ai_pipeline.py             # Pipeline AI tổng thể
│   │   ├── camera_reader.py           # Đọc luồng MJPEG từ ESP32-CAM
│   │   ├── face_recognizer.py         # Nhận diện khuôn mặt InsightFace
│   │   ├── pose_estimator.py          # MediaPipe Pose Estimation
│   │   ├── servo_controller.py        # PID + điều khiển Servo
│   │   ├── event_logger.py            # Ghi log sự kiện
│   │   ├── exercise_manager.py        # Phân tích hành vi/tư thế
│   │   └── door_ws_server.py          # WebSocket giao tiếp thời gian thực
│   │
│   ├── registration/                  # Module đăng ký người dùng
│   │   │
│   │   ├── face_registrar.py          # Thu thập ảnh khuôn mặt
│   │   ├── face_recognizer.py         # Sinh embedding khuôn mặt
│   │   ├── image_preprocessor.py      # Tiền xử lý ảnh
│   │   └── migrate_to_sqlite.py       # Lưu dữ liệu đăng ký
│   │
│   ├── static/                        # CSS/JS giao diện Dashboard
│   │   ├── dashboard.css
│   │   ├── dashboard.js
│   │   ├── people.css
│   │   └── people.js
│   │
│   └── templates/                     # HTML Dashboard
│       ├── dashboard.html
│       ├── people.html
│       └── register.html
│
├── app_operation.py                   # Chạy hệ thống nhận diện chính
├── app_registration.py                # Chạy giao diện đăng ký khuôn mặt
├── app_dashboard.py                   # Dashboard giám sát
│
├── config.py                          # Cấu hình hệ thống
├── face_database.py                   # Quản lý CSDL khuôn mặt
├── download_model.py                  # Tải mô hình AI
│
├── README.md
└── requirements.txt
```


---

## ⚙️ Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.9+
- Windows 10/11


### 2. Cấu hình ESP32 và ESP32-CAM

Upload firmware stream MJPEG lên ESP32 và ESP32-CAM, sau đó cập nhật địa chỉ IP trong `config.py`:

### 3. Kết nối servo

SG90x2


---



## 🕺 Nhận diện hành vi

| Hành vi | Điều kiện phát hiện |
|---|---|
| Đứng yên | Không thỏa điều kiện nào khác |
| Di chuyển | Tọa độ hông thay đổi liên tục giữa các frame |
| Nhảy | Mắt cá chân (keypoint 27, 28) cao hơn ngưỡng bình thường |
| Dơ 2 tay | Cổ tay (keypoint 15, 16) cao hơn vai (keypoint 11, 12) |
| Nằm | Vai và hông gần cùng độ cao (y xấp xỉ bằng nhau) |

---

## 📊 Chỉ số hiệu năng mục tiêu

| Chỉ số | Mục tiêu |
|---|---|
| FPS | 15–20 FPS |
| Latency | < 120ms |
| Tracking error | < 30px |
| Thời gian đăng ký | 20–30 giây/người |

---

## 🔧 Tham số có thể điều chỉnh

---

## ❗ Lỗi thường gặp


---

## 📦 Thư viện sử dụng

```
ultralytics>=8.0        # YOLOv8
insightface>=0.7        # Face recognition
onnxruntime>=1.16       # InsightFace backend
opencv-python>=4.8      # Camera, tracker, image processing
mediapipe>=0.10         # Pose estimation
streamlit>=1.28         # Dashboard
pyserial>=3.5           # Serial communication
numpy>=1.24             # Array operations
scipy>=1.11             # Signal processing
```

---

## 👥 Thành viên nhóm

| Họ tên | MSSV | Vai trò |
|---|---|---|
| | | |
| | | |

**Học phần:** Triển khai ứng dụng AI và IoT

---

## 📄 License

MIT License — Dự án học thuật, chỉ dùng cho mục đích giáo dục.
