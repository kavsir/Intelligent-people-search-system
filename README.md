# 🎯 AIoT Face Tracking & Behavior Recognition System

Hệ thống AIoT nhận diện và bám theo khuôn mặt người đã đăng ký, kết hợp nhận diện hành vi realtime, điều khiển cơ cấu pan-tilt servo MG996R qua ESP32-S3-CAM.

---

## 📋 Tính năng chính

- **Face Registration** — Đăng ký khuôn mặt qua video trực tiếp với pose-triggered capture (~15 góc tự động)
- **Face Recognition** — Nhận diện người đã đăng ký bằng InsightFace buffalo_sc
- **Face Tracking** — Bám theo khuôn mặt mục tiêu với CSRT Tracker khi mất mặt
- **Behavior Recognition** — Nhận diện hành vi: đứng, di chuyển, nhảy, dơ 2 tay, nằm
- **Patrol Mode** — Tự động quét trái/phải khi không tìm thấy mục tiêu
- **Pan-Tilt Control** — Điều khiển 2 servo MG996R mượt bằng Kalman Filter + PID
- **Dashboard Realtime** — Hiển thị video, skeleton, trạng thái, log sự kiện

---

## 🔧 Phần cứng yêu cầu

| Thiết bị | Vai trò |
|---|---|
| ESP32-S3-CAM | Camera, stream MJPEG qua WiFi |
| ESP32 / Arduino | Nhận lệnh Serial, điều khiển servo PWM |
| MG996R x2 | Servo Pan (trái/phải) + Tilt (lên/xuống) |
| Nguồn 5V/2A (riêng) | Cấp điện cho 2 servo MG996R |
| PC Windows | Xử lý AI, dashboard |

> ⚠️ **Quan trọng:** Servo MG996R cần nguồn riêng 5V/2A, **không** lấy điện từ ESP32 vì servo kéo dòng mạnh gây reset board.

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


## ⚙️ Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.9+
- Windows 10/11
- Webcam hoặc ESP32-S3-CAM đã cấu hình stream MJPEG

### 2. Cài thư viện

```bash
pip install ultralytics       # YOLOv8n-face + person detection
pip install insightface       # Face recognition
pip install onnxruntime       # Backend cho InsightFace
pip install opencv-python     # Camera + CSRT tracker
pip install mediapipe         # Pose estimation + behavior
pip install streamlit         # Dashboard UI
pip install pyserial          # Giao tiếp servo qua Serial
pip install numpy scipy       # Kalman filter + embedding
```

### 3. Cấu hình ESP32-S3-CAM

Upload firmware stream MJPEG lên ESP32-S3-CAM, sau đó cập nhật địa chỉ IP trong `camera.py`:

```python
STREAM_URL = "http://192.168.x.x:81/stream"  # Thay bằng IP thực của ESP32
```

### 4. Kết nối servo

```
ESP32/Arduino  →  MG996R Pan
Pin D9         →  Signal (dây vàng/cam)
GND            →  GND (dây nâu/đen)
5V nguồn riêng →  VCC (dây đỏ)

ESP32/Arduino  →  MG996R Tilt
Pin D10        →  Signal
GND            →  GND
5V nguồn riêng →  VCC
```

---

## 🚀 Hướng dẫn chạy

### Bước 1: Đăng ký khuôn mặt

```bash
python register_face.py --name "nguyen_van_a"
```

- Nhìn vào camera, xoay mặt từ từ theo hướng dẫn trên màn hình
- Hệ thống tự động chụp khi phát hiện góc mặt thay đổi > 15°
- Thu đủ ~15 ảnh trong khoảng 20–30 giây → tự động lưu vào `face_db/`
- Lặp lại cho từng người cần đăng ký

### Bước 2: Kết nối phần cứng

```bash
# Kiểm tra cổng Serial của ESP32/Arduino
python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"
```

Cập nhật cổng trong `controller.py`:
```python
SERIAL_PORT = "COM3"  # Thay bằng cổng thực
```

### Bước 3: Chạy hệ thống

```bash
python main.py
```

### Bước 4: Mở Dashboard

```bash
streamlit run dashboard.py
```

Truy cập `http://localhost:8501` trên trình duyệt.

---

## 🔄 Trạng thái hệ thống

| Trạng thái | Màu | Mô tả |
|---|---|---|
| FACE_TRACK | 🟢 Xanh lá | Thấy mặt rõ, đang nhận diện và bám |
| BODY_TRACK | 🟡 Vàng | Mất mặt (quay ngang/mờ), CSRT tiếp quản |
| PERSON_TRACK | 🟠 Cam | Quay lưng, bám theo toàn thân |
| LOST | 🔴 Đỏ | Mất mục tiêu, servo giữ nguyên, chờ 2s |
| PATROL | 🔍 Trắng | Không thấy ai, quét trái/phải tìm kiếm |
| IGNORE | ⚪ Xám | Phát hiện người lạ, bỏ qua |

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

## 🔍 Patrol Mode

Khi không tìm thấy người đăng ký trong **5 giây**, hệ thống tự động chuyển sang Patrol Mode:

```
Pan quét: -60° → -30° → 0° → +30° → +60° → lặp lại
Dừng mỗi vị trí: 0.5 giây (để AI detect kịp)
Tilt: giữ nguyên hoặc quét nhẹ ±15°
Tốc độ: 5°/step (chậm hơn bám target)
Thoát: phát hiện người đăng ký → FACE_TRACK ngay
```

---

## 📊 Chỉ số hiệu năng mục tiêu

| Chỉ số | Mục tiêu |
|---|---|
| FPS | 15–20 FPS |
| Latency | < 120ms |
| Tracking error | < 30px |
| Servo jitter | Không rung (deadzone 15px, max 3°/frame) |
| Thời gian đăng ký | 20–30 giây/người |
| Số góc đăng ký | ~15 góc tự động |
| Face recognition threshold | Cosine similarity > 0.5 |

---

## 🔧 Tham số có thể điều chỉnh

| Tham số | File | Mặc định | Mô tả |
|---|---|---|---|
| `STREAM_URL` | camera.py | — | IP ESP32-S3-CAM |
| `SERIAL_PORT` | controller.py | COM3 | Cổng Serial servo |
| `DETECT_EVERY_N` | detector.py | 3 | Chạy InsightFace mỗi N frame |
| `FACE_THRESHOLD` | detector.py | 0.5 | Ngưỡng nhận diện khuôn mặt |
| `DEADZONE_PX` | controller.py | 15 | Ngưỡng pixel không điều khiển servo |
| `MAX_DEGREE_STEP` | controller.py | 3 | Góc thay đổi tối đa mỗi frame |
| `KP` | controller.py | 0.05 | Hệ số PID proportional |
| `LOST_TIMEOUT` | tracker.py | 2.0 | Giây chờ trước khi chuyển LOST |
| `PATROL_TIMEOUT` | tracker.py | 5.0 | Giây chờ trước khi Patrol |
| `PATROL_STEP` | controller.py | 5 | Độ quét mỗi bước Patrol |

---

## ❗ Lỗi thường gặp

**Camera không kết nối được**
- Kiểm tra ESP32-S3-CAM và PC cùng mạng WiFi
- Thử truy cập `STREAM_URL` trực tiếp trên trình duyệt
- Dùng webcam USB thay thế để test: đổi `STREAM_URL = 0` trong `camera.py`

**InsightFace không nhận diện được**
- Đảm bảo đã đăng ký đủ góc mặt (chạy lại `register_face.py`)
- Kiểm tra ánh sáng đủ sáng khi đăng ký và vận hành
- Thử tăng threshold: `FACE_THRESHOLD = 0.4`

**Servo rung nhiều**
- Giảm `KP` (ví dụ: 0.03)
- Tăng `DEADZONE_PX` (ví dụ: 20)
- Giảm `MAX_DEGREE_STEP` (ví dụ: 2)

**FPS thấp hơn mục tiêu**
- Giảm frame size: đổi sang 320x240 trong ESP32 firmware
- Tăng `DETECT_EVERY_N` (ví dụ: 5)
- Đảm bảo không có ứng dụng nặng chạy nền

**Servo không phản hồi**
- Kiểm tra lại `SERIAL_PORT` (chạy lệnh kiểm tra cổng ở Bước 2)
- Đảm bảo nguồn 5V/2A riêng đã kết nối cho servo
- Kiểm tra dây tín hiệu từ ESP32/Arduino đến servo

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
