# AIoT Multi-Camera Dangerous Person Recognition & Tracking System

## Hệ Thống Camera An Ninh AIoT Đa Camera Ứng Dụng Nhận Diện Và Truy Vết Đối Tượng Nguy Hiểm

Hệ thống AIoT sử dụng nhiều camera ESP32-CAM cố định để thu nhận hình ảnh theo thời gian thực, kết hợp các mô hình Trí tuệ nhân tạo hiện đại nhằm phát hiện, nhận diện, theo dõi và truy vết đối tượng nguy hiểm trong khu vực giám sát.

Hệ thống được xây dựng dựa trên các công nghệ Computer Vision, Deep Learning và IoT, cho phép nhận diện khuôn mặt, theo dõi đối tượng trên nhiều camera, phân tích hành vi cơ bản, ghi nhận lịch sử di chuyển, phát cảnh báo thời gian thực và điều khiển thiết bị IoT như khóa cửa hoặc còi báo động khi phát hiện đối tượng thuộc danh sách theo dõi.

---

## 1. Tổng quan kiến trúc

Hệ thống được chia thành các thành phần chính:

| Thành phần | Vai trò |
|------------|----------|
| ESP32-CAM | Thu nhận hình ảnh |
| CameraReader | Đọc luồng MJPEG |
| AI Pipeline | Nhận diện, theo dõi, phân tích hành vi |
| SQLite Database | Lưu trữ dữ liệu |
| Dashboard Web | Giao diện giám sát |
| IoT Controller | Điều khiển khóa cửa/cảnh báo |

```text
ESP32-CAM
      │
      ▼
CameraReader
      │
      ▼
YOLOv8
(Phát hiện người & khuôn mặt)
      │
      ▼
InsightFace
(Nhận diện danh tính)
      │
      ▼
Tracking FSM
(SEARCHING → TRACKING → LOST)
      │
      ▼
Kalman Filter
(Làm mượt vị trí)
      │
      ▼
MediaPipe Pose
(Phân tích hành vi)
      │
      ▼
Event Logger
(Ghi nhật ký)
      │
      ▼
SQLite Database
      │
      ▼
Dashboard + Alert System
      │
      ▼
IoT Device Control
(ESP32 Door Lock / Alarm)
```

---

## 2. Tính năng chính

### Nhận diện khuôn mặt

- Đăng ký khuôn mặt người dùng.
- Sinh embedding 512 chiều bằng InsightFace.
- So khớp bằng Cosine Similarity.
- Nhận diện thời gian thực.

### Theo dõi đối tượng

- Theo dõi liên tục trên từng camera.
- FSM Tracking gồm:
  - SEARCHING
  - TRACKING
  - LOST
- Dự phòng theo thân người khi mất khuôn mặt.

### Truy vết đa camera

- Lưu lịch sử xuất hiện.
- Ghi nhận thời gian.
- Ghi nhận camera phát hiện.
- Hỗ trợ tái hiện lộ trình di chuyển.

### Phân tích hành vi

MediaPipe Pose nhận dạng:

- Đứng
- Đi bộ
- Chạy
- Nằm
- Giơ tay
- Hành vi bất thường

### Cảnh báo an ninh

Khi phát hiện:

- Đối tượng nguy hiểm
- Đối tượng trong danh sách theo dõi
- Hành vi bất thường

Hệ thống sẽ:

- Hiển thị popup cảnh báo.
- Gửi thông báo thời gian thực.
- Ghi sự kiện vào cơ sở dữ liệu.
- Kích hoạt thiết bị IoT.

### Điều khiển IoT

Thiết bị ESP32 có thể:

- Đóng cửa tự động.
- Mở cửa từ Dashboard.
- Kích hoạt còi báo động.
- Gửi phản hồi trạng thái về Server.

---

## 3. Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Object Detection | YOLOv8 |
| Face Recognition | InsightFace |
| Tracking | FSM + Kalman Filter |
| Pose Estimation | MediaPipe Pose |
| Backend | Flask |
| Realtime | Flask-SocketIO |
| Database | SQLite |
| Camera Stream | OpenCV |
| IoT Communication | WebSocket |
| Hardware | ESP32-CAM, ESP32 |

---

## 4. Cấu trúc thư mục

```text
project/
│
├── models/
│   ├── yolov8n.pt
│   └── yolov8n-face.pt
│
├── data/
│   └── face_dataset.db
│
├── logs/
│
├── src/
│   │
│   ├── operation/
│   │   ├── ai_pipeline.py
│   │   ├── camera_reader.py
│   │   ├── face_recognizer.py
│   │   ├── tracker.py
│   │   ├── kalman.py
│   │   ├── pose_estimator.py
│   │   ├── event_logger.py
│   │   ├── alert_manager.py
│   │   └── door_ws_server.py
│   │
│   ├── registration/
│   │   ├── face_registrar.py
│   │   ├── image_preprocessor.py
│   │   └── migrate_to_sqlite.py
│   │
│   ├── templates/
│   │   ├── dashboard.html
│   │   └── register.html
│   │
│   └── static/
│
├── app_registration.py
├── app_dashboard.py
├── config.py
├── face_database.py
├── README.md
└── requirements.txt
```

---

## 5. Quy trình hoạt động

### Bước 1

ESP32-CAM gửi luồng MJPEG.

### Bước 2

CameraReader đọc khung hình mới nhất.

### Bước 3

YOLOv8 phát hiện người và khuôn mặt.

### Bước 4

InsightFace nhận diện danh tính.

### Bước 5

FSM Tracking duy trì theo dõi đối tượng.

### Bước 6

Kalman Filter làm mượt vị trí.

### Bước 7

MediaPipe Pose phân tích hành vi.

### Bước 8

Event Logger ghi nhận sự kiện.

### Bước 9

Lưu dữ liệu vào SQLite.

### Bước 10

Dashboard hiển thị kết quả và phát cảnh báo.

---

## 6. Chạy hệ thống

### Cài đặt

```bash
pip install -r requirements.txt
```

### Khởi động đăng ký khuôn mặt

```bash
python app_registration.py
```

### Khởi động Dashboard

```bash
python app_dashboard.py
```

---

## 7. Kết quả đầu ra

Hệ thống cung cấp:

- Video trực tiếp từ camera.
- Thông tin nhận diện.
- Bounding Box Tracking.
- Lịch sử truy vết.
- Nhật ký sự kiện.
- Danh sách cảnh báo.
- Điều khiển thiết bị IoT.
- Báo cáo thống kê.

---

## 8. Hướng phát triển

- Multi-Camera ReID.
- Theo dõi xuyên camera.
- Cảnh báo Email/Zalo/Telegram.
- Hỗ trợ GPU.
- Triển khai Cloud.
- Phân tích hành vi bất thường nâng cao.
- Tích hợp thêm cảm biến IoT.
- Nhận diện biển số xe.
- Face Anti-Spoofing.
- Edge AI Deployment.
