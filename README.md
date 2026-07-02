# AIoT Multi-Camera Face Recognition & Access Control System

Hệ thống AIoT giám sát nhiều phòng bằng camera ESP32-CAM cố định (không pan/tilt), tự động **nhận diện khuôn mặt**, **theo dõi target**, **nhận diện hành vi/bài tập** (squat, hít đất, đứng, di chuyển, nhảy, giơ tay, nằm), và **điều khiển khóa cửa** tự động qua ESP32 servo — kèm dashboard web thời gian thực.

---

## 1. Tổng quan kiến trúc

Hệ thống gồm **2 ứng dụng Flask độc lập** dùng chung một tầng dữ liệu (`config.py` + `face_database.py`):

| Ứng dụng | File chạy | Port | Vai trò |
|---|---|---|---|
| Đăng ký khuôn mặt | `app_registration.py` | `5000` | Web đăng ký người mới (upload ảnh hoặc chụp webcam 5 góc) |
| Dashboard vận hành | `app_dashboard.py` | `5001` | Xem live camera, sơ đồ tầng, gán bài tập, điều khiển cửa |
| (Chế độ standalone) | `app_operation.py` | – | Bản desktop (OpenCV window) tương đương dashboard, không cần trình duyệt |

```
                       ┌────────────────────┐
   ESP32-CAM (Phòng 1) │                    │      MJPEG stream
   ESP32-CAM (Phòng 2) │──── HTTP MJPEG ───▶│  CameraReader (1 thread/camera)
                       │                    │
                       └────────────────────┘
                                 │ frame
                                 ▼
                     ┌────────────────────────┐
                     │      AIPipeline         │  FSM: SEARCHING → TRACKING → LOST
                     │ (1 thread / camera)     │
                     │  YOLOv8-face  (detect)  │
                     │  InsightFace  (identify)│
                     │  YOLOv8-person(fallback)│
                     │  MediaPipe Pose (hành vi)│
                     │  Kalman Filter (smoothing)│
                     └───────┬─────────┬──────┘
                             │         │
                 behavior_manager   exercise_manager
                 (7 hành vi, mọi     (squat/pushup có
                  người đã đăng ký)   gán bài tập)
                             │         │
                             ▼         ▼
                     face_database.py (SQLite: data/face_dataset.db)
                             ▲
                             │
                  app_registration.py (đăng ký người mới)

   ESP32 Dev Module (2 servo cửa) ⇄ WebSocket (port 8765) ⇄ app_dashboard.py
```

### Vòng đời một target trong `AIPipeline` (máy trạng thái hữu hạn)

```
   SEARCHING ──(nhận diện được mặt đã đăng ký)──▶ TRACKING
       ▲                                             │
       │                                    mất mặt liên tục
       │                                  (≥ LOST_THRESHOLD frame,
       │                                   đã thử fallback theo THÂN
       │                                   người qua YOLO-person + IOU)
       │                                             ▼
       └──────────── tìm lại được ai đó ────────  LOST
```

- **SEARCHING**: chạy YOLO-face + InsightFace trên toàn khung hình, chọn người gần tâm khung hình nhất trong số các khuôn mặt đã đăng ký.
- **TRACKING**: ưu tiên giữ nguyên định danh đã khóa; nếu mặt không nhận diện được (quay đi, xa camera), thử khóa tiếp theo **thân người** (YOLO-person + so khớp IOU với bbox cũ) để không mất target ngay lập tức.
- **LOST**: sau khi mất cả mặt lẫn thân trong `LOST_THRESHOLD` frame liên tiếp, quét lại toàn khung hình để tìm bất kỳ ai đã đăng ký (ưu tiên đúng người cũ nếu xuất hiện lại).

---

## 2. Tính năng chính

- **Nhận diện & theo dõi đa camera** (mỗi phòng 1 camera cố định, độc lập hoàn toàn).
- **Sơ đồ tầng 2D** với suy luận vị trí (`InferenceEngine`): nếu một phòng có camera vừa mất target, các phòng **liền kề không có camera** sẽ được tô sáng "suy luận đang có người", tự xóa sau `INFERRED_PRESENCE_TIMEOUT_SEC` giây hoặc khi có camera xác nhận lại.
- **Nhận diện hành vi** (luôn chạy cho mọi người đã đăng ký, không cần gán gì): Đứng, Di chuyển, Nhảy, Giơ tay, Nằm — lưu lịch sử theo thời gian (`behavior_log`) mỗi `BEHAVIOR_SNAPSHOT_INTERVAL_SEC` giây.
- **Gán & chấm bài tập** (squat/hít đất): đặt số rep mục tiêu cho một người; hệ thống tự đếm rep bằng góc khớp (MediaPipe). Sai động tác hoặc mất kết nối trước khi đạt đủ số lần → **FAIL** → tự động **xóa dữ liệu khuôn mặt** người đó khỏi hệ thống (cơ chế "phạt", không cần thao tác thủ công).
- **Khóa cửa tự động (lockdown)**: bình thường nút mở/đóng cửa hoạt động tự do; ngay khi có người đã đăng ký xuất hiện ở **bất kỳ phòng nào**, toàn bộ cửa trong hệ thống tự **đóng lại** (không tự mở lại, phải bấm tay).
- **Dashboard thời gian thực** (Socket.IO): video MJPEG từng phòng, sơ đồ tầng, log sự kiện dạng timeline, bảng bài tập, trạng thái cửa — tất cả cập nhật live.
- **Trang "Theo dõi" (`/people`)**: tổng hợp theo từng người — đang ở phòng nào, cửa phòng đó ra sao, tiến độ bài tập.

---

## 3. Công nghệ sử dụng

| Thành phần | Thư viện |
|---|---|
| Phát hiện khuôn mặt / người | `ultralytics` (YOLOv8n-face, YOLOv8n) |
| Nhận diện danh tính | `insightface` (buffalo_sc, embedding 512-D, cosine similarity) |
| Ước lượng tư thế / đếm rep | `mediapipe` (Pose, model_complexity=0) |
| Xóa nền ảnh đăng ký | `rembg` |
| Web framework | `Flask`, `Flask-SocketIO` (async_mode="threading") |
| Giao tiếp cửa ESP32 | `websockets` (raw WebSocket, không qua Socket.IO) |
| Camera stream | `opencv-python` (đọc MJPEG qua FFMPEG backend) |
| Lưu trữ | `sqlite3` (file `data/face_dataset.db`) |
| Lọc mượt vị trí | Kalman Filter 2D (`cv2.KalmanFilter`) |

---

## 4. Cấu trúc thư mục

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

## 5. Cài đặt

```bash
# Python 3.10+ khuyến nghị
pip install ultralytics insightface mediapipe rembg opencv-python \
            flask flask-socketio websockets numpy --break-system-packages

# GPU (tùy chọn, khuyến nghị nếu chạy nhiều camera cùng lúc):
pip install onnxruntime-gpu torch --break-system-packages
```

Tải sẵn 2 model YOLO vào `models/`:
- `yolov8n-face.pt` — phát hiện khuôn mặt
- `yolov8n.pt` — phát hiện người (fallback theo thân, và pose)

Nếu đã có dữ liệu đăng ký cũ theo layout thư mục (`data/face_db/`, `data/processed/`), chạy 1 lần:

```bash
python migrate_to_sqlite.py
```

---

## 6. Cấu hình (`config.py`)

| Nhóm | Biến | Ý nghĩa |
|---|---|---|
| Thiết bị | `USE_GPU` | Ưu tiên CUDA nếu có, tự fallback CPU |
| Camera | `CAMERAS` | Danh sách `{id, room_name, url}` — mỗi ESP32-CAM 1 entry |
| Khung hình | `FRAME_WIDTH/HEIGHT`, `TARGET_FPS` | Kích thước & FPS mục tiêu dashboard |
| Nhận diện | `AI_CONF_THRESHOLD`, `FACE_RECOGNITION_THRESHOLD` | Ngưỡng tin cậy YOLO / cosine similarity |
| An toàn mất target | `LOST_GRACE_PERIOD_SEC` | Thời gian ân hạn trước khi coi là LOST |
| Hành vi | `BEHAVIOR_SNAPSHOT_INTERVAL_SEC` | Chu kỳ ghi running-total vào DB |
| Sơ đồ tầng | `FLOOR_PLAN`, `INFERRED_PRESENCE_TIMEOUT_SEC` | Cấu trúc phòng + thời gian tự xóa suy luận |
| Cửa | `DOOR_WS_HOST/PORT` | Địa chỉ WebSocket server cho ESP32 cửa |
| Cross-app | `REGISTRATION_APP_PORT` | Để dashboard build link sang app đăng ký |

> ⚠️ **Lưu ý:** `SEARCHING_SKIP_FRAMES`, `IDENTITY_RECHECK_INTERVAL`, `STEP_SLEEP_SEC`, `STEP_SLEEP_SKIP_IF_STEP_TOOK_LONGER_THAN_SEC` hiện được khai báo trong `config.py` nhưng **chưa được áp dụng thực tế** trong `ai_pipeline.py` (vòng lặp vẫn `time.sleep(0.01)` cố định). Xem mục 8.

---

## 7. Chạy hệ thống

```bash
# Terminal 1 — app đăng ký khuôn mặt
python app_registration.py        # http://localhost:5000

# Terminal 2 — dashboard vận hành (khởi động camera + AI + door WS)
python app_dashboard.py           # http://localhost:5001
```

Hoặc chạy bản desktop (không cần trình duyệt, có cửa sổ OpenCV, nhấn `q` để thoát an toàn):

```bash
python app_operation.py
```

Firmware ESP32 tương ứng: `esp32_servo.ino` (2 servo, kết nối vào `ws://<host>:8765` dưới dạng client, multiplex 2 cửa qua trường `door` trong JSON).

---

## 8. API chính (`app_dashboard.py`)

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/video_feed/<room_id>` | MJPEG stream đã overlay bbox/FPS/latency |
| GET | `/api/room_status` | Trạng thái mọi phòng trong sơ đồ tầng (có/suy luận/trống) |
| GET | `/api/config` | Config phía client cần (floor plan, fps, port đăng ký) |
| POST | `/api/reset_inference` | Xóa mọi highlight "suy luận" thủ công |
| GET | `/api/events?n=` | N sự kiện gần nhất |
| GET | `/api/registered_people` | Danh sách tên đã đăng ký |
| GET / POST | `/api/exercises`, `/api/exercises/assign`, `/api/exercises/<name>` (DELETE) | Quản lý gán bài tập |
| GET | `/api/behaviors`, `/api/behaviors/history?name=&limit=` | Bảng hành vi live / lịch sử 1 người |
| GET | `/api/people_overview` | Gộp thông tin người + phòng + cửa + bài tập |
| Socket.IO | `toggle_door {room_id}` → `door_response`, `door_status` | Điều khiển & đẩy trạng thái cửa realtime |

`app_registration.py`: `GET /`, `POST /get_landmarks`, `POST /register_from_image`, `POST /register_final` (5 góc webcam).

---

## 9. Ghi chú kỹ thuật / hạn chế đã biết

- **Tối ưu CPU chưa hoàn thiện**: các cờ giảm tải CPU trong `config.py` (bỏ qua frame khi SEARCHING, sleep có điều kiện) đã được khai báo nhưng chưa nối vào `ai_pipeline.py`. Chạy 2+ camera đồng thời trên CPU (YOLO-face + YOLO-person + InsightFace + MediaPipe mỗi thread) sẽ tốn tài nguyên hơn mức cần thiết.
- **`LOST_THRESHOLD` (10 frame) đang hardcode** trong `AIPipeline`, không được tính từ `config.LOST_GRACE_PERIOD_SEC` — đổi config này hiện không có tác dụng.
- **`camera_reader.py`**: set `CAP_PROP_OPEN_TIMEOUT_MSEC`/`READ_TIMEOUT_MSEC` sau khi `cv2.VideoCapture(url)` đã mở — có thể không áp dụng cho lần mở đầu tiên tùy backend; cân nhắc `cap = cv2.VideoCapture(); cap.set(...); cap.open(url)`.
- **`app_registration.py`** chạy `debug=True, host="0.0.0.0"` — nên tắt debug mode hoặc giới hạn host khi triển khai ngoài môi trường dev cục bộ (Werkzeug debugger có thể bị lợi dụng để thực thi mã từ xa).
- File `operation/face-recognizer.py` (dấu gạch ngang) là bản trùng lặp của `face_recognizer.py`, không thể `import` được theo tên module hợp lệ — nên xóa để tránh nhầm lẫn khi bảo trì.
- `app_operation.py` có `EventLogger()` + `behavior_manager.start()` chạy ở top-level module — khi bị `import` bởi `app_dashboard.py` sẽ tạo thêm 1 file log CSV thừa mỗi lần khởi động dashboard; nên bọc trong `if __name__ == "__main__":`.

---

## 10. Lộ trình có thể mở rộng

- Hoàn thiện các cờ tối ưu CPU trong `config.py` để chạy mượt hơn với nhiều camera.
- Thêm xác thực (login) cho dashboard/API trước khi triển khai ngoài mạng nội bộ.
- Gộp `app_registration.py` và `app_dashboard.py` thành một Flask app duy nhất (đã được ghi chú là dự định trong code).
- Thêm cơ chế reconnect/backoff có giới hạn cho `CameraReader` thay vì retry vô hạn.
