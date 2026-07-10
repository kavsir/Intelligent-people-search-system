"""
Background thread that continuously reads frames from an ESP32-CAM MJPEG
stream and exposes the latest frame in a thread-safe way.

In the multi-camera setup, each ESP32-CAM streams directly to this machine
(no longer relayed through the ESP32-S3). The ESP32-S3 is now only a
coordination gateway (HTTP/MQTT) that tells each ESP32-CAM when to
stream/snapshot/sleep -- it does not carry video. Create one CameraReader
instance per physical camera, passing that camera's own url/name.
"""

import os
import sys
import threading
import time

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Cấu hình FFmpeg để xử lý MJPEG stream từ ESP32-CAM ổn định hơn.
# ESP32-CAM gửi MJPEG qua HTTP không có container chuẩn, gói tin dễ bị lỗi
# khi qua Wi-Fi. Các flags này giúp FFmpeg:
#   - nobuffer:       không tích luỹ buffer lớn, đọc frame mới nhất ngay
#   - low_delay:      ưu tiên độ trễ thấp (quan trọng cho real-time)
#   - max_delay:      giới hạn thời gian chờ tối đa 500ms cho 1 packet
#   - analyzeduration: bỏ qua phân tích stream kéo dài
#   - probesize:      giới hạn kích thước probe để mở stream nhanh hơn
#   - err_detect:     bỏ qua các lỗi bitstream nhẹ thay vì crash
# ---------------------------------------------------------------------------
_FFMPEG_OPTIONS = (
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;500000|"
    "analyzeduration;0|"
    "probesize;32|"
    "err_detect;ignore_err"
)

# Đặt biến môi trường MỘT LẦN khi module được import.
# OPENCV_FFMPEG_CAPTURE_OPTIONS được OpenCV đọc khi mở VideoCapture với
# backend FFmpeg (cv2.CAP_FFMPEG). Định dạng: "key1;val1|key2;val2|..."
_ALREADY_SET = "_OPENCV_FFMPEG_OPTIONS_SET"
if not os.environ.get(_ALREADY_SET):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _FFMPEG_OPTIONS
    os.environ[_ALREADY_SET] = "1"


class CameraReader(threading.Thread):
    def __init__(self, url=None, name="camera"):
        """
        url:  MJPEG stream URL for this specific ESP32-CAM. Defaults to
              config.CAMERA_URL for backward compatibility with single-camera
              setups, but multi-camera setups should always pass this
              explicitly (e.g. from config.CAMERAS[i]["url"]).
        name: Human-readable label used only in log/print messages, so it's
              obvious which physical camera a given log line refers to when
              several CameraReader threads are running at once.
        """
        super().__init__()
        self.url = url or config.CAMERA_URL
        self.cam_name = name
        self.cap = None
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = True
        self.daemon = True
        self._reconnect_count = 0
        self._max_reconnect_delay = 5  # giây – max backoff

    def _calc_backoff(self):
        """Exponential backoff: 1s, 2s, 4s, capped at _max_reconnect_delay."""
        delay = min(2 ** self._reconnect_count, self._max_reconnect_delay)
        return delay

    def _open_capture(self):
        """
        Mở MJPEG stream từ ESP32-CAM với các thiết lập chịu lỗi.

        - Dùng backend FFmpeg tường minh (cv2.CAP_FFMPEG) thay vì auto-detect,
          để chắc chắn các flags trong OPENCV_FFMPEG_CAPTURE_OPTIONS có hiệu lực.
        - set buffersize=1 để luôn lấy frame mới nhất, giảm độ trễ.
        - set timeout ngắn để fail-fast khi camera mất kết nối.
        """
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        print(f"[CameraReader:{self.cam_name}] Connecting to ESP32-CAM at: {self.url} ...")
        self.cap = self._open_capture()

        while self.running:
            if self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                except cv2.error as e:
                    # Bắt lỗi C++ assertion từ FFmpeg backend (vd:
                    # "pkt->stream_index < (unsigned)s->nb_streams")
                    # khi ESP32-CAM gửi packet MJPEG bị hỏng/cụt qua Wi-Fi.
                    # Thay vì crash process, ta reconnect stream.
                    print(
                        f"[CameraReader:{self.cam_name}] OpenCV error reading frame: {e}"
                    )
                    ret = False
                    # Đảm bảo cap được release để tránh rò rỉ resource
                    try:
                        self.cap.release()
                    except Exception:
                        pass

                if ret:
                    with self.lock:
                        self.frame = frame
                        self.ret = True
                    # Reset backoff khi đọc frame thành công
                    self._reconnect_count = 0
                else:
                    with self.lock:
                        self.ret = False
                    delay = self._calc_backoff()
                    self._reconnect_count += 1
                    print(
                        f"[CameraReader:{self.cam_name}] Warning: dropped frame. "
                        f"Reconnecting in {delay}s (attempt #{self._reconnect_count})..."
                    )
                    time.sleep(delay)
                    self.cap = self._open_capture()
            else:
                delay = self._calc_backoff()
                self._reconnect_count += 1
                print(
                    f"[CameraReader:{self.cam_name}] ERROR: could not open stream. "
                    f"Retrying in {delay}s (attempt #{self._reconnect_count})..."
                )
                with self.lock:
                    self.ret = False
                time.sleep(delay)
                self.cap = self._open_capture()

        if self.cap:
            self.cap.release()

    def get_frame(self):
        """Thread-safe accessor for the most recent frame."""
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.running = False