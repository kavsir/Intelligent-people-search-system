import threading
import time
import cv2
import config

class CameraReader(threading.Thread):
    def __init__(self):
        super().__init__()
        self.url = config.CAMERA_URL
        self.cap = None
        self.frame = None
        self.ret = False
        self.lock = threading.Lock() # Khóa an toàn đa luồng
        self.running = True
        self.daemon = True           # Tự động tắt thread này khi kết thúc hàm main

    def run(self):
        print(f"[CameraReader] Đang kết nối tới ESP32-S3-CAM tại: {self.url} ...")
        self.cap = cv2.VideoCapture(self.url)

        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                        self.ret = True
                else:
                    print("[CameraReader] Cảnh báo: Mất gói tin hoặc khung hình lỗi. Đang kết nối lại...")
                    self.cap.release()
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.url)
            else:
                print("[CameraReader] LỖI: Không thể mở luồng stream. Đang thử lại sau 2 giây...")
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url)

        if self.cap:
            self.cap.release()

    def get_frame(self):
        """Hàm an toàn giúp luồng chính lấy ra khung hình mới nhất"""
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy() # Trả về bản sao để tránh xung đột vùng nhớ
            return False, None

    def stop(self):
        self.running = False