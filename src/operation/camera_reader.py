"""
Background thread that continuously reads frames from the ESP32-S3-CAM
MJPEG stream and exposes the latest frame in a thread-safe way.
"""

import os
import sys
import threading
import time

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CameraReader(threading.Thread):
    def __init__(self):
        super().__init__()
        self.url = config.CAMERA_URL
        self.cap = None
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = True
        self.daemon = True

    def run(self):
        print(f"[CameraReader] Connecting to ESP32-S3-CAM at: {self.url} ...")
        self.cap = cv2.VideoCapture(self.url)

        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                        self.ret = True
                else:
                    print("[CameraReader] Warning: dropped frame. Reconnecting...")
                    self.cap.release()
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.url)
            else:
                print("[CameraReader] ERROR: could not open stream. Retrying in 2s...")
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url)

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
