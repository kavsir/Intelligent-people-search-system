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

    def _open_capture(self):
        """Open the stream with short open/read timeouts so a missing
        camera fails fast (a few seconds) instead of hanging on the
        default OS-level TCP timeout (which can be 20-60+ seconds on
        Windows for an unreachable host)."""
        cap = cv2.VideoCapture(self.url)
        # These properties are honored by the FFMPEG backend (used for
        # http:// URLs); harmless no-ops on backends that ignore them.
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
        return cap

    def run(self):
        print(f"[CameraReader:{self.cam_name}] Connecting to ESP32-CAM at: {self.url} ...")
        self.cap = self._open_capture()

        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                        self.ret = True
                else:
                    print(f"[CameraReader:{self.cam_name}] Warning: dropped frame. Reconnecting...")
                    with self.lock:
                        self.ret = False
                    self.cap.release()
                    time.sleep(1)
                    self.cap = self._open_capture()
            else:
                print(f"[CameraReader:{self.cam_name}] ERROR: could not open stream. Retrying in 2s...")
                with self.lock:
                    self.ret = False
                time.sleep(2)
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