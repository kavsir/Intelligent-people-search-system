"""
Background thread that continuously reads frames from an ESP32-CAM MJPEG
stream and exposes the latest frame in a thread-safe way.

In the multi-camera setup, each ESP32-CAM streams directly to this machine
(no longer relayed through the ESP32-S3). The ESP32-S3 is now only a
coordination gateway (HTTP/MQTT) that tells each ESP32-CAM when to
stream/snapshot/sleep -- it does not carry video. Create one CameraReader
instance per physical camera, passing that camera's own url/name.

--------------------------------------------------------------------------
Why a watchdog thread (fixes "signal comes back but the app never notices")
--------------------------------------------------------------------------
The naive reconnect loop below (release -> sleep -> reopen on a failed
read) only runs if cap.read() actually RETURNS. In practice, once a
chunked MJPEG-over-HTTP stream dies mid-connection (camera rebooted,
Wi-Fi hiccup, wrong IP for a moment, etc.), cv2.VideoCapture.read() can
block far longer than CAP_PROP_READ_TIMEOUT_MSEC actually enforces --
this property isn't reliably honored by every OpenCV/FFmpeg build for
this kind of stream. When that happens, run() is frozen INSIDE that one
read() call and never reaches the "dropped frame, reconnecting" branch --
so even once the camera/network is reachable again, nothing ever retries.

The watchdog (a second thread) tracks how long it's been since the last
frame. If that exceeds CAMERA_STALL_TIMEOUT_SEC while a connection is
still nominally "good" (self.ret == True), it calls cap.release() from
THIS other thread -- which forces the blocked cap.read() in run() to
return/raise, letting run()'s normal reconnect path take back over
(including automatically picking the stream up again the moment it's
actually reachable).
"""

import os
import sys
import threading
import time

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CameraReader(threading.Thread):
    def __init__(self, url=None, name="camera", stall_timeout_sec=None):
        """
        url:  MJPEG stream URL for this specific ESP32-CAM. Defaults to
              config.CAMERA_URL for backward compatibility with single-camera
              setups, but multi-camera setups should always pass this
              explicitly (e.g. from config.CAMERAS[i]["url"]).
        name: Human-readable label used only in log/print messages, so it's
              obvious which physical camera a given log line refers to when
              several CameraReader threads are running at once.
        stall_timeout_sec: how long with no new frame before the watchdog
              force-reconnects. Defaults to config.CAMERA_STALL_TIMEOUT_SEC
              (or 5s if that isn't set).
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

        self._last_frame_time = time.time()
        self.stall_timeout_sec = stall_timeout_sec or getattr(
            config, "CAMERA_STALL_TIMEOUT_SEC", 5.0
        )
        self._watchdog_thread = None

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
        # Reset the stall clock on every (re)connect attempt too, not just
        # on a successful frame -- otherwise the watchdog could judge a
        # brand-new connection "stalled" before it even had a chance to
        # start receiving frames.
        with self.lock:
            self._last_frame_time = time.time()
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
                        self._last_frame_time = time.time()
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

    def _watchdog_run(self):
        """
        Runs in a SEPARATE thread from run(). See module docstring for why
        this exists -- in short, it unsticks run() if it's frozen inside a
        blocking cap.read() call on a connection that died without
        actually returning an error.
        """
        while self.running:
            time.sleep(1.0)

            with self.lock:
                currently_ok = self.ret
                seconds_since_frame = time.time() - self._last_frame_time
            cap = self.cap  # plain attribute read; worst case we release a
                             # moment-old reference, which is harmless

            if currently_ok and seconds_since_frame > self.stall_timeout_sec and cap is not None:
                print(
                    f"[CameraReader:{self.cam_name}] No frame for "
                    f"{seconds_since_frame:.1f}s despite an apparently-open "
                    f"connection -- forcing a reconnect..."
                )
                with self.lock:
                    self.ret = False
                try:
                    cap.release()
                except Exception as e:
                    print(f"[CameraReader:{self.cam_name}] watchdog release() error: {e}")

    def start(self):
        super().start()
        self._watchdog_thread = threading.Thread(target=self._watchdog_run, daemon=True)
        self._watchdog_thread.start()

    def get_frame(self):
        """Thread-safe accessor for the most recent frame."""
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.running = False