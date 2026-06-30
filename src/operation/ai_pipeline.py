"""
AI tracking pipeline – ổn định target, hiển thị tất cả khuôn mặt đã đăng ký.
"""

import os
import sys
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from operation.event_logger import EventLogger
from operation.face_recognizer import FaceRecognizer


class KalmanFilter2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2, 0)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def predict_and_correct(self, cx, cy):
        if not self.initialized:
            self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            self.initialized = True
        meas = np.array([[np.float32(cx)], [np.float32(cy)]], np.float32)
        self.kf.predict()
        corrected = self.kf.correct(meas)
        return int(corrected[0][0]), int(corrected[1][0])

    def reset(self):
        self.initialized = False


def _crop_safe(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


class AIPipeline(threading.Thread):
    def __init__(self, camera_thread, event_logger=None, room_name="camera"):
        super().__init__()
        self.camera_thread = camera_thread
        self.running = True
        self.daemon = True
        self.room_name = room_name
        self.logger = event_logger or EventLogger()

        self.device = config.get_torch_device()
        print(f"[AI:{self.room_name}] YOLO running on {self.device.upper()}.")
        self.yolo_face = YOLO(config.YOLO_FACE_MODEL_PATH)
        if not os.path.exists(config.YOLO_FACE_MODEL_PATH):
            print(f"[AI:{self.room_name}] WARNING: face model not found, using person model.")
            self.yolo_face = YOLO(config.YOLO_PERSON_MODEL_PATH)

        self.recognizer = FaceRecognizer()
        self.recognizer.load_database()
        if self.recognizer.is_empty():
            print(f"[AI:{self.room_name}] WARNING: face_db empty.")

        self.kalman = KalmanFilter2D()

        # FSM
        self.STATE_SEARCHING = "SEARCHING"
        self.STATE_TRACKING = "TRACKING"
        self.STATE_LOST = "LOST"
        self.current_state = self.STATE_SEARCHING

        # Target chính
        self.locked_identity = None
        self.target_bbox = None
        self.target_center = None

        # Tất cả khuôn mặt đã đăng ký trong frame hiện tại
        self.all_faces = []   # list of dict {name, bbox, score, center}

        # Đếm frame mất target liên tục (để chuyển sang LOST)
        self.lost_counter = 0
        self.LOST_THRESHOLD = 10   # 10 frame mất -> coi là lost

        # Output lock
        self._lock = threading.Lock()
        self.has_target = False
        self.raw_bbox = None
        self.smoothed_bbox = None
        self.smoothed_center = None
        self.last_step_latency_ms = 0.0
        self.last_detect_ms = 0.0
        self.last_recognize_ms = 0.0

    def _set_output(self, has_target, raw_bbox, smoothed_bbox, center):
        with self._lock:
            self.has_target = has_target
            self.raw_bbox = raw_bbox
            self.smoothed_bbox = smoothed_bbox
            self.smoothed_center = center

    def _clear_target(self):
        self.locked_identity = None
        self.target_bbox = None
        self.target_center = None
        self.kalman.reset()
        self._set_output(False, None, None, None)

    def run(self):
        print(f"[AI:{self.room_name}] FSM started.")
        self.logger.log("PIPELINE_STARTED", room=self.room_name)

        while self.running:
            ret, frame = self.camera_thread.get_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            step_start = time.time()
            try:
                self._step(frame)
            except Exception as e:
                print(f"[AI:{self.room_name}] Error: {e}, resetting.")
                self.logger.log("PIPELINE_ERROR", error=str(e), room=self.room_name)
                self.current_state = self.STATE_SEARCHING
                self._clear_target()

            with self._lock:
                self.last_step_latency_ms = (time.time() - step_start) * 1000.0
            time.sleep(0.01)

    def _step(self, frame):
        if self.current_state == self.STATE_SEARCHING:
            self._step_searching(frame)
        elif self.current_state == self.STATE_TRACKING:
            self._step_tracking(frame)
        elif self.current_state == self.STATE_LOST:
            self._step_lost(frame)

    # =============================================
    # SEARCHING
    # =============================================
    def _step_searching(self, frame):
        detect_start = time.time()
        results = self.yolo_face(frame, verbose=False, device=self.device)
        detect_ms = (time.time() - detect_start) * 1000.0

        candidates = []   # (bbox, name, score, center)
        all_faces = []

        recognize_start = time.time()
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                crop = _crop_safe(frame, xyxy)
                if crop is None:
                    continue
                name, score = self.recognizer.identify(crop)
                if name is not None:
                    cx = (xyxy[0] + xyxy[2]) // 2
                    cy = (xyxy[1] + xyxy[3]) // 2
                    center = (cx, cy)
                    candidates.append((xyxy, name, score, center))
                    all_faces.append({
                        "name": name,
                        "bbox": xyxy.tolist(),
                        "score": score,
                        "center": center
                    })
        recognize_ms = (time.time() - recognize_start) * 1000.0
        self.last_detect_ms = detect_ms
        self.last_recognize_ms = recognize_ms

        with self._lock:
            self.all_faces = all_faces

        # In số lượng face nhận diện được (debug)
        print(f"[DEBUG] SEARCHING: found {len(all_faces)} registered faces")

        if candidates:
            # Chọn người gần tâm nhất + score cao
            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2
            best = max(candidates, key=lambda c: c[2] / (abs(c[3][0]-center_x) + abs(c[3][1]-center_y) + 1))
            bbox, name, score, center = best
            self.locked_identity = name
            self.target_bbox = list(bbox)
            self.target_center = center
            self.kalman.reset()
            kx, ky = self.kalman.predict_and_correct(center[0], center[1])
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
            self.current_state = self.STATE_TRACKING
            self.lost_counter = 0
            self.logger.log("TARGET_ACQUIRED", name=name, score=f"{score:.2f}", room=self.room_name)
            print(f"[FSM:{self.room_name}] Acquired '{name}' -> TRACKING")
        else:
            self._clear_target()

    # =============================================
    # TRACKING – ổn định, không nhảy lung tung
    # =============================================
    def _step_tracking(self, frame):
        detect_start = time.time()
        results = self.yolo_face(frame, verbose=False, device=self.device)
        detect_ms = (time.time() - detect_start) * 1000.0

        candidates = []   # (bbox, name, score, center)
        all_faces = []

        recognize_start = time.time()
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                crop = _crop_safe(frame, xyxy)
                if crop is None:
                    continue
                name, score = self.recognizer.identify(crop)
                if name is not None:
                    cx = (xyxy[0] + xyxy[2]) // 2
                    cy = (xyxy[1] + xyxy[3]) // 2
                    center = (cx, cy)
                    candidates.append((xyxy, name, score, center))
                    all_faces.append({
                        "name": name,
                        "bbox": xyxy.tolist(),
                        "score": score,
                        "center": center
                    })
        recognize_ms = (time.time() - recognize_start) * 1000.0
        self.last_detect_ms = detect_ms
        self.last_recognize_ms = recognize_ms

        with self._lock:
            self.all_faces = all_faces

        print(f"[DEBUG] TRACKING: found {len(all_faces)} registered faces, locked='{self.locked_identity}'")

        # Kiểm tra xem target hiện tại có trong danh sách không
        current_found = None
        for cand in candidates:
            if cand[1] == self.locked_identity:
                current_found = cand
                break

        if current_found is None:
            # Mất target
            self.lost_counter += 1
            if self.lost_counter >= self.LOST_THRESHOLD:
                # Thực sự lost
                self.current_state = self.STATE_LOST
                self.logger.log("TARGET_LOST", name=self.locked_identity, room=self.room_name)
                print(f"[FSM:{self.room_name}] Lost '{self.locked_identity}' -> LOST")
                self._clear_target()
            else:
                # Vẫn trong thời gian chờ, giữ target cũ (không đổi)
                # Nhưng vẫn cập nhật vị trí nếu có mặt của target trong frame (dù không nhận diện được?) 
                # Không có, giữ nguyên output
                pass
        else:
            # Target hiện diện
            self.lost_counter = 0
            bbox, name, score, center = current_found
            self.target_bbox = list(bbox)
            self.target_center = center
            kx, ky = self.kalman.predict_and_correct(center[0], center[1])
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))

            # KHÔNG CHUYỂN TARGET KHI CÓ NGƯỜI KHÁC – chỉ chuyển khi target hiện tại biến mất
            # => giữ target ổn định

    # =============================================
    # LOST – tìm lại bất kỳ ai đã đăng ký
    # =============================================
    def _step_lost(self, frame):
        results = self.yolo_face(frame, verbose=False, device=self.device)
        all_faces = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                crop = _crop_safe(frame, xyxy)
                if crop is None:
                    continue
                name, score = self.recognizer.identify(crop)
                if name is not None:
                    cx = (xyxy[0] + xyxy[2]) // 2
                    cy = (xyxy[1] + xyxy[3]) // 2
                    all_faces.append({
                        "name": name,
                        "bbox": xyxy.tolist(),
                        "score": score,
                        "center": (cx, cy)
                    })
        with self._lock:
            self.all_faces = all_faces

        print(f"[DEBUG] LOST: found {len(all_faces)} registered faces")

        # Tìm target cũ hoặc bất kỳ ai
        if all_faces:
            # Ưu tiên target cũ nếu có
            chosen = None
            for face in all_faces:
                if face["name"] == self.locked_identity:
                    chosen = face
                    break
            if chosen is None:
                chosen = all_faces[0]  # lấy người đầu tiên
            name = chosen["name"]
            bbox = chosen["bbox"]
            center = chosen["center"]
            self.locked_identity = name
            self.target_bbox = bbox
            self.target_center = center
            self.kalman.reset()
            kx, ky = self.kalman.predict_and_correct(center[0], center[1])
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
            self.current_state = self.STATE_TRACKING
            self.lost_counter = 0
            self.logger.log("TARGET_REACQUIRED", name=name, room=self.room_name)
            print(f"[FSM:{self.room_name}] Reacquired '{name}' -> TRACKING")
        else:
            self._clear_target()

    # =============================================
    # Public accessors
    # =============================================
    def get_ai_result(self):
        with self._lock:
            return self.has_target, self.raw_bbox, self.smoothed_bbox, self.smoothed_center, self.current_state

    def get_all_faces(self):
        with self._lock:
            return self.all_faces.copy()

    def get_locked_identity(self):
        return self.locked_identity

    def get_last_latency_ms(self):
        with self._lock:
            return self.last_step_latency_ms

    def get_latency_breakdown(self):
        with self._lock:
            return self.last_detect_ms, self.last_recognize_ms

    def stop(self):
        self.running = False