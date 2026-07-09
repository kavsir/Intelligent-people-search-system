"""
AI tracking pipeline -- ổn định target, hiển thị tất cả khuôn mặt đã đăng ký.
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
import face_database
from operation.event_logger import EventLogger
from operation.face_recognizer import FaceRecognizer
from operation.pose_estimator import PoseEstimator
from operation.body_features import BodyFeatureExtractor
from operation.servo_controller import ServoController
from operation.exercise_manager import exercise_manager
from operation.behavior_manager import behavior_manager
def _bbox_iou(a, b):
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
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
    def predict_only(self):
        """Predict next position WITHOUT a measurement correction.
        Used on skip-frames in TRACKING to keep the smoothed position
        moving smoothly while saving the cost of YOLO+InsightFace.
        Returns (cx, cy) or (None, None) if the filter was never seeded."""
        if not self.initialized:
            return None, None
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[1])
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
    def __init__(self, camera_thread, event_logger=None, room_name="camera", room_id=None,
                 door_ws=None):
        super().__init__()
        self.camera_thread = camera_thread
        self.running = True
        self.daemon = True
        self.room_name = room_name
        self.room_id = room_id or room_name
        # DoorWebSocketServer instance shared with app_dashboard.py's door
        # control -- passed through to ServoController when this room has
        # a servo (config.SERVO_ENABLED_ROOMS), since pan/tilt commands
        # now ride the SAME WebSocket connection as door commands (see
        # operation/servo_controller.py). None under app_operation.py's
        # standalone CLI dashboard (no door server there) -- servo angles
        # still compute normally, just don't get sent anywhere.
        self.door_ws = door_ws
        self.logger = event_logger or EventLogger()
        self.device = config.get_torch_device()
        print(f"[AI:{self.room_name}] YOLO running on {self.device.upper()}.")
        self.yolo_face = YOLO(config.YOLO_FACE_MODEL_PATH)
        if not os.path.exists(config.YOLO_FACE_MODEL_PATH):
            print(f"[AI:{self.room_name}] WARNING: face model not found, using person model.")
            self.yolo_face = YOLO(config.YOLO_PERSON_MODEL_PATH)
        # Used only as a fallback in TRACKING when the locked person's
        # face can no longer be recognized (too far / bad angle) -- lets
        # us keep the lock on their BODY instead of losing them outright,
        # and is also what feeds the exercise pose estimator.
        self.yolo_person = YOLO(config.YOLO_PERSON_MODEL_PATH)
        self.recognizer = FaceRecognizer()
        self.recognizer.load_database()
        if self.recognizer.is_empty():
            print(f"[AI:{self.room_name}] WARNING: face_db empty.")
        # One PoseEstimator per room (mediapipe Pose objects are not
        # thread-safe, so this must never be shared across AIPipeline
        # threads).
        self.pose_estimator = PoseEstimator()
        # Separate mediapipe Pose instance (also not thread-safe, also one
        # per room) used ONLY to extract long-term body-shape ratios for
        # EVERY registered face seen this step -- not just the locked
        # target. See operation/body_features.py.
        self.body_extractor = BodyFeatureExtractor()
        self._last_body_update = {}  # name -> time.time() of last profile update
        self.BODY_PROFILE_INTERVAL_SEC = getattr(
            config, "BODY_PROFILE_UPDATE_INTERVAL_SEC", 2.0
        )
        # Throttle for FaceRecognizer.reload_if_changed() -- see run().
        self._last_db_check = 0.0
        self.DB_RELOAD_CHECK_INTERVAL_SEC = getattr(
            config, "FACE_DB_RELOAD_CHECK_INTERVAL_SEC", 2.0
        )
        # Frame-skip counters for SEARCHING / TRACKING state throttling.
        # _frame_counter increments once per _step() call regardless of
        # state; each state method decides whether to skip heavy work
        # based on its own modulo cadence.
        self._frame_counter = 0
        self.SEARCHING_SKIP_FRAMES = getattr(config, "SEARCHING_SKIP_FRAMES", 0)
        self.TRACKING_SKIP_FRAMES = getattr(config, "TRACKING_SKIP_FRAMES", 0)
        self.POSE_EVERY_N_FRAMES = getattr(config, "POSE_EVERY_N_FRAMES", 1)
        self.kalman = KalmanFilter2D()
        # Pan/tilt servo -- CAM2-ONLY add-on, see config.SERVO_ENABLED_ROOMS.
        # None everywhere else, which _drive_servo() below treats as "this
        # room has no servo, do nothing" -- so cam1 (and any other room
        # not listed) is completely unaffected.
        self.servo = None
        self._servo_lost_counter = 0
        self._servo_was_tracking = False
        self._last_frame_shape = None  # (h, w) of the current step's frame
        self.SERVO_RETURN_TO_CENTER_AFTER_LOST_FRAMES = getattr(
            config, "SERVO_RETURN_TO_CENTER_AFTER_LOST_FRAMES", 15
        )
        if self.room_id in getattr(config, "SERVO_ENABLED_ROOMS", []):
            try:
                self.servo = ServoController(config.SERVO_CONFIG, door_ws=self.door_ws)
                if self.door_ws is None:
                    print(
                        f"[AI:{self.room_name}] Pan/tilt servo ENABLED (room id='{self.room_id}') "
                        f"but no door_ws was passed in -- running in SIMULATE mode (angles "
                        f"compute, nothing gets sent)."
                    )
                else:
                    print(f"[AI:{self.room_name}] Pan/tilt servo ENABLED (room id='{self.room_id}').")
            except Exception as e:
                print(f"[AI:{self.room_name}] Servo init failed ({e}) -- continuing WITHOUT servo.")
                self.servo = None
        # FSM
        self.STATE_SEARCHING = "SEARCHING"
        self.STATE_TRACKING = "TRACKING"
        self.STATE_LOST = "LOST"
        self.current_state = self.STATE_SEARCHING
        # Target chính
        self.locked_identity = None
        self.target_bbox = None
        self.target_center = None
        # Last non-None locked_identity, kept even after _clear_target()
        # wipes locked_identity -- used as the first guess when trying to
        # reacquire someone by BODY SHAPE ALONE (no face visible) in
        # _step_lost(), since "whoever we were just tracking" is the most
        # likely candidate for "whoever just reappeared".
        self._last_locked_identity = None
        # "FACE" while the locked person's face is still recognizable each
        # frame. "BODY" while we're keeping the lock via YOLO-person + IOU
        # continuity because their face faded out (they walked further
        # away) but they haven't actually left the frame. "BODY_SHAPE"
        # when neither face NOR IOU continuity found them, but their
        # long-term body-shape profile (face_database.body_recognition)
        # matched someone currently visible with high confidence --
        # weakest of the 3 signals, since body shape alone is a softer
        # biometric than face recognition, but the only one that works
        # when someone's face genuinely isn't visible (turned away,
        # occluded, too far).
        self.lock_mode = None
        # Tất cả khuôn mặt đã đăng ký trong frame hiện tại
        self.all_faces = []   # list of dict {name, bbox, score, center}
        # Đếm frame mất target liên tục (để chuyển sang LOST)
        self.lost_counter = 0
        self.LOST_THRESHOLD = 10   # 10 frame mất -> coi là lost
        # --- Handoff (Phòng 1 <-> Phòng 2) ---
        self.on_target_lost_callback = None   # do HandoffManager gán (app_dashboard.py)
        self._boundary_exit_streak = 0        # đếm số frame liên tiếp servo "dính" biên
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
        self._drive_servo(has_target, center)
    def _drive_servo(self, has_target, center):
        """
        CAM2-ONLY (see config.SERVO_ENABLED_ROOMS): physically steer the
        pan/tilt mount so the currently-locked target's smoothed center
        stays near the middle of the frame. This is the ONE place every
        _step_* branch's _set_output(True, ..., center) call funnels
        through -- FACE lock, BODY (IOU continuity) lock, and BODY_SHAPE
        (re-acquired with no face visible) lock all drive the servo the
        exact same way, since it only cares WHERE the target is, never
        WHO or HOW they were found. Does nothing at all if self.servo is
        None (every room except the ones listed in SERVO_ENABLED_ROOMS).
        """
        if self.servo is None:
            return
        if has_target and center is not None and self._last_frame_shape is not None:
            self._servo_lost_counter = 0
            self._servo_was_tracking = True
            h, w = self._last_frame_shape
            cx, cy = w / 2.0, h / 2.0
            error_x = center[0] - cx
            error_y = center[1] - cy
            self.servo.update(error_x, error_y)
            return
        # Không có mục tiêu frame này.
        if self._servo_was_tracking:
            # Vừa rớt khóa -- tránh integral windup, giống lock_interrupted
            # ở project pan-tilt gốc.
            self.servo.reset_integral()
            self._servo_was_tracking = False
        self._servo_lost_counter += 1
        if (self._servo_lost_counter == self.SERVO_RETURN_TO_CENTER_AFTER_LOST_FRAMES
                and not self.servo.scanning):
            # Không tự đưa servo về giữa nếu đang ở chế độ quét chủ động
            # tìm người vừa được bàn giao (xem HandoffManager.check_pending
            # trong app_dashboard.py) -- việc dừng quét + về giữa chỉ nên
            # do chính handoff quyết định (khi phòng đích đã tìm ra người).
            self.servo.go_to_center(config.SERVO_CONFIG)
    def get_servo_status(self):
        """(enabled, pan_angle, tilt_angle) -- None angles if this room has
        no servo. Handy for a dashboard overlay/status readout."""
        if self.servo is None:
            return False, None, None
        return True, self.servo.pan_angle, self.servo.tilt_angle
    def _clear_target(self):
        if self.locked_identity is not None:
            self._last_locked_identity = self.locked_identity
        self.locked_identity = None
        self.target_bbox = None
        self.target_center = None
        self.lock_mode = None
        self.kalman.reset()
        self._set_output(False, None, None, None)
    def force_clear_target(self):
        """
        Thread-safe external request to drop whatever this room is
        currently locked onto and go back to SEARCHING. Used when the
        locked person's registration data was just deleted (they failed
        an assigned exercise), so we stop tracking someone who no longer
        exists in face_db.
        """
        self.current_state = self.STATE_SEARCHING
        self._clear_target()
    def get_lock_mode(self):
        return self.lock_mode
    def set_on_target_lost_callback(self, fn):
        """fn(name, room_id, direction, last_center) -- gọi khi target
        chuyển sang LOST, kèm hướng suy luận được (LEFT/RIGHT/UNKNOWN)."""
        self.on_target_lost_callback = fn
    def _estimate_exit_direction(self):
        """
        Ước lượng HƯỚNG mất dấu ngay TRƯỚC thời điểm chuyển sang LOST.
        Trả về (direction: "LEFT"|"RIGHT"|"UNKNOWN", meta: dict để log).
        Phòng "static" (cam1) không còn chia biên trái/phải -- hướng ở đây
        chỉ để LOG/debug (center_x cuối cùng), KHÔNG dùng để quyết định
        handoff nữa (xem HandoffManager.on_target_lost: cam1 mất dấu là
        LUÔN đón đầu servo cam2, bất kể hướng). Phòng "dynamic" (cam2) vẫn
        dùng góc Pan để suy hướng, vì đó là tọa độ đáng tin của phòng này.
        """
        cfg = config.HANDOFF_CONFIG.get(self.room_id)
        if cfg is None:
            return "UNKNOWN", {}
        if cfg["type"] == "static":
            cx = self.target_center[0] if self.target_center is not None else None
            return "UNKNOWN", {"center_x": cx}
        # dynamic (servo) room
        if self.servo is None:
            return "UNKNOWN", {}
        pan = self.servo.pan_angle
        if pan >= cfg["pan_right_boundary"]:
            return "RIGHT", {"pan_angle": pan}
        if pan <= cfg["pan_left_boundary"]:
            return "LEFT", {"pan_angle": pan}
        return "UNKNOWN", {"pan_angle": pan}
    def _check_dynamic_boundary_exit(self):
        """
        Chỉ áp dụng cho phòng có servo. Coi mục tiêu đã ra khỏi PHẠM VI
        LOGIC của phòng (không phải giới hạn cơ khí servo) ngay khi góc
        Pan cần để bám mục tiêu vượt biên cấu hình đủ N frame liên tiếp
        -- NGAY CẢ KHI khuôn mặt vẫn đang được nhận diện bình thường.
        Trả về True nếu vừa xác nhận đủ điều kiện chốt LOST do vượt biên.
        """
        cfg = config.HANDOFF_CONFIG.get(self.room_id)
        if cfg is None or cfg.get("type") != "dynamic" or self.servo is None:
            self._boundary_exit_streak = 0
            return False
        pan = self.servo.pan_angle
        beyond = pan <= cfg["pan_left_boundary"] or pan >= cfg["pan_right_boundary"]
        self._boundary_exit_streak = self._boundary_exit_streak + 1 if beyond else 0
        return self._boundary_exit_streak >= cfg.get("boundary_confirm_frames", 8)
    def _declare_lost_with_handoff(self, reason_event):
        """
        Điểm CHUNG để chốt LOST + suy luận hướng + gọi callback handoff --
        dùng cho cả 2 kịch bản: (a) mất dấu khuôn mặt bình thường sau
        LOST_THRESHOLD frame, (b) vượt biên logic của phòng động dù vẫn
        thấy mặt (xem _check_dynamic_boundary_exit()).
        """
        name = self.locked_identity
        direction, meta = self._estimate_exit_direction()
        self.current_state = self.STATE_LOST
        self.logger.log(reason_event, name=name, room=self.room_name,
                         direction=direction, **meta)
        print(f"[FSM:{self.room_name}] Lost '{name}' ({reason_event}) "
              f"-> LOST, direction={direction}")
        exercise_manager.set_online(name, False)
        if self.on_target_lost_callback:
            try:
                self.on_target_lost_callback(name, self.room_id, direction, self.target_center)
            except Exception as e:
                print(f"[AI:{self.room_name}] on_target_lost_callback error: {e}")
        self._clear_target()
    def run(self):
        print(f"[AI:{self.room_name}] FSM started.")
        self.logger.log("PIPELINE_STARTED", room=self.room_name)
        while self.running:
            ret, frame = self.camera_thread.get_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            # Cheap, throttled check: has app_registration.py (a SEPARATE
            # process) registered or deleted someone since we last loaded
            # the embeddings? If so, pick it up right away -- this is what
            # removes the "must restart app_dashboard.py" requirement.
            now_check = time.time()
            if now_check - self._last_db_check >= self.DB_RELOAD_CHECK_INTERVAL_SEC:
                self._last_db_check = now_check
                self.recognizer.reload_if_changed()
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
        self._last_frame_shape = frame.shape[:2]  # (h, w) -- for _drive_servo()
        self._frame_counter += 1
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
        # Frame-skip throttle: skip heavy YOLO-face + InsightFace on some
        # steps while scanning -- still finds new arrivals quickly while
        # cutting CPU usage. 0 = run every step (no skip).
        if self.SEARCHING_SKIP_FRAMES > 0 and self._frame_counter % (self.SEARCHING_SKIP_FRAMES + 1) != 0:
            return
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
        self._update_body_profiles(frame, all_faces)
        if candidates:
            # Chọn người gần tâm nhất + score cao
            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2
            best = max(candidates, key=lambda c: c[2] / (abs(c[3][0]-center_x) + abs(c[3][1]-center_y) + 1))
            bbox, name, score, center = best
            self.locked_identity = name
            self.target_bbox = list(bbox)
            self.target_center = center
            self.lock_mode = "FACE"
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
    # TRACKING -- ổn định, không nhảy lung tung
    # =============================================
    def _step_tracking(self, frame):
        # --- Skip-frame fast path ---
        # When TRACKING_SKIP_FRAMES > 0, skip the expensive YOLO-face +
        # InsightFace pass on some steps and just keep the Kalman
        # prediction alive. The smoothed centre still updates smoothly
        # (so the dashboard crosshair + servo tracking don't stutter),
        # but we avoid re-detecting a face we already know is there.
        # This is the single biggest CPU saving in the whole pipeline
        # because it cuts YOLO-face + InsightFace + body-profile updates
        # all at once, while the Kalman filter bridges the gap.
        if self.TRACKING_SKIP_FRAMES > 0 and self._frame_counter % (self.TRACKING_SKIP_FRAMES + 1) != 0:
            if self.target_center is not None and self.target_bbox is not None:
                pred = self.kalman.predict_only()
                if pred[0] is not None:
                    self._set_output(True, self.target_bbox, self.target_bbox, pred)
            # Pose estimation (MediaPipe) is also heavy -- throttle it
            # separately so it doesn't run every single frame either.
            if self.target_bbox is not None:
                self._run_exercise_tracking(frame, self.target_bbox, center=self.target_center)
            return

        # --- Normal (full-detection) path below ---
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
        self._update_body_profiles(frame, all_faces)
        # Kiểm tra xem target hiện tại có trong danh sách không
        current_found = None
        for cand in candidates:
            if cand[1] == self.locked_identity:
                current_found = cand
                break
        if current_found is not None:
            # Mặt vẫn nhận diện được bình thường -- đường đi cũ, không đổi.
            self.lost_counter = 0
            self.lock_mode = "FACE"
            bbox, name, score, center = current_found
            self.target_bbox = list(bbox)
            self.target_center = center
            kx, ky = self.kalman.predict_and_correct(center[0], center[1])
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
            self._run_exercise_tracking(frame, self.target_bbox, center=(kx, ky))
            if self._check_dynamic_boundary_exit():
                self._declare_lost_with_handoff("TARGET_LOST_BOUNDARY")
                return
            # KHÔNG CHUYỂN TARGET KHI CÓ NGƯỜI KHÁC -- chỉ chuyển khi target hiện tại biến mất
            return
        # Không nhận diện được mặt target ở frame này -- thử khoá theo
        # THÂN NGƯỜI (người đó có thể đã đi xa hơn, mặt quá nhỏ/lệch góc
        # để nhận diện, nhưng vẫn còn trong khung hình ở vị trí gần với
        # bbox cũ).
        body_bbox = self._find_body_continuity(frame)
        if body_bbox is not None:
            self.lost_counter = 0
            self.lock_mode = "BODY"
            cx = (body_bbox[0] + body_bbox[2]) // 2
            cy = (body_bbox[1] + body_bbox[3]) // 2
            self.target_bbox = list(body_bbox)
            self.target_center = (cx, cy)
            kx, ky = self.kalman.predict_and_correct(cx, cy)
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
            self._run_exercise_tracking(frame, self.target_bbox, center=(kx, ky))
            if self._check_dynamic_boundary_exit():
                self._declare_lost_with_handoff("TARGET_LOST_BOUNDARY")
                return
            return
        # Cả mặt lẫn thân (IOU) đều không tìm thấy -- trước khi coi là mất
        # hẳn, thử nhận lại CHÍNH người này bằng HÌNH DÁNG thân người (họ
        # có thể đã di chuyển hẳn sang vị trí khác trong khung, không còn
        # overlap với bbox cũ, và mặt đang quay đi chỗ khác).
        shape_bbox, shape_name, shape_score = self._find_identity_by_body_shape(
            frame, candidate_names=[self.locked_identity]
        )
        if shape_bbox is not None:
            self.lost_counter = 0
            self.lock_mode = "BODY_SHAPE"
            cx = (shape_bbox[0] + shape_bbox[2]) // 2
            cy = (shape_bbox[1] + shape_bbox[3]) // 2
            self.target_bbox = shape_bbox
            self.target_center = (cx, cy)
            kx, ky = self.kalman.predict_and_correct(cx, cy)
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
            self._run_exercise_tracking(frame, self.target_bbox, center=(kx, ky))
            self.logger.log(
                "BODY_SHAPE_MATCH", name=self.locked_identity,
                similarity=f"{shape_score:.2f}", room=self.room_name,
            )
            if self._check_dynamic_boundary_exit():
                self._declare_lost_with_handoff("TARGET_LOST_BOUNDARY")
                return
            return
        # Cả 3 cách đều không tìm thấy -- thực sự có thể đã mất.
        self.lost_counter += 1
        if self.lost_counter >= self.LOST_THRESHOLD:
            self._declare_lost_with_handoff("TARGET_LOST")
        # else: vẫn trong thời gian chờ (grace period), giữ nguyên output cũ.
    def _update_body_profiles(self, frame, all_faces):
        """
        Long-term body-shape tracking for PROBLEM #2 -- runs for EVERY
        registered face seen this step (all_faces), not just whichever
        one is currently locked as the main target. This is what makes
        the body_recognition profile keep improving even for people the
        FSM isn't actively "tracking" right now.
        For each registered face:
          1. Match it to a YOLO-person body box (one YOLO-person pass is
             shared across every face this step -- cheaper than one pass
             per face).
          2. Extract clothing-invariant skeleton ratios from that body
             crop (operation/body_features.py).
          3. Fold the reading into face_database.body_recognition via an
             exponential moving average.
        Throttled per-person (BODY_PROFILE_UPDATE_INTERVAL_SEC, default
        2s) so this doesn't run mediapipe Pose on every single frame for
        every face -- that would be far too expensive stacked on top of
        YOLO-face + YOLO-person + InsightFace + the existing PoseEstimator,
        especially with multiple rooms running at once (see config.py's
        CPU-tuning notes).
        """
        if not all_faces:
            return
        now = time.time()
        due_faces = [
            f for f in all_faces
            if now - self._last_body_update.get(f["name"], 0.0) >= self.BODY_PROFILE_INTERVAL_SEC
        ]
        if not due_faces:
            return
        try:
            person_results = self.yolo_person(frame, verbose=False, device=self.device, classes=[0])
        except Exception as e:
            print(f"[AI:{self.room_name}] body-profile yolo_person error: {e}")
            return
        person_boxes = [
            box.xyxy[0].cpu().numpy().astype(int)
            for r in person_results for box in r.boxes
        ]
        if not person_boxes:
            return
        for face in due_faces:
            name = face["name"]
            fx1, fy1, fx2, fy2 = face["bbox"]
            fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            # A face sits inside its own body's box, near the top -- pick
            # the person box whose region contains the face center.
            body_box = None
            for pb in person_boxes:
                if pb[0] <= fcx <= pb[2] and pb[1] <= fcy <= pb[3]:
                    body_box = pb
                    break
            if body_box is None:
                continue
            crop = _crop_safe(frame, body_box)
            features = self.body_extractor.extract(crop)
            self._last_body_update[name] = now  # mark attempted regardless of success
            if features is None:
                continue  # partial/occluded skeleton this time -- try again next interval
            try:
                face_database.update_body_profile(name, features)
            except Exception as e:
                print(f"[AI:{self.room_name}] body profile save failed for '{name}': {e}")
    def _find_identity_by_body_shape(self, frame, candidate_names=None, min_similarity=None):
        """
        Try to find a registered person among ALL bodies YOLO-person can
        see in `frame`, using ONLY their long-term body-shape profile
        (face_database.body_recognition) -- no face required. This is the
        fallback for when face recognition genuinely can't see a face at
        all (turned away, too far, bad angle), so a registered person
        doesn't get treated as "gone" just because their face isn't
        pointed at the camera right now.
        candidate_names: None = cold match against every profile with
            enough history (stricter threshold, see config); a list/set =
            targeted re-check against only those names (looser threshold,
            since context already narrows it down -- e.g. "probably
            whoever we were just tracking here").
        Returns (bbox: list[int,4], name: str, similarity: float), or
        (None, None, 0.0) if nothing clears the bar.
        """
        try:
            person_results = self.yolo_person(frame, verbose=False, device=self.device, classes=[0])
        except Exception as e:
            print(f"[AI:{self.room_name}] body-shape match yolo_person error: {e}")
            return None, None, 0.0
        if min_similarity is None:
            min_similarity = (
                config.BODY_MATCH_MIN_SIMILARITY if candidate_names
                else config.BODY_MATCH_MIN_SIMILARITY_COLD
            )
        min_samples = getattr(config, "BODY_MATCH_MIN_SAMPLES", 5)
        best_box, best_name, best_score = None, None, -1.0
        for r in person_results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                crop = _crop_safe(frame, xyxy)
                features = self.body_extractor.extract(crop)
                if features is None:
                    continue
                name, score = face_database.match_body_profile(
                    features, candidate_names=candidate_names, min_sample_count=min_samples
                )
                if name is not None and score > best_score:
                    best_box, best_name, best_score = xyxy, name, score
        if best_box is not None and best_score >= min_similarity:
            return list(best_box), best_name, best_score
        return None, None, 0.0
    def _find_body_continuity(self, frame):
        """
        Khi mặt của target đang khoá không còn nhận diện được ở frame này,
        thử khoá tiếp theo THÂN NGƯỜI: chạy YOLO-person, so khớp IOU với
        bbox cuối cùng đã biết. Trả về bbox mới (list[int,4]) nếu khớp đủ
        tốt, ngược lại trả về None.
        """
        if self.target_bbox is None:
            return None
        try:
            results = self.yolo_person(frame, verbose=False, device=self.device, classes=[0])
        except Exception as e:
            print(f"[AI:{self.room_name}] yolo_person error: {e}")
            return None
        best_iou = 0.0
        best_box = None
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                iou = _bbox_iou(xyxy, self.target_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_box = xyxy
        if best_box is not None and best_iou >= 0.25:
            return list(best_box)
        return None
    def _run_exercise_tracking(self, frame, bbox, center=None):
        """
        Feed the currently-locked (recognized, registered) person's body
        crop into this room's PoseEstimator every frame. This now covers
        two independent things:
          1) Exercise rep counting (squat/pushup), scored against
             exercise_manager ONLY if that person currently has an
             assignment -- unchanged behavior.
          2) General behavior recognition (Đứng/Di chuyển/Nhảy/Giơ tay/
             Nằm), which always runs for ANY recognized registered
             person, assignment or not -- these are ordinary behaviors,
             not exercises, so they're tracked unconditionally in
             behavior_manager.
        NOTE on CPU cost: this used to skip mediapipe entirely for
        unassigned people to save CPU. Now that behavior recognition must
        run for everyone recognized (not just assigned people), mediapipe
        runs on every frame a registered person is locked, assigned or
        not. POSE_EVERY_N_FRAMES (config, default 2) throttles this to
        every Nth frame -- MediaPipe Pose is one of the heaviest per-step
        costs, and running it at half or one-third rate still catches
        every rep/behavior while cutting CPU usage significantly.
        """
        # Throttle: only run MediaPipe Pose every N frames.
        if self.POSE_EVERY_N_FRAMES > 1 and self._frame_counter % self.POSE_EVERY_N_FRAMES != 0:
            return
        name = self.locked_identity
        if name is None:
            return
        is_assigned = exercise_manager.is_assigned(name)
        if is_assigned:
            exercise_manager.set_online(name, True)
        # IMPORTANT: pass the Kalman-SMOOTHED center (the same one drawn as
        # the crosshair on the dashboard), not the raw per-frame detection
        # center. Raw YOLO box centers jitter by several pixels frame-to-
        # frame even while the person stands still, which used to make
        # jump/move detection fire randomly on pure detector noise.
        crop = _crop_safe(frame, bbox)
        result = self.pose_estimator.process(name, crop, bbox_center=center)
        if is_assigned and result["rep_completed"]:
            exercise_manager.register_rep(name, result["rep_completed"])
            self.logger.log(
                "EXERCISE_REP", name=name, exercise=result["rep_completed"], room=self.room_name
            )
            print(f"[Exercise:{self.room_name}] '{name}' completed a {result['rep_completed']} rep")
        if result["behavior_events"]:
            behavior_manager.record_many(name, result["behavior_events"])
            self.logger.log(
                "BEHAVIOR_DETECTED",
                name=name,
                events=",".join(result["behavior_events"]),
                room=self.room_name,
            )
    # =============================================
    # LOST -- tìm lại bất kỳ ai đã đăng ký
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
        self._update_body_profiles(frame, all_faces)
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
            self.lock_mode = "FACE"
            self.kalman.reset()
            kx, ky = self.kalman.predict_and_correct(center[0], center[1])
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
            self.current_state = self.STATE_TRACKING
            self.lost_counter = 0
            self.logger.log("TARGET_REACQUIRED", name=name, room=self.room_name)
            print(f"[FSM:{self.room_name}] Reacquired '{name}' -> TRACKING")
            return
        # Không thấy mặt AI cả -- trước khi tiếp tục chịu ở SEARCHING/LOST,
        # thử nhận lại bằng HÌNH DÁNG thân người (không cần mặt). Ưu tiên
        # kiểm tra đúng người vừa mất (nếu có) với ngưỡng dễ hơn, sau đó
        # mới thử khớp lạnh (cold match) với TẤT CẢ hồ sơ đã đăng ký, với
        # ngưỡng khắt khe hơn vì không có ngữ cảnh thu hẹp ứng viên.
        shape_bbox = shape_name = None
        shape_score = 0.0
        if self._last_locked_identity is not None:
            shape_bbox, shape_name, shape_score = self._find_identity_by_body_shape(
                frame, candidate_names=[self._last_locked_identity]
            )
        if shape_bbox is None:
            shape_bbox, shape_name, shape_score = self._find_identity_by_body_shape(
                frame, candidate_names=None
            )
        if shape_bbox is not None:
            cx = (shape_bbox[0] + shape_bbox[2]) // 2
            cy = (shape_bbox[1] + shape_bbox[3]) // 2
            self.locked_identity = shape_name
            self.target_bbox = shape_bbox
            self.target_center = (cx, cy)
            self.lock_mode = "BODY_SHAPE"
            self.kalman.reset()
            kx, ky = self.kalman.predict_and_correct(cx, cy)
            self._set_output(True, self.target_bbox, self.target_bbox, (kx, ky))
            self.current_state = self.STATE_TRACKING
            self.lost_counter = 0
            self.logger.log(
                "TARGET_REACQUIRED_BY_BODY", name=shape_name,
                similarity=f"{shape_score:.2f}", room=self.room_name,
            )
            print(f"[FSM:{self.room_name}] Reacquired '{shape_name}' by body shape "
                  f"(similarity={shape_score:.2f}) -> TRACKING")
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
        try:
            self.body_extractor.close()
        except Exception:
            pass
        if self.servo is not None:
            try:
                self.servo.close()
            except Exception:
                pass