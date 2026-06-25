"""
AI tracking pipeline: a finite state machine that prioritizes face tracking
(cheap CSRT tracker, periodically re-validated by YOLO-face) and falls back
to person tracking (YOLO-person) when the face is lost.

Unlike the first version of this file, this pipeline only locks onto a
*registered* person (matched via FaceRecognizer against data/face_db/) --
an unrecognized face is reported but never tracked. This is what connects
the registration app's output to the operation app, which was previously
missing entirely.

States:
    SEARCHING       -> no locked target yet, scanning every frame
    TRACKING_FACE   -> locked onto a registered face, tracked via CSRT
    FALLBACK_PERSON -> face lost, tracking the same person's body instead
    LOST            -> target missing for longer than the grace period;
                       servo holds its last position (see ServoController)

IGNORE is not a separate FSM state here -- it's simply what happens when a
detected face does not match anyone in face_db: it's logged but ignored,
and the FSM stays in SEARCHING.
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
    """Simple constant-velocity 2D Kalman filter used to smooth target center."""

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
    """Crop a bbox out of frame, clamped to image bounds. Returns None if empty."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


class AIPipeline(threading.Thread):
    def __init__(self, camera_thread, event_logger=None):
        super().__init__()
        self.camera_thread = camera_thread
        self.running = True
        self.daemon = True

        self.logger = event_logger or EventLogger()

        # --- Load models ---
        print("[AI] Loading AI models...")
        self.device = config.get_torch_device()
        print(f"[AI] YOLO models running on {self.device.upper()}.")

        self.yolo_person = YOLO(config.YOLO_PERSON_MODEL_PATH)
        try:
            self.yolo_face = YOLO(config.YOLO_FACE_MODEL_PATH)
            self._face_model_ok = True
        except Exception:
            print(
                "[AI] WARNING: face model not found at "
                f"{config.YOLO_FACE_MODEL_PATH}. Falling back to the person "
                "model for face detection -- face tracking will NOT work "
                "correctly until yolov8n-face.pt is in place (see "
                "download_model.py)."
            )
            self.yolo_face = self.yolo_person
            self._face_model_ok = False

        self.recognizer = FaceRecognizer()
        self.recognizer.load_database()
        if self.recognizer.is_empty():
            print(
                "[AI] WARNING: face_db is empty. No registered person can be "
                "tracked until at least one person is registered via the "
                "registration app."
            )

        self.kalman = KalmanFilter2D()

        # --- Finite state machine ---
        self.STATE_SEARCHING = "SEARCHING"
        self.STATE_TRACKING_FACE = "TRACKING_FACE"
        self.STATE_FALLBACK_PERSON = "FALLBACK_PERSON"
        self.STATE_LOST = "LOST"
        self.current_state = self.STATE_SEARCHING

        # --- Classic tracker (CSRT) ---
        self.tracker = None
        self.validate_counter = 0
        self.VALIDATE_INTERVAL = 20  # re-run YOLO every N frames to correct drift

        # Frame counter used to throttle the face re-check while in fallback,
        # instead of relying on wall-clock time (which is non-deterministic
        # with respect to the loop).
        self.fallback_face_check_counter = 0
        self.FALLBACK_FACE_CHECK_INTERVAL = 3

        # --- Identity tracking ---
        self.locked_identity = None  # name of the person currently locked on

        # --- Lost-target debounce (avoids 1-2 frame flicker flapping state) ---
        self.LOST_GRACE_PERIOD_SEC = config.LOST_GRACE_PERIOD_SEC
        self._target_missing_since = None

        # --- Output state (protected by _lock since the servo/dashboard
        # threads read it concurrently with this thread writing it) ---
        self._lock = threading.Lock()
        self.has_target = False
        self.raw_bbox = None
        self.smoothed_bbox = None
        self.target_center = None
        self.last_step_latency_ms = 0.0

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _init_csrt_tracker(self, frame, bbox):
        """(Re)initialize the OpenCV CSRT tracker on the given bbox."""
        try:
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.TrackerCSRT.create()

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        self.tracker.init(frame, (x1, y1, w, h))
        self.validate_counter = 0

    def _set_output(self, has_target, raw_bbox, smoothed_bbox, target_center):
        with self._lock:
            self.has_target = has_target
            self.raw_bbox = raw_bbox
            self.smoothed_bbox = smoothed_bbox
            self.target_center = target_center

    def _clear_target(self, reset_kalman=True):
        self._set_output(False, None, None, None)
        if reset_kalman:
            self.kalman.reset()

    def _mark_target_seen(self):
        """Call whenever a locked target is successfully found this frame."""
        self._target_missing_since = None

    def _mark_target_missing_and_check_lost(self):
        """
        Call when the locked target could not be found this frame.
        Returns True once the grace period has elapsed (caller should then
        transition to LOST); returns False while still within the grace
        period (caller should leave the last known output untouched so the
        servo holds position rather than snapping to "no target").
        """
        now = time.time()
        if self._target_missing_since is None:
            self._target_missing_since = now
            return False

        return (now - self._target_missing_since) >= self.LOST_GRACE_PERIOD_SEC

    # -----------------------------------------------------------------
    # Thread main loop
    # -----------------------------------------------------------------
    def run(self):
        print(f"[AI] FSM thread started. Initial state: {self.current_state}")
        self.logger.log("PIPELINE_STARTED", state=self.current_state)

        while self.running:
            ret, frame = self.camera_thread.get_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            step_start = time.time()
            try:
                self._step(frame)
            except Exception as exc:
                # Never let a single bad frame kill the whole AI thread.
                print(f"[AI] Unexpected error during step, resetting to SEARCHING: {exc}")
                self.logger.log("PIPELINE_ERROR", error=str(exc))
                self.current_state = self.STATE_SEARCHING
                self.tracker = None
                self.locked_identity = None
                self._target_missing_since = None
                self._clear_target()

            with self._lock:
                self.last_step_latency_ms = (time.time() - step_start) * 1000.0

            time.sleep(0.01)

    def _step(self, frame):
        if self.current_state == self.STATE_SEARCHING:
            self._step_searching(frame)
        elif self.current_state == self.STATE_TRACKING_FACE:
            self._step_tracking_face(frame)
        elif self.current_state == self.STATE_FALLBACK_PERSON:
            self._step_fallback_person(frame)
        elif self.current_state == self.STATE_LOST:
            self._step_lost(frame)

    # -----------------------------------------------------------------
    # State: SEARCHING
    # -----------------------------------------------------------------
    def _step_searching(self, frame):
        face_results = self.yolo_face(frame, verbose=False, device=self.device)

        for r in face_results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                crop = _crop_safe(frame, xyxy)
                name, score = self.recognizer.identify(crop)

                if name is None:
                    # Either nobody is registered, or this face doesn't match
                    # anyone registered closely enough -- log once and move
                    # on without locking onto a stranger.
                    self.logger.log("FACE_IGNORED", score=f"{score:.2f}")
                    continue

                self.raw_bbox = list(xyxy)
                self.locked_identity = name
                self.current_state = self.STATE_TRACKING_FACE
                self._target_missing_since = None
                self._init_csrt_tracker(frame, self.raw_bbox)
                self.logger.log(
                    "TARGET_ACQUIRED", name=name, score=f"{score:.2f}", mode="FACE_TRACK"
                )
                print(f"[FSM] Recognized '{name}' (score={score:.2f}) -> TRACKING_FACE")
                return

        # No registered face found this frame.
        self._clear_target(reset_kalman=False)

    # -----------------------------------------------------------------
    # State: TRACKING_FACE
    # -----------------------------------------------------------------
    def _step_tracking_face(self, frame):
        self.validate_counter += 1

        success, tracker_box = self.tracker.update(frame)

        if not success:
            self._handle_face_track_failure(reason="CSRT lost the face")
            return

        tx, ty, tw, th = map(int, tracker_box)
        smoothed_bbox = [tx, ty, tx + tw, ty + th]

        raw_cx, raw_cy = tx + tw // 2, ty + th // 2
        kx, ky = self.kalman.predict_and_correct(raw_cx, raw_cy)

        self._mark_target_seen()
        self._set_output(True, self.raw_bbox, smoothed_bbox, (kx, ky))

        if self.validate_counter >= self.VALIDATE_INTERVAL:
            self.validate_counter = 0
            self._revalidate_face_identity(frame)

    def _revalidate_face_identity(self, frame):
        """
        Periodically re-run YOLO-face + recognition to correct CSRT drift and
        confirm we're still tracking the same registered person (not someone
        who has since walked into the CSRT box).
        """
        face_results = self.yolo_face(frame, verbose=False, device=self.device)

        for r in face_results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                crop = _crop_safe(frame, xyxy)
                name, score = self.recognizer.identify(crop)

                if name != self.locked_identity:
                    continue

                self.raw_bbox = list(xyxy)
                self._init_csrt_tracker(frame, self.raw_bbox)
                return

        self._handle_face_track_failure(
            reason="re-check found no face matching the locked identity"
        )

    def _handle_face_track_failure(self, reason):
        print(f"[FSM] {reason} -> trying FALLBACK_PERSON")
        self.logger.log("FACE_TRACK_LOST", reason=reason, name=self.locked_identity)
        self.current_state = self.STATE_FALLBACK_PERSON
        self.fallback_face_check_counter = 0

    # -----------------------------------------------------------------
    # State: FALLBACK_PERSON
    # -----------------------------------------------------------------
    def _step_fallback_person(self, frame):
        person_results = self.yolo_person(frame, verbose=False, device=self.device)
        person_found = False

        for r in person_results:
            person_boxes = [
                b for b in r.boxes if self.yolo_person.names[int(b.cls[0])] == "person"
            ]
            if len(person_boxes) == 0:
                continue

            px1, py1, px2, py2 = person_boxes[0].xyxy[0].cpu().numpy().astype(int)
            raw_bbox = [px1, py1, px2, py2]
            person_found = True

            body_cx = px1 + (px2 - px1) // 2
            body_cy = py1 + (py2 - py1) // 4  # bias toward chest/head, not navel

            kx, ky = self.kalman.predict_and_correct(body_cx, body_cy)
            self._mark_target_seen()
            self._set_output(True, raw_bbox, raw_bbox, (kx, ky))

            # Periodically re-check for the registered face within the
            # person box, using a frame counter rather than wall-clock time
            # for a deterministic 1-in-N check rate.
            self.fallback_face_check_counter += 1
            if self.fallback_face_check_counter >= self.FALLBACK_FACE_CHECK_INTERVAL:
                self.fallback_face_check_counter = 0
                if self._try_reacquire_face(frame):
                    return
            break

        if not person_found:
            self._handle_target_missing(reason="lost person in FALLBACK_PERSON")

    def _try_reacquire_face(self, frame):
        """Look for the locked identity's face again; switch back to
        TRACKING_FACE if found. Returns True if reacquired."""
        face_results = self.yolo_face(frame, verbose=False, device=self.device)

        for fr in face_results:
            for box in fr.boxes:
                fbox = box.xyxy[0].cpu().numpy().astype(int)
                crop = _crop_safe(frame, fbox)
                name, score = self.recognizer.identify(crop)

                if name != self.locked_identity:
                    continue

                self.raw_bbox = list(fbox)
                self._init_csrt_tracker(frame, self.raw_bbox)
                self.current_state = self.STATE_TRACKING_FACE
                self.logger.log("FACE_REACQUIRED", name=name, score=f"{score:.2f}")
                print(f"[FSM] Face reacquired for '{name}' -> TRACKING_FACE")
                return True

        return False

    # -----------------------------------------------------------------
    # State: LOST
    # -----------------------------------------------------------------
    def _step_lost(self, frame):
        """
        Target has been missing longer than the grace period. Servo holds
        its last commanded position (has_target=False tells ServoController
        not to move). Keep scanning for the registered face so we can
        recover automatically without operator intervention.
        """
        if self._try_reacquire_face(frame):
            self._mark_target_seen()
            self.logger.log("TARGET_REACQUIRED", name=self.locked_identity)
            return

        # Stay in LOST; output already reflects "no target" (set in
        # _handle_target_missing right before transitioning here).

    # -----------------------------------------------------------------
    # Shared "target missing" handling with debounce
    # -----------------------------------------------------------------
    def _handle_target_missing(self, reason):
        should_declare_lost = self._mark_target_missing_and_check_lost()

        if not should_declare_lost:
            # Still within the grace period: don't touch has_target/target
            # output -- the dashboard/servo simply keep using the last
            # known position for a moment, instead of snapping to "no
            # target" on a single dropped frame.
            return

        print(f"[FSM] {reason}, grace period elapsed -> LOST")
        self.logger.log("TARGET_LOST", name=self.locked_identity, reason=reason)
        self.current_state = self.STATE_LOST
        self.tracker = None
        self._clear_target(reset_kalman=True)

    # -----------------------------------------------------------------
    # Public accessors
    # -----------------------------------------------------------------
    def get_ai_result(self):
        """Return (has_target, raw_bbox, smoothed_bbox, target_center, state)."""
        with self._lock:
            return (
                self.has_target,
                self.raw_bbox,
                self.smoothed_bbox,
                self.target_center,
                self.current_state,
            )

    def get_locked_identity(self):
        return self.locked_identity

    def get_last_latency_ms(self):
        with self._lock:
            return self.last_step_latency_ms

    def reload_face_database(self):
        """Call this if a new person is registered while operation is running."""
        self.recognizer.load_database()

    def stop(self):
        self.running = False