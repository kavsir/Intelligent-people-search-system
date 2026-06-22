"""
AI tracking pipeline: a finite state machine that prioritizes face
tracking (cheap CSRT tracker, periodically re-validated by YOLO-face) and
falls back to person tracking (YOLO-person) when the face is lost.

States:
    SEARCHING       -> no target yet, scanning every frame
    TRACKING_FACE   -> locked onto a face, tracked cheaply via CSRT
    FALLBACK_PERSON -> face lost, tracking the person's body instead
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


class AIPipeline(threading.Thread):
    def __init__(self, camera_thread):
        super().__init__()
        self.camera_thread = camera_thread
        self.running = True

        # --- Load models ---
        print("[AI] Loading AI models...")
        self.device = config.get_torch_device()
        print(f"[AI] YOLO models running on {self.device.upper()}.")

        self.yolo_person = YOLO(config.YOLO_PERSON_MODEL_PATH)
        try:
            self.yolo_face = YOLO(config.YOLO_FACE_MODEL_PATH)
        except Exception:
            print("[AI] Face model not found, using person model as fallback.")
            self.yolo_face = self.yolo_person

        self.kalman = KalmanFilter2D()

        # --- Finite state machine ---
        self.STATE_SEARCHING = "SEARCHING"
        self.STATE_TRACKING_FACE = "TRACKING_FACE"
        self.STATE_FALLBACK_PERSON = "FALLBACK_PERSON"
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

        # --- Output state ---
        self.has_target = False
        self.raw_bbox = None
        self.smoothed_bbox = None
        self.target_center = None

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

    def run(self):
        print(f"[AI] FSM thread started. Initial state: {self.current_state}")

        while self.running:
            ret, frame = self.camera_thread.get_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            try:
                self._step(frame)
            except Exception as exc:
                # Never let a single bad frame kill the whole AI thread.
                print(f"[AI] Unexpected error during step, resetting to SEARCHING: {exc}")
                self.current_state = self.STATE_SEARCHING
                self.has_target = False
                self.raw_bbox = None
                self.smoothed_bbox = None
                self.target_center = None
                self.kalman.reset()

            time.sleep(0.01)

    def _step(self, frame):
        if self.current_state == self.STATE_SEARCHING:
            self._step_searching(frame)
        elif self.current_state == self.STATE_TRACKING_FACE:
            self._step_tracking_face(frame)
        elif self.current_state == self.STATE_FALLBACK_PERSON:
            self._step_fallback_person(frame)

    # -----------------------------------------------------------------
    # State: SEARCHING
    # -----------------------------------------------------------------
    def _step_searching(self, frame):
        face_results = self.yolo_face(frame, verbose=False, device=self.device)
        face_found = False

        for r in face_results:
            if len(r.boxes) > 0:
                box = r.boxes[0].xyxy[0].cpu().numpy().astype(int)
                self.raw_bbox = list(box)
                self.current_state = self.STATE_TRACKING_FACE
                self._init_csrt_tracker(frame, self.raw_bbox)
                face_found = True
                print("[FSM] Face detected -> TRACKING_FACE (CSRT engaged)")
                break

        if not face_found:
            person_results = self.yolo_person(frame, verbose=False, device=self.device)
            for r in person_results:
                person_boxes = [b for b in r.boxes if int(b.cls[0]) == 0]
                if len(person_boxes) > 0:
                    box = person_boxes[0].xyxy[0].cpu().numpy().astype(int)
                    self.raw_bbox = list(box)
                    self.current_state = self.STATE_FALLBACK_PERSON
                    self.kalman.reset()
                    print("[FSM] No face but person detected -> FALLBACK_PERSON")
                    break

        if self.current_state == self.STATE_SEARCHING:
            self.has_target = False
            self.raw_bbox = None
            self.smoothed_bbox = None
            self.target_center = None

    # -----------------------------------------------------------------
    # State: TRACKING_FACE
    # -----------------------------------------------------------------
    def _step_tracking_face(self, frame):
        self.validate_counter += 1

        success, tracker_box = self.tracker.update(frame)

        if not success:
            print("[FSM] CSRT lost the face.")
            self.current_state = self.STATE_FALLBACK_PERSON
            return

        tx, ty, tw, th = map(int, tracker_box)
        self.smoothed_bbox = [tx, ty, tx + tw, ty + th]

        raw_cx, raw_cy = tx + tw // 2, ty + th // 2
        kx, ky = self.kalman.predict_and_correct(raw_cx, raw_cy)

        self.has_target = True
        self.target_center = (kx, ky)

        if self.validate_counter >= self.VALIDATE_INTERVAL:
            self.validate_counter = 0
            face_results = self.yolo_face(frame, verbose=False, device=self.device)
            verified = False

            for r in face_results:
                if len(r.boxes) > 0:
                    box = r.boxes[0].xyxy[0].cpu().numpy().astype(int)
                    self.raw_bbox = list(box)
                    self._init_csrt_tracker(frame, self.raw_bbox)
                    verified = True
                    break

            if not verified:
                print("[FSM] CSRT drifted or user turned away (re-check found no face).")
                self.current_state = self.STATE_FALLBACK_PERSON

    # -----------------------------------------------------------------
    # State: FALLBACK_PERSON
    # -----------------------------------------------------------------
    def _step_fallback_person(self, frame):
        person_results = self.yolo_person(frame, verbose=False, device=self.device)
        person_found = False

        for r in person_results:
            person_boxes = [b for b in r.boxes if int(b.cls[0]) == 0]
            if len(person_boxes) == 0:
                continue

            px1, py1, px2, py2 = person_boxes[0].xyxy[0].cpu().numpy().astype(int)
            self.raw_bbox = [px1, py1, px2, py2]
            self.smoothed_bbox = self.raw_bbox
            person_found = True

            body_cx = px1 + (px2 - px1) // 2
            body_cy = py1 + (py2 - py1) // 4  # bias toward chest/head, not navel

            kx, ky = self.kalman.predict_and_correct(body_cx, body_cy)
            self.has_target = True
            self.target_center = (kx, ky)

            # Periodically re-check for a face within the person box, using a
            # frame counter rather than wall-clock time for a deterministic
            # 1-in-N check rate.
            self.fallback_face_check_counter += 1
            if self.fallback_face_check_counter >= self.FALLBACK_FACE_CHECK_INTERVAL:
                self.fallback_face_check_counter = 0
                face_results = self.yolo_face(frame, verbose=False, device=self.device)
                for fr in face_results:
                    if len(fr.boxes) > 0:
                        fbox = fr.boxes[0].xyxy[0].cpu().numpy().astype(int)
                        self.raw_bbox = list(fbox)
                        self._init_csrt_tracker(frame, self.raw_bbox)
                        self.current_state = self.STATE_TRACKING_FACE
                        print("[FSM] Face reacquired -> TRACKING_FACE")
                        break
            break

        if not person_found:
            print("[FSM] Lost target entirely -> SEARCHING")
            self.current_state = self.STATE_SEARCHING
            self.kalman.reset()

    def get_ai_result(self):
        """Return (has_target, raw_bbox, smoothed_bbox, target_center, state)."""
        return (
            self.has_target,
            self.raw_bbox,
            self.smoothed_bbox,
            self.target_center,
            self.current_state,
        )

    def stop(self):
        self.running = False
