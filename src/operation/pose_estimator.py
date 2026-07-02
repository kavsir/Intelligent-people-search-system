"""
Pose estimation + rep counting for the two supported exercises: squat and
push-up. Uses MediaPipe Pose to extract body landmarks from a person's
bounding-box crop, classifies which movement is being performed from torso
orientation, and counts reps via a simple UP/DOWN angle state machine.

IMPORTANT: a `PoseEstimator` instance wraps a single `mediapipe.solutions
.pose.Pose` object, which is NOT thread-safe. Create exactly ONE
PoseEstimator per AIPipeline (i.e. per room/camera thread) and never share
it across threads.

Install: pip install mediapipe --break-system-packages
"""

import math
import time

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose


class _PersonExerciseState:
    """Per-person (by registered name) UP/DOWN phase + rep counters, scoped
    to a single PoseEstimator (= a single room)."""

    def __init__(self):
        self.squat_phase = "UP"
        self.pushup_phase = "UP"
        self.squat_reps = 0
        self.pushup_reps = 0
        self.last_update = time.time()


def _angle(a, b, c):
    """Angle ABC (at vertex b) in degrees, given (x, y) tuples."""
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    return ang


class PoseEstimator:
    """
    Per-room pose estimator. Call process(name, crop) once per frame for
    the currently-locked target; it returns which exercise is being
    performed (if any) and whether a rep was JUST completed on this frame.
    """

    # Angle thresholds (degrees). Tuned to be forgiving rather than strict
    # -- this is a heuristic classifier, not a trained action-recognition
    # model, so some slack avoids missed reps on imperfect form/camera
    # angles. Tighten if you want stricter form checking.
    PUSHUP_DOWN_ELBOW_ANGLE = 95
    PUSHUP_UP_ELBOW_ANGLE = 155
    SQUAT_DOWN_KNEE_ANGLE = 110
    SQUAT_UP_KNEE_ANGLE = 160
    # Torso-vs-vertical angle (degrees) above which we classify the
    # movement as push-up (body roughly horizontal) rather than squat
    # (body roughly upright).
    TORSO_HORIZONTAL_THRESHOLD = 55

    def __init__(self):
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # "lite" model -- keep CPU cost down, we
                                  # already run YOLO + InsightFace per frame
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._states = {}  # name -> _PersonExerciseState

    def _state_for(self, name):
        if name not in self._states:
            self._states[name] = _PersonExerciseState()
        return self._states[name]

    def process(self, name, crop_bgr):
        """
        Run pose estimation on one person's body crop and update their rep
        counters for this room.

        Returns:
            {"exercise": "squat"|"pushup"|None,
             "rep_completed": "squat"|"pushup"|None}
        `rep_completed` is set only on the exact frame a rep finishes.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return {"exercise": None, "rep_completed": None}

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return {"exercise": None, "rep_completed": None}

        lm = result.pose_landmarks.landmark
        h, w = crop_bgr.shape[:2]

        def pt(landmark):
            p = lm[landmark.value]
            return (p.x * w, p.y * h)

        L = mp_pose.PoseLandmark
        try:
            shoulder = pt(L.LEFT_SHOULDER)
            elbow = pt(L.LEFT_ELBOW)
            wrist = pt(L.LEFT_WRIST)
            hip = pt(L.LEFT_HIP)
            knee = pt(L.LEFT_KNEE)
            ankle = pt(L.LEFT_ANKLE)
        except (IndexError, AttributeError):
            return {"exercise": None, "rep_completed": None}

        # Torso orientation relative to vertical: near 0 deg = standing
        # upright (squat), near 90 deg = lying horizontal (push-up).
        torso_dx = hip[0] - shoulder[0]
        torso_dy = hip[1] - shoulder[1]
        torso_angle = math.degrees(math.atan2(abs(torso_dx), abs(torso_dy) + 1e-6))

        state = self._state_for(name)
        exercise = None
        rep_completed = None

        if torso_angle > self.TORSO_HORIZONTAL_THRESHOLD:
            exercise = "pushup"
            elbow_angle = _angle(shoulder, elbow, wrist)
            if elbow_angle < self.PUSHUP_DOWN_ELBOW_ANGLE:
                state.pushup_phase = "DOWN"
            elif elbow_angle > self.PUSHUP_UP_ELBOW_ANGLE and state.pushup_phase == "DOWN":
                state.pushup_phase = "UP"
                state.pushup_reps += 1
                rep_completed = "pushup"
        else:
            exercise = "squat"
            knee_angle = _angle(hip, knee, ankle)
            if knee_angle < self.SQUAT_DOWN_KNEE_ANGLE:
                state.squat_phase = "DOWN"
            elif knee_angle > self.SQUAT_UP_KNEE_ANGLE and state.squat_phase == "DOWN":
                state.squat_phase = "UP"
                state.squat_reps += 1
                rep_completed = "squat"

        state.last_update = time.time()
        return {"exercise": exercise, "rep_completed": rep_completed}

    def reset(self, name):
        """Drop a person's local UP/DOWN phase state (e.g. they were
        unassigned or just failed/passed and got re-assigned fresh)."""
        self._states.pop(name, None)