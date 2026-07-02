"""
Pose estimation + rep counting for the two supported exercises (squat,
push-up) AND general behavior classification (Đứng/stand, Di chuyển/move,
Nhảy/jump, Giơ tay/raise_hand, Nằm/lie). Uses MediaPipe Pose to extract
body landmarks from a person's bounding-box crop.

IMPORTANT: a `PoseEstimator` instance wraps a single `mediapipe.solutions
.pose.Pose` object, which is NOT thread-safe. Create exactly ONE
PoseEstimator per AIPipeline (i.e. per room/camera thread) and never share
it across threads.

--------------------------------------------------------------------------
Behavior vs exercise, and why bbox_center matters
--------------------------------------------------------------------------
squat/pushup reps are detected from JOINT ANGLES inside the crop (elbow,
knee), which works fine regardless of where the crop itself sits in the
full camera frame.

"Nhảy" (jump) and "Di chuyển" (move), however, are about the person's
POSITION changing over time. If the crop AIPipeline passes in is a
tracked bounding box that re-centers on the person every frame, the
person's landmarks stay roughly still *inside the crop* even while they
are jumping or walking across the room -- the box moves with them. So
jump/move detection needs the bbox's center in FULL-FRAME coordinates
(the same `target_center` app_operation.py already draws the crosshair
from), not crop-local landmarks.

process() therefore takes an optional `bbox_center=(x, y)` in full-frame
pixel coordinates. If it's not supplied, jump/move detection is simply
skipped (stand/lie/raise_hand/squat/pushup still work, since those only
depend on body shape, not absolute position) -- so this stays a
non-breaking addition until AIPipeline is updated to pass it through.

Install: pip install mediapipe --break-system-packages
"""

import math
import time
from collections import deque

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# ---------------------------------------------------------------------------
# Behavior/posture tuning knobs
# ---------------------------------------------------------------------------
# How many recent full-frame bbox-center samples to keep for jump/move
# detection. At ~10-15 pipeline steps/sec per room this is roughly a
# half-second window.
POSTURE_HISTORY_LEN = 10

# Jump: bbox-center Y must rise (decrease, since image Y grows downward) by
# at least this fraction of FRAME_HEIGHT from the window's baseline, and
# then come back down near baseline within the same window -- i.e. an
# up-then-land trajectory rather than just walking to a higher spot.
JUMP_RISE_RATIO = 0.08
JUMP_LANDING_TOLERANCE_RATIO = 0.03
# Cooldown so one jump doesn't get counted multiple times while airborne.
JUMP_COOLDOWN_SEC = 0.6

# Move: bbox-center horizontal range across the history window, as a
# fraction of FRAME_WIDTH, above which we call it "moving" rather than
# standing still (small in-place sway shouldn't count).
MOVE_DISPLACEMENT_RATIO = 0.12

# Torso-vs-vertical angle (degrees) above which the body is roughly
# horizontal. Shared with the existing pushup classifier below; a
# horizontal torso is classified as "lie" UNLESS an active push-up
# elbow motion is also happening this frame (see process()).
LYING_TORSO_ANGLE = 60

# Hand is "raised" once the wrist is above the shoulder (smaller image Y)
# by at least this fraction of the crop height. A margin (rather than
# wrist.y < shoulder.y) avoids flicker right at the shoulder line.
HAND_RAISE_MARGIN_RATIO = 0.05

# Consecutive frames a NEW posture candidate must persist before we
# confirm the transition and emit a behavior event -- avoids counting a
# "move" or "jump" event on every single noisy frame.
POSTURE_DEBOUNCE_FRAMES = 5

# "jump" used to fire the instant the window-based rise/land condition was
# true for a SINGLE frame -- one noisy bbox_center sample was enough to
# trigger it. Now requires the condition to hold for this many consecutive
# process() calls before actually emitting the event.
JUMP_CONFIRM_FRAMES = 2

# Same idea for hand-raise: require the wrist-above-shoulder condition to
# persist for this many consecutive frames before confirming, so a single
# noisy MediaPipe landmark near the shoulder line doesn't flicker the
# event on and off.
HAND_RAISE_DEBOUNCE_FRAMES = 3


def _angle(a, b, c):
    """Angle ABC (at vertex b) in degrees, given (x, y) tuples."""
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    return ang


class _PersonExerciseState:
    """Per-person (by registered name) state, scoped to a single
    PoseEstimator (= a single room). Covers both exercise rep counting
    (squat/pushup) and general behavior/posture classification."""

    def __init__(self):
        # --- exercise rep counting (existing) ---
        self.squat_phase = "UP"
        self.pushup_phase = "UP"
        self.squat_reps = 0
        self.pushup_reps = 0

        # --- posture (Đứng/Di chuyển/Nhảy/Nằm) ---
        self.posture = "stand"          # confirmed current posture
        self.pending_posture = None     # candidate posture being debounced
        self.pending_posture_frames = 0
        self.bbox_history = deque(maxlen=POSTURE_HISTORY_LEN)  # (t, x, y)
        self.last_jump_time = 0.0
        self.jump_pending_frames = 0

        # --- hand raise (Giơ tay) ---
        self.hand_raised = False
        self.pending_hand_state = None
        self.pending_hand_frames = 0

        self.last_update = time.time()


class PoseEstimator:
    """
    Per-room pose estimator. Call process(name, crop) once per frame for
    the currently-locked/recognized target; it returns which exercise is
    being performed (if any), whether a rep JUST completed, the person's
    current posture, and a list of any behavior events newly confirmed
    on this exact frame (feed these straight into
    behavior_manager.behavior_manager.record_many(name, events)).
    """

    PUSHUP_DOWN_ELBOW_ANGLE = 95
    PUSHUP_UP_ELBOW_ANGLE = 155
    SQUAT_DOWN_KNEE_ANGLE = 110
    SQUAT_UP_KNEE_ANGLE = 160
    TORSO_HORIZONTAL_THRESHOLD = 55

    def __init__(self, frame_width=None, frame_height=None):
        """
        frame_width/frame_height: full-camera-frame dimensions, used only
        to turn bbox_center pixel deltas into the *_RATIO thresholds above.
        Defaults to config.FRAME_WIDTH/FRAME_HEIGHT if not given.
        """
        if frame_width is None or frame_height is None:
            import config
            frame_width = frame_width or config.FRAME_WIDTH
            frame_height = frame_height or config.FRAME_HEIGHT
        self.frame_width = frame_width
        self.frame_height = frame_height

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

    # ------------------------------------------------------------------
    # Posture classification helpers
    # ------------------------------------------------------------------
    def _classify_motion(self, state, bbox_center, now):
        """Returns 'jump', 'move', or None based on recent bbox_center
        history in full-frame coordinates. None means "no motion signal
        strong enough to call it jump/move" -- caller falls back to
        stand/lie based on body shape instead."""
        if bbox_center is None:
            return None

        state.bbox_history.append((now, bbox_center[0], bbox_center[1]))
        if len(state.bbox_history) < POSTURE_HISTORY_LEN:
            return None  # not enough history yet

        ys = [p[2] for p in state.bbox_history]
        xs = [p[1] for p in state.bbox_history]

        # Average a small window at each end instead of comparing single
        # raw samples -- a lone noisy detection (a few px of YOLO/Kalman
        # jitter) can no longer masquerade as "baseline" or "current".
        edge = max(1, POSTURE_HISTORY_LEN // 4)
        baseline_y = sum(ys[:edge]) / edge
        current_y = sum(ys[-edge:]) / edge
        min_y = min(ys)

        rise = baseline_y - min_y  # positive = rose upward at some point
        landed_back = abs(current_y - baseline_y) <= JUMP_LANDING_TOLERANCE_RATIO * self.frame_height
        jump_cooldown_ok = (now - state.last_jump_time) > JUMP_COOLDOWN_SEC

        jump_condition = (
            rise > JUMP_RISE_RATIO * self.frame_height
            and landed_back
            and jump_cooldown_ok
        )

        # Require the condition to hold for a couple of consecutive frames
        # before actually emitting -- a single-frame coincidence (e.g. a
        # small jitter spike right as the window slides) no longer counts
        # as a full jump.
        if jump_condition:
            state.jump_pending_frames += 1
        else:
            state.jump_pending_frames = 0

        if state.jump_pending_frames >= JUMP_CONFIRM_FRAMES:
            state.last_jump_time = now
            state.jump_pending_frames = 0
            return "jump"

        x_range = max(xs) - min(xs)
        if x_range > MOVE_DISPLACEMENT_RATIO * self.frame_width:
            return "move"

        return None

    def _update_posture(self, state, candidate):
        """Debounced posture state machine. Returns the confirmed new
        posture name if a transition was just confirmed this frame, else
        None. 'jump' is momentary (never "held"), so it's always reported
        immediately without changing the underlying stand/move/lie state."""
        if candidate == "jump":
            return "jump"

        if candidate == state.posture:
            state.pending_posture = None
            state.pending_posture_frames = 0
            return None

        if candidate == state.pending_posture:
            state.pending_posture_frames += 1
        else:
            state.pending_posture = candidate
            state.pending_posture_frames = 1

        if state.pending_posture_frames >= POSTURE_DEBOUNCE_FRAMES:
            state.posture = candidate
            state.pending_posture = None
            state.pending_posture_frames = 0
            return candidate

        return None

    def _update_hand_raise(self, state, shoulder, wrist, crop_h):
        """Returns 'raise_hand' the moment a raise is confirmed (transition
        not-raised -> raised, held for HAND_RAISE_DEBOUNCE_FRAMES
        consecutive frames); returns None otherwise, including while the
        hand stays raised or after it's lowered again. Debounced the same
        way posture is, so a single noisy MediaPipe landmark sample right
        at the shoulder line doesn't fire (or flicker) the event."""
        is_raised = (shoulder[1] - wrist[1]) > HAND_RAISE_MARGIN_RATIO * crop_h

        if is_raised == state.hand_raised:
            state.pending_hand_state = None
            state.pending_hand_frames = 0
            return None

        if is_raised == state.pending_hand_state:
            state.pending_hand_frames += 1
        else:
            state.pending_hand_state = is_raised
            state.pending_hand_frames = 1

        if state.pending_hand_frames < HAND_RAISE_DEBOUNCE_FRAMES:
            return None

        state.hand_raised = is_raised
        state.pending_hand_state = None
        state.pending_hand_frames = 0

        return "raise_hand" if is_raised else None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def process(self, name, crop_bgr, bbox_center=None):
        """
        Run pose estimation on one person's body crop, update their rep
        counters, and classify their current behavior for this room.

        bbox_center: (x, y) center of this person's bounding box in
            FULL CAMERA FRAME pixel coordinates this frame (e.g. AIPipeline's
            `target_center`). Optional -- pass it to enable jump/move
            detection; omit it and those two are simply never reported.

        Returns:
            {
              "exercise": "squat"|"pushup"|None,
              "rep_completed": "squat"|"pushup"|None,
              "posture": "stand"|"move"|"jump"|"lie",
              "behavior_events": [...],  # 0+ of BEHAVIORS, newly confirmed
                                          # THIS frame -- feed straight into
                                          # behavior_manager.record_many()
            }
        """
        empty = {"exercise": None, "rep_completed": None, "posture": None, "behavior_events": []}
        if crop_bgr is None or crop_bgr.size == 0:
            return empty

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return empty

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
            return empty

        state = self._state_for(name)
        now = time.time()
        behavior_events = []

        # Torso orientation relative to vertical: near 0 deg = standing
        # upright, near 90 deg = lying/horizontal.
        torso_dx = hip[0] - shoulder[0]
        torso_dy = hip[1] - shoulder[1]
        torso_angle = math.degrees(math.atan2(abs(torso_dx), abs(torso_dy) + 1e-6))

        # ------------------------------------------------------------
        # 1) Exercise rep counting (squat / pushup) -- unchanged logic.
        # ------------------------------------------------------------
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
                behavior_events.append("pushup")
        else:
            exercise = "squat"
            knee_angle = _angle(hip, knee, ankle)
            if knee_angle < self.SQUAT_DOWN_KNEE_ANGLE:
                state.squat_phase = "DOWN"
            elif knee_angle > self.SQUAT_UP_KNEE_ANGLE and state.squat_phase == "DOWN":
                state.squat_phase = "UP"
                state.squat_reps += 1
                rep_completed = "squat"
                behavior_events.append("squat")

        # ------------------------------------------------------------
        # 2) Posture (Đứng / Di chuyển / Nhảy / Nằm).
        #    "lie" only applies when the person is horizontal AND not
        #    actively mid-pushup-rep-motion (phase == "DOWN" means we
        #    just saw them push down/up, i.e. genuinely exercising).
        # ------------------------------------------------------------
        motion_candidate = self._classify_motion(state, bbox_center, now)

        if motion_candidate == "jump":
            behavior_events.append("jump")
            posture_candidate = state.posture  # jump doesn't replace stand/move/lie
        elif motion_candidate == "move":
            posture_candidate = "move"
        elif torso_angle > LYING_TORSO_ANGLE and exercise != "pushup":
            posture_candidate = "lie"
        else:
            posture_candidate = "stand"

        confirmed = self._update_posture(state, posture_candidate)
        if confirmed:
            behavior_events.append(confirmed)

        # ------------------------------------------------------------
        # 3) Giơ tay (raise hand) -- independent overlay event.
        # ------------------------------------------------------------
        hand_event = self._update_hand_raise(state, shoulder, wrist, h)
        if hand_event:
            behavior_events.append(hand_event)

        state.last_update = now

        return {
            "exercise": exercise,
            "rep_completed": rep_completed,
            "posture": state.posture,
            "behavior_events": behavior_events,
        }

    def reset(self, name):
        """Drop a person's local state (e.g. they were unassigned or just
        failed/passed and got re-assigned fresh, or were unregistered)."""
        self._states.pop(name, None)