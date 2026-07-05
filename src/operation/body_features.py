"""
Long-term, CLOTHING-INVARIANT body-shape features for cross-session /
cross-clothing person continuity.

Design goal (why ratios, never colors):
    Người A mặc áo siêu nhân hôm nay, áo khác ngày mai -> vẫn phải khớp
    hồ sơ của A. Người B mặc ĐÚNG áo siêu nhân đó -> không được nhận nhầm
    thành A. The only way to satisfy both at once is to never look at
    clothing color/texture at all -- so every feature here is a purely
    geometric RATIO between skeleton segment lengths (MediaPipe Pose
    landmarks), which is:
      - invariant to distance from the camera (no absolute pixel length
        is ever stored, only one length divided by another)
      - invariant to clothing color/texture/pattern entirely (skeleton
        only)
      - roughly stable for the same person across sightings (shoulder
        width vs hip width, leg-to-torso proportion, etc. don't change
        day to day)

Honesty about limits: this is a soft-biometric SUPPORTING signal, not a
replacement for face recognition. At 320x240 from a single fixed
webcam with no depth sensing, skeleton-ratio re-identification is
noisier and far less discriminative than face embeddings -- it's used
here (see ai_pipeline.py's _update_body_profiles) to keep building a
per-person profile over time and to corroborate cross-room movement
events, NOT as a stand-alone identity source. See
face_database.body_recognition for the persisted profile.

Install: pip install mediapipe --break-system-packages   (already a
dependency of operation/pose_estimator.py)
"""

import math

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class BodyFeatureExtractor:
    """
    Wraps one mediapipe.solutions.pose.Pose instance. NOT thread-safe --
    create exactly ONE instance per AIPipeline (per room/camera thread),
    same rule as PoseEstimator, and never share across threads.
    """

    def __init__(self):
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # keep CPU cost down -- already running
                                 # YOLO-face + YOLO-person + InsightFace +
                                 # PoseEstimator's own Pose instance
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, body_crop_bgr):
        """
        Run pose estimation on a body crop and return a dict of 5
        dimensionless ratios, or None if a full skeleton (both shoulders,
        both hips, both knees, both ankles, all with decent visibility)
        wasn't found -- a partial/occluded view produces unreliable
        ratios, so we skip it rather than pollute the running average.

        Returns:
            {
              "shoulder_hip_ratio":   shoulder width / hip width,
              "torso_leg_ratio":      torso length / (thigh+shin) length,
              "thigh_shin_ratio":     thigh length / shin length,
              "shoulder_torso_ratio": shoulder width / torso length,
              "body_aspect_ratio":    crop height / crop width -- a crude
                                      thin/stocky proxy. Noisiest of the
                                      5 -- weight it lowest in any future
                                      matching logic built on top of this.
            }
        """
        if body_crop_bgr is None or body_crop_bgr.size == 0:
            return None

        rgb = cv2.cvtColor(body_crop_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks.landmark
        h, w = body_crop_bgr.shape[:2]

        def pt(landmark, min_visibility=0.5):
            p = lm[landmark.value]
            if p.visibility < min_visibility:
                return None
            return (p.x * w, p.y * h)

        L = mp_pose.PoseLandmark
        l_sh, r_sh = pt(L.LEFT_SHOULDER), pt(L.RIGHT_SHOULDER)
        l_hip, r_hip = pt(L.LEFT_HIP), pt(L.RIGHT_HIP)
        l_knee, r_knee = pt(L.LEFT_KNEE), pt(L.RIGHT_KNEE)
        l_ankle, r_ankle = pt(L.LEFT_ANKLE), pt(L.RIGHT_ANKLE)

        required = (l_sh, r_sh, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle)
        if any(p is None for p in required):
            return None  # partial skeleton this frame -- caller retries later

        shoulder_width = _dist(l_sh, r_sh)
        hip_width = _dist(l_hip, r_hip)
        mid_shoulder = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
        mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        torso_len = _dist(mid_shoulder, mid_hip)
        thigh_len = (_dist(l_hip, l_knee) + _dist(r_hip, r_knee)) / 2
        shin_len = (_dist(l_knee, l_ankle) + _dist(r_knee, r_ankle)) / 2
        leg_len = thigh_len + shin_len

        if torso_len < 1e-3 or leg_len < 1e-3 or shin_len < 1e-3 or hip_width < 1e-3:
            return None  # degenerate geometry (landmarks collapsed) -- skip

        return {
            "shoulder_hip_ratio": shoulder_width / hip_width,
            "torso_leg_ratio": torso_len / leg_len,
            "thigh_shin_ratio": thigh_len / shin_len,
            "shoulder_torso_ratio": shoulder_width / torso_len,
            "body_aspect_ratio": h / w,
        }

    def close(self):
        self._pose.close()