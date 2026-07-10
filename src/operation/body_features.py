"""
Long-term, CLOTHING-INVARIANT body-shape features for cross-session /
cross-clothing person continuity -- UPPER-BODY ONLY (vai, ngực, khuỷu
tay, cổ tay). Trước đây bản gốc dùng toàn thân (cần thấy cả đầu gối +
mắt cá chân), nhưng khi người đăng ký đứng gần camera (vừa bước vào
phòng) chân rất hay bị cắt khỏi khung hình hoặc bị bàn/ghế che khuất --
nên bản này CHỈ dựa vào NỬA THÂN TRÊN, thứ gần như luôn thấy được ở
khoảng cách đó.

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
        width vs hip width, upper-arm-to-torso proportion, etc. don't
        change day to day)

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

def upper_body_box(bbox, ratio=0.55):
    """
    Cắt bbox thân người ĐẦY ĐỦ (từ YOLO-person, đầu -> chân) xuống còn
    phần NỬA THÂN TRÊN (vai, ngực, khuỷu tay, cổ tay) -- bỏ hẳn phần
    hông dưới/chân, vì 2 lý do:
      1) Khi người đăng ký đứng đủ gần camera để "nửa thân trên" hiện rõ
         (vừa bước vào phòng), chân của họ rất hay bị cắt khỏi khung
         hình hoặc bị bàn/ghế che khuất.
      2) Vai + khuỷu tay + cổ tay đã đủ tạo một bộ tỉ lệ hình học riêng
         cho từng người mà không cần thấy chân.

    bbox: [x1, y1, x2, y2] -- bbox thân người đầy đủ (vd từ yolo_person).
    ratio: giữ lại `ratio` phần chiều cao TÍNH TỪ ĐỈNH bbox xuống (mặc
        định 55% -- đủ để luôn thấy khuỷu tay/cổ tay ở tư thế đứng
        thẳng bình thường, hiếm khi lấn xuống tới đầu gối).

    QUAN TRỌNG: dùng CÙNG MỘT ratio này ở cả lúc XÂY hồ sơ
    (_update_body_profiles) lẫn lúc SO KHỚP (_find_identity_by_body_shape)
    -- nếu không, "body_aspect_ratio" (crop height/width) sẽ không còn
    so sánh được giữa 2 lần đo.
    """
    x1, y1, x2, y2 = bbox
    new_y2 = y1 + int((y2 - y1) * ratio)
    return [int(x1), int(y1), int(x2), int(new_y2)]

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
        Run pose estimation on a body crop -- crop này PHẢI đã được cắt
        xuống nửa thân trên bằng upper_body_box() trước khi gọi hàm này
        (caller's trách nhiệm, xem ai_pipeline.py). Trả về dict 5 tỉ lệ
        không thứ nguyên, hoặc None nếu không thấy đủ landmark NỬA THÂN
        TRÊN (2 vai, 2 hông, 2 khuỷu tay, 2 cổ tay, đều đủ visibility) --
        một khung hình bị che một phần cho ra tỉ lệ không đáng tin nên bỏ
        qua thay vì làm nhiễu running average.

        Returns:
            {
              "shoulder_hip_ratio":     shoulder width / hip width,
              "shoulder_torso_ratio":   shoulder width / torso length,
              "upperarm_torso_ratio":   avg(vai->khuỷu tay) / torso length,
              "forearm_upperarm_ratio": avg(khuỷu tay->cổ tay) / avg(vai->khuỷu tay),
              "body_aspect_ratio":      crop height / crop width -- một
                                        proxy thô cho dáng gầy/đậm.
                                        Noisiest trong 5 tỉ lệ -- weight
                                        thấp nhất trong logic so khớp.
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
        l_el, r_el = pt(L.LEFT_ELBOW), pt(L.RIGHT_ELBOW)
        l_wr, r_wr = pt(L.LEFT_WRIST), pt(L.RIGHT_WRIST)

        required = (l_sh, r_sh, l_hip, r_hip, l_el, r_el, l_wr, r_wr)
        if any(p is None for p in required):
            return None  # thiếu landmark nửa thân trên frame này -- caller thử lại sau

        shoulder_width = _dist(l_sh, r_sh)
        hip_width = _dist(l_hip, r_hip)
        mid_shoulder = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
        mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        torso_len = _dist(mid_shoulder, mid_hip)
        upperarm_len = (_dist(l_sh, l_el) + _dist(r_sh, r_el)) / 2
        forearm_len = (_dist(l_el, l_wr) + _dist(r_el, r_wr)) / 2

        if torso_len < 1e-3 or hip_width < 1e-3 or upperarm_len < 1e-3:
            return None  # degenerate geometry (landmarks collapsed) -- skip

        return {
            "shoulder_hip_ratio": shoulder_width / hip_width,
            "shoulder_torso_ratio": shoulder_width / torso_len,
            "upperarm_torso_ratio": upperarm_len / torso_len,
            "forearm_upperarm_ratio": forearm_len / upperarm_len,
            "body_aspect_ratio": h / w,
        }

    def close(self):
        self._pose.close()