# van_hanh/ai_pipeline.py
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Giả lập bộ lọc Kalman từ Bước 2 của bạn để code không bị lỗi
class KalmanFilter2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2, 0)
        self.kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
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
        
        # --- TẢI CÁC MÔ HÌNH AI ---
        print("[AI] Đang tải các mô hình AI...")
        # Sử dụng YOLOv8n tiêu chuẩn (Class 0 là Person) làm Fallback bám thân người
        self.yolo_person = YOLO("yolov8n.pt") 
        # Sử dụng mô hình YOLO Face của bạn để detect mặt
        try:
            self.yolo_face = YOLO("yolov8n-face.pt")
        except:
            self.yolo_face = self.yolo_person # Fallback nếu thiếu file
            
        # Khởi tạo bộ lọc Kalman (từ Bước 2)
        self.kalman = KalmanFilter2D()

        # --- ĐỊNH NGHĨA MÁY TRẠNG THÁI (FSM) ---
        self.STATE_SEARCHING = "SEARCHING"
        self.STATE_TRACKING_FACE = "TRACKING_FACE"
        self.STATE_FALLBACK_PERSON = "FALLBACK_PERSON"
        self.current_state = self.STATE_SEARCHING

        # --- KHỞI TẠO BỘ TRACKER TRUYỀN THỐNG ---
        self.tracker = None
        self.validate_counter = 0
        self.VALIDATE_INTERVAL = 20 # Cứ 20 frame dùng CSRT thì bật AI check lại 1 lần xem có bị trôi không

        # --- BIẾN ĐẦU RA KẾT QUẢ ---
        self.has_target = False
        self.raw_bbox = None        # Khung đỏ (AI)
        self.smoothed_bbox = None   # Khung xanh (Kalman / Tracker)
        self.target_center = None   # Chấm vàng gửi xuống Servo

    def _init_csrt_tracker(self, frame, bbox):
        """Khởi tạo hoặc tái thiết lập OpenCV CSRT Tracker"""
        # OpenCV mới dùng cv2.TrackerCSRT.create(), bản cũ hoặc opencv-contrib dùng cv2.TrackerCSRT_create()
        try:
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.TrackerCSRT.create()
            
        # Ép kiểu dữ liệu bbox về dạng chuẩn tuple (x, y, w, h)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        self.tracker.init(frame, (x1, y1, w, h))
        self.validate_counter = 0

    def run(self):
        print(f"[AI] Luồng FSM Máy trạng thái bắt đầu hoạt động. Trạng thái gốc: {self.current_state}")
        
        while self.running:
            ret, frame = self.camera_thread.get_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # =========================================================================
            # TRẠNG THÁI 1: SEARCHING - TÌM KIẾM MỤC TIÊU BAN ĐẦU
            # =========================================================================
            if self.current_state == self.STATE_SEARCHING:
                # Ưu tiên số 1: Quét tìm Khuôn mặt
                face_results = self.yolo_face(frame, verbose=False)
                face_found = False
                
                for r in face_results:
                    if len(r.boxes) > 0:
                        # Lấy khuôn mặt có độ tự tin cao nhất đầu tiên
                        box = r.boxes[0].xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = box
                        
                        self.raw_bbox = [x1, y1, x2, y2]
                        self.current_state = self.STATE_TRACKING_FACE
                        self._init_csrt_tracker(frame, self.raw_bbox)
                        face_found = True
                        print("[FSM] 🎯 Đã phát hiện Mặt -> Chuyển sang: TRACKING_FACE (CSRT kích hoạt)")
                        break
                
                # Ưu tiên số 2: Nếu không thấy mặt, quét tìm Thân người luôn để neo giữ góc máy
                if not face_found:
                    person_results = self.yolo_person(frame, verbose=False)
                    for r in person_results:
                        # Lọc lấy class 0 (Person)
                        person_boxes = [b for b in r.boxes if int(b.cls[0]) == 0]
                        if len(person_boxes) > 0:
                            box = person_boxes[0].xyxy[0].cpu().numpy().astype(int)
                            self.raw_bbox = list(box)
                            self.current_state = self.STATE_FALLBACK_PERSON
                            self.kalman.reset()
                            print("[FSM] 🚶 Không thấy mặt nhưng thấy Dáng người -> Chuyển sang: FALLBACK_PERSON")
                            break
                            
                if self.current_state == self.STATE_SEARCHING:
                    # Hoàn toàn không thấy gì
                    self.has_target = False
                    self.raw_bbox = None
                    self.smoothed_bbox = None
                    self.target_center = None

            # =========================================================================
            # TRẠNG THÁI 2: TRACKING_FACE - BÁM MẶT SIÊU NHẸ BẰNG CSRT TRACKER
            # =========================================================================
            elif self.current_state == self.STATE_TRACKING_FACE:
                self.validate_counter += 1
                
                # Cập nhật tọa độ bằng Tracker (Không tốn CPU chạy Deep Learning)
                success, tracker_box = self.tracker.update(frame)
                
                if success:
                    tx, ty, tw, th = map(int, tracker_box)
                    self.smoothed_bbox = [tx, ty, tx + tw, ty + th]
                    
                    # Tính toán tâm mục tiêu và đẩy qua Kalman làm mượt
                    raw_cx, raw_cy = tx + tw // 2, ty + th // 2
                    kx, ky = self.kalman.predict_and_correct(raw_cx, raw_cy)
                    
                    self.has_target = True
                    self.target_center = (kx, ky)
                    
                    # --- ĐỊNH KỲ KIỂM CHỨNG (VALIDATION CHỐNG TRÔI) ---
                    if self.validate_counter >= self.VALIDATE_INTERVAL:
                        self.validate_counter = 0
                        # Chạy AI mặt cắt một vùng nhỏ quanh tracker (để tăng tốc) hoặc chạy cả frame
                        face_results = self.yolo_face(frame, verbose=False)
                        verified = False
                        for r in face_results:
                            if len(r.boxes) > 0:
                                # Nếu AI vẫn xác nhận có mặt trong frame, cập nhật lại tọa độ gốc cho tracker chuẩn hóa
                                box = r.boxes[0].xyxy[0].cpu().numpy().astype(int)
                                self.raw_bbox = list(box)
                                # Tái tạo lại tracker bằng tọa độ AI mới nhất để triệt tiêu sai số tích lũy
                                self._init_csrt_tracker(frame, self.raw_bbox)
                                verified = True
                                break
                        if not verified:
                            print("[FSM] ⚠️ CSRT bị trôi hoặc người dùng quay đi (AI check lại không thấy mặt).")
                            self.current_state = self.STATE_FALLBACK_PERSON
                else:
                    # CSRT thất bại (Mất dấu đột ngột, che khuất hoàn toàn)
                    print("[FSM] ❌ CSRT Tracker mất dấu mặt.")
                    self.current_state = self.STATE_FALLBACK_PERSON

            # =========================================================================
            # TRẠNG THÁI 3: FALLBACK_PERSON - CƠ CHẾ DỰ PHÒNG BÁM THÂN NGƯỜI
            # =========================================================================
            elif self.current_state == self.STATE_FALLBACK_PERSON:
                # Chạy YOLOv8n tìm thân người
                person_results = self.yolo_person(frame, verbose=False)
                person_found = False
                
                for r in person_results:
                    person_boxes = [b for b in r.boxes if int(b.cls[0]) == 0]
                    if len(person_boxes) > 0:
                        px1, py1, px2, py2 = person_boxes[0].xyxy[0].cpu().numpy().astype(int)
                        self.raw_bbox = [px1, py1, px2, py2]
                        self.smoothed_bbox = self.raw_bbox
                        person_found = True
                        
                        # Lấy tâm thân người (hoặc dịch lên 1/4 phía trên thân người để hướng camera gần vùng đầu hơn)
                        body_cx = px1 + (px2 - px1) // 2
                        body_cy = py1 + (py2 - py1) // 4 # Hướng về phía ngực/đầu thay vì rốn
                        
                        kx, ky = self.kalman.predict_and_correct(body_cx, body_cy)
                        self.has_target = True
                        self.target_center = (kx, ky)
                        
                        # [CHIẾN LƯỢC QUAY LẠI] Quét tìm lại mặt bên trong vùng thân người này
                        # Để tiết kiệm tài nguyên, chỉ quét mặt mỗi 3 frame một lần khi đang fallback
                        if int(time.time() * 100) % 3 == 0:
                            face_results = self.yolo_face(frame, verbose=False)
                            for fr in face_results:
                                if len(fr.boxes) > 0:
                                    fbox = fr.boxes[0].xyxy[0].cpu().numpy().astype(int)
                                    self.raw_bbox = list(fbox)
                                    # Người dùng đã quay mặt lại! Khởi động lại CSRT ngay lập tức
                                    self._init_csrt_tracker(frame, self.raw_bbox)
                                    self.current_state = self.STATE_TRACKING_FACE
                                    print("[FSM] 🎉 Người dùng đã quay mặt lại! Tái kích hoạt TRACKING_FACE.")
                                    break
                        break
                        
                if not person_found:
                    # Mất dấu hoàn toàn cả thân lẫn mặt
                    print("[FSM] ❌ Mất dấu toàn bộ mục tiêu. Quay về trạng thái SEARCHING.")
                    self.current_state = self.STATE_SEARCHING
                    self.kalman.reset()

            # Giữ nhịp độ luồng AI đồng bộ
            time.sleep(0.01)

    def get_ai_result(self):
        """Trả dữ liệu ra cho Luồng Servo và Luồng Dashboard chính"""
        return self.has_target, self.raw_bbox, self.smoothed_bbox, self.target_center, self.current_state

    def stop(self):
        self.running = False