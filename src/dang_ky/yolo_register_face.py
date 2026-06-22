import os
import numpy as np
import cv2
from ultralytics import YOLO

# Khởi tạo mô hình YOLOv8-Face
try:
    yolo_model = YOLO('yolov8n-face.pt') 
except Exception:
    yolo_model = YOLO('yolov8n.pt') 

def _find_best_face_index(result):
    """
    HÀM TRỢ GIÚP: Duyệt qua tất cả các mặt phát hiện được,
    tính điểm = Diện tích * Độ tự tin để tìm ra mặt TO NHẤT và RÕ NHẤT.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None
        
    best_idx = 0
    max_score = -1
    
    for i, box in enumerate(result.boxes):
        # Lấy tọa độ phẳng x1, y1, x2, y2
        xyxy = box.xyxy[0].cpu().numpy()
        # Lấy độ tự tin rõ nét của khuôn mặt (0.0 -> 1.0)
        conf = box.conf[0].cpu().item() if box.conf is not None else 0.5
        
        # Tính kích thước diện tích hộp bao quanh mặt
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        area = w * h
        
        # Công thức định vị mục tiêu ưu tiên chính diện/gần cam nhất
        score = area * conf
        
        if score > max_score:
            max_score = score
            best_idx = i
            
    return best_idx

def get_face_landmarks(image_bgr):
    """Sử dụng YOLO để tìm mặt tối ưu nhất và trả về tọa độ landmark vẽ lưới"""
    if image_bgr is None:
        return None
        
    results = yolo_model(image_bgr, verbose=False)
    for result in results:
        best_idx = _find_best_face_index(result)
        if best_idx is None:
            continue
            
        # Nếu có điểm landmark (keypoints) từ YOLOv8-Face
        if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints) > 0:
            kp = result.keypoints.xy[best_idx].cpu().numpy()
            if len(kp) > 0: 
                return kp
                
        # Phương án dự phòng tạo landmark giả lập từ bounding box tốt nhất
        box = result.boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = box[:4]
        return np.array([
            [(x1+x2)/2 - 20, (y1+y2)/2 - 20], [(x1+x2)/2 + 20, (y1+y2)/2 - 20],
            [(x1+x2)/2, (y1+y2)/2],
            [(x1+x2)/2 - 15, (y1+y2)/2 + 20], [(x1+x2)/2 + 15, (y1+y2)/2 + 20]
        ])
    return None

def get_face_embedding(image_bgr):
    """Trích xuất vector đặc trưng toán học của khuôn mặt RÕ NHẤT / GẦN NHẤT"""
    if image_bgr is None:
        return None
        
    results = yolo_model(image_bgr, verbose=False)
    for result in results:
        best_idx = _find_best_face_index(result)
        if best_idx is None:
            continue
            
        box = result.boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, box[:4])
        
        face_crop = image_bgr[y1:y2, x1:x2]
        if face_crop.size == 0: 
            continue
            
        face_resized = cv2.resize(face_crop, (112, 112))
        hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
        
    return None

def save_face_data(name, embedding_list, image_list):
    """Tạo folder ngoài src và lưu trữ cấu trúc dữ liệu vật lý"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    person_dir = os.path.join(BASE_DIR, 'face_db', name)
    os.makedirs(person_dir, exist_ok=True)
    
    embeddings = np.array(embedding_list)
    npy_path = os.path.join(person_dir, 'embedding.npy')
    np.save(npy_path, embeddings)
    
    for index, img_bgr in enumerate(image_list):
        img_name = f"goc_{index + 1}.jpg" if len(image_list) > 1 else "anh_dai_dien.jpg"
        img_path = os.path.join(person_dir, img_name)
        cv2.imwrite(img_path, img_bgr)
        
    print(f"[YOLO CHỌN LỌC] Đã lưu mục tiêu rõ nhất của {name} vào: {person_dir}")
    return person_dir