import os
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

# Khởi tạo InsightFace sử dụng mô hình buffalo_sc
app = FaceAnalysis(name='buffalo_sc')
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1: Chạy trên CPU

def get_face_embedding(image_bgr):
    """Trả về embedding (512D) của khuôn mặt đầu tiên, hoặc None nếu không có."""
    faces = app.get(image_bgr)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding

def get_face_landmarks(image_bgr):
    """Trả về mảng (106, 2) các landmark khuôn mặt phục vụ hiển thị vẽ khung."""
    faces = app.get(image_bgr)
    if len(faces) == 0:
        return None
    return faces[0].landmark_2d_106  # (106, 2)

def save_embedding(name, embedding_list):
    """Lưu tập hợp ma trận danh sách embedding vào thư mục face_db dưới dạng .npy"""
    os.makedirs('face_db', exist_ok=True)
    embeddings = np.array(embedding_list)  # Kích thước ma trận (N, 512)
    path = os.path.join('face_db', f'{name}.npy')
    np.save(path, embeddings)
    return path