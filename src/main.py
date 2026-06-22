from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from dang_ky.yolo_register_face import get_face_embedding, get_face_landmarks, save_face_data
# IMPORT FILE TIỀN XỬ LÝ BACKGROUND VỪA TẠO
from dang_ky.image_preprocessor import process_person_background 

import os

app = Flask(
    __name__,
    template_folder=os.path.join(
        os.path.dirname(__file__),
        "dang_ky",
        "templates"
    ),
    static_folder=os.path.join(
        os.path.dirname(__file__),
        "dang_ky",
        "static"
    )
)

@app.route('/')
def index():
    return render_template('register.html')

@app.route('/get_landmarks', methods=['POST'])
def get_landmarks():
    data = request.get_json()
    image_data = data['image']
    header, encoded = image_data.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status': 'error', 'message': 'Không đọc được ảnh'})
    
    landmarks = get_face_landmarks(img)
    if landmarks is None:
        return jsonify({'status': 'error', 'message': 'Không tìm thấy khuôn mặt hợp lệ'})
    return jsonify({'status': 'success', 'landmarks': landmarks.tolist()})

# API Đăng ký từ Tab Upload ảnh tĩnh
@app.route('/register_from_image', methods=['POST'])
def register_from_image():
    data = request.get_json()
    image_data = data['image']
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'status': 'error', 'message': 'Thiếu tên người đăng ký'})
    
    header, encoded = image_data.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status': 'error', 'message': 'Không đọc được ảnh'})
    
    emb = get_face_embedding(img)
    if emb is None:
        return jsonify({'status': 'error', 'message': 'Mô hình không nhận diện được mặt'})
    
    # 1. Lưu dữ liệu gốc vào face_db
    save_face_data(name, [emb], [img])
    
    # 2. GỌI HOẠT ĐỘNG TIỀN XỬ LÝ ẢNH SANG THƯ MỤC RIÊNG
    process_person_background(name)
    
    return jsonify({'status': 'success', 'message': f'Đã đăng ký và tạo dữ liệu tiền xử lý cho [{name}]'})

# API Đăng ký từ Tab chụp 5 góc độ Webcam
@app.route('/register_final', methods=['POST'])
def register_final():
    data = request.get_json()
    name = data.get('name', '').strip()
    images = data.get('images', []) 
    if not name:
        return jsonify({'status': 'error', 'message': 'Thiếu tên người đăng ký'})
    if not images or len(images) < 5:
        return jsonify({'status': 'error', 'message': 'Chưa chụp đủ dữ liệu 5 góc'})
    
    embeddings = []
    valid_images = []
    
    for img_base64 in images:
        header, encoded = img_base64.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue
            
        emb = get_face_embedding(img)
        if emb is not None:
            embeddings.append(emb)
            valid_images.append(img)
    
    if not embeddings:
        return jsonify({'status': 'error', 'message': 'Không trích xuất được đặc trưng hợp lệ'})
    
    # 1. Lưu dữ liệu gốc 5 góc vào face_db
    save_face_data(name, embeddings, valid_images)
    
    # 2. GỌI HOẠT ĐỘNG TIỀN XỬ LÝ ĐỒNG BỘ 5 GÓC ẢNH SANG THƯ MỤC TIEN_SU_LY
    process_person_background(name)
    
    return jsonify({'status': 'success', 'message': f'Hệ thống đã xử lý xong toàn bộ 5 góc ảnh nền đen sắc nét của [{name}].'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)