# main_vh.py
import cv2
import time
from van_hanh.camera_reader import CameraReader
from van_hanh.ai_pipeline import AIPipeline
from van_hanh.servo_controller import ServoController

def main():
    # =========================================================================
    # 1. KHỞI ĐỘNG TẤT CẢ CÁC LUỒNG NỀN (3 THREADS)
    # =========================================================================
    # Luồng 1: Đọc Camera từ ESP32-S3 ngầm
    cam_thread = CameraReader()
    cam_thread.start()

    # Luồng 2: Xử lý AI, Máy trạng thái nâng cao (FSM) & Bộ lọc Kalman ngầm
    ai_thread = AIPipeline(cam_thread)
    ai_thread.start()

    # Luồng 3: Tính toán sai số góc quay và gửi lệnh xuống mạch qua Serial
    # LƯU Ý: Thay đổi 'COM8' bằng cổng COM thực tế của bạn (Ví dụ: 'COM5', '/dev/ttyUSB0')
    servo_thread = ServoController(ai_thread, port='COM8', baudrate=115200, frame_size=(320, 240))
    servo_thread.start()

    print("\n--- HỆ THỐNG VẬN HÀNH TOÀN DIỆN (NÂNG CẤP FSM BƯỚC 5) ---")
    print("Khóa cố định hiệu năng ở mức: 30 FPS")
    print("Nhấn phím 'q' tại màn hình hiển thị để DỪNG hệ thống an toàn.\n")

    # Cấu hình quản lý FPS cứng ở mức 30
    TARGET_FPS = 30
    IDEAL_FRAME_TIME = 1.0 / TARGET_FPS

    # Các biến theo dõi FPS thực tế
    fps_start_time = time.time()
    fps_counter = 0
    fps_text = "FPS: 0"

    # =========================================================================
    # 2. VÒNG LẶP DASHBOARD CHÍNH
    # =========================================================================
    while True:
        frame_start_time = time.time()

        # Lấy khung hình gốc từ luồng camera
        ret, frame = cam_thread.get_frame()

        if ret and frame is not None:
            h, w, _ = frame.shape
            cx, cy = w // 2, h // 2 # Tâm của khung hình camera

            # Tính toán hiển thị FPS thực tế
            fps_counter += 1
            if (time.time() - fps_start_time) > 1.0:
                fps_text = f"FPS: {fps_counter}"
                fps_counter = 0
                fps_start_time = time.time()

            # --- ĐIỂM CẬP NHẬT BƯỚC 5: Trích xuất thêm trạng thái current_state (5 tham số) ---
            has_target, raw_bbox, smoothed_bbox, target_center, current_state = ai_thread.get_ai_result()
            
            # Trích xuất góc servo hiện tại từ Luồng 3 để giám sát dữ liệu đám mây/màn hình
            current_pan, current_tilt = servo_thread.get_current_angles()

            # 🎯 Vẽ hồng tâm (Crosshair) cố định chính giữa màn hình (Mục tiêu cần hướng về đây)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (255, 255, 255), 1)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (255, 255, 255), 1)

            if has_target:
                # --- THIẾT LẬP MÀU SẮC & TIÊU ĐỀ THEO TRẠNG THÁI MÁY ---
                if current_state == "TRACKING_FACE":
                    status_color = (0, 255, 0)   # 🟢 Xanh lá cây - Đang bám nét mặt cực mượt (CSRT)
                    status_text = "STATE: TRACKING FACE (CSRT)"
                elif current_state == "FALLBACK_PERSON":
                    status_color = (0, 165, 255) # 🟠 Màu cam - Mất mặt, tự động dự phòng bám thân (YOLO)
                    status_text = "STATE: FALLBACK PERSON (YOLO)"
                else:
                    status_color = (0, 255, 255) # 🟡 Màu vàng dự phòng chung
                    status_text = f"STATE: {current_state}"

                # 🔴 Vẽ khung thô phát hiện tức thời từ mô hình YOLO (Nếu có dữ liệu)
                if raw_bbox:
                    rx1, ry1, rx2, ry2 = raw_bbox
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 1)

                # 🟢/🟠 Vẽ khung tracking mượt mà kết hợp Text trạng thái linh hoạt
                if smoothed_bbox:
                    sx1, sy1, sx2, sy2 = smoothed_bbox
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), status_color, 2)
                    cv2.putText(frame, status_text, (sx1, sy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

                # 🟡 Vẽ chấm tròn tâm mục tiêu di động và đường chỉ hướng kéo servo
                if target_center:
                    tx, ty = target_center
                    cv2.circle(frame, (tx, ty), 4, (0, 255, 255), -1)
                    cv2.line(frame, (tx, ty), (cx, cy), (0, 255, 255), 1)
            else:
                # Trạng thái chưa tìm thấy bất kỳ đối tượng nào
                cv2.putText(frame, "STATE: SEARCHING TARGET...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 📊 In thông số hệ thống lên góc màn hình
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Servo Pan: {current_pan} | Tilt: {current_tilt}", (10, h - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Hiển thị lên màn hình kiểm thử tổng hợp
            cv2.imshow("ESP32-S3 AIoT Pan-Tilt Control Dashboard", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Main] Đang phát tín hiệu tắt hệ thống...")
            break

        # --- BỘ ĐIỀU TIẾT KHÓA 30 FPS ---
        elapsed_time = time.time() - frame_start_time
        sleep_time = IDEAL_FRAME_TIME - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    # =========================================================================
    # 3. GIẢI PHÓNG TÀI NGUYÊN AN TOÀN TRÁNH RÁC LUỒNG
    # =========================================================================
    print("[Main] Đang dừng luồng Servo, AI và Camera...")
    servo_thread.stop()
    ai_thread.stop()    
    cam_thread.stop()
    
    servo_thread.join()
    ai_thread.join()
    cam_thread.join()
    
    cv2.destroyAllWindows()
    print("[Main] Toàn bộ hệ thống kết thúc vận hành an toàn.")

if __name__ == "__main__":
    main()