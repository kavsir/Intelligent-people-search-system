# van_hanh/servo_controller.py
import threading
import time
import serial

class PID:
    def __init__(self, kp, ki, kd, max_i_clip=10):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_i_clip = max_i_clip # Anti-windup: Giới hạn tối đa của khâu tích phân để tránh tràn số
        
        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 0.001 # Tránh lỗi chia cho 0
            
        # 1. Khâu tỷ lệ (P)
        p_out = self.kp * error
        
        # 2. Khâu tích phân (I) kèm bộ khử tràn (Anti-windup)
        self.integral += error * dt
        self.integral = max(-self.max_i_clip, min(self.max_i_clip, self.integral))
        i_out = self.ki * self.integral
        
        # 3. Khâu vi phân (D)
        derivative = (error - self.last_error) / dt
        d_out = self.kd * derivative
        
        # Cập nhật trạng thái cho chu kỳ sau
        self.last_error = error
        self.last_time = now
        
        # Tổng đầu ra điều chỉnh (Delta Angle)
        return p_out + i_out + d_out

    def reset(self):
        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()


class ServoController(threading.Thread):
    def __init__(self, ai_pipeline, port='COM3', baudrate=115200, frame_size=(320, 240)):
        super().__init__()
        self.ai_pipeline = ai_pipeline
        self.running = True
        
        # Cấu hình kích thước khung hình
        self.frame_width, self.frame_height = frame_size
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        # Góc hiện tại của hệ thống servo (bắt đầu ở trung tâm)
        self.pan_angle = 90.0
        self.tilt_angle = 90.0
        
        # Vùng chết (Deadzone) - nếu lệch dưới mức này thì bỏ qua (giúp ổn định hệ thống)
        self.deadzone = 10 

        # =========================================================================
        # KHỞI TẠO BỘ ĐIỀU KHIỂN PID CHO 2 TRỤC (Hãy tinh chỉnh bộ thông số này tại đây)
        # =========================================================================
        # Trục X (Trái - Phải)
        self.pid_x = PID(kp=0.04, ki=0.01, kd=0.002, max_i_clip=15)
        # Trục Y (Lên - Xuống)
        self.pid_y = PID(kp=0.04, ki=0.01, kd=0.002, max_i_clip=15)

        # Khởi tạo cổng kết nối Serial phần cứng
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            time.sleep(2) # Đợi mạch khởi động ổn định kết nối
            print(f"[Servo PID] Kết nối thành công cổng {port}")
        except Exception as e:
            self.ser = None
            print(f"[Servo PID] ⚠️ Giả lập Serial được kích hoạt (Không tìm thấy {port}).")

    def run(self):
        print("[Servo PID] Luồng điều khiển PID bắt đầu vận hành...")
        while self.running:
            # Lấy kết quả từ pipeline AI
            has_target, _, _, target_center = self.ai_pipeline.get_ai_result()
            
            if has_target and target_center:
                tx, ty = target_center
                
                # Tính toán sai số vị trí thực tế so với hồng tâm camera
                error_x = tx - self.center_x
                error_y = ty - self.center_y
                
                # --- Xử lý trục ngang PAN ---
                if abs(error_x) > self.deadzone:
                    # Tính toán Delta góc cần thay đổi từ bộ PID
                    delta_pan = self.pid_x.compute(error_x)
                    # Áp dụng hiệu chỉnh (Dấu - hoặc + tùy thuộc vào chiều đặt phần cứng camera)
                    self.pan_angle -= delta_pan
                else:
                    self.pid_x.reset() # Reset tích phân khi đã vào vùng an toàn

                # --- Xử lý trục đứng TILT ---
                if abs(error_y) > self.deadzone:
                    delta_tilt = self.pid_y.compute(error_y)
                    self.tilt_angle += delta_tilt
                else:
                    self.pid_y.reset()

                # Giới hạn góc phần cứng (tránh quá giới hạn cơ khí gây gãy/kẹt MG996R)
                self.pan_angle = max(10.0, min(170.0, self.pan_angle))
                self.tilt_angle = max(10.0, min(170.0, self.tilt_angle))
                
                # Đóng gói và gửi lệnh dạng văn bản chuẩn xuống Arduino/ESP32
                cmd = f"P:{int(self.pan_angle)},T:{int(self.tilt_angle)}\n"
                
                if self.ser and self.ser.is_open:
                    self.ser.write(cmd.encode('utf-8'))
            else:
                # Nếu mất mục tiêu, reset trạng thái bộ PID để tránh hiện tượng cộng dồn tích phân rác
                self.pid_x.reset()
                self.pid_y.reset()
                    
            # Tần suất gửi xung điều khiển phần cứng (50ms ~ 20Hz là hoàn hảo cho phản hồi của servo)
            time.sleep(0.05) 

    def get_current_angles(self):
        return int(self.pan_angle), int(self.tilt_angle)

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            # Đưa hệ thống về vị trí an toàn 90, 90 trước khi đóng kết nối
            self.ser.write(b"P:90,T:90\n")
            time.sleep(0.2)
            self.ser.close()
        print("[Servo PID] Đã tắt luồng điều khiển.")