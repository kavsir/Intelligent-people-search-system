"""
PID-based pan/tilt servo controller. Reads the latest target center from
the AI pipeline and sends pan/tilt angle commands over serial.
"""

import os
import sys
import threading
import time

import serial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PID:
    def __init__(self, kp, ki, kd, max_i_clip=10):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_i_clip = max_i_clip  # anti-windup clamp on the integral term

        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 0.001  # avoid division by zero

        p_out = self.kp * error

        self.integral += error * dt
        self.integral = max(-self.max_i_clip, min(self.max_i_clip, self.integral))
        i_out = self.ki * self.integral

        derivative = (error - self.last_error) / dt
        d_out = self.kd * derivative

        self.last_error = error
        self.last_time = now

        return p_out + i_out + d_out

    def reset(self):
        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()


class ServoController(threading.Thread):
    def __init__(
        self,
        ai_pipeline,
        port=None,
        baudrate=None,
        frame_size=(320, 240),
    ):
        super().__init__()
        self.ai_pipeline = ai_pipeline
        self.running = True

        self.frame_width, self.frame_height = frame_size
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2

        self.pan_angle = 90.0
        self.tilt_angle = 90.0

        self.deadzone = 10  # ignore small offsets to keep the system stable

        # Pan axis (left/right)
        self.pid_x = PID(kp=0.04, ki=0.01, kd=0.002, max_i_clip=15)
        # Tilt axis (up/down)
        self.pid_y = PID(kp=0.04, ki=0.01, kd=0.002, max_i_clip=15)

        port = port or config.SERVO_PORT
        baudrate = baudrate or config.SERVO_BAUDRATE

        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            time.sleep(2)  # allow the board to finish booting
            print(f"[Servo PID] Connected on port {port}")
        except Exception:
            self.ser = None
            print(f"[Servo PID] Serial port {port} not found, running in simulation mode.")

    def run(self):
        print("[Servo PID] Control loop started.")
        while self.running:
            # get_ai_result() returns 5 values:
            # (has_target, raw_bbox, smoothed_bbox, target_center, state)
            has_target, _, _, target_center, _ = self.ai_pipeline.get_ai_result()

            if has_target and target_center:
                tx, ty = target_center

                error_x = tx - self.center_x
                error_y = ty - self.center_y

                if abs(error_x) > self.deadzone:
                    delta_pan = self.pid_x.compute(error_x)
                    self.pan_angle -= delta_pan
                else:
                    self.pid_x.reset()

                if abs(error_y) > self.deadzone:
                    delta_tilt = self.pid_y.compute(error_y)
                    self.tilt_angle += delta_tilt
                else:
                    self.pid_y.reset()

                # Clamp to safe mechanical limits (avoid stalling the MG996R).
                self.pan_angle = max(10.0, min(170.0, self.pan_angle))
                self.tilt_angle = max(10.0, min(170.0, self.tilt_angle))

                cmd = f"P:{int(self.pan_angle)},T:{int(self.tilt_angle)}\n"

                if self.ser and self.ser.is_open:
                    self.ser.write(cmd.encode("utf-8"))
            else:
                # No target: reset PID state to avoid stale integral windup.
                self.pid_x.reset()
                self.pid_y.reset()

            time.sleep(0.05)  # ~20 Hz update rate

    def get_current_angles(self):
        return int(self.pan_angle), int(self.tilt_angle)

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.write(b"P:90,T:90\n")
            time.sleep(0.2)
            self.ser.close()
        print("[Servo PID] Control loop stopped.")
