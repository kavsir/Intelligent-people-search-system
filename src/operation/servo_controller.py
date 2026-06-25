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
from operation.event_logger import EventLogger


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


def _clamp_step(delta, max_step):
    """Clamp a single-frame angle change to +/- max_step degrees, so a
    sudden large PID output (e.g. right after switching tracking targets)
    can't jerk the servo -- it ramps instead."""
    return max(-max_step, min(max_step, delta))


class ServoController(threading.Thread):
    def __init__(
        self,
        ai_pipeline,
        port=None,
        baudrate=None,
        frame_size=(320, 240),
        event_logger=None,
    ):
        super().__init__()
        self.ai_pipeline = ai_pipeline
        self.running = True
        self.daemon = True

        self.logger = event_logger or EventLogger()

        self.frame_width, self.frame_height = frame_size
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2

        self._angle_lock = threading.Lock()
        self.pan_angle = 90.0
        self.tilt_angle = 90.0

        self.deadzone = config.SERVO_DEADZONE_PX
        self.max_degree_step = config.SERVO_MAX_DEGREE_STEP

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
        last_logged_pan, last_logged_tilt = self.pan_angle, self.tilt_angle

        while self.running:
            # get_ai_result() returns 5 values:
            # (has_target, raw_bbox, smoothed_bbox, target_center, state)
            has_target, _, _, target_center, _ = self.ai_pipeline.get_ai_result()

            if has_target and target_center:
                tx, ty = target_center

                error_x = tx - self.center_x
                error_y = ty - self.center_y

                with self._angle_lock:
                    if abs(error_x) > self.deadzone:
                        delta_pan = _clamp_step(
                            self.pid_x.compute(error_x), self.max_degree_step
                        )
                        self.pan_angle -= delta_pan
                    else:
                        self.pid_x.reset()

                    if abs(error_y) > self.deadzone:
                        delta_tilt = _clamp_step(
                            self.pid_y.compute(error_y), self.max_degree_step
                        )
                        self.tilt_angle += delta_tilt
                    else:
                        self.pid_y.reset()

                    # Clamp to safe mechanical limits (avoid stalling the MG996R).
                    self.pan_angle = max(10.0, min(170.0, self.pan_angle))
                    self.tilt_angle = max(10.0, min(170.0, self.tilt_angle))

                    pan_to_send = int(self.pan_angle)
                    tilt_to_send = int(self.tilt_angle)

                cmd = f"P:{pan_to_send},T:{tilt_to_send}\n"

                if self.ser and self.ser.is_open:
                    self.ser.write(cmd.encode("utf-8"))

                # Only log when the commanded angle actually changes by a
                # visible amount, so the event log isn't flooded at ~20Hz.
                if (
                    abs(pan_to_send - last_logged_pan) >= 1
                    or abs(tilt_to_send - last_logged_tilt) >= 1
                ):
                    self.logger.log(
                        "SERVO_ANGLE_UPDATED", pan=pan_to_send, tilt=tilt_to_send
                    )
                    last_logged_pan, last_logged_tilt = pan_to_send, tilt_to_send
            else:
                # No target: reset PID state to avoid stale integral windup.
                # Servo deliberately holds its last commanded angle rather
                # than re-centering, per the assignment's "stop or hold
                # position" safety requirement.
                self.pid_x.reset()
                self.pid_y.reset()

            time.sleep(0.05)  # ~20 Hz update rate

    def get_current_angles(self):
        with self._angle_lock:
            return int(self.pan_angle), int(self.tilt_angle)

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.write(b"P:90,T:90\n")
            time.sleep(0.2)
            self.ser.close()
        self.logger.log("SERVO_RESET_TO_CENTER", pan=90, tilt=90)
        print("[Servo PID] Control loop stopped.")