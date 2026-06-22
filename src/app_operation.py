"""
Operation entry point: runs the camera reader, AI tracking pipeline, and
servo controller as background threads, and displays a live debug
dashboard window.

Press 'q' on the dashboard window to stop the system safely.
"""

import os
import sys
import time

import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from operation.ai_pipeline import AIPipeline
from operation.camera_reader import CameraReader
from operation.servo_controller import ServoController


def main():
    # -----------------------------------------------------------------
    # 1. Start background threads
    # -----------------------------------------------------------------
    cam_thread = CameraReader()
    cam_thread.start()

    ai_thread = AIPipeline(cam_thread)
    ai_thread.start()

    servo_thread = ServoController(
        ai_thread,
        port=config.SERVO_PORT,
        baudrate=config.SERVO_BAUDRATE,
        frame_size=(config.FRAME_WIDTH, config.FRAME_HEIGHT),
    )
    servo_thread.start()

    print("\n--- OPERATION SYSTEM RUNNING ---")
    print(f"Target FPS: {config.TARGET_FPS}")
    print("Press 'q' on the dashboard window to stop safely.\n")

    ideal_frame_time = 1.0 / config.TARGET_FPS

    fps_start_time = time.time()
    fps_counter = 0
    fps_text = "FPS: 0"

    # -----------------------------------------------------------------
    # 2. Main dashboard loop
    # -----------------------------------------------------------------
    while True:
        frame_start_time = time.time()

        ret, frame = cam_thread.get_frame()

        if ret and frame is not None:
            h, w, _ = frame.shape
            cx, cy = w // 2, h // 2

            fps_counter += 1
            if (time.time() - fps_start_time) > 1.0:
                fps_text = f"FPS: {fps_counter}"
                fps_counter = 0
                fps_start_time = time.time()

            has_target, raw_bbox, smoothed_bbox, target_center, current_state = (
                ai_thread.get_ai_result()
            )
            current_pan, current_tilt = servo_thread.get_current_angles()

            # Crosshair at the center of the frame (the target the robot aims for)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (255, 255, 255), 1)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (255, 255, 255), 1)

            if has_target:
                if current_state == "TRACKING_FACE":
                    status_color = (0, 255, 0)  # green
                    status_text = "STATE: TRACKING FACE (CSRT)"
                elif current_state == "FALLBACK_PERSON":
                    status_color = (0, 165, 255)  # orange
                    status_text = "STATE: FALLBACK PERSON (YOLO)"
                else:
                    status_color = (0, 255, 255)  # yellow
                    status_text = f"STATE: {current_state}"

                if raw_bbox:
                    rx1, ry1, rx2, ry2 = raw_bbox
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 1)

                if smoothed_bbox:
                    sx1, sy1, sx2, sy2 = smoothed_bbox
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), status_color, 2)
                    cv2.putText(
                        frame,
                        status_text,
                        (sx1, sy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        status_color,
                        2,
                    )

                if target_center:
                    tx, ty = target_center
                    cv2.circle(frame, (tx, ty), 4, (0, 255, 255), -1)
                    cv2.line(frame, (tx, ty), (cx, cy), (0, 255, 255), 1)
            else:
                cv2.putText(
                    frame,
                    "STATE: SEARCHING TARGET...",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"Servo Pan: {current_pan} | Tilt: {current_tilt}",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow("ESP32-S3 AIoT Pan-Tilt Control Dashboard", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Main] Shutdown signal received...")
            break

        elapsed_time = time.time() - frame_start_time
        sleep_time = ideal_frame_time - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    # -----------------------------------------------------------------
    # 3. Clean shutdown
    # -----------------------------------------------------------------
    print("[Main] Stopping servo, AI, and camera threads...")
    servo_thread.stop()
    ai_thread.stop()
    cam_thread.stop()

    servo_thread.join()
    ai_thread.join()
    cam_thread.join()

    cv2.destroyAllWindows()
    print("[Main] System shut down safely.")


if __name__ == "__main__":
    main()
