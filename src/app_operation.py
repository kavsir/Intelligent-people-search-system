"""
Operation entry point: runs one camera reader + AI recognition pipeline per
physical ESP32-CAM (one per room, see config.CAMERAS). Displays a live
debug dashboard with all camera feeds side-by-side and a simple 2D top-down
floor map that lights up red for whichever room currently has a detected
target.

Cameras are fixed-position -- there is no pan/tilt servo. Each room's
pipeline only detects and recognizes whoever is in its own frame; nothing
moves the camera to follow a person.

Architecture note: each ESP32-CAM streams MJPEG directly to this machine.
The ESP32-S3 is a separate coordination gateway (HTTP/MQTT) that tells each
ESP32-CAM when to stream/snapshot/sleep -- it does not relay video, so it
has no role in this file.

Press 'q' on the dashboard window to stop the system safely.
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from operation.ai_pipeline import AIPipeline
from operation.camera_reader import CameraReader
from operation.event_logger import EventLogger


# ---------------------------------------------------------------------------
# Per-room runtime bundle
# ---------------------------------------------------------------------------
class RoomUnit:
    """Holds every thread/object that belongs to one physical camera/room."""

    def __init__(self, cam_config, shared_logger):
        self.id = cam_config["id"]
        self.room_name = cam_config["room_name"]

        self.camera_thread = CameraReader(url=cam_config["url"], name=self.room_name)
        self.ai_thread = AIPipeline(
            self.camera_thread, event_logger=shared_logger, room_name=self.room_name
        )

        # Per-room FPS counter state (each camera can run at a slightly
        # different real FPS, so these are tracked independently).
        self._fps_start_time = time.time()
        self._fps_counter = 0
        self._fps_text = "FPS: 0"

    def start(self):
        self.camera_thread.start()
        self.ai_thread.start()

    def stop(self):
        self.ai_thread.stop()
        self.camera_thread.stop()

    def join(self):
        self.ai_thread.join()
        self.camera_thread.join()

    def tick_fps(self):
        self._fps_counter += 1
        if (time.time() - self._fps_start_time) > 1.0:
            self._fps_text = f"FPS: {self._fps_counter}"
            self._fps_counter = 0
            self._fps_start_time = time.time()
        return self._fps_text


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def validate_camera_config(cameras):
    """Catch config mistakes before any thread is started."""
    errors = []

    ids = [c["id"] for c in cameras]
    if len(ids) != len(set(ids)):
        errors.append(f"Duplicate camera 'id' values in config.CAMERAS: {ids}")

    urls = [c["url"] for c in cameras]
    if len(urls) != len(set(urls)):
        errors.append(
            f"Duplicate camera 'url' values in config.CAMERAS: {urls} -- "
            "two rooms are pointing at the same stream."
        )

    if errors:
        print("\n[CONFIG ERROR] Refusing to start -- fix config.CAMERAS first:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_event_feed(frame, events, origin=(10, 90), line_height=16, max_lines=6):
    """Draw the most recent log events as small text in the corner."""
    x, y = origin
    for ts, event, details in events[-max_lines:]:
        text = f"{ts} {event} {details}"
        cv2.putText(
            frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1
        )
        y += line_height


def render_room_panel(frame, room, fps_text):
    """
    Draw overlays: crosshair, bbox for ALL registered faces (yellow),
    and a thicker bbox for the main target (green/red).
    """
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    # Lấy thông tin mục tiêu chính
    has_target, raw_bbox, smoothed_bbox, target_center, current_state = (
        room.ai_thread.get_ai_result()
    )
    # Lấy danh sách tất cả khuôn mặt đã đăng ký
    all_faces = room.ai_thread.get_all_faces()

    latency_ms = room.ai_thread.get_last_latency_ms()
    identity = room.ai_thread.get_locked_identity()

    # Vẽ tên phòng
    cv2.putText(
        frame, room.room_name, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    # Crosshair
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (255, 255, 255), 1)

    # ===== Vẽ TẤT CẢ các khuôn mặt đã đăng ký (trừ target chính) =====
    for face in all_faces:
        name = face["name"]
        bbox = face["bbox"]  # [x1,y1,x2,y2]
        score = face["score"]
        # Nếu là target chính, bỏ qua vì sẽ vẽ đậm sau
        if name == identity and has_target:
            continue
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)
        label = f"{name} ({score:.2f})"
        cv2.putText(
            frame, label, (bbox[0], bbox[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
        )

    # ===== Vẽ mục tiêu chính (nổi bật) =====
    if has_target:
        if current_state == "TRACKING":
            status_color = (0, 255, 0)  # green
            status_text = f"TRACKING ({identity or '?'})"
        else:
            status_color = (0, 165, 255)  # orange
            status_text = f"{current_state}"

        if raw_bbox:
            rx1, ry1, rx2, ry2 = raw_bbox
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 1)

        if smoothed_bbox:
            sx1, sy1, sx2, sy2 = smoothed_bbox
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), status_color, 3)
            cv2.putText(
                frame,
                status_text,
                (sx1, max(sy1 - 8, 35)),
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
        color = (0, 0, 255) if current_state == "LOST" else (0, 255, 255)
        cv2.putText(
            frame,
            f"STATE: {current_state}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    # --- Overlay info: FPS, latency, (u, v) ---
    cv2.putText(frame, fps_text, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(
        frame,
        f"Latency: {latency_ms:.1f} ms",
        (110, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
    )

    uv_text = f"(u,v)=({target_center[0]},{target_center[1]})" if target_center else "(u,v)=N/A"
    cv2.putText(
        frame, uv_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
    )

    return has_target


def draw_topdown_map(room_has_target, room_names, cell_size=140, gap=12):
    """
    Draw a simple 2D top-down floor map: one square per room, side by side.
    A room's square turns red while a target is currently detected in it,
    otherwise it stays neutral gray. room_has_target and room_names must be
    the same length and in the same order.
    """
    n = len(room_names)
    margin = 20
    width = margin * 2 + n * cell_size + (n - 1) * gap
    height = margin * 2 + cell_size + 30  # extra space for the title

    canvas = np.full((height, width, 3), (40, 40, 40), dtype=np.uint8)

    cv2.putText(
        canvas, "Floor Map (Top-Down)", (margin, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
    )

    for i, (room_name, has_target) in enumerate(zip(room_names, room_has_target)):
        x1 = margin + i * (cell_size + gap)
        y1 = margin + 22
        x2 = x1 + cell_size
        y2 = y1 + cell_size

        if has_target:
            fill_color = (0, 0, 200)     # red = target detected here
            border_color = (0, 0, 255)
        else:
            fill_color = (70, 70, 70)    # neutral = empty/no detection
            border_color = (120, 120, 120)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill_color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 2)

        text_size = cv2.getTextSize(room_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x1 + (cell_size - text_size[0]) // 2
        text_y = y1 + cell_size // 2
        cv2.putText(
            canvas, room_name, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return canvas


def tile_horizontally(frames):
    """Stack frames side-by-side, padding heights so they line up evenly."""
    max_h = max(f.shape[0] for f in frames)
    resized = []
    for f in frames:
        if f.shape[0] != max_h:
            scale = max_h / f.shape[0]
            f = cv2.resize(f, (int(f.shape[1] * scale), max_h))
        resized.append(f)
    return cv2.hconcat(resized)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # -----------------------------------------------------------------
    # 0. Validate config BEFORE starting any thread.
    # -----------------------------------------------------------------
    validate_camera_config(config.CAMERAS)

    # -----------------------------------------------------------------
    # 1. Start background threads: one RoomUnit (camera + AI) per entry
    #    in config.CAMERAS.
    # -----------------------------------------------------------------
    shared_logger = EventLogger()

    rooms = [RoomUnit(cam_cfg, shared_logger) for cam_cfg in config.CAMERAS]

    for room in rooms:
        room.start()

    print("\n--- OPERATION SYSTEM RUNNING (multi-camera, fixed cameras) ---")
    print(f"Rooms: {[r.room_name for r in rooms]}")
    print(f"Target FPS: {config.TARGET_FPS}")
    print(f"Event log: {shared_logger.log_path}")
    print("Press 'q' on the dashboard window to stop safely.\n")

    ideal_frame_time = 1.0 / config.TARGET_FPS

    # -----------------------------------------------------------------
    # 2. Main dashboard loop
    # -----------------------------------------------------------------
    while True:
        frame_start_time = time.time()

        panels = []
        room_has_target = []

        for room in rooms:
            ret, frame = room.camera_thread.get_frame()

            if not ret or frame is None:
                # Camera not ready yet / disconnected: show a placeholder
                # panel instead of skipping it, so the layout doesn't jump
                # around and the missing room is obvious on the dashboard.
                placeholder = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder, f"{room.room_name}: NO SIGNAL", (10, config.FRAME_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                )
                panels.append(placeholder)
                room_has_target.append(False)
                continue

            fps_text = room.tick_fps()
            has_target = render_room_panel(frame, room, fps_text)
            draw_event_feed(frame, shared_logger.get_recent(6))

            panels.append(frame)
            room_has_target.append(has_target)

        if panels:
            combined = tile_horizontally(panels)
            cv2.imshow("AIoT Multi-Camera Recognition Dashboard", combined)

            topdown = draw_topdown_map(room_has_target, [r.room_name for r in rooms])
            cv2.imshow("Floor Map (Top-Down)", topdown)

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
    print("[Main] Stopping all camera/AI threads...")
    for room in rooms:
        room.stop()
    for room in rooms:
        room.join()

    shared_logger.close()
    cv2.destroyAllWindows()
    print("[Main] System shut down safely.")


if __name__ == "__main__":
    main()