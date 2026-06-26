"""
Web dashboard for the multi-camera operation system.

Runs on http://localhost:5001 and shows:
  - Two live MJPEG video feeds (one per room/camera), with the same
    bbox/state/FPS/latency overlays as the old cv2.imshow dashboard.
  - A 2D top-down floor map that highlights a room red whenever a target
    is currently detected in it.

This is a separate Flask app from app_registration.py (port 5000) so the
two can run side-by-side during development. They can be merged into one
app later once this is stable.

Run:
    python app_dashboard.py
Then open http://localhost:5001 in a browser. Ctrl+C in the terminal stops
the system (camera/AI/servo threads are stopped on shutdown).
"""

import os
import sys
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from operation.event_logger import EventLogger
from app_operation import RoomUnit, render_room_panel, draw_topdown_map, draw_event_feed

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "operation", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "operation", "static"),
)

# ---------------------------------------------------------------------------
# Shared state, built once at startup (see start_system() below).
# ---------------------------------------------------------------------------
shared_logger = None
rooms = []  # list[RoomUnit], same order as config.CAMERAS

# Tracks the latest "has_target" flag per room so the /api/room_status
# endpoint (polled by the browser for the 2D map) doesn't need to touch
# cv2 drawing code -- it's updated by the same loop that encodes frames.
_room_status_lock = threading.Lock()
_room_has_target = {}


def start_system():
    """Start one RoomUnit (camera + AI [+ servo]) per entry in config.CAMERAS.
    Called once, before the Flask server starts handling requests."""
    global shared_logger, rooms

    shared_logger = EventLogger()
    rooms = [RoomUnit(cam_cfg, shared_logger) for cam_cfg in config.CAMERAS]

    for room in rooms:
        room.start()
        _room_has_target[room.id] = False

    servo_rooms = [r.room_name for r in rooms if r.has_servo]
    print("\n--- WEB DASHBOARD SYSTEM RUNNING (multi-camera) ---")
    print(f"Rooms: {[r.room_name for r in rooms]}")
    print(f"Servo attached to: {servo_rooms or 'NONE'}")
    print(f"Event log: {shared_logger.log_path}")
    print("Open http://localhost:5001 in your browser.")
    print("Press Ctrl+C in this terminal to stop safely.\n")


def stop_system():
    print("\n[Main] Stopping all camera/AI/servo threads...")
    for room in rooms:
        room.stop()
    for room in rooms:
        room.join()
    if shared_logger:
        shared_logger.close()
    print("[Main] System shut down safely.")


# ---------------------------------------------------------------------------
# MJPEG streaming
# ---------------------------------------------------------------------------
def _generate_mjpeg(room):
    """Yield one multipart/x-mixed-replace JPEG frame at a time for a
    single room. If the camera has no signal, yields a placeholder frame
    with "NO SIGNAL" instead of stalling the HTTP response."""
    target_frame_time = 1.0 / config.TARGET_FPS

    while True:
        loop_start = time.time()
        ret, frame = room.camera_thread.get_frame()

        if not ret or frame is None:
            frame = cv2.cvtColor(
                cv2.UMat(config.FRAME_HEIGHT, config.FRAME_WIDTH, cv2.CV_8UC1).get(),
                cv2.COLOR_GRAY2BGR,
            )
            cv2.putText(
                frame,
                f"{room.room_name}: NO SIGNAL",
                (10, config.FRAME_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "Check camera power / Wi-Fi / IP in config.py",
                (10, config.FRAME_HEIGHT // 2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (180, 180, 180),
                1,
            )
            has_target = False
        else:
            fps_text = room.tick_fps()
            has_target = render_room_panel(frame, room, fps_text)
            draw_event_feed(frame, shared_logger.get_recent(6))

        with _room_status_lock:
            _room_has_target[room.id] = has_target

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )

        elapsed = time.time() - loop_start
        sleep_time = target_frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


@app.route("/video_feed/<room_id>")
def video_feed(room_id):
    room = next((r for r in rooms if r.id == room_id), None)
    if room is None:
        return f"Unknown room id: {room_id}", 404
    return Response(
        _generate_mjpeg(room), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------------------------
# Status API (polled by the browser to color the 2D floor map)
# ---------------------------------------------------------------------------
@app.route("/api/room_status")
def room_status():
    with _room_status_lock:
        statuses = dict(_room_has_target)

    result = []
    for room in rooms:
        has_target, _, _, _, current_state = room.ai_thread.get_ai_result()
        result.append(
            {
                "id": room.id,
                "room_name": room.room_name,
                "has_target": statuses.get(room.id, False),
                "state": current_state,
                "identity": room.ai_thread.get_locked_identity(),
                "has_servo": room.has_servo,
            }
        )
    return jsonify({"rooms": result})


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "dashboard.html",
        rooms=[{"id": r.id, "room_name": r.room_name, "has_servo": r.has_servo} for r in rooms],
    )


if __name__ == "__main__":
    start_system()
    try:
        # use_reloader=False is required: the reloader would otherwise spawn
        # a second process and start every camera/AI thread twice.
        app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False, threaded=True)
    finally:
        stop_system()