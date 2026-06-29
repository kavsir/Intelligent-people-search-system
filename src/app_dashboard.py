"""
Web dashboard for the multi-camera operation system.

Runs on http://localhost:5001 and shows:
  - Two live MJPEG video feeds (one per room/camera), with bbox/state/FPS/
    latency overlays. Cameras are fixed-position (no pan/tilt servo).
  - A 2D top-down floor map that highlights a room red whenever a target
    is currently detected in it.

This is a separate Flask app from app_registration.py (port 5000) so the
two can run side-by-side during development. They can be merged into one
app later once this is stable.

Run:
    python app_dashboard.py
Then open http://localhost:5001 in a browser. Ctrl+C in the terminal stops
the system (camera/AI threads are stopped on shutdown).
"""

import os
import sys
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template, request

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from operation.event_logger import EventLogger
from app_operation import (
    RoomUnit,
    render_room_panel,
    draw_topdown_map,
    draw_event_feed,
    validate_camera_config,
)


# ---------------------------------------------------------------------------
# Inferred-presence engine
# ---------------------------------------------------------------------------
# Tracks which no-cam rooms are currently "suspected" to contain the target,
# based on where the target was last seen by a real camera.
#
# Data structure: dict  room_id -> {"since": float, "identity": str|None}
# "since" is the time.time() when the inference was first triggered.
# ---------------------------------------------------------------------------
class InferenceEngine:
    """
    When a camera loses its target (state transitions to LOST), this engine
    highlights the neighboring no-cam rooms as "inferred presence".

    Logic (option B from spec):
      - Only no-cam neighbors are highlighted (if a neighboring room has a
        cam and that cam doesn't see the target, the cam itself is the source
        of truth – no inference needed there).
      - Auto-clears after config.INFERRED_PRESENCE_TIMEOUT_SEC seconds.
      - Also cleared immediately when any camera picks up the target again.
      - Manual reset via reset() (called by /api/reset_inference).
    """

    def __init__(self, floor_plan, timeout_sec):
        # Build lookup maps from floor plan config
        self._room_cfg = {r["id"]: r for r in floor_plan}
        self._cam_rooms = {r["id"] for r in floor_plan if r["cam_id"] is not None}
        self._timeout = timeout_sec  # 0 = never auto-clear

        self._lock = threading.Lock()
        # room_id -> {"since": float, "identity": str|None}
        self._inferred = {}

    def update(self, cam_room_statuses):
        """
        Called every polling cycle with the latest camera results.
        cam_room_statuses: list of dicts with keys: room_id, has_target,
                           identity, state.
        """
        now = time.time()

        with self._lock:
            # 1. Find rooms where a cam currently HAS a target → clear any
            #    inferred state for all rooms (target is confirmed somewhere).
            confirmed_ids = {s["room_id"] for s in cam_room_statuses if s["has_target"]}
            if confirmed_ids:
                self._inferred.clear()
                return

            # 2. No cam has the target. For each cam room that just lost its
            #    target (state == LOST), infer presence in its no-cam neighbors.
            for status in cam_room_statuses:
                if status["state"] != "LOST":
                    continue
                room_id = status["room_id"]
                cfg = self._room_cfg.get(room_id)
                if cfg is None:
                    continue
                for neighbor_id in cfg["neighbors"]:
                    neighbor_cfg = self._room_cfg.get(neighbor_id)
                    if neighbor_cfg is None:
                        continue
                    # Only infer into no-cam rooms (option B)
                    if neighbor_cfg["cam_id"] is not None:
                        continue
                    if neighbor_id not in self._inferred:
                        self._inferred[neighbor_id] = {
                            "since": now,
                            "identity": status.get("identity"),
                        }

            # 3. Auto-clear entries that have exceeded the timeout
            if self._timeout > 0:
                expired = [
                    rid for rid, info in self._inferred.items()
                    if (now - info["since"]) >= self._timeout
                ]
                for rid in expired:
                    del self._inferred[rid]

    def get_inferred(self):
        """Return a copy of the current inferred-presence dict."""
        with self._lock:
            return dict(self._inferred)

    def reset(self):
        """Manual clear (dashboard Reset button)."""
        with self._lock:
            self._inferred.clear()

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
inference_engine = None  # InferenceEngine, built in start_system()

# Tracks the latest "has_target" flag per room so the /api/room_status
# endpoint (polled by the browser for the 2D map) doesn't need to touch
# cv2 drawing code -- it's updated by the same loop that encodes frames.
_room_status_lock = threading.Lock()
_room_has_target = {}

# Build a fast lookup: cam_id -> room_id using FLOOR_PLAN
_cam_to_room = {
    r["cam_id"]: r["id"]
    for r in getattr(config, "FLOOR_PLAN", [])
    if r["cam_id"] is not None
}


def start_system():
    """Start one RoomUnit (camera + AI) per entry in config.CAMERAS.
    Called once, before the Flask server starts handling requests."""
    global shared_logger, rooms, inference_engine

    validate_camera_config(config.CAMERAS)

    shared_logger = EventLogger()
    rooms = [RoomUnit(cam_cfg, shared_logger) for cam_cfg in config.CAMERAS]

    for room in rooms:
        room.start()
        _room_has_target[room.id] = False

    floor_plan = getattr(config, "FLOOR_PLAN", [])
    timeout = getattr(config, "INFERRED_PRESENCE_TIMEOUT_SEC", 60)
    inference_engine = InferenceEngine(floor_plan, timeout)

    print("\n--- WEB DASHBOARD SYSTEM RUNNING (multi-camera, fixed cameras) ---")
    print(f"Rooms: {[r.room_name for r in rooms]}")
    print(f"Floor plan: {[r['id'] for r in floor_plan]}")
    print(f"Inferred presence timeout: {timeout}s")
    print(f"Event log: {shared_logger.log_path}")
    print("Open http://localhost:5001 in your browser.")
    print("Press Ctrl+C in this terminal to stop safely.\n")


def stop_system():
    print("\n[Main] Stopping all camera/AI threads...")
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

    # Build cam-room statuses for the inference engine
    cam_statuses = []
    cam_results = {}
    for room in rooms:
        has_target, _, _, _, current_state = room.ai_thread.get_ai_result()
        identity = room.ai_thread.get_locked_identity()
        latency = room.ai_thread.get_last_latency_ms()

        # Map camera room id → floor plan room id
        floor_room_id = _cam_to_room.get(room.id, room.id)

        cam_statuses.append({
            "room_id": floor_room_id,
            "has_target": statuses.get(room.id, False),
            "state": current_state,
            "identity": identity,
        })
        cam_results[floor_room_id] = {
            "has_target": statuses.get(room.id, False),
            "state": current_state,
            "identity": identity,
            "latency_ms": round(latency, 1),
            "has_cam": True,
        }

    # Run inference engine
    if inference_engine:
        inference_engine.update(cam_statuses)
    inferred = inference_engine.get_inferred() if inference_engine else {}

    # Build full floor plan result
    floor_plan = getattr(config, "FLOOR_PLAN", [])
    now = time.time()
    result = []
    for room_cfg in floor_plan:
        rid = room_cfg["id"]
        if rid in cam_results:
            entry = cam_results[rid].copy()
            entry["id"] = rid
            entry["room_name"] = room_cfg["name"]
            entry["inferred"] = False
            entry["inferred_since"] = None
        elif rid in inferred:
            info = inferred[rid]
            elapsed = now - info["since"]
            timeout = getattr(config, "INFERRED_PRESENCE_TIMEOUT_SEC", 60)
            entry = {
                "id": rid,
                "room_name": room_cfg["name"],
                "has_target": False,
                "state": "INFERRED",
                "identity": info["identity"],
                "latency_ms": None,
                "has_cam": False,
                "inferred": True,
                "inferred_since": round(elapsed),
                "inferred_timeout": timeout,
            }
        else:
            entry = {
                "id": rid,
                "room_name": room_cfg["name"],
                "has_target": False,
                "state": "EMPTY",
                "identity": None,
                "latency_ms": None,
                "has_cam": room_cfg["cam_id"] is not None,
                "inferred": False,
                "inferred_since": None,
            }
        result.append(entry)

    return jsonify({"rooms": result})


@app.route("/api/config")
def get_config():
    """Expose relevant config values to the browser."""
    return jsonify({
        "inferred_presence_timeout_sec": getattr(config, "INFERRED_PRESENCE_TIMEOUT_SEC", 60),
        "floor_plan": getattr(config, "FLOOR_PLAN", []),
        "target_fps": config.TARGET_FPS,
    })


@app.route("/api/reset_inference", methods=["POST"])
def reset_inference():
    """Manual clear of all inferred-presence highlights."""
    if inference_engine:
        inference_engine.reset()
        shared_logger.log("INFERENCE_RESET", source="dashboard")
    return jsonify({"status": "ok"})


@app.route("/api/events")
def get_events():
    """Return recent event log entries for the timeline panel."""
    n = int(request.args.get("n", 50))
    events = shared_logger.get_recent(n)
    return jsonify({
        "events": [
            {"time": ts, "event": ev, "details": det}
            for ts, ev, det in events
        ]
    })


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "dashboard.html",
        rooms=[{"id": r.id, "room_name": r.room_name} for r in rooms],
    )


if __name__ == "__main__":
    start_system()
    try:
        # use_reloader=False is required: the reloader would otherwise spawn
        # a second process and start every camera/AI thread twice.
        app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False, threaded=True)
    finally:
        stop_system()