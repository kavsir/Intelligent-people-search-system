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
from flask_socketio import SocketIO, emit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import face_database
from operation.event_logger import EventLogger
from operation.door_ws_server import DoorWebSocketServer
from operation.exercise_manager import exercise_manager
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
# async_mode="threading" on purpose: door_ws_server.py runs its own asyncio
# event loop in a plain background thread. eventlet/gevent (the other
# Flask-SocketIO async modes) monkey-patch the stdlib socket/threading
# layer, which would silently break that asyncio loop. "threading" mode
# keeps everything on normal OS threads so the two coexist safely.
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

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

# ---------------------------------------------------------------------------
# Door (ESP32 servo) state -- ONE DOOR PER ROOM/CAMERA
# ---------------------------------------------------------------------------
# Doors are keyed by the SAME id as config.CAMERAS[*]["id"] (e.g. "cam1",
# "cam2"). BOTH doors are driven by a SINGLE physical ESP32 Dev Module
# board with two servos, multiplexed over one WebSocket connection using
# a "door" field in every message (see esp32_servo.ino / door_ws_server.py).
# Each door can only be enabled/toggled based on ITS OWN room's presence;
# rooms are never allowed to unlock each other's door, even though both
# doors share the same underlying ESP32 connection.
door_ws = None                  # DoorWebSocketServer, built in start_system()
_door_lock = threading.Lock()
_door_enabled = {}              # room_id -> bool (True while THAT room has a confirmed target)
_door_last_seen_present = {}    # room_id -> time.time() of last confirmed presence
_door_safety_thread = None
_door_safety_running = False


def _door_room_ids():
    """Every room id that has a door -- currently: every camera room."""
    return [cam["id"] for cam in config.CAMERAS]


def _room_registered_present(room_id):
    """
    True if THIS SPECIFIC room currently has a CAMERA-CONFIRMED registered
    target (i.e. its AIPipeline reached TRACKING with a locked identity).
    Deliberately does NOT count InferenceEngine's "inferred presence" in
    no-cam rooms, and deliberately does NOT look at any OTHER room -- a
    door only ever unlocks from its own room's real face match.
    """
    with _room_status_lock:
        return _room_has_target.get(room_id, False)


def _push_door_status(room_id):
    """Emit one room's door enable-flag/state/connection to all browsers."""
    with _door_lock:
        enabled = _door_enabled.get(room_id, False)
    state = door_ws.get_state(room_id) if door_ws else "UNKNOWN"
    connected = door_ws.is_connected(room_id) if door_ws else False
    socketio.emit("door_status", {
        "room_id": room_id,
        "enabled": enabled,
        "state": state,
        "esp32_connected": connected,
    })


def _update_door_presence():
    """
    For EVERY room independently: recompute whether that room's door
    button should be enabled, push an update to browsers if it changed,
    and auto-close that room's door if nobody registered has been seen in
    it for config.DOOR_AUTO_CLOSE_SEC seconds -- a safety net so a door
    can't be left open indefinitely just because no one pressed 'close'.
    """
    now = time.time()

    for room_id in _door_room_ids():
        present = _room_registered_present(room_id)

        with _door_lock:
            was_enabled = _door_enabled.get(room_id, False)
            _door_enabled[room_id] = present
            if present:
                _door_last_seen_present[room_id] = now
            since_present = now - _door_last_seen_present.get(room_id, now)

        if present != was_enabled:
            _push_door_status(room_id)
            if shared_logger:
                shared_logger.log(
                    "DOOR_ACCESS_ENABLED" if present else "DOOR_ACCESS_DISABLED",
                    room=room_id,
                )

        timeout = getattr(config, "DOOR_AUTO_CLOSE_SEC", 15)
        if (
            timeout > 0
            and not present
            and since_present >= timeout
            and door_ws is not None
            and door_ws.get_state(room_id) == "OPEN"
        ):
            if door_ws.send_command(room_id, "CLOSE"):
                if shared_logger:
                    shared_logger.log(
                        "DOOR_AUTO_CLOSED", room=room_id, after_sec=round(since_present, 1)
                    )
                _push_door_status(room_id)


def _door_safety_loop():
    """Background tick so presence/auto-close logic runs even if no
    browser tab is open polling /api/room_status."""
    while _door_safety_running:
        try:
            _update_door_presence()
        except Exception as e:
            print(f"[Door] safety loop error: {e}")
        time.sleep(1.0)


def _on_exercise_fail(name):
    """
    Called by exercise_manager the instant a person's result becomes
    'fail' (wrong movement performed, or they went offline before
    reaching their target rep count). Permanently deletes their face_db +
    processed data and forces every room to drop any active lock on them,
    so they can never be tracked again unless re-registered from scratch.
    """
    try:
        face_database.delete_person(name)
        print(f"[Exercise] FAIL -> deleted '{name}' from the face database")
    except Exception as e:
        print(f"[Exercise] Error deleting data for '{name}': {e}")

    for room in rooms:
        room.ai_thread.recognizer.load_database()
        room.ai_thread.pose_estimator.reset(name)
        if room.ai_thread.get_locked_identity() == name:
            room.ai_thread.force_clear_target()

    if shared_logger:
        shared_logger.log("EXERCISE_FAILED_DELETED", name=name)


def start_system():
    """Start one RoomUnit (camera + AI) per entry in config.CAMERAS.
    Called once, before the Flask server starts handling requests."""
    global shared_logger, rooms, inference_engine
    global door_ws, _door_safety_thread, _door_safety_running

    validate_camera_config(config.CAMERAS)

    shared_logger = EventLogger()
    rooms = [RoomUnit(cam_cfg, shared_logger) for cam_cfg in config.CAMERAS]

    for room in rooms:
        room.start()
        _room_has_target[room.id] = False
    
    floor_plan = getattr(config, "FLOOR_PLAN", [])
    timeout = getattr(config, "INFERRED_PRESENCE_TIMEOUT_SEC", 60)
    inference_engine = InferenceEngine(floor_plan, timeout)

    exercise_manager.set_on_fail_callback(_on_exercise_fail)

    # --- Door WebSocket server: ONE physical ESP32 (Dev Module, 2 servos)
    # connects IN to us as a single WebSocket client, and multiplexes
    # BOTH doors' commands/states over that one connection using a
    # "door" field per message (door id == the room's camera id). ---
    for room_id in _door_room_ids():
        _door_enabled[room_id] = False
        _door_last_seen_present[room_id] = time.time()

    def _on_door_state_change(room_id, state, connected):
        # Called from the DoorWS asyncio thread. _push_door_status() only
        # reads shared state under lock and emits, so this is safe to call
        # directly from another thread.
        _push_door_status(room_id)
        if shared_logger:
            shared_logger.log(
                "DOOR_ESP32_CONNECTED" if connected else "DOOR_ESP32_DISCONNECTED",
                room=room_id, state=state,
            )

    door_ws = DoorWebSocketServer(
        host=getattr(config, "DOOR_WS_HOST", "0.0.0.0"),
        port=getattr(config, "DOOR_WS_PORT", 8765),
        door_ids=_door_room_ids(),
        on_state_change=_on_door_state_change,
    )
    door_ws.start()

    _door_safety_running = True
    _door_safety_thread = threading.Thread(target=_door_safety_loop, daemon=True)
    _door_safety_thread.start()

    print("\n--- WEB DASHBOARD SYSTEM RUNNING (multi-camera, fixed cameras) ---")
    print(f"Rooms: {[r.room_name for r in rooms]}")
    print(f"Floor plan: {[r['id'] for r in floor_plan]}")
    print(f"Inferred presence timeout: {timeout}s")
    print(f"Door WebSocket: ws://{door_ws.host}:{door_ws.port}")
    print(f"Event log: {shared_logger.log_path}")
    print("Open http://localhost:5001 in your browser.")
    print("Press Ctrl+C in this terminal to stop safely.\n")


def stop_system():
    global _door_safety_running
    print("\n[Main] Stopping all camera/AI threads...")
    _door_safety_running = False
    for room in rooms:
        room.stop()
    for room in rooms:
        room.join()
    if door_ws:
        door_ws.stop()
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
        "registration_port": getattr(config, "REGISTRATION_APP_PORT", 5000),
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
# Door control (Socket.IO) -- bridges browser clicks to the single door
# ESP32 over door_ws (a separate raw WebSocket connection carrying BOTH
# doors, multiplexed by room_id -- see operation/door_ws_server.py)
# ---------------------------------------------------------------------------
@socketio.on("connect")
def handle_connect():
    """Send the current status of EVERY door right away so a freshly-opened
    browser tab doesn't sit showing 'unknown' until the next presence
    change in some room."""
    for room_id in _door_room_ids():
        with _door_lock:
            enabled = _door_enabled.get(room_id, False)
        state = door_ws.get_state(room_id) if door_ws else "UNKNOWN"
        connected = door_ws.is_connected(room_id) if door_ws else False
        emit("door_status", {
            "room_id": room_id,
            "enabled": enabled,
            "state": state,
            "esp32_connected": connected,
        })


@socketio.on("toggle_door")
def handle_toggle_door(data):
    """
    Browser asked to open/close ONE SPECIFIC room's door. We re-check that
    room's presence server-side -- never trust that the button was
    actually disabled client-side -- and toggle relative to that room's
    ESP32's last known real state. A room's presence can NEVER be used to
    toggle another room's door.
    """
    room_id = (data or {}).get("room_id")
    if not room_id or room_id not in _door_room_ids():
        emit("door_response", {
            "room_id": room_id,
            "status": "error",
            "message": "Phòng không hợp lệ.",
        })
        return

    if not _room_registered_present(room_id):
        emit("door_response", {
            "room_id": room_id,
            "status": "error",
            "message": "Không phát hiện người đã đăng ký trong phòng này.",
        })
        return

    if door_ws is None or not door_ws.is_connected(room_id):
        emit("door_response", {
            "room_id": room_id,
            "status": "error",
            "message": "ESP32 cửa của phòng này chưa kết nối tới server.",
        })
        return

    target = "CLOSE" if door_ws.get_state(room_id) == "OPEN" else "OPEN"
    ok = door_ws.send_command(room_id, target)

    if ok:
        if shared_logger:
            shared_logger.log(f"DOOR_MANUAL_{target}", room=room_id, source="dashboard")
        emit("door_response", {
            "room_id": room_id,
            "status": "success",
            "message": "Cửa đã mở!" if target == "OPEN" else "Cửa đã đóng!",
        })
        _push_door_status(room_id)
    else:
        emit("door_response", {
            "room_id": room_id,
            "status": "error",
            "message": "Không nhận được phản hồi từ ESP32 (timeout).",
        })


# ---------------------------------------------------------------------------
# Exercise assignment API
# ---------------------------------------------------------------------------
@app.route("/api/registered_people")
def registered_people():
    """List every name currently in face_db, for the assignment dropdown."""
    return jsonify({"names": face_database.list_people()})


@app.route("/api/exercises")
def get_exercises():
    """Full assignment table for the dashboard panel."""
    return jsonify({"rows": exercise_manager.get_table()})


@app.route("/api/exercises/assign", methods=["POST"])
def assign_exercise():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    exercise = (data.get("exercise") or "").strip().lower()
    target_reps = data.get("target_reps")

    if not name:
        return jsonify({"status": "error", "message": "Thiếu tên người đăng ký."}), 400
    if not face_database.person_exists(name):
        return jsonify({"status": "error", "message": f"'{name}' chưa được đăng ký khuôn mặt."}), 400
    if exercise not in ("squat", "pushup"):
        return jsonify({"status": "error", "message": "Động tác phải là 'squat' hoặc 'pushup'."}), 400
    try:
        target_reps = int(target_reps)
        if target_reps < 1:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Số lần thực hiện phải là số nguyên dương."}), 400

    exercise_manager.assign(name, exercise, target_reps)
    for room in rooms:
        room.ai_thread.pose_estimator.reset(name)
    if shared_logger:
        shared_logger.log("EXERCISE_ASSIGNED", name=name, exercise=exercise, target_reps=target_reps)

    return jsonify({"status": "ok", "rows": exercise_manager.get_table()})


@app.route("/api/exercises/<name>", methods=["DELETE"])
def delete_exercise_row(name):
    """
    Per-row delete button: removes the assignment. Per the spec, this
    resets the person to 'off' with their rep count starting over from
    zero -- it does NOT delete their face_db/processed data (that only
    happens automatically on FAIL, never from this manual row-delete).
    """
    exercise_manager.unassign(name)
    for room in rooms:
        room.ai_thread.pose_estimator.reset(name)
    if shared_logger:
        shared_logger.log("EXERCISE_ROW_DELETED", name=name)
    return jsonify({"status": "ok", "rows": exercise_manager.get_table()})


# ---------------------------------------------------------------------------
# People overview page -- combined view of every registered person: which
# room they're currently seen in, that room's door status, and their
# exercise assignment/progress/result.
# ---------------------------------------------------------------------------
def _room_lookup_by_cam_id(room_id):
    return next((r for r in rooms if r.id == room_id), None)


@app.route("/api/people_overview")
def people_overview():
    """
    One row per registered person, merging:
      - face_database (who is registered)
      - each room's AIPipeline locked identity (which room currently sees them)
      - door_ws (that room's door state, only meaningful if a room was found)
      - exercise_manager (assigned exercise / reps / online / result)
    """
    names = face_database.list_people()
    exercise_by_name = {row["name"]: row for row in exercise_manager.get_table()}

    # Which room (if any) currently has each name locked as its target.
    name_to_room_id = {}
    for room in rooms:
        identity = room.ai_thread.get_locked_identity()
        if identity:
            name_to_room_id[identity] = room.id

    people = []
    for name in names:
        room_id = name_to_room_id.get(name)
        room_obj = _room_lookup_by_cam_id(room_id) if room_id else None

        door_state = "UNKNOWN"
        door_enabled = False
        door_connected = False
        if room_obj is not None:
            door_state = door_ws.get_state(room_id) if door_ws else "UNKNOWN"
            door_connected = door_ws.is_connected(room_id) if door_ws else False
            with _door_lock:
                door_enabled = _door_enabled.get(room_id, False)

        ex = exercise_by_name.get(name)

        people.append({
            "name": name,
            "room_id": room_id,
            "room_name": room_obj.room_name if room_obj else None,
            "door_state": door_state,       # "OPEN" | "CLOSED" | "UNKNOWN"
            "door_enabled": door_enabled,    # this room's presence-gate flag
            "door_connected": door_connected,
            "assigned": ex is not None,
            "exercise": ex["exercise"] if ex else None,           # "squat" | "pushup" | None
            "target_reps": ex["target_reps"] if ex else None,
            "count": ex["count"] if ex else 0,
            "online": ex["online"] if ex else False,
            "result": ex["result"] if ex else None,               # None | "success" | "fail"
        })

    return jsonify({"people": people})


@app.route("/people")
def people_page():
    return render_template("people.html")


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
        # a second process and start every camera/AI thread (and the door
        # WebSocket server) twice.
        socketio.run(app, host="0.0.0.0", port=5001, debug=False, use_reloader=False)
    finally:
        stop_system()