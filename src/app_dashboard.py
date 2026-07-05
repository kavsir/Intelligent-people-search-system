# app_dashboard.py
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

import base64
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
from operation.behavior_manager import behavior_manager
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
class InferenceEngine:
    """
    When a camera loses its target (state transitions to LOST), this engine
    highlights the neighboring no-cam rooms as "inferred presence".
    """

    def __init__(self, floor_plan, timeout_sec):
        self._room_cfg = {r["id"]: r for r in floor_plan}
        self._cam_rooms = {r["id"] for r in floor_plan if r["cam_id"] is not None}
        self._timeout = timeout_sec

        self._lock = threading.Lock()
        self._inferred = {}

    def update(self, cam_room_statuses):
        now = time.time()

        with self._lock:
            confirmed_ids = {s["room_id"] for s in cam_room_statuses if s["has_target"]}
            if confirmed_ids:
                self._inferred.clear()
                return

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
                    if neighbor_cfg["cam_id"] is not None:
                        continue
                    if neighbor_id not in self._inferred:
                        self._inferred[neighbor_id] = {
                            "since": now,
                            "identity": status.get("identity"),
                        }

            if self._timeout > 0:
                expired = [
                    rid for rid, info in self._inferred.items()
                    if (now - info["since"]) >= self._timeout
                ]
                for rid in expired:
                    del self._inferred[rid]

    def get_inferred(self):
        with self._lock:
            return dict(self._inferred)

    def reset(self):
        with self._lock:
            self._inferred.clear()


app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "operation", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "operation", "static"),
)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
shared_logger = None
rooms = []
inference_engine = None

_room_status_lock = threading.Lock()
_room_has_target = {}

_cam_to_room = {
    r["cam_id"]: r["id"]
    for r in getattr(config, "FLOOR_PLAN", [])
    if r["cam_id"] is not None
}

# ---------------------------------------------------------------------------
# Door state
# ---------------------------------------------------------------------------
door_ws = None
_door_lock = threading.Lock()
_door_person_present = {}
_door_safety_thread = None
_door_safety_running = False

# ---------------------------------------------------------------------------
# Tracking snapshot state
# ---------------------------------------------------------------------------
# So sánh {tên người: tên phòng} hiện tại vs lần trước để quyết định có chụp không.
# Quy tắc:
#   - Thay đổi (thêm/xóa/di chuyển phòng) → chụp
#   - Từ có người → không còn ai → KHÔNG chụp
#   - Không thay đổi → không chụp
_tracking_lock = threading.Lock()
_last_tracking_state = {}  # {name: room_name}
_tracking_running = False
_tracking_thread = None

# ---------------------------------------------------------------------------
# Cross-room movement notifications (PROBLEM #2)
# ---------------------------------------------------------------------------
# Unlike _last_tracking_state (which _check_and_capture_tracking() resets
# to {} the moment every room goes empty, since that's what its own
# "don't photograph an empty room" rule needs), _last_seen_room is NEVER
# cleared -- it always remembers the last room each registered name was
# confirmed in, even across a gap where nobody has a camera on them (e.g.
# walking through a no-cam room in FLOOR_PLAN). That's what lets a
# same-person A→(no-cam room)→B transition still be reported correctly.
_movement_lock = threading.Lock()
_last_seen_room = {}  # {name: room_name}


def _door_room_ids():
    return [cam["id"] for cam in config.CAMERAS]


def _room_registered_present(room_id):
    with _room_status_lock:
        return _room_has_target.get(room_id, False)


def _push_door_status(room_id):
    with _door_lock:
        person_present = _door_person_present.get(room_id, False)
    state = door_ws.get_state(room_id) if door_ws else "UNKNOWN"
    connected = door_ws.is_connected(room_id) if door_ws else False
    socketio.emit("door_status", {
        "room_id": room_id,
        "state": state,
        "esp32_connected": connected,
        "person_present": person_present,
    })


def _lockdown_all_doors(triggered_by_room_id):
    for room_id in _door_room_ids():
        if door_ws is None or not door_ws.is_connected(room_id):
            continue
        if door_ws.get_state(room_id) == "CLOSED":
            continue

        if door_ws.send_command(room_id, "CLOSE"):
            if shared_logger:
                shared_logger.log(
                    "DOOR_LOCKDOWN_CLOSED", room=room_id, triggered_by=triggered_by_room_id
                )
            _push_door_status(room_id)


def _update_door_presence():
    for room_id in _door_room_ids():
        present = _room_registered_present(room_id)

        with _door_lock:
            was_present = _door_person_present.get(room_id, False)
            _door_person_present[room_id] = present

        if present != was_present:
            _push_door_status(room_id)
            if shared_logger:
                shared_logger.log(
                    "REGISTERED_PERSON_DETECTED" if present else "REGISTERED_PERSON_CLEARED",
                    room=room_id,
                )

        if present and not was_present:
            _lockdown_all_doors(triggered_by_room_id=room_id)


def _door_safety_loop():
    while _door_safety_running:
        try:
            _update_door_presence()
        except Exception as e:
            print(f"[Door] safety loop error: {e}")
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Movement notification logic
# ---------------------------------------------------------------------------
def _check_movement(current_state):
    """
    current_state: {name: room_name} -- who is CURRENTLY locked as a
    room's target, same shape _check_and_capture_tracking() already
    builds every 0.5s.

    For each registered name currently seen somewhere, compare against
    the last room we ever confirmed them in. If it's a DIFFERENT room,
    emit a "[name] di chuyển từ [room A] sang [room B]" notification --
    both to the event log (shows up in the dashboard's timeline via the
    existing shared_logger -> /api/events pipeline, no frontend changes
    needed) and as a Socket.IO event for anyone listening live.
    """
    global _last_seen_room

    for name, room_name in current_state.items():
        with _movement_lock:
            prev_room = _last_seen_room.get(name)
            _last_seen_room[name] = room_name

        if prev_room is None or prev_room == room_name:
            continue  # first-ever sighting, or still in the same room -- not a move

        message = f"{name} di chuyển từ {prev_room} sang {room_name}"
        print(f"[Movement] {message}")
        if shared_logger:
            shared_logger.log(
                "PERSON_MOVED", name=name, from_room=prev_room, to_room=room_name
            )
        socketio.emit("movement_notification", {
            "name": name,
            "from_room": prev_room,
            "to_room": room_name,
            "message": message,
        })


# ---------------------------------------------------------------------------
# Tracking snapshot logic
# ---------------------------------------------------------------------------
def _check_and_capture_tracking():
    """
    So sánh trạng thái hiện tại (ai đang ở phòng nào) với lần trước.
    Nếu thay đổi → chụp ảnh từng phòng có người → lưu vào bảng theo_doi.
    """
    global _last_tracking_state

    # Xây dựng trạng thái hiện tại: {name: room_name}
    current_state = {}
    for room in rooms:
        identity = room.ai_thread.get_locked_identity()
        if identity:
            current_state[identity] = room.room_name

    # Phát hiện di chuyển giữa các phòng -- chạy độc lập với logic chụp
    # ảnh bên dưới, vì _last_seen_room không bao giờ bị xóa về {} như
    # _last_tracking_state.
    _check_movement(current_state)

    with _tracking_lock:
        last_state = dict(_last_tracking_state)

    # Không thay đổi → bỏ qua
    if current_state == last_state:
        return

    # Từ "có người" → "không còn ai" → KHÔNG chụp (theo yêu cầu)
    if not current_state and last_state:
        with _tracking_lock:
            _last_tracking_state = {}
        if shared_logger:
            shared_logger.log("TRACKING_ALL_CLEARED")
        return

    # Có thay đổi → chụp ảnh
    # 1. Xây dựng danh sách entries
    entries = [{"name": name, "room_name": room_name}
               for name, room_name in current_state.items()]

    # 2. Lấy frame hiện tại của từng phòng có người
    image_by_room = {}
    rooms_with_people = {e["room_name"] for e in entries}
    for room in rooms:
        if room.room_name in rooms_with_people:
            ret, frame = room.camera_thread.get_frame()
            if ret and frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    image_by_room[room.room_name] = buf.tobytes()

    # 3. Lưu vào database
    if entries and image_by_room:
        face_database.save_tracking_snapshot(entries, image_by_room)
        if shared_logger:
            shared_logger.log(
                "TRACKING_SNAPSHOT",
                people=list(current_state.keys()),
                rooms=list(image_by_room.keys()),
            )
        print(f"[Tracking] Captured: {list(current_state.keys())} in {list(image_by_room.keys())}")

    # 4. Cập nhật trạng thái lần trước
    with _tracking_lock:
        _last_tracking_state = dict(current_state)


def _tracking_loop():
    """Background thread: check tracking changes every 0.5s."""
    while _tracking_running:
        try:
            _check_and_capture_tracking()
        except Exception as e:
            print(f"[Tracking] loop error: {e}")
        time.sleep(0.5)


def _on_exercise_fail(name):
    try:
        face_database.delete_person(name)
        face_database.delete_tracking_history(name)
        face_database.delete_body_profile(name)
        print(f"[Exercise] FAIL -> deleted '{name}' from face database + tracking + body profile")
    except Exception as e:
        print(f"[Exercise] Error deleting data for '{name}': {e}")

    for room in rooms:
        room.ai_thread.recognizer.load_database()
        room.ai_thread.pose_estimator.reset(name)
        if room.ai_thread.get_locked_identity() == name:
            room.ai_thread.force_clear_target()

    behavior_manager.forget(name)
    with _movement_lock:
        _last_seen_room.pop(name, None)

    if shared_logger:
        shared_logger.log("EXERCISE_FAILED_DELETED", name=name)


def start_system():
    global shared_logger, rooms, inference_engine
    global door_ws, _door_safety_thread, _door_safety_running
    global _tracking_running, _tracking_thread

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

    behavior_manager.start()

    # Door WebSocket server
    for room_id in _door_room_ids():
        _door_person_present[room_id] = False

    def _on_door_state_change(room_id, state, connected):
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

    # Tracking snapshot thread
    _tracking_running = True
    _tracking_thread = threading.Thread(target=_tracking_loop, daemon=True)
    _tracking_thread.start()

    print("\n--- WEB DASHBOARD SYSTEM RUNNING (multi-camera, fixed cameras) ---")
    print(f"Rooms: {[r.room_name for r in rooms]}")
    print(f"Floor plan: {[r['id'] for r in floor_plan]}")
    print(f"Inferred presence timeout: {timeout}s")
    print(f"Door WebSocket: ws://{door_ws.host}:{door_ws.port}")
    print(f"Tracking snapshot: 0.5s interval")
    print(f"Event log: {shared_logger.log_path}")
    print("Open http://localhost:5001 in your browser.")
    print("Press Ctrl+C in this terminal to stop safely.\n")


def stop_system():
    global _door_safety_running, _tracking_running
    print("\n[Main] Stopping all camera/AI threads...")
    _door_safety_running = False
    _tracking_running = False
    behavior_manager.stop()
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
# Status API
# ---------------------------------------------------------------------------
@app.route("/api/room_status")
def room_status():
    with _room_status_lock:
        statuses = dict(_room_has_target)

    cam_statuses = []
    cam_results = {}
    for room in rooms:
        has_target, _, _, _, current_state = room.ai_thread.get_ai_result()
        identity = room.ai_thread.get_locked_identity()
        latency = room.ai_thread.get_last_latency_ms()

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

    if inference_engine:
        inference_engine.update(cam_statuses)
    inferred = inference_engine.get_inferred() if inference_engine else {}

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
    return jsonify({
        "inferred_presence_timeout_sec": getattr(config, "INFERRED_PRESENCE_TIMEOUT_SEC", 60),
        "floor_plan": getattr(config, "FLOOR_PLAN", []),
        "target_fps": config.TARGET_FPS,
        "registration_port": getattr(config, "REGISTRATION_APP_PORT", 5000),
    })


@app.route("/api/reset_inference", methods=["POST"])
def reset_inference():
    if inference_engine:
        inference_engine.reset()
        shared_logger.log("INFERENCE_RESET", source="dashboard")
    return jsonify({"status": "ok"})


@app.route("/api/events")
def get_events():
    n = int(request.args.get("n", 50))
    events = shared_logger.get_recent(n)
    return jsonify({
        "events": [
            {"time": ts, "event": ev, "details": det}
            for ts, ev, det in events
        ]
    })


# ---------------------------------------------------------------------------
# Door control (Socket.IO)
# ---------------------------------------------------------------------------
@socketio.on("connect")
def handle_connect():
    for room_id in _door_room_ids():
        with _door_lock:
            person_present = _door_person_present.get(room_id, False)
        state = door_ws.get_state(room_id) if door_ws else "UNKNOWN"
        connected = door_ws.is_connected(room_id) if door_ws else False
        emit("door_status", {
            "room_id": room_id,
            "state": state,
            "esp32_connected": connected,
            "person_present": person_present,
        })


@socketio.on("toggle_door")
def handle_toggle_door(data):
    room_id = (data or {}).get("room_id")
    if not room_id or room_id not in _door_room_ids():
        emit("door_response", {
            "room_id": room_id,
            "status": "error",
            "message": "Phòng không hợp lệ.",
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
    return jsonify({"names": face_database.list_people()})


@app.route("/api/exercises")
def get_exercises():
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
    exercise_manager.unassign(name)
    for room in rooms:
        room.ai_thread.pose_estimator.reset(name)
    if shared_logger:
        shared_logger.log("EXERCISE_ROW_DELETED", name=name)
    return jsonify({"status": "ok", "rows": exercise_manager.get_table()})


# ---------------------------------------------------------------------------
# Behavior recognition API
# ---------------------------------------------------------------------------
@app.route("/api/behaviors")
def get_behaviors():
    return jsonify({"rows": behavior_manager.get_table()})


@app.route("/api/behaviors/history")
def get_behaviors_history():
    name = (request.args.get("name") or "").strip()
    if not name:
        return jsonify({"status": "error", "message": "Thiếu tham số 'name'."}), 400
    limit = int(request.args.get("limit", 200))
    return jsonify({"name": name, "history": face_database.get_behavior_history(name, limit=limit)})


# ---------------------------------------------------------------------------
# People overview page -- BỔ SUNG cột ảnh theo dõi
# ---------------------------------------------------------------------------
def _room_lookup_by_cam_id(room_id):
    return next((r for r in rooms if r.id == room_id), None)


@app.route("/api/people_overview")
def people_overview():
    """
    One row per registered person, merging:
      - face_database (who is registered)
      - each room's AIPipeline locked identity (which room currently sees them)
      - door_ws (that room's door state)
      - exercise_manager (assigned exercise / reps / online / result)
      - face_database.get_latest_tracking() (ảnh theo dõi mới nhất)
    """
    names = face_database.list_people()
    exercise_by_name = {row["name"]: row for row in exercise_manager.get_table()}

    name_to_room_id = {}
    for room in rooms:
        identity = room.ai_thread.get_locked_identity()
        if identity:
            name_to_room_id[identity] = room.id

    # Lấy ảnh theo dõi mới nhất cho mỗi người
    tracking_data = face_database.get_latest_tracking()

    people = []
    for name in names:
        room_id = name_to_room_id.get(name)
        room_obj = _room_lookup_by_cam_id(room_id) if room_id else None

        door_state = "UNKNOWN"
        person_present = False
        door_connected = False
        if room_obj is not None:
            door_state = door_ws.get_state(room_id) if door_ws else "UNKNOWN"
            door_connected = door_ws.is_connected(room_id) if door_ws else False
            with _door_lock:
                person_present = _door_person_present.get(room_id, False)

        ex = exercise_by_name.get(name)

        # Ảnh theo dõi
        track = tracking_data.get(name)
        tracking_image_b64 = None
        tracking_time = None
        if track and track.get("image_data"):
            tracking_image_b64 = base64.b64encode(track["image_data"]).decode("utf-8")
            tracking_time = track["captured_at"]

        people.append({
            "name": name,
            "room_id": room_id,
            "room_name": room_obj.room_name if room_obj else None,
            "door_state": door_state,
            "person_present": person_present,
            "door_connected": door_connected,
            "assigned": ex is not None,
            "exercise": ex["exercise"] if ex else None,
            "target_reps": ex["target_reps"] if ex else None,
            "count": ex["count"] if ex else 0,
            "online": ex["online"] if ex else False,
            "result": ex["result"] if ex else None,
            # Thêm trường mới cho ảnh theo dõi
            "tracking_image": tracking_image_b64,
            "tracking_time": tracking_time,
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
        socketio.run(app, host="0.0.0.0", port=5001, debug=False, use_reloader=False)
    finally:
        stop_system()