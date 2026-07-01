"""
Raw WebSocket server that the SINGLE door ESP32 (Dev Module, driving BOTH
servos -- one per room/door) connects to AS A CLIENT.

This is a plain `websockets` server (NOT Flask-SocketIO) because the
ESP32 side speaks plain RFC6455 WebSocket frames with simple JSON text
payloads -- replicating the Socket.IO sub-protocol on an Arduino board is
fragile, so we keep this connection completely separate from the
browser<->server Socket.IO channel.

Multi-door protocol (ONE physical ESP32, ONE WebSocket connection,
messages multiplexed by a "door" field):

    ESP32 -> Server (on connect, and after every move):
        {"door": "cam1", "state": "closed"}
        {"door": "cam2", "state": "closed"}

    Server -> ESP32 (whenever the dashboard button is pressed):
        {"door": "cam1", "cmd": "OPEN"}
        {"door": "cam2", "cmd": "CLOSE"}

`door` must match a config.CAMERAS[*]["id"] (e.g. "cam1", "cam2") -- this
is what ties a servo to the room whose camera decides whether that
specific door is allowed to be opened. See esp32_servo.ino, which drives
two Servo objects (DOOR_CAM1_PIN, DOOR_CAM2_PIN) off this single board.

Runs its own asyncio event loop in a background thread, so it can live
inside the same process as the Flask app regardless of which async_mode
Flask-SocketIO uses (we use async_mode="threading" in app_dashboard.py
specifically so eventlet/gevent monkey-patching never gets a chance to
fight with this asyncio loop).

Requires: pip install websockets --break-system-packages   (v11+)
"""

import asyncio
import json
import threading
import time

import websockets


class _DoorState:
    """Per-door (not per-connection) bookkeeping. Both doors share the
    same underlying ESP32 WebSocket connection."""

    def __init__(self):
        self.state = "UNKNOWN"   # "OPEN" | "CLOSED" | "UNKNOWN"
        self.pending_ack = None  # asyncio.Future, set while a command is in flight


class DoorWebSocketServer:
    def __init__(self, host="0.0.0.0", port=8765, door_ids=None,
                 on_state_change=None, command_timeout=3.0):
        """
        door_ids: list of every expected door id (e.g. ["cam1", "cam2"]),
            so the dashboard can show both doors as "not connected" even
            before the ESP32 has said anything about them yet.
        on_state_change: optional callable(door_id: str, state: str,
            connected: bool), called (from the asyncio thread -- keep it
            fast/non-blocking) whenever a door's reported state changes,
            or whenever the single ESP32 connects/disconnects (applied to
            every door_id at once on connect/disconnect, since they share
            one physical board/connection).
        command_timeout: seconds to wait for the ESP32 to ack a command
            before send_command() gives up and returns False.
        """
        self.host = host
        self.port = port
        self.on_state_change = on_state_change
        self.command_timeout = command_timeout

        self._loop = None
        self._thread = None
        self._server = None

        self._lock = threading.Lock()
        self._doors = {did: _DoorState() for did in (door_ids or [])}
        self._esp32_ws = None       # the single ESP32 connection, if any
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Start the WebSocket server in a background daemon thread."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Give the loop a brief moment to actually start listening before
        # returning, so a send_command() call right after start() doesn't
        # race the bind.
        time.sleep(0.2)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        # websockets.serve() (v13+) calls asyncio.get_running_loop()
        # internally at construction time, so it MUST be created while a
        # loop is actually running -- not just set via set_event_loop().
        self._loop.run_until_complete(self._start_server())
        print(f"[DoorWS] Listening for the door ESP32 on ws://{self.host}:{self.port}")
        self._loop.run_forever()

    async def _start_server(self):
        self._server = await websockets.serve(self._handle_client, self.host, self.port)

    def stop(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------------------
    # ESP32 connection handler (runs inside the asyncio loop)
    # ------------------------------------------------------------------
    async def _handle_client(self, ws):
        # Only one physical door-controller ESP32 is expected. If a second
        # connection shows up (e.g. board rebooted before the old TCP
        # connection timed out), just replace the old one.
        print(f"[DoorWS] Door ESP32 connected from {ws.remote_address}")
        with self._lock:
            self._esp32_ws = ws
            self._connected = True
        self._notify_all(True)

        try:
            async for message in ws:
                self._handle_message(message)
        except Exception as e:
            print(f"[DoorWS] Connection error: {e}")
        finally:
            with self._lock:
                if self._esp32_ws is ws:
                    self._esp32_ws = None
                    self._connected = False
            print("[DoorWS] Door ESP32 disconnected")
            self._notify_all(False)

    def _handle_message(self, message):
        """Handle one JSON text frame from the ESP32, e.g.
        {"door": "cam1", "state": "open"}."""
        try:
            data = json.loads(message)
        except (TypeError, ValueError):
            print(f"[DoorWS] Ignoring non-JSON message: {message!r}")
            return

        door_id = data.get("door")
        state = data.get("state")
        if not door_id:
            print(f"[DoorWS] Message missing 'door' field: {data!r}")
            return

        changed = False
        with self._lock:
            door = self._doors.setdefault(door_id, _DoorState())
            if state:
                state_upper = state.upper()
                changed = state_upper != door.state
                door.state = state_upper
            pending = door.pending_ack

        if changed:
            self._notify(door_id, door.state, True)

        # Wake up a thread blocked in send_command() waiting for this ack.
        if pending and not pending.done():
            pending.set_result(data)

    def _notify(self, door_id, state, connected):
        if self.on_state_change:
            try:
                self.on_state_change(door_id, state, connected)
            except Exception as e:
                print(f"[DoorWS] on_state_change callback error: {e}")

    def _notify_all(self, connected):
        """Fired on the single ESP32 connecting/disconnecting -- applies
        to every known door since they share one physical board."""
        with self._lock:
            door_ids = list(self._doors.keys())
        for did in door_ids:
            self._notify(did, self.get_state(did), connected)

    # ------------------------------------------------------------------
    # Public, thread-safe API (call from Flask request / Socket.IO
    # handler threads, or any other thread -- never call from outside
    # an event loop context directly).
    # ------------------------------------------------------------------
    def is_connected(self, door_id=None):
        """Connection is per-PHYSICAL-BOARD, so it's the same for every
        door_id -- the door_id arg exists only for call-site symmetry
        with get_state()/send_command()."""
        with self._lock:
            return self._connected

    def get_state(self, door_id):
        with self._lock:
            door = self._doors.get(door_id)
            return door.state if door else "UNKNOWN"

    def get_all_states(self):
        """door_id -> {"state":..., "connected":...} snapshot for every
        known door (used to build the initial dashboard payload)."""
        with self._lock:
            connected = self._connected
            return {
                did: {"state": d.state, "connected": connected}
                for did, d in self._doors.items()
            }

    def send_command(self, door_id, command):
        """
        command: "OPEN" or "CLOSE".
        Blocks the calling thread briefly waiting for the ESP32 to
        acknowledge that specific door's new state. Returns True on
        success, False on timeout or if the ESP32 isn't connected.
        """
        with self._lock:
            ws = self._esp32_ws
        if ws is None or self._loop is None:
            print(f"[DoorWS] send_command: ESP32 not connected (door '{door_id}')")
            return False

        future = asyncio.run_coroutine_threadsafe(
            self._send_and_wait(door_id, ws, command), self._loop
        )
        try:
            return future.result(timeout=self.command_timeout + 1)
        except Exception as e:
            print(f"[DoorWS] send_command failed (door '{door_id}'): {e}")
            return False

    async def _send_and_wait(self, door_id, ws, command):
        ack_future = self._loop.create_future()
        with self._lock:
            door = self._doors.setdefault(door_id, _DoorState())
            door.pending_ack = ack_future

        payload = json.dumps({"door": door_id, "cmd": command})
        await ws.send(payload)
        try:
            await asyncio.wait_for(ack_future, timeout=self.command_timeout)
            return True
        except asyncio.TimeoutError:
            print(f"[DoorWS] Timed out waiting for ESP32 ack (door '{door_id}')")
            return False