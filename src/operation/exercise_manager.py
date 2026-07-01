"""
Global registry of "exercise assignments": which registered person must
perform which exercise (squat / push-up), how many reps they need, and
their live online/offline + pass/fail status.

This is intentionally ONE shared object (not per-room), because the same
registered person can be seen by either camera -- an assignment follows
the *person*, not the room/camera that happens to see them.

Thread-safety: every public method takes an internal lock. AIPipeline
threads (one per room) call set_online()/register_rep() every frame they
have a locked, assigned target. The Flask app calls assign()/unassign()/
get_table() from request-handling threads.

Pass/fail rules (as specified):
  - Performing the WRONG exercise (a completed rep that doesn't match the
    assigned one) -> immediate FAIL.
  - Reaching the assigned target rep count of the CORRECT exercise -> SUCCESS.
  - Going OFFLINE (no longer detected/tracked) before reaching the target
    -> FAIL ("chưa đạt đủ số lần").
  - On FAIL, the on_fail callback (registered by app_dashboard.py) is
    invoked, which deletes that person's face_db + processed data and
    forces every room to drop any active lock on them.
"""

import threading


class ExerciseManager:
    EXERCISES = ("squat", "pushup")

    def __init__(self):
        self._lock = threading.Lock()
        # name -> {
        #   "exercise": "squat"|"pushup", "target_reps": int, "count": int,
        #   "online": bool, "result": None|"success"|"fail",
        # }
        self._rows = {}
        self._on_fail = None  # callback(name) for cross-room cleanup + deletion

    def set_on_fail_callback(self, fn):
        self._on_fail = fn

    # ------------------------------------------------------------------
    # Admin actions (called from Flask routes)
    # ------------------------------------------------------------------
    def assign(self, name, exercise, target_reps):
        exercise = exercise.lower()
        if exercise not in self.EXERCISES:
            raise ValueError(f"Unknown exercise: {exercise}")
        target_reps = max(1, int(target_reps))
        with self._lock:
            self._rows[name] = {
                "exercise": exercise,
                "target_reps": target_reps,
                "count": 0,
                "online": False,
                "result": None,
            }

    def unassign(self, name):
        """Per-row delete button: removes the row entirely. The person
        reverts to not-being-tracked-for-exercise (off, count reset) until
        re-assigned from scratch."""
        with self._lock:
            self._rows.pop(name, None)

    def is_assigned(self, name):
        with self._lock:
            return name in self._rows

    # ------------------------------------------------------------------
    # Called every frame by AIPipeline for the currently locked target
    # ------------------------------------------------------------------
    def set_online(self, name, online):
        fail_triggered = False
        with self._lock:
            row = self._rows.get(name)
            if row is None:
                return
            was_online = row["online"]
            row["online"] = online

            if (
                was_online and not online
                and row["result"] is None
                and row["count"] < row["target_reps"]
            ):
                row["result"] = "fail"
                fail_triggered = True

        if fail_triggered and self._on_fail:
            self._on_fail(name)

    def register_rep(self, name, exercise_done):
        """exercise_done: "squat" or "pushup" -- whatever movement was
        actually just completed, regardless of what's assigned."""
        fail_triggered = False
        with self._lock:
            row = self._rows.get(name)
            if row is None or row["result"] is not None:
                return  # not assigned, or already finished

            if exercise_done != row["exercise"]:
                row["result"] = "fail"
                fail_triggered = True
            else:
                row["count"] += 1
                if row["count"] >= row["target_reps"]:
                    row["result"] = "success"

        if fail_triggered and self._on_fail:
            self._on_fail(name)

    # ------------------------------------------------------------------
    # Read access (Flask /api/exercises)
    # ------------------------------------------------------------------
    def get_table(self):
        with self._lock:
            return [
                {
                    "name": name,
                    "exercise": row["exercise"],
                    "target_reps": row["target_reps"],
                    "count": row["count"],
                    "online": row["online"],
                    "result": row["result"],
                }
                for name, row in self._rows.items()
            ]


# Single shared instance -- import this, don't instantiate your own.
exercise_manager = ExerciseManager()