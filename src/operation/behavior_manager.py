"""
Global registry of "behavior" activity counts for every registered person
-- Đứng (stand), Di chuyển (move), Nhảy (jump), Giơ tay (raise_hand),
Nằm (lie), squat, hít đất (pushup).

This is deliberately separate from exercise_manager.py:
  - exercise_manager tracks PASS/FAIL against an assigned exercise
    (squat/pushup only), and only while that person has an active
    assignment.
  - behavior_manager just counts how many times each of the 7 behaviors
    happened, for EVERY recognized registered person, all the time,
    whether or not they currently have an exercise assigned. Standing /
    moving / jumping / raising a hand / lying down are ordinary behaviors,
    not exercises, so they are never scored pass/fail and never fed into
    ExerciseManager.

Thread-safety: every public method takes an internal lock. AIPipeline
threads (one per room) call record(name, behavior) whenever
PoseEstimator.process() reports a newly-confirmed behavior event for the
currently recognized person in that room.

Every BEHAVIOR_SNAPSHOT_INTERVAL_SEC seconds, a background thread writes
each person's current running totals into face_database.behavior_log,
stamped with the real wall-clock time -- see face_database.py's
log_behavior_snapshot().
"""

import threading

import config
import face_database

BEHAVIORS = ("stand", "move", "jump", "raise_hand", "lie", "squat", "pushup")

SNAPSHOT_INTERVAL_SEC = getattr(config, "BEHAVIOR_SNAPSHOT_INTERVAL_SEC", 5)


class BehaviorManager:
    def __init__(self, snapshot_interval=SNAPSHOT_INTERVAL_SEC):
        self._lock = threading.Lock()
        self._counts = {}  # name -> {behavior: int}, running totals
        self._snapshot_interval = snapshot_interval
        self._stop_event = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------
    # Called every frame by AIPipeline (via PoseEstimator's returned
    # behavior_events) for the currently recognized registered person.
    # ------------------------------------------------------------------
    def record(self, name, behavior):
        """Increment one behavior counter for one person by 1. Unknown
        behavior strings are ignored (defensive against typos)."""
        if behavior not in BEHAVIORS:
            print(f"[BehaviorManager] Ignoring unknown behavior '{behavior}' for '{name}'")
            return
        with self._lock:
            row = self._counts.setdefault(name, {b: 0 for b in BEHAVIORS})
            row[behavior] += 1

    def record_many(self, name, behaviors):
        """Convenience for PoseEstimator.process(), which can report more
        than one confirmed event on the same frame (e.g. a squat rep
        finishing on the same frame a hand-raise is confirmed)."""
        for b in behaviors:
            self.record(name, b)

    # ------------------------------------------------------------------
    # Read access (Flask /api/behaviors, dashboard table)
    # ------------------------------------------------------------------
    def get_counts(self, name):
        with self._lock:
            return dict(self._counts.get(name, {b: 0 for b in BEHAVIORS}))

    def get_all_counts(self):
        with self._lock:
            return {name: dict(c) for name, c in self._counts.items()}

    def get_table(self):
        """Row-list shape matching exercise_manager.get_table()'s style,
        handy for a dashboard endpoint that mirrors /api/exercises."""
        with self._lock:
            return [
                {"name": name, **counts} for name, counts in self._counts.items()
            ]

    def forget(self, name):
        """Drop a person's in-memory running counts (e.g. they were
        deleted/unregistered). Rows already written to behavior_log are
        left alone -- that table is a historical report, not live state."""
        with self._lock:
            self._counts.pop(name, None)

    # ------------------------------------------------------------------
    # Background snapshot loop: every N seconds, persist current totals.
    # ------------------------------------------------------------------
    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(
            f"[BehaviorManager] Snapshotting behavior counts every "
            f"{self._snapshot_interval}s -> face_database.behavior_log"
        )

    def stop(self):
        self._stop_event.set()

    def _run(self):
        while not self._stop_event.wait(self._snapshot_interval):
            snapshot = self.get_all_counts()
            if not snapshot:
                continue
            try:
                face_database.log_behavior_snapshot(snapshot)
            except Exception as e:
                print(f"[BehaviorManager] snapshot write failed: {e}")


# Single shared instance -- import this, don't instantiate your own.
behavior_manager = BehaviorManager()