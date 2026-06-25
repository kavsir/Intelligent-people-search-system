"""
Structured event logging shared across the operation pipeline.

The assignment explicitly requires logging events such as "target detected",
"target lost", and "servo angle updated" (see Bao_Cao_Dinh_Huong, section 3
and 7.4). This module gives every component (AI pipeline, servo controller,
dashboard) a single, thread-safe place to:

  1. Write structured event rows to a CSV log file on disk (for the
     technical report / experiment results).
  2. Keep a small in-memory ring buffer of the most recent events, so the
     live dashboard can show a scrolling event feed without re-reading the
     file from disk every frame.

Usage:
    logger = EventLogger()
    logger.log("TARGET_LOST", state="SEARCHING")
    logger.log("IDENTITY_MATCH", name="alice", score=0.71)
    recent = logger.get_recent(20)   # for the dashboard
"""

import csv
import os
import threading
import time
from collections import deque


class EventLogger:
    def __init__(self, log_dir=None, max_buffer=200):
        if log_dir is None:
            # src/operation/event_logger.py -> project root is two levels up
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            log_dir = os.path.join(project_root, "logs")

        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"session_{timestamp}.csv")

        self._lock = threading.Lock()
        self._buffer = deque(maxlen=max_buffer)

        self._file = open(self.log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "elapsed_sec", "event", "details"])
        self._file.flush()

        self._start_time = time.time()
        print(f"[EventLogger] Writing session log to {self.log_path}")

    def log(self, event, **details):
        """Record one event. Extra keyword args are stored as 'k=v' pairs."""
        now = time.time()
        elapsed = now - self._start_time
        ts_str = time.strftime("%H:%M:%S", time.localtime(now))
        details_str = ", ".join(f"{k}={v}" for k, v in details.items())

        with self._lock:
            self._writer.writerow([ts_str, f"{elapsed:.2f}", event, details_str])
            self._file.flush()
            self._buffer.append((ts_str, event, details_str))

    def get_recent(self, n=20):
        """Return up to the last n (timestamp, event, details) tuples."""
        with self._lock:
            return list(self._buffer)[-n:]

    def close(self):
        with self._lock:
            if self._file and not self._file.closed:
                self._file.close()