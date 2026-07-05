"""
Identity recognition: matches a detected face crop against the embeddings
saved by the registration app under data/face_db/<name>/embedding.npy.

This module is what was missing from the operation pipeline -- without it,
ai_pipeline.py could detect "a face" but never knew *whose* face it was, so
it tracked anybody rather than the registered target(s).

Usage:
    recognizer = FaceRecognizer()
    recognizer.load_database()           # call once at startup
    name, score = recognizer.identify(face_crop_bgr)
    # name is None if no registered person matched above the threshold
"""

import os
import sys

import numpy as np
from insightface.app import FaceAnalysis

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import face_database

# Cosine similarity threshold above which a face crop is considered a match
# for a given registered person. InsightFace's normed_embedding makes cosine
# similarity equivalent to a plain dot product. 0.5 matches the threshold
# documented in README.md; tune via config if needed.
RECOGNITION_THRESHOLD = getattr(config, "FACE_RECOGNITION_THRESHOLD", 0.5)


class FaceRecognizer:
    """Loads data/face_db/* once, then matches face crops against it cheaply."""

    def __init__(self):
        self.device_ctx_id = config.get_insightface_ctx_id()
        self._face_app = FaceAnalysis(name="buffalo_sc")
        try:
            self._face_app.prepare(ctx_id=self.device_ctx_id, det_size=(320, 320))
            print(
                f"[FaceRecognizer] InsightFace running on "
                f"{'GPU' if self.device_ctx_id >= 0 else 'CPU'}."
            )
        except Exception as exc:
            if self.device_ctx_id != -1:
                print(f"[FaceRecognizer] GPU init failed ({exc}); falling back to CPU.")
                self.device_ctx_id = -1
                self._face_app.prepare(ctx_id=-1, det_size=(320, 320))
            else:
                raise

        # name -> np.ndarray of shape (N, 512), one row per registered angle
        self.database = {}

        # face_db_version we last loaded (see face_database.get_face_db_version()).
        # -1 guarantees the very first reload_if_changed() call reloads.
        self._loaded_version = -1

    def load_database(self):
        """
        Load every registered person's embeddings from the SQLite face
        dataset (data/face_dataset.db) -- see face_database.py. Safe to
        call again later to pick up newly registered people without
        restarting the operation app.
        """
        self.database = face_database.load_all_embeddings()
        self._loaded_version = face_database.get_face_db_version()

        total_people = len(self.database)
        total_angles = sum(e.shape[0] for e in self.database.values())
        print(
            f"[FaceRecognizer] Loaded {total_people} registered person(s), "
            f"{total_angles} embedding(s) total."
        )

    def reload_if_changed(self):
        """
        Cheap check (one small SELECT): compare face_database's
        face_db_version counter against what we last loaded. If it moved
        -- meaning app_registration.py (a SEPARATE process) just
        registered or deleted someone -- reload the in-memory embeddings
        right away.

        This is what makes a newly-registered person recognizable
        immediately, without restarting app_dashboard.py / app_operation.py.
        Call this periodically (e.g. every couple of seconds) from
        AIPipeline's loop -- NOT every frame, since it hits the database.

        Returns True if a reload actually happened.
        """
        try:
            current_version = face_database.get_face_db_version()
        except Exception as e:
            print(f"[FaceRecognizer] version check failed: {e}")
            return False

        if current_version != self._loaded_version:
            print(
                f"[FaceRecognizer] face_db_version changed "
                f"({self._loaded_version} -> {current_version}), reloading database..."
            )
            self.load_database()
            return True
        return False

    def is_empty(self):
        return len(self.database) == 0

    def get_embedding(self, face_crop_bgr):
        """Run InsightFace on a face crop and return its 512-D embedding, or None."""
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None

        faces = self._face_app.get(face_crop_bgr)
        if not faces:
            return None

        # If multiple faces somehow appear in the crop, take the largest one.
        best_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        return best_face.normed_embedding

    def identify(self, face_crop_bgr, threshold=None):
        """
        Identify the person in a face crop.

        Returns (name, score) where name is None if nobody in the database
        matches above the threshold (including the case where the database
        is empty or no face embedding could be extracted from the crop).
        """
        threshold = RECOGNITION_THRESHOLD if threshold is None else threshold

        if self.is_empty():
            return None, 0.0

        embedding = self.get_embedding(face_crop_bgr)
        if embedding is None:
            return None, 0.0

        return self.identify_from_embedding(embedding, threshold=threshold)

    def identify_from_embedding(self, embedding, threshold=None):
        """Same as identify(), but takes an already-computed embedding."""
        threshold = RECOGNITION_THRESHOLD if threshold is None else threshold

        if self.is_empty() or embedding is None:
            return None, 0.0

        best_name = None
        best_score = -1.0

        for name, stored_embeddings in self.database.items():
            # normed_embedding rows -> cosine similarity is a plain dot product
            similarities = stored_embeddings @ embedding
            score = float(np.max(similarities))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= threshold:
            return best_name, best_score
        return None, best_score