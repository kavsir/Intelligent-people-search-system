"""
One-time migration: import the old folder-based dataset
    data/face_db/<name>/embedding.npy + *.jpg
    data/processed/<name>/*.jpg
into the new SQLite database (data/face_dataset.db, see face_database.py).

Run once, from the src/ directory, after pulling the SQLite-backed code:

    python migrate_to_sqlite.py

Safe to re-run -- each person's data in the DB is overwritten from disk
again. The old folders under data/face_db and data/processed are left
untouched; delete them manually once you've confirmed the DB looks right.
"""

import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import face_database


def _load_raw_person(name):
    person_dir = os.path.join(config.FACE_DB_DIR, name)
    embedding_path = os.path.join(person_dir, "embedding.npy")
    if not os.path.isfile(embedding_path):
        print(f"  [skip] no embedding.npy for '{name}'")
        return None, None

    embeddings = np.load(embedding_path)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    valid_ext = (".jpg", ".jpeg", ".png")
    image_files = sorted(
        f for f in os.listdir(person_dir) if f.lower().endswith(valid_ext)
    )
    images = []
    for fname in image_files:
        img = cv2.imread(os.path.join(person_dir, fname))
        if img is not None:
            images.append(img)

    if len(images) != len(embeddings):
        print(
            f"  [warn] '{name}': {len(embeddings)} embedding(s) vs "
            f"{len(images)} image(s) -- importing the overlapping count only"
        )

    return embeddings, images


def _load_processed_person(name):
    proc_dir = os.path.join(config.PROCESSED_DIR, name)
    if not os.path.isdir(proc_dir):
        return []
    valid_ext = (".jpg", ".jpeg", ".png")
    image_files = sorted(
        f for f in os.listdir(proc_dir) if f.lower().endswith(valid_ext)
    )
    images = []
    for fname in image_files:
        img = cv2.imread(os.path.join(proc_dir, fname))
        if img is not None:
            images.append(img)
    return images


def main():
    if not os.path.isdir(config.FACE_DB_DIR):
        print(f"No folder-based dataset found at {config.FACE_DB_DIR}, nothing to migrate.")
        return

    names = sorted(
        n for n in os.listdir(config.FACE_DB_DIR)
        if os.path.isdir(os.path.join(config.FACE_DB_DIR, n))
    )
    print(f"Found {len(names)} person folder(s): {names}")

    for name in names:
        print(f"Migrating '{name}'...")
        embeddings, images = _load_raw_person(name)
        if embeddings is None or not images:
            continue

        n = min(len(embeddings), len(images))
        face_database.save_face_data(name, list(embeddings[:n]), images[:n])

        processed = _load_processed_person(name)
        if processed:
            face_database.save_processed_images(name, processed)

    print(
        "\nMigration finished -> "
        f"{face_database.DB_PATH}\n"
        "Old folders under data/face_db and data/processed were left "
        "untouched -- delete them manually once you've verified the DB "
        "looks right (e.g. via /api/registered_people)."
    )


if __name__ == "__main__":
    main()