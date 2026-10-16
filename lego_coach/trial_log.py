"""One CSV row per checked step, plus the photo the VLM saw."""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path

import cv2

FIELDS = [
    "trial_id", "timestamp", "participant", "build", "step", "condition", "model",
    "ground_truth", "predicted_correct", "predicted_error_type",
    "detector_counts_match", "n_bricks_detected", "brick_detector",
    "trigger", "hands_clear_wait_s", "vlm_latency_s", "response_latency_s",
    "parse_ok", "feedback", "raw_reply", "image",
]


class TrialLog:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.images_dir = data_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "trials.csv"
        self._lock = threading.Lock()
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, FIELDS).writeheader()

    def next_trial_id(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def save_images(self, trial_id: str, frame, annotated=None) -> str:
        path = self.images_dir / f"{trial_id}.jpg"
        cv2.imwrite(str(path), frame)
        if annotated is not None:
            cv2.imwrite(str(self.images_dir / f"{trial_id}_annotated.jpg"), annotated)
        return str(path.relative_to(self.data_dir))

    def write(self, row: dict):
        row = {k: row.get(k, "") for k in FIELDS}
        with self._lock, self.path.open("a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerow(row)

    def recent(self, n: int = 10) -> list[dict]:
        with self._lock, self.path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        return rows[-n:][::-1]


def counts_match(bricks, expected: dict | None) -> str:
    """Detector-only baseline: do the detected color counts equal the expected ones?"""
    if not expected:
        return ""
    found: dict[str, int] = {}
    for b in bricks:
        found[b.color] = found.get(b.color, 0) + 1
    return str(found == expected)
