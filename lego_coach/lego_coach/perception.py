"""Perception with Google MediaPipe.

Two jobs:

1. HandWatcher (MediaPipe Gesture Recognizer)
   - A held thumbs-up means "I finished this step, please check it" (non-verbal input).
   - The robot waits until the student's hands leave the camera view before taking the
     photo it checks, so hands don't hide the bricks.

2. BrickDetector
   - With a custom MediaPipe Object Detector trained on LEGO bricks (LEGO_DETECTOR_MODEL),
     bricks are detected by that model.
   - Without one, bricks are found by color (OpenCV), skipping any area covered by a hand.
   MediaPipe's stock object detector is trained on everyday objects (COCO) and has no
   LEGO class, which is why the color detector is the default.
"""

from __future__ import annotations

import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config


# ---------------------------------------------------------------- model files

def ensure_model(url: str, name: str) -> Path:
    """Download a MediaPipe model file once and cache it."""
    path = config.MODELS_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe model {name} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def _mp_image(frame_bgr: np.ndarray):
    import mediapipe as mp

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


# ---------------------------------------------------------------- hands

@dataclass
class HandState:
    n_hands: int = 0
    thumbs_up: bool = False
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # x, y, w, h in pixels


class HandWatcher:
    """Wraps the MediaPipe Gesture Recognizer (hand landmarks + gesture labels)."""

    def __init__(self, min_confidence: float = 0.5):
        from mediapipe.tasks.python import BaseOptions, vision

        model = ensure_model(config.GESTURE_MODEL_URL, "gesture_recognizer.task")
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._last_ts = 0

    def observe(self, frame_bgr: np.ndarray) -> HandState:
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # VIDEO mode needs increasing timestamps
        self._last_ts = ts
        result = self._recognizer.recognize_for_video(_mp_image(frame_bgr), ts)

        h, w = frame_bgr.shape[:2]
        state = HandState(n_hands=len(result.hand_landmarks))
        for landmarks in result.hand_landmarks:
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            pad = 0.25 * max(max(xs) - min(xs), max(ys) - min(ys))  # cover the palm and fingers
            x0, y0 = int(max(0, min(xs) - pad)), int(max(0, min(ys) - pad))
            x1, y1 = int(min(w, max(xs) + pad)), int(min(h, max(ys) + pad))
            state.boxes.append((x0, y0, x1 - x0, y1 - y0))
        for gestures in result.gestures:
            if gestures and gestures[0].category_name == "Thumb_Up" and gestures[0].score > 0.6:
                state.thumbs_up = True
        return state

    def close(self):
        self._recognizer.close()


# ---------------------------------------------------------------- bricks

@dataclass
class Brick:
    label: str  # e.g. "red" or a custom detector class such as "red_2x4"
    color: str
    box: tuple[int, int, int, int]  # x, y, w, h in pixels
    score: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h / 2


# OpenCV hue runs 0-179. Values tuned for bright LEGO plastic under indoor light;
# check them with scripts/check_colors.py on your own table and adjust if needed.
HSV_RANGES = {
    "red": [((0, 120, 70), (8, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "orange": [((9, 140, 100), (20, 255, 255))],
    "yellow": [((21, 100, 100), (35, 255, 255))],
    "green": [((36, 80, 50), (85, 255, 255))],
    "blue": [((90, 100, 50), (130, 255, 255))],
}


class BrickDetector:
    def __init__(self, min_area_frac: float = 0.0015):
        self.min_area_frac = min_area_frac
        self._detector = None
        self.source = "color"
        if config.LEGO_DETECTOR_MODEL:
            from mediapipe.tasks.python import BaseOptions, vision

            options = vision.ObjectDetectorOptions(
                base_options=BaseOptions(model_asset_path=config.LEGO_DETECTOR_MODEL),
                running_mode=vision.RunningMode.IMAGE,
                score_threshold=0.4,
                max_results=30,
            )
            self._detector = vision.ObjectDetector.create_from_options(options)
            self.source = "mediapipe"

    def detect(self, frame_bgr: np.ndarray, hand_boxes=()) -> list[Brick]:
        if self._detector is not None:
            return self._detect_mediapipe(frame_bgr)
        return self._detect_color(frame_bgr, hand_boxes)

    def _detect_mediapipe(self, frame_bgr):
        result = self._detector.detect(_mp_image(frame_bgr))
        bricks = []
        for det in result.detections:
            cat = det.categories[0]
            b = det.bounding_box
            color = cat.category_name.split("_")[0]  # "red_2x4" -> "red"
            bricks.append(Brick(cat.category_name, color, (b.origin_x, b.origin_y, b.width, b.height), cat.score))
        return bricks

    def _detect_color(self, frame_bgr, hand_boxes):
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
        ignore = np.zeros((h, w), np.uint8)
        for x, y, bw, bh in hand_boxes:
            ignore[y:y + bh, x:x + bw] = 255
        kernel = np.ones((5, 5), np.uint8)
        min_area = self.min_area_frac * h * w

        bricks = []
        for color, ranges in HSV_RANGES.items():
            mask = np.zeros((h, w), np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask &= cv2.bitwise_not(ignore)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) >= min_area:
                    bricks.append(Brick(color, color, cv2.boundingRect(c)))
        return bricks


def summarize(bricks: list[Brick], frame_shape) -> str:
    """Short text description of the detected bricks, added to the VLM prompt."""
    if not bricks:
        return "No bricks detected."
    h, w = frame_shape[:2]
    counts = Counter(b.label for b in bricks)
    count_text = ", ".join(f"{n} {label}" for label, n in counts.most_common())
    lines = [f"{len(bricks)} bricks detected: {count_text}."]
    for b in sorted(bricks, key=lambda b: (b.center[1], b.center[0])):
        cx, cy = b.center
        horiz = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
        vert = "top" if cy < h / 3 else "bottom" if cy > 2 * h / 3 else "middle"
        lines.append(f"- {b.label} at {vert}-{horiz} of the image")
    return "\n".join(lines)


def annotate(frame_bgr: np.ndarray, bricks: list[Brick], hand_boxes=()) -> np.ndarray:
    """Copy of the frame with boxes drawn on it, saved next to each trial for checking."""
    out = frame_bgr.copy()
    for b in bricks:
        x, y, w, h = b.box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, b.label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    for x, y, w, h in hand_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 2)
    return out
