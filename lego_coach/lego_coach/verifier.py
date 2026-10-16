"""Step verification with a vision-language model (VLM).

For each step the VLM gets the current camera photo of the student's build, the step
instruction and, depending on the experimental condition, a reference photo of the
correctly finished step. It answers in JSON: is the step correct, which kind of mistake,
and one short hint for the student.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

from . import config

ERROR_TYPES = ("none", "missing_brick", "wrong_color", "wrong_position", "wrong_orientation", "other")

SYSTEM_PROMPT = """You are the eyes of a friendly robot that coaches a middle school student through a LEGO build.
You check ONE step at a time by looking closely at the photo of the student's build.
Be strict: a step is correct only if every brick asked for in this step has the right color,
position and orientation. Bricks from earlier steps should still be in place."""

PROMPT = """Step {number} of {total}: "{instruction}"
{reference_note}{brick_note}
Look at the photo of the student's build{current_note} and decide if this step was done correctly.

Answer with JSON only, no other text:
{{"correct": true or false,
  "error_type": one of {error_types},
  "feedback": "one or two short, encouraging sentences for a 12-year-old. If there is a mistake, say which brick or area to look at, but do not give the full answer."}}"""


@dataclass
class Verdict:
    correct: bool
    error_type: str
    feedback: str
    raw: str
    parse_ok: bool
    vlm_latency_s: float


def encode_image(frame: np.ndarray, max_width: int = 640) -> str:
    h, w = frame.shape[:2]
    if w > max_width:  # smaller image = faster and cheaper (same as the course demos)
        frame = cv2.resize(frame, (max_width, h * max_width // w))
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


def load_build(name: str) -> dict:
    path = config.BUILDS_DIR / f"{name}.json"
    build = json.loads(path.read_text())
    build["id"] = name
    return build


def list_builds() -> list[str]:
    return sorted(p.stem for p in config.BUILDS_DIR.glob("*.json"))


def reference_path(build: dict, step_index: int) -> Path:
    """Where the reference photo for a step lives (it may not exist yet)."""
    step = build["steps"][step_index]
    rel = step.get("reference_image") or f"{build['id']}/step{step_index + 1}.jpg"
    return config.BUILDS_DIR / rel


def parse_reply(text: str) -> tuple[bool, str, str, bool]:
    """Pull correct / error_type / feedback out of the model's reply."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            correct = data.get("correct")
            if isinstance(correct, str):
                correct = correct.strip().lower() == "true"
            error = str(data.get("error_type", "other")).strip().lower()
            if error not in ERROR_TYPES:
                error = "other"
            if correct:
                error = "none"
            elif error == "none":
                error = "other"
            return bool(correct), error, str(data.get("feedback", "")).strip(), True
        except (json.JSONDecodeError, AttributeError):
            pass
    # Fallback for small models that ignore the JSON format.
    lowered = text.lower()
    correct = '"correct": true' in lowered or (
        "correct" in lowered and not any(w in lowered for w in ("incorrect", "not correct", "mistake", "wrong", "false"))
    )
    error = "none" if correct else next((e for e in ERROR_TYPES[1:] if e in lowered), "other")
    return correct, error, text.strip()[:200], False


class StepVerifier:
    def __init__(self):
        self.client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)
        self.model = config.VLM_MODEL

    def verify(self, frame: np.ndarray, build: dict, step_index: int, condition: str,
               brick_summary: str | None = None) -> Verdict:
        step = build["steps"][step_index]
        content: list[dict] = []

        reference_note, current_note = "", ""
        if condition == "reference_image":
            ref_path = reference_path(build, step_index)
            ref = cv2.imread(str(ref_path))
            if ref is None:
                raise FileNotFoundError(
                    f"No reference photo for step {step_index + 1} at {ref_path}. "
                    "Capture one from the control panel first."
                )
            reference_note = "The FIRST photo shows this step done correctly (the reference).\n"
            current_note = " (the SECOND photo) and compare it with the reference"
            content.append({"type": "image_url", "image_url": {"url": encode_image(ref)}})

        brick_note = ""
        if brick_summary:
            brick_note = ("A brick detector (it can be wrong, trust the photo first) found:\n"
                          f"{brick_summary}\n")

        prompt = PROMPT.format(
            number=step_index + 1, total=len(build["steps"]), instruction=step["instruction"],
            reference_note=reference_note, brick_note=brick_note, current_note=current_note,
            error_types=", ".join(f'"{e}"' for e in ERROR_TYPES),
        )
        content.append({"type": "image_url", "image_url": {"url": encode_image(frame)}})
        content.insert(0, {"type": "text", "text": prompt})

        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
            temperature=0,
        )
        latency = time.perf_counter() - t0
        raw = response.choices[0].message.content or ""
        correct, error, feedback, ok = parse_reply(raw)
        if not feedback:
            feedback = "Great job, that step looks right!" if correct else "Hmm, take another look at this step."
        return Verdict(correct, error, feedback, raw, ok, latency)
