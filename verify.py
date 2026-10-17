"""The model checks, shared by the live app (lego_coach.py) and the offline replay (replay.py).

    identify_piece(frame)                  which LEGO piece is held up (VLM)
    check_build(step, frame, condition)    is the step built correctly, for one study condition:
        vlm_image   the VLM compares the camera image with a reference photo of the correct step
        vlm_text    the VLM compares the camera image with the written instructions
        llm_judge   the VLM only describes the build; the LLM compares that with the instructions and decides
"""

import json
import re

import cv2

from common import LLM_MODEL, VLM_MODEL, ask, images_content
from steps import BRICK_NAMES, STEPS

CONDITIONS = ["vlm_image", "vlm_text", "llm_judge"]
VERDICT_FORMAT = ('"correct": true or false, "feedback": "one short, kind sentence to the student; '
                  'if something is wrong, say exactly what to fix"')


def models_for(condition):
    return f"{VLM_MODEL} + {LLM_MODEL}" if condition == "llm_judge" else VLM_MODEL


def parse_json(reply):
    """The first {...} object in a model reply, or {} if there is none."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def as_bool(value):
    """true/false from a model, also when it answers "true" as a string; None if missing."""
    if isinstance(value, str):
        return {"true": True, "false": False}.get(value.strip().lower())
    return None if value is None else bool(value)


def ask_vlm(images, prompt):
    reply = ask(VLM_MODEL, [{"role": "user", "content": images_content(images, prompt)}])
    print("  vlm:", reply.strip().replace("\n", " "))
    return reply


def ask_llm(prompt):
    reply = ask(LLM_MODEL, [{"role": "user", "content": prompt}])
    print("  llm:", reply.strip().replace("\n", " "))
    return reply


def instructions_until(step):
    return "\n".join(f"{s['number']}. {s['text']}" for s in STEPS[:step["number"]])


def identify_piece(frame):
    """Which piece is held up: (brick label or 'none', what the VLM saw)."""
    options = "\n".join(f"- {label}: {name}" for label, name in BRICK_NAMES.items())
    data = parse_json(ask_vlm([frame], (
        "A student is holding up one LEGO piece to a robot's camera. Which piece is it?\n"
        f"The possible pieces are:\n{options}\n"
        "Look at its color, size, number of studs and shape. Ignore hands and background.\n"
        'Reply with only JSON: {"observation": "what you see in the hand", '
        '"piece": "one label from the list, or none if no piece is clearly visible"}')))
    answer = re.sub(r"[\s_]", "", str(data.get("piece", ""))).lower()
    piece = next((label for label in BRICK_NAMES if re.sub(r"[\s_]", "", label).lower() == answer), "none")
    return piece, str(data.get("observation", ""))


def check_build_vlm(step, frame, with_reference):
    """The VLM checks the build against the reference photo or the instructions."""
    images = [frame]
    if with_reference:
        images.insert(0, cv2.imread(step["reference"]))
        intro = (f"Image 1 is the REFERENCE: the correct LEGO build after step {step['number']}.\n"
                 "Image 2 is the STUDENT'S build right now, seen from a robot's camera.\n")
    else:
        intro = "The image is the STUDENT'S LEGO build right now, seen from a robot's camera.\n"
    data = parse_json(ask_vlm(images, (
        "You are a friendly robot coach checking a middle school student's LEGO build.\n" + intro
        + f"The instructions so far are:\n{instructions_until(step)}\n\n"
        f"Has the student correctly completed step {step['number']}? Check which pieces are there, their "
        "colors, positions and orientation. Ignore hands, background, lighting and camera angle.\n"
        'Reply with only JSON: {"observation": "the pieces you see on the student\'s build, with colors '
        'and positions", ' + VERDICT_FORMAT + "}")))
    return {"correct": as_bool(data.get("correct")), "feedback": str(data.get("feedback", "")),
            "vlm_said": str(data.get("observation", "")), "llm_said": ""}


def check_build_llm(step, frame):
    """The VLM only describes what it sees; the LLM, which cannot see, decides from that description."""
    description = ask_vlm([frame], (
        "This is a LEGO build seen from a robot's camera. Describe it precisely: every piece with its color "
        "and size (for example blue 2x4 brick, black 1x2 slope), and where each sits relative to the others "
        "(on top of, in front of, left or right of). Ignore hands, background and lighting. "
        "Describe only; do not judge it.")).strip()
    data = parse_json(ask_llm(
        "You are the decision module of a robot LEGO coach for middle school students. You cannot see the "
        f"build; a vision model described it for you:\n\"{description}\"\n\n"
        f"The student should have just finished step {step['number']}. "
        f"Instructions so far:\n{instructions_until(step)}\n\n"
        f"Has the student correctly completed step {step['number']}? Compare the description with the "
        "instructions: which pieces, colors, positions and orientation. If the description is unclear or "
        "incomplete, say it is not correct and ask the student to show the build again.\n"
        'Reply with only JSON: {"reason": "one sentence on how you decided", ' + VERDICT_FORMAT + "}"))
    return {"correct": as_bool(data.get("correct")), "feedback": str(data.get("feedback", "")),
            "vlm_said": description, "llm_said": str(data.get("reason", ""))}


def check_build(step, frame, condition):
    """Returns dict with correct (bool), feedback (what the robot says), vlm_said, llm_said."""
    if condition == "llm_judge":
        result = check_build_llm(step, frame)
    else:
        result = check_build_vlm(step, frame, with_reference=condition == "vlm_image")
    if result["correct"] is None:  # unreadable reply: ask to look again
        result["feedback"] = "I couldn't quite tell. Can you move your hands away so I can look again?"
    elif not result["feedback"]:
        result["feedback"] = "Great job!" if result["correct"] else "Something doesn't look right. Check the instructions again."
    result["correct"] = bool(result["correct"])
    return result
