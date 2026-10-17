"""LEGO Coach: Reachy Mini watches a student build a LEGO model and checks every step.

For each step in steps.py:
  1. Piece check: the robot asks for the next piece; the student holds it up and the VLM names it.
  2. Build check: the robot reads the instruction; the student builds and presses Enter; the robot
     looks down at the build and checks it, then nods, or tilts its head and says what to fix.
     A wrong step is retried until it is right.

How the build is checked (--verify, the study condition):
    vlm_image   the VLM compares the camera image with a reference photo of the correct step (default)
    vlm_text    the VLM compares the camera image with the written instructions
    llm_judge   the VLM only describes the build; the LLM compares that with the instructions and decides

    python lego_coach.py                       # free Ollama models, laptop voice
    python lego_coach.py --api                 # OpenAI models, robot voice
    python lego_coach.py --verify vlm_text
    python lego_coach.py --study P01           # log ground truth for the evaluation
    python lego_coach.py --start 3             # start at step 3
    python lego_coach.py --no-piece-check      # skip holding up each piece

Every check is logged to trials.csv and its camera frame saved in captures/; replay.py then checks
the same photos with all three conditions. At the prompt: Enter = check, s = skip this check, q = quit.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2

from common import LOOK_DOWN_PITCH, connect_robot, get_frame, move_antennas, move_head, speak, wiggle_antennas
from steps import BRICK_NAMES, STEPS
from verify import CONDITIONS, check_build, identify_piece, models_for

LOG_FILE = "trials.csv"
LOG_FIELDS = ["time", "participant", "condition", "model", "phase", "step", "attempt", "expected",
              "robot_correct", "feedback", "vlm_said", "llm_said",
              "latency_s", "ground_truth", "mistake_type", "image"]

parser = argparse.ArgumentParser()
parser.add_argument("--verify", choices=CONDITIONS, default="vlm_image")
parser.add_argument("--study", metavar="PARTICIPANT", help="ask the experimenter for ground truth after every check")
parser.add_argument("--start", type=int, default=1, help="step number to start at")
parser.add_argument("--no-piece-check", action="store_true", help="skip holding up each piece")
parser.add_argument("--api", action="store_true", help="use OpenAI models (read in common.py)")
args = parser.parse_args()


# ---------------------------------------------------------------- robot behaviors

head_pitch = 0  # where the head is looking now; gestures happen around it, not from center


def say(mini, text):
    print("reachy>", text)
    speak(mini, text)


def look(mini, pitch):
    """Turn the head to a pitch, only if it is not already there."""
    global head_pitch
    if pitch != head_pitch:
        move_head(mini, pitch=pitch, duration=0.8)
        head_pitch = pitch


def look_at_student(mini):
    look(mini, 0)
    move_antennas(mini, 0, 0)


def look_at_build(mini):
    look(mini, LOOK_DOWN_PITCH)


def neutral(mini):
    """Head centered and level, antennas up: the resting pose when the program ends."""
    global head_pitch
    move_head(mini, duration=1.0)
    move_antennas(mini, 0, 0, duration=0.5)
    head_pitch = 0


def thinking(mini):
    """Antennas perk up while the models work; the head stays still."""
    move_antennas(mini, 0.4, 0.4)


def celebrate(mini):
    """Small nod where the head is looking, plus an antenna wiggle."""
    move_head(mini, pitch=head_pitch + 6, duration=0.3)
    move_head(mini, pitch=head_pitch, duration=0.3)
    wiggle_antennas(mini)


def concerned(mini):
    """Small head tilt with drooping antennas: something is not right."""
    move_head(mini, pitch=head_pitch, roll=-8, duration=0.6)
    move_antennas(mini, 1.0, -1.0, duration=0.6)
    move_head(mini, pitch=head_pitch, duration=0.6)


# ---------------------------------------------------------------- logging

def ground_truth():
    """In study mode the experimenter records what was really true."""
    if not args.study:
        return "", ""
    truth = ""
    while truth not in ("y", "n"):
        truth = input("  [experimenter] was it actually correct? (y/n) ").strip().lower()
    mistake = input("  [experimenter] mistake type (Enter = none): ").strip()
    return truth == "y", mistake


def record(phase, step, attempt, expected, frame, latency, robot_correct, feedback, vlm_said="", llm_said=""):
    """Append one check to trials.csv and save its camera frame in captures/."""
    truth, mistake = ground_truth()
    os.makedirs("captures", exist_ok=True)
    image = f"captures/{args.study or 'demo'}_{args.verify}_step{step}_{phase}{attempt}_{datetime.now():%Y%m%d-%H%M%S}.jpg"
    cv2.imwrite(image, frame)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, newline="") as f:
            header = next(csv.reader(f), [])
        if header != LOG_FIELDS:  # log from an older version: keep it, start a new one (file must be closed first)
            os.replace(LOG_FILE, f"trials_old_{datetime.now():%Y%m%d-%H%M%S}.csv")
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "time": datetime.now().isoformat(timespec="seconds"), "participant": args.study or "",
            "condition": args.verify,
            "model": models_for(args.verify),
            "phase": phase, "step": step, "attempt": attempt, "expected": expected,
            "robot_correct": robot_correct, "feedback": feedback, "vlm_said": vlm_said, "llm_said": llm_said,
            "latency_s": round(latency, 2), "ground_truth": truth, "mistake_type": mistake, "image": image,
        })


# ---------------------------------------------------------------- interaction

def prompt():
    """Wait for the student. Returns 'check', 'skip' or 'quit'."""
    key = input("  [Enter = check, s = skip, q = quit] ").strip().lower()
    return {"s": "skip", "q": "quit"}.get(key, "check")


def piece_check(mini, step):
    expected = step["brick"]
    name = BRICK_NAMES[expected]
    look_at_student(mini)
    say(mini, f"Step {step['number']}. Find the {name}, and hold it up so I can see it.")
    attempt = 0
    while (action := prompt()) == "check":
        attempt += 1
        start = time.time()
        frame = get_frame(mini)
        thinking(mini)
        piece, observation = identify_piece(frame)
        latency = time.time() - start
        correct = piece == expected
        if correct:
            feedback = f"Yes, that's the {name}!"
            wiggle_antennas(mini)
        elif piece == "none":
            feedback = "I can't see it clearly. Hold it a bit closer to my camera."
            concerned(mini)
        else:
            feedback = f"That looks like the {BRICK_NAMES[piece]}. We need the {name}."
            concerned(mini)
        say(mini, feedback)
        record("piece", step["number"], attempt, expected, frame, latency, correct, feedback,
               vlm_said=f"{piece}: {observation}")
        look_at_student(mini)
        if correct:
            return "next"
    return action


def build_check(mini, step):
    say(mini, step["text"])
    attempt = 0
    while (action := prompt()) == "check":
        attempt += 1
        look_at_build(mini)
        start = time.time()
        frame = get_frame(mini)
        thinking(mini)
        result = check_build(step, frame, args.verify)
        latency = time.time() - start
        print(f"  verdict: {'correct' if result['correct'] else 'not correct'} ({latency:.1f} s)")
        if result["correct"]:
            celebrate(mini)
        else:
            concerned(mini)
        say(mini, result["feedback"])
        record("build", step["number"], attempt, f"STEP_{step['number']}", frame, latency, result["correct"],
               result["feedback"], result["vlm_said"], result["llm_said"])
        move_antennas(mini, 0, 0)  # the head keeps watching the build until the step is right
        if result["correct"]:
            return "next"
    return action


def main():
    if args.verify == "vlm_image":
        missing = [s["reference"] for s in STEPS if not os.path.exists(s["reference"])]
        if missing:
            raise SystemExit("Missing reference photos for --verify vlm_image:\n  " + "\n  ".join(missing))
    print(f"Condition: {args.verify}")
    with connect_robot() as mini:
        try:
            look_at_student(mini)
            wiggle_antennas(mini)
            say(mini, "Hi! I'm your LEGO coach. Let's build together, one step at a time. "
                      "I'll check each step for you.")
            for step in STEPS[args.start - 1:]:
                if not args.no_piece_check and piece_check(mini, step) == "quit":
                    break
                if build_check(mini, step) == "quit":
                    break
            else:
                look_at_student(mini)
                celebrate(mini)
                say(mini, "You finished the whole build. Amazing work!")
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            neutral(mini)  # on q, Ctrl+C or a crash


if __name__ == "__main__":
    main()
