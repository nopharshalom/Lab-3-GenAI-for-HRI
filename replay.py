"""Replay: check every build photo from the live sessions with all three conditions, on the same images.

Each live build check (lego_coach.py --study) saved its camera photo and the experimenter's ground truth
in trials.csv. This sends every one of those photos through vlm_image, vlm_text and llm_judge, so the
conditions are compared on identical images (a paired comparison).

    python replay.py --api                         # OpenAI models (use the same models as the live run)
    python replay.py --api --conditions vlm_image vlm_text
    python analyze.py replay.csv                   # accuracy per condition + paired test

Results go to replay.csv, one row per (photo, condition). Running again skips what is already done,
so an interrupted replay picks up where it stopped.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2

from steps import STEPS
from verify import CONDITIONS, check_build, models_for

FIELDS = ["time", "participant", "condition", "model", "phase", "step", "attempt", "expected",
          "robot_correct", "feedback", "vlm_said", "llm_said",
          "latency_s", "ground_truth", "mistake_type", "image", "live_condition"]

parser = argparse.ArgumentParser()
parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
parser.add_argument("--input", default="trials.csv", help="log of the live sessions")
parser.add_argument("--output", default="replay.csv")
parser.add_argument("--api", action="store_true", help="use OpenAI models (read in common.py)")
args = parser.parse_args()

with open(args.input, newline="") as f:
    photos = [r for r in csv.DictReader(f)
              if r["phase"] == "build" and r["ground_truth"] in ("True", "False") and os.path.exists(r["image"])]
done = set()
if os.path.exists(args.output):
    with open(args.output, newline="") as f:
        done = {(r["image"], r["condition"]) for r in csv.DictReader(f)}
todo = [(live, condition) for live in photos for condition in args.conditions if (live["image"], condition) not in done]
print(f"{len(photos)} build photos with ground truth, {len(todo)} checks to run")

new_file = not os.path.exists(args.output)
with open(args.output, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
    for i, (live, condition) in enumerate(todo, start=1):
        print(f"\n[{i}/{len(todo)}] {condition}: {live['image']}")
        start = time.time()
        result = check_build(STEPS[int(live["step"]) - 1], cv2.imread(live["image"]), condition)
        latency = time.time() - start
        writer.writerow({
            "time": datetime.now().isoformat(timespec="seconds"), "participant": live["participant"],
            "condition": condition, "model": models_for(condition), "phase": "build",
            "step": live["step"], "attempt": live["attempt"], "expected": live["expected"],
            "robot_correct": result["correct"], "feedback": result["feedback"],
            "vlm_said": result["vlm_said"], "llm_said": result["llm_said"], "latency_s": round(latency, 2),
            "ground_truth": live["ground_truth"], "mistake_type": live["mistake_type"],
            "image": live["image"], "live_condition": live["condition"],
        })
        f.flush()  # keep finished checks if the replay is interrupted
print(f"\nDone. Results in {args.output}; summarize with: python analyze.py {args.output}")
