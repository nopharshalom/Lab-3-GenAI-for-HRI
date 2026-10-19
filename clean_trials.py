"""Correct the experimenter's ground-truth entries in trials.csv and write trials_clean.csv.

Every correction is listed here so it can be reported. The original trials.csv is not changed.

    python clean_trials.py
    python replay.py --api --input trials_clean.csv
    python analyze.py trials_clean.csv
"""

import csv

# (participant, phase, step, attempt) -> corrected fields, with the reason
CORRECTIONS = {
    # experimenter typed n because the robot was wrong, but the build itself was correct (robot false alarms)
    ("P02", "build", "1", "2"): {"ground_truth": "True", "mistake_type": ""},  # typed "false alarm"
    ("P02", "build", "5", "3"): {"ground_truth": "True", "mistake_type": ""},  # typed "wrong verdict"
    ("P02", "build", "6", "2"): {"ground_truth": "True", "mistake_type": ""},  # typed "wrong verdict"
    ("P02", "build", "6", "3"): {"ground_truth": "True", "mistake_type": ""},  # typed "wrong verdict"
    # mistake type left empty; confirmed by the experimenter afterwards
    ("P04", "build", "6", "1"): {"mistake_type": "wrong_position"},
    # planned wrong-color piece (green instead of red) was entered as correct; same as P02's step 3
    ("P01", "piece", "3", "1"): {"ground_truth": "False"},
    # right piece, robot could not see it: correct, not a mistake
    ("P04", "piece", "1", "2"): {"mistake_type": ""},
}

# typos and variants of the mistake labels
LABELS = {"none": "", "rotate": "rotated", "wrong position": "wrong_position", "wrong color": "wrong_color"}
BUILD_LABELS = {"wrong peice": "wrong_color"}  # P04 step 1: orange brick instead of the blue 1x4
PIECE_LABELS = {"wrong peice": "wrong_piece"}

with open("trials.csv", newline="") as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, list(reader)

applied = 0
for r in rows:
    key = (r["participant"], r["phase"], r["step"], r["attempt"])
    if key in CORRECTIONS:
        r.update(CORRECTIONS[key])
        applied += 1
    label = r["mistake_type"].strip()
    label = LABELS.get(label, label)
    r["mistake_type"] = (BUILD_LABELS if r["phase"] == "build" else PIECE_LABELS).get(label, label)
assert applied == len(CORRECTIONS), f"only {applied} of {len(CORRECTIONS)} corrections matched a row"

with open("trials_clean.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"Applied {applied} corrections; wrote trials_clean.csv ({len(rows)} rows)")
