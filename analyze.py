"""Summarize a log for the report: accuracy, missed mistakes, false alarms and latency per condition
(only rows with ground truth, i.e. runs with --study). For replay.csv, where every condition checked
the same photos, it also compares each pair of conditions with an exact McNemar test.

    python analyze.py                # the live sessions (trials.csv)
    python analyze.py replay.csv     # all conditions on the same photos
"""

import csv
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from math import comb

path = sys.argv[1] if len(sys.argv) > 1 else "trials.csv"
with open(path, newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["ground_truth"] in ("True", "False")]
if not rows:
    sys.exit(f"No rows with ground truth in {path} (run lego_coach.py with --study).")


def mean_sd(values):
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.2f} ± {sd:.2f}"


def summarize(title, group_rows):
    truth = [r["ground_truth"] == "True" for r in group_rows]
    said = [r["robot_correct"] == "True" for r in group_rows]
    accuracy = sum(t == s for t, s in zip(truth, said)) / len(group_rows)
    mistakes = [s for t, s in zip(truth, said) if not t]  # it was wrong
    good = [s for t, s in zip(truth, said) if t]  # it was right
    missed = f"{sum(mistakes) / len(mistakes):.1%}" if mistakes else "-"  # robot said OK on a mistake
    false_alarm = f"{(len(good) - sum(good)) / len(good):.1%}" if good else "-"  # robot flagged a correct one
    latency = mean_sd([float(r["latency_s"]) for r in group_rows])
    print(f"{title:<12} n={len(group_rows):<4} accuracy={accuracy:.1%}  missed mistakes={missed}  "
          f"false alarms={false_alarm}  latency(s)={latency}")


for phase in ("build", "piece"):
    by_condition = defaultdict(list)
    for r in rows:
        if r["phase"] == phase:
            by_condition[r["condition"]].append(r)
    if by_condition:
        print(f"\n== {phase} checks ==")
        for condition, group in sorted(by_condition.items()):
            summarize(condition, group)

# RQ1: which kinds of mistakes are missed most often (robot said correct on a wrong build)
mistakes = defaultdict(lambda: [0, 0])
for r in rows:
    if r["phase"] == "build" and r["ground_truth"] == "False":
        kind = f"{r['condition']} / {r['mistake_type'] or 'unlabeled'}"
        mistakes[kind][0] += r["robot_correct"] == "True"
        mistakes[kind][1] += 1
if mistakes:
    print("\n== missed mistakes by type ==")
    for kind, (missed, total) in sorted(mistakes.items()):
        print(f"{kind:<40} missed {missed}/{total}")

# H2: paired comparison on photos that every condition checked (replay.csv)
right = defaultdict(dict)  # photo -> condition -> was the robot right?
for r in rows:
    if r["phase"] == "build":
        right[r["image"]][r["condition"]] = (r["robot_correct"] == "True") == (r["ground_truth"] == "True")
conditions = sorted({c for by_condition in right.values() for c in by_condition})
pairs = [(a, b) for a, b in combinations(conditions, 2)
         if any(a in by_condition and b in by_condition for by_condition in right.values())]
if pairs:
    print("\n== paired comparison on the same photos (exact McNemar test) ==")
    for a, b in pairs:
        both = [by_condition for by_condition in right.values() if a in by_condition and b in by_condition]
        only_a = sum(c[a] and not c[b] for c in both)  # a right, b wrong
        only_b = sum(c[b] and not c[a] for c in both)  # b right, a wrong
        n = only_a + only_b
        p = min(1.0, 2 * sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
        print(f"{a} vs {b}: {len(both)} photos, only {a} right: {only_a}, only {b} right: {only_b}, p = {p:.3f}")
