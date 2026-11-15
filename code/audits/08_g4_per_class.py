
import csv
import json
import math
import sys
import zipfile
from pathlib import Path
from statistics import mean, stdev

ZIP = "g4_predictions_and_run_records.zip"
CONDITIONS = [
    "real_only", "random_50", "clip_50",
    "random_150", "clip_150"
]
SEEDS = [0, 1, 2]

with zipfile.ZipFile(ZIP) as z:
    index = json.loads(z.read("handoff/prediction_index.json"))
    report = json.loads(z.read("handoff/report.json"))
    classes = index["official_classes"]

    official = {
        (r["condition_id"], r["seed"]): r
        for r in report["results"]
    }

    scores = {}

    for entry in index["predictions"]:
        condition = entry["condition_id"]
        seed = entry["seed"]

        if condition == "clip_zero_shot":
            continue

        rows = json.loads(z.read(entry["relative_path"]))
        counts = [0] * 10
        correct = [0] * 10

        for row in rows:
            k = row["true_label"]
            counts[k] += 1
            correct[k] += int(
                k == row["predicted_label"]
            )

        assert counts == [1000] * 10

        for k, name in enumerate(classes):
            accuracy = correct[k] / counts[k]

            assert math.isclose(
                accuracy,
                official[condition, seed]["per_class_accuracy"][name],
                abs_tol=1e-12
            )

            scores[condition, seed, k] = accuracy

assert len(scores) == 150

summary = []

for condition in CONDITIONS:
    for k, name in enumerate(classes):

        values = [
            scores[condition, seed, k]
            for seed in SEEDS
        ]

        gains = [
            100 * (
                scores[condition, seed, k]
                - scores["real_only", seed, k]
            )
            for seed in SEEDS
        ]

        summary.append([
            condition,
            name,
            round(100 * mean(values), 3),
            round(100 * stdev(values), 3),
            round(mean(gains), 3),
            round(stdev(gains), 3)
        ])

output = Path("C_evidence/08_g4_per_class.csv")

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "condition",
        "class_name",
        "mean_accuracy_pct",
        "sd_accuracy_pp",
        "mean_gain_pp",
        "sd_gain_pp"
    ])
    writer.writerows(summary)

print("Python:", sys.version.split()[0])
print("Conditions:", len(CONDITIONS))
print("Classes:", len(classes))
print("Per-class run checks:", len(scores))
print("Summary rows:", len(summary))
print("RESULT: ALL PER-CLASS CHECKS PASS")
print("Saved:", output)
