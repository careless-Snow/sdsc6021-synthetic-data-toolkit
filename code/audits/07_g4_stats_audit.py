
import csv
import json
import math
import sys
import zipfile
from pathlib import Path
from statistics import mean, stdev

ZIP = "g4_predictions_and_run_records.zip"
CONDITIONS = [
    "real_only",
    "random_50",
    "clip_50",
    "random_150",
    "clip_150"
]
SEEDS = [0, 1, 2]

with zipfile.ZipFile(ZIP) as z:
    index = json.loads(z.read("handoff/prediction_index.json"))
    report = json.loads(z.read("handoff/report.json"))

    accuracies = {}
    zero_shot = None

    # Independently recompute accuracy from predictions
    for entry in index["predictions"]:
        rows = json.loads(z.read(entry["relative_path"]))
        accuracy = sum(
            row["true_label"] == row["predicted_label"]
            for row in rows
        ) / len(rows)

        condition = entry["condition_id"]
        seed = entry["seed"]

        if condition == "clip_zero_shot":
            zero_shot = accuracy
        else:
            key = (condition, seed)
            assert key not in accuracies
            accuracies[key] = accuracy

assert len(accuracies) == 15

official = {
    row["condition_id"]: row
    for row in report["summary"]
}

results = []

print("Python:", sys.version.split()[0])
print("G4 OFFICIAL PREDICTION STATISTICS")
print("-" * 68)

for condition in CONDITIONS:
    values = [accuracies[condition, s] for s in SEEDS]

    gains = [
        100 * (accuracies[condition, s] -
               accuracies["real_only", s])
        for s in SEEDS
    ]

    avg = mean(values)
    sd = stdev(values)
    gain_avg = mean(gains)
    gain_sd = stdev(gains)

    # Cross-check against A's official summary
    expected = official[condition]

    assert math.isclose(
        avg, expected["mean_accuracy"], abs_tol=1e-12
    )
    assert math.isclose(
        sd, expected["std_accuracy_ddof1"], abs_tol=1e-12
    )
    assert math.isclose(
        gain_avg, expected["paired_delta_pp_mean"],
        abs_tol=1e-12
    )
    assert math.isclose(
        gain_sd, expected["paired_delta_pp_std_ddof1"],
        abs_tol=1e-12
    )

    results.append([
        condition,
        round(100 * avg, 4),
        round(100 * sd, 4),
        round(gain_avg, 4),
        round(gain_sd, 4)
    ])

    print(
        f"{condition:12s} "
        f"Accuracy={100*avg:6.2f}% "
        f"SD={100*sd:.3f}pp "
        f"Gain={gain_avg:+.3f}pp "
        f"PASS"
    )

assert math.isclose(
    zero_shot,
    report["clip_zero_shot"]["accuracy"],
    abs_tol=1e-12
)
print(f"CLIP zero-shot: {100*zero_shot:.2f}% PASS")

# Save independently recomputed summary
output = Path("C_evidence/07_g4_summary.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "condition",
        "mean_accuracy_pct",
        "sd_accuracy_pct",
        "paired_gain_pp",
        "sd_gain_pp"
    ])
    writer.writerows(results)

print("-" * 68)
print("RESULT: ALL OFFICIAL STATISTICS CHECKS PASS")
print("Saved:", output)
