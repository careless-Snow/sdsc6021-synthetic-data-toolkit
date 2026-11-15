
import csv
import json
import math
import zipfile
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

base = Path("C_evidence")
bundle = Path("g4_predictions_and_run_records.zip")

conditions = [
    "random_50",
    "clip_50",
    "random_150",
    "clip_150"
]

labels = [
    "Random 50",
    "CLIP 50",
    "Random 150",
    "CLIP 150"
]

seeds = [0, 1, 2]
accuracies = {}

# 1. Read actual G4 prediction records
with zipfile.ZipFile(bundle) as z:
    index = json.loads(
        z.read("handoff/prediction_index.json")
    )

    for entry in index["predictions"]:
        condition = entry["condition_id"]
        seed = entry["seed"]

        if condition == "clip_zero_shot":
            continue

        rows = json.loads(
            z.read(entry["relative_path"])
        )

        assert len(rows) == 10000

        accuracy = sum(
            r["true_label"] == r["predicted_label"]
            for r in rows
        ) / len(rows)

        key = (condition, seed)
        assert key not in accuracies
        accuracies[key] = accuracy

assert len(accuracies) == 15

# 2. Read the independently verified summary
with (base / "07_g4_summary.csv").open(
    encoding="utf-8-sig",
    newline=""
) as f:
    summary = {
        row["condition"]: row
        for row in csv.DictReader(f)
    }

# 3. Create the paired-gain figure
fig, ax = plt.subplots(figsize=(9, 5))

# Keep seed colors consistent across conditions
seed_colors = ["C0", "C1", "C2"]

for i, condition in enumerate(conditions):

    gains = [
        100 * (
            accuracies[condition, seed]
            - accuracies["real_only", seed]
        )
        for seed in seeds
    ]

    avg = mean(gains)
    sd = stdev(gains)

    # Verify against the independent G4 summary
    assert math.isclose(
        avg,
        float(summary[condition]["paired_gain_pp"]),
        abs_tol=0.0001
    )

    assert math.isclose(
        sd,
        float(summary[condition]["sd_gain_pp"]),
        abs_tol=0.0001
    )

    # Plot individual paired-seed gains
    for j, value in enumerate(gains):
        ax.scatter(
            i + (j - 1) * 0.12,
            value,
            color=seed_colors[j],
            marker="o",
            alpha=0.75,
            s=48,
            label=f"Seed {j}" if i == 0 else None,
            zorder=3
        )

    # Plot mean and sample SD
    ax.errorbar(
        i,
        avg,
        yerr=sd,
        fmt="D",
        color="black",
        capsize=6,
        markersize=8,
        linewidth=1.5,
        label="Mean +/- SD" if i == 0 else None,
        zorder=4
    )

    # Place the annotation above the upper error bar
    ax.text(
        i,
        avg + sd + 0.22,
        f"+{avg:.2f} pp",
        ha="center",
        va="bottom",
        fontsize=10
    )

    # Print values for reproducibility
    print(
        f"{condition}: "
        f"gains={[round(x, 3) for x in gains]}, "
        f"mean={avg:.3f} pp, "
        f"SD={sd:.3f} pp"
    )

# 4. Format the chart
ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    color="gray"
)

ax.set_xticks(range(4), labels)

ax.set_ylabel(
    "Paired Accuracy Gain (percentage points)"
)

ax.set_title(
    "Synthetic Data: Paired Gain over Real-Only Baseline"
)

# Leave extra space for upper annotations
ax.set_ylim(-0.2, 5.7)

ax.grid(
    axis="y",
    alpha=0.25
)

# Put the legend in the empty lower-right area
ax.legend(
    loc="lower right",
    frameon=True
)

fig.text(
    0.5,
    0.01,
    "Each dot represents one paired seed; "
    "diamonds show mean +/- sample SD.",
    ha="center",
    fontsize=9
)

fig.tight_layout(
    rect=[0, 0.04, 1, 1]
)

# 5. Save figure
output = base / "11_g4_paired_gain.png"

fig.savefig(
    output,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("RESULT: PAIRED GAIN FIGURE SAVED")
print("Saved:", output)
