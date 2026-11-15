
import csv
import json
import zipfile
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

bundle = Path("g4_predictions_and_run_records.zip")
output_dir = Path("C_evidence")
output_dir.mkdir(exist_ok=True)

comparisons = [
    ("CLIP vs Random (50)", "clip_50", "random_50"),
    ("CLIP vs Random (150)", "clip_150", "random_150"),
    ("150 vs 50 (Random)", "random_150", "random_50"),
    ("150 vs 50 (CLIP)", "clip_150", "clip_50"),
]

seeds = (0, 1, 2)
accuracies = {}

# 1. Recompute accuracy from actual G4 predictions
with zipfile.ZipFile(bundle) as z:
    index = json.loads(
        z.read("handoff/prediction_index.json")
    )

    for entry in index["predictions"]:
        condition = entry["condition_id"]
        seed = entry["seed"]

        if condition == "clip_zero_shot":
            continue

        records = json.loads(
            z.read(entry["relative_path"])
        )

        assert len(records) == 10000

        key = (condition, seed)
        assert key not in accuracies

        accuracies[key] = sum(
            r["true_label"] == r["predicted_label"]
            for r in records
        ) / len(records)

assert len(accuracies) == 15

# 2. Calculate paired comparisons
fig, ax = plt.subplots(figsize=(10, 5.6))
results = []

for i, (label, condition_a, condition_b) in enumerate(comparisons):

    deltas = [
        100 * (
            accuracies[condition_a, seed]
            - accuracies[condition_b, seed]
        )
        for seed in seeds
    ]

    avg = mean(deltas)
    sd = stdev(deltas)

    results.append([
        label,
        condition_a,
        condition_b,
        *(round(d, 4) for d in deltas),
        round(avg, 4),
        round(sd, 4)
    ])

    # Individual paired seeds
    for j, value in enumerate(deltas):
        ax.scatter(
            i + (j - 1) * 0.11,
            value,
            color=f"C{j}",
            s=55,
            label=f"Seed {j}" if i == 0 else None,
            zorder=3
        )

    # Mean and sample standard deviation
    ax.errorbar(
        i, avg, yerr=sd,
        fmt="D",
        color="black",
        markersize=7,
        capsize=5,
        label="Mean +/- sample SD" if i == 0 else None,
        zorder=4
    )

    label_y = (
        avg + sd + 0.07
        if avg >= 0
        else avg - sd - 0.07
    )

    ax.text(
        i, label_y,
        f"{avg:+.3f} pp",
        ha="center",
        va="bottom" if avg >= 0 else "top",
        fontsize=10
    )

    print(
        f"{label}: "
        f"seed deltas={[round(x, 3) for x in deltas]} | "
        f"mean={avg:+.3f} pp | "
        f"SD={sd:.3f} pp"
    )

# 3. Format the figure
ax.axhline(
    0, linewidth=1,
    linestyle="--",
    color="gray"
)

ax.set_ylim(-0.60, 0.80)

ax.set_xticks(
    range(4),
    [
        "CLIP - Random\n50 / class",
        "CLIP - Random\n150 / class",
        "150 - 50\nRandom",
        "150 - 50\nCLIP"
    ]
)

ax.set_ylabel(
    "Paired Accuracy Difference (percentage points)"
)

ax.set_title(
    "Direct Comparisons: Selection Method and Synthetic Quantity"
)

ax.grid(axis="y", alpha=0.25)
ax.legend(loc="upper right")

fig.text(
    0.5, 0.01,
    "Each dot is one matched seed. Error bars show "
    "sample SD (n = 3), not confidence intervals.",
    ha="center",
    fontsize=9
)

fig.tight_layout(rect=[0, 0.04, 1, 1])

# 4. Save the figure
figure_path = output_dir / "13_g4_direct_comparisons.png"

fig.savefig(
    figure_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

# 5. Save the independent statistical results
csv_path = output_dir / "13_g4_direct_comparisons.csv"

with csv_path.open(
    "w", newline="", encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "comparison",
        "condition_a",
        "condition_b",
        "seed_0_delta_pp",
        "seed_1_delta_pp",
        "seed_2_delta_pp",
        "mean_delta_pp",
        "sample_sd_pp"
    ])

    writer.writerows(results)

print("RESULT: DIRECT COMPARISON ANALYSIS PASSED")
print("CSV:", csv_path)
print("FIG:", figure_path)
