
import csv
from pathlib import Path
import matplotlib.pyplot as plt

base = Path("C_evidence")
source = base / "07_g4_summary.csv"

with source.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

labels = {
    "real_only": "Real Only",
    "random_50": "Random 50",
    "clip_50": "CLIP 50",
    "random_150": "Random 150",
    "clip_150": "CLIP 150",
}

order = list(labels)

data = {row["condition"]: row for row in rows}
assert set(data) == set(order), "Unexpected conditions"

means = [float(data[k]["mean_accuracy_pct"]) for k in order]
sds = [float(data[k]["sd_accuracy_pct"]) for k in order]
names = [labels[k] for k in order]

fig, ax = plt.subplots(figsize=(9, 5))

positions = range(len(order))

ax.errorbar(
    positions,
    means,
    yerr=sds,
    fmt="o",
    capsize=5,
    markersize=8,
    linewidth=1.5
)

for i, (value, sd) in enumerate(zip(means, sds)):
    ax.text(
        i,
        value + sd + 0.18,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

ax.set_xticks(list(positions), names)
ax.set_ylabel("Top-1 Accuracy (%)")
ax.set_title("CIFAR-10: Accuracy Across Five Conditions")
ax.set_ylim(68, 79)
ax.grid(axis="y", alpha=0.25)

fig.text(
    0.5, 0.01,
    "Mean ± sample SD across 3 paired seeds; not confidence intervals.",
    ha="center", fontsize=9
)

fig.tight_layout(rect=[0, 0.04, 1, 1])

output = base / "10_g4_accuracy.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)

print("RESULT: ACCURACY FIGURE SAVED")
print("Saved:", output)
print("Conditions:", len(order))
