import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

input_csv = Path("C_evidence/08_g4_per_class.csv")
output_png = Path("C_evidence/14_g4_per_class_gain.png")

rows = []
with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

conditions = ["random_50", "clip_50", "random_150", "clip_150"]

# collect class names from clip_50 ordering
clip50_rows = [r for r in rows if r["condition"] == "clip_50"]

# sort classes by clip_50 mean gain descending
clip50_rows.sort(key=lambda r: float(r["mean_gain_pp"]), reverse=True)
class_order = [r["class_name"] for r in clip50_rows]

# build lookup
lookup = {}
for r in rows:
    key = (r["condition"], r["class_name"])
    lookup[key] = float(r["mean_gain_pp"])

# matrix: each condition has gains across classes
data = {cond: [lookup[(cond, cls)] for cls in class_order] for cond in conditions}

y = np.arange(len(class_order))
bar_h = 0.18
offsets = [-1.5 * bar_h, -0.5 * bar_h, 0.5 * bar_h, 1.5 * bar_h]

fig, ax = plt.subplots(figsize=(11, 7))

for cond, off in zip(conditions, offsets):
    label = {
        "random_50": "Random 50",
        "clip_50": "CLIP 50",
        "random_150": "Random 150",
        "clip_150": "CLIP 150",
    }[cond]
    ax.barh(y + off, data[cond], height=bar_h, label=label)

ax.axvline(0, linestyle="--", linewidth=1, color="gray")

ax.set_yticks(y)
ax.set_yticklabels(class_order)
ax.invert_yaxis()

ax.set_xlabel("Mean Accuracy Gain over Real-Only Baseline (percentage points)")
ax.set_ylabel("CIFAR-10 Class")
ax.set_title("Per-Class Gain from Synthetic Data Across Four Conditions")

ax.legend(loc="lower right", frameon=True)
ax.grid(axis="x", alpha=0.25)

fig.text(
    0.5,
    0.01,
    "Classes are ordered by CLIP 50 mean gain. Positive values indicate improvement over real-only baseline.",
    ha="center",
    fontsize=9
)

fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(output_png, dpi=200, bbox_inches="tight")
plt.close(fig)

print("RESULT: PER-CLASS GAIN FIGURE SAVED")
print(f"Saved: {output_png}")
print(f"Classes plotted: {len(class_order)}")
print("Order:", ", ".join(class_order))