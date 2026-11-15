
import csv
import json
import math
import zipfile
from pathlib import Path
from datetime import datetime
import sys

base = Path("C_evidence")
cost_file = Path("stage_costs.json")
g4_zip = Path("g4_predictions_and_run_records.zip")

print("Time:", datetime.now().astimezone().isoformat())
print("Python:", sys.version.split()[0])

# 1. Read official G3 cost records
with cost_file.open(encoding="utf-8") as f:
    g3 = json.load(f)

assert g3["run_id"] == "synthetic_v2"
assert g3["status"] == "G3_production_costs_only"

generation = sum(
    x["seconds"] for x in g3["generation_invocations"]
)

assert math.isclose(
    generation,
    g3["candidate_pool_production_seconds"],
    abs_tol=1e-6
)

selection = g3["stages"]["select"]["seconds"]
features = g3["stages"]["features"]["seconds"]
training = g3["stages"]["train"]["seconds"]

g3_total = generation + selection + features + training

print("PASS: G3 timing records reconciled")

# 2. Read official G4 timing records
with zipfile.ZipFile(g4_zip) as z:
    report = json.loads(z.read("handoff/report.json"))
    invocation = json.loads(
        z.read("handoff/invocation.json")
    )

assert report["status"] == "success"
assert report["mode"] == "final_test"
assert invocation["run_id"] == g3["run_id"]
assert invocation["exit_code"] == 0

g4_evaluation = report["seconds"]
g4_cli_wall = invocation["cli_wall_seconds"]

assert g4_cli_wall >= g4_evaluation

print("PASS: G4 official timing records checked")

# 3. Calculate stage costs
stages = [
    ("G3", "SD1.5 generation", generation),
    ("G3", "CLIP selection", selection),
    ("G3", "ResNet features", features),
    ("G3", "Classifier training", training),
    ("G4", "Final evaluation", g4_evaluation),
]

measured_sum = g3_total + g4_evaluation

print("\nRECORDED STAGE COSTS")
print("-" * 55)

for phase, name, seconds in stages:
    print(
        f"{phase} | {name:22s} | "
        f"{seconds:9.2f} s | "
        f"{seconds / 60:6.2f} min"
    )

print("-" * 55)
print(f"G3 stage subtotal: {g3_total / 60:.2f} min")
print(f"G4 evaluation:      {g4_evaluation / 60:.2f} min")
print(f"G4 CLI wall time:   {g4_cli_wall / 60:.2f} min")
print(f"Measured stage sum: {measured_sum / 60:.2f} min")

# 4. Save independently checked costs
output = base / "12_g4_cost_summary.csv"

with output.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "phase", "stage", "seconds", "minutes"
    ])
    for phase, name, seconds in stages:
        writer.writerow([
            phase, name,
            round(seconds, 4),
            round(seconds / 60, 4)
        ])

print("\nRESULT: COST CHECKS PASS")
print("Saved:", output)
print(
    "Scope: recorded stage invocations only; "
    "not total project cost."
)
