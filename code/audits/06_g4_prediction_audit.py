
import json
import zipfile
import hashlib
import math
import sys
import platform
from collections import Counter
from datetime import datetime

ZIP = "g4_predictions_and_run_records.zip"

CONDITIONS = (
    "real_only", "random_50", "clip_50",
    "random_150", "clip_150"
)

print("Time:", datetime.now().astimezone().isoformat())
print("Python:", sys.version.split()[0])
print("OS:", platform.platform())

with zipfile.ZipFile(ZIP) as z:

    index = json.loads(
        z.read("handoff/prediction_index.json")
    )
    report = json.loads(
        z.read("handoff/report.json")
    )

    entries = index["predictions"]

    assert len(entries) == 16
    assert index["rows_total"] == 160000

    expected_ids = {
        f"real_test_{i:05d}" for i in range(10000)
    }

    expected_runs = {
        (c, s) for c in CONDITIONS for s in range(3)
    }

    report_runs = {
        (r["condition_id"], r["seed"]): r
        for r in report["results"]
    }

    assert set(report_runs) == expected_runs

    reference_labels = None
    seen_runs = set()
    total_rows = 0

    for entry in entries:

        path = entry["relative_path"]
        raw = z.read(path)

        # Verify the original prediction file hash
        actual_hash = hashlib.sha256(raw).hexdigest()
        assert actual_hash == entry["sha256"], path

        rows = json.loads(raw)
        assert len(rows) == entry["rows"] == 10000

        # Verify sample IDs and class balance
        ids = [r["sample_id"] for r in rows]

        assert len(set(ids)) == 10000
        assert set(ids) == expected_ids

        labels = {
            r["sample_id"]: r["true_label"]
            for r in rows
        }

        assert Counter(labels.values()) == {
            i: 1000 for i in range(10)
        }

        assert all(
            type(r["predicted_label"]) == int
            and 0 <= r["predicted_label"] <= 9
            for r in rows
        )

        # Check true-label consistency across files
        if reference_labels is None:
            reference_labels = labels
        else:
            assert labels == reference_labels

        # Independently recompute top-1 accuracy
        accuracy = sum(
            r["true_label"] == r["predicted_label"]
            for r in rows
        ) / len(rows)

        assert math.isclose(
            accuracy,
            entry["accuracy_recomputed"],
            abs_tol=1e-12
        )

        condition = entry["condition_id"]
        seed = entry["seed"]

        if condition == "clip_zero_shot":

            assert seed is None
            assert math.isclose(
                accuracy,
                report["clip_zero_shot"]["accuracy"],
                abs_tol=1e-12
            )

        else:

            key = (condition, seed)

            assert key in expected_runs
            assert key not in seen_runs

            assert all(
                r["seed"] == seed
                and r["condition_id"] == condition
                and r["config_hash"] == index["config_hash"]
                for r in rows
            )

            assert math.isclose(
                accuracy,
                report_runs[key]["accuracy"],
                abs_tol=1e-12
            )

            seen_runs.add(key)

        total_rows += len(rows)

        print(
            f"{condition:15s} "
            f"seed={str(seed):>4s} "
            f"accuracy={accuracy:.4f} PASS"
        )

    assert seen_runs == expected_runs
    assert total_rows == 160000

print("RESULT: ALL 160000 PREDICTION RECORD CHECKS PASS")
print("Scope: stored predictions and report consistency.")
