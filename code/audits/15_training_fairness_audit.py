
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

root = Path("project_source")
shared = root / "shared" / "synthetic_v2"

conditions = {
    "real_only": 0,
    "random_50": 50,
    "clip_50": 50,
    "random_150": 150,
    "clip_150": 150,
}
seeds = (0, 1, 2)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


print("Python:", sys.version.split()[0])
print("OS:", platform.platform())
print("Source:", root.resolve())

assert (root / "code/classifier.py").is_file(), \
    "Project source not found"

config = load(root / "configs/study.json")
frozen = load(shared / "models/frozen.json")
features_report = load(shared / "features_report.json")
jobs = load(shared / "models/training_jobs.json")

# 1. Frozen protocol and configuration
assert config["protocol_version"] == "0.4-frozen"
assert config["protocol_status"] == "frozen"
assert config["dataset"]["real_per_class"] == 20
assert config["dataset"]["split_seeds"] == list(seeds)

cl = config["classifier"]
hp = cl["hyperparameters"]

assert cl["architecture"] == "resnet18"
assert cl["weights"] == "IMAGENET1K_V1"
assert cl["freeze_backbone"] is True
assert cl["freeze_batchnorm"] is True
assert cl["train_views"] == 4
assert cl["synthetic_loss_weight"] == 0.5
assert hp["updates"] == 500
assert hp["images_per_class_per_batch"] == 2

print("PASS: frozen protocol and fairness parameters")

# 2. Verify source files match frozen record
for name in ("classifier.py", "data_features.py"):
    actual = sha256(root / "code" / name)
    expected = frozen["code_sha256"][name]
    assert actual == expected, name

print("PASS: frozen classifier and feature source hashes")

# 3. Verify real 20-shot splits
for seed in seeds:
    path = shared / "splits" / f"seed{seed}.json"
    rows = load(path)

    assert len(rows) == 200

    counts = Counter(r["class_id"] for r in rows)
    assert counts == Counter({i: 20 for i in range(10)})

    assert len({r["image_id"] for r in rows}) == 200
    assert len({
        r["original_train_index"] for r in rows
    }) == 200

    assert all(
        r["split_seed"] == seed
        and r["split"] == "train"
        and r["source"] == "real"
        for r in rows
    )

    assert all(
        r["image_id"] ==
        f"real_train_{r['original_train_index']:05d}"
        for r in rows
    )

    pair = features_report["paired_splits"][str(seed)]

    assert sha256(path) == pair["real_manifest_sha256"]
    assert pair["data"]["dev_count"] == 0
    assert pair["data"]["official_test_labels_read"] is False
    assert pair["features"]["parameters_frozen"] is True
    assert pair["features"]["batchnorm_unchanged"] is True

    print(f"PASS: seed {seed} real split and feature records")

# 4. Verify all 15 training records
job_map = {
    (j["condition_id"], j["seed"]): j for j in jobs
}

model_map = {
    (m["condition_id"], m["seed"]): m
    for m in frozen["models"]
}

expected = {
    (condition, seed)
    for condition in conditions
    for seed in seeds
}

assert len(jobs) == len(job_map) == 15
assert len(frozen["models"]) == len(model_map) == 15
assert set(job_map) == set(model_map) == expected

for seed in seeds:
    seed_jobs = [
        job_map[(condition, seed)]
        for condition in conditions
    ]

    # Paired initialization and real batches
    assert len({
        j["initial_state_sha256"]
        for j in seed_jobs
    }) == 1

    assert len({
        j["real_schedule_sha256"]
        for j in seed_jobs
    }) == 1

    for condition, count in conditions.items():
        job = job_map[(condition, seed)]
        model = model_map[(condition, seed)]

        assert job["mode"] == "main"
        assert job["evaluation_split"] == \
            "train_reload_check_only"

        assert job["reference_count"] == 200
        assert job["dev_count"] == 0
        assert job["accuracy"] is None
        assert job["synthetic_per_class"] == count
        assert job["updates"] == model["updates"] == 500
        assert job["reload_identical"] is True

        assert job["initial_state_sha256"] == \
            model["initial_state_sha256"]

        assert job["real_schedule_sha256"] == \
            model["real_schedule_sha256"]

    print(f"PASS: seed {seed} paired heads and equal updates")

# 5. Static inspection of sealed training source
classifier = (
    root / "code/classifier.py"
).read_text(encoding="utf-8")

features = (
    root / "code/data_features.py"
).read_text(encoding="utf-8")

assert "return 0.5*real + 0.5*source_loss" in classifier
assert ".reshape(n,views).mean(1).mean()" in classifier
assert "head.load_state_dict(initial)" in classifier
assert "for step,indices in enumerate(schedule):" in classifier
assert "optimizer.step()" in classifier

assert "backbone.requires_grad_(False)" in features
assert 'backbone.to("cuda").eval()' in features
assert "with torch.inference_mode():" in features

print("PASS: frozen source fairness logic")

print("RESULT: TRAINING FAIRNESS RECORD/SOURCE AUDIT PASS")
print("Scope: saved splits, metadata, hashes and source inspection.")
print("No model retraining or independent GPU inference performed.")
