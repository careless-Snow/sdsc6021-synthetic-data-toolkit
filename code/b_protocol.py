"""B-owned paired splits and strict hand-off validation (standard library only)."""
from __future__ import annotations

import hashlib
from collections import Counter

from a_protocol import CLASSES, CONDITIONS, config_hash


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256(config_hash(list(parts)).encode()).hexdigest()[:15], 16)


def paired_splits(labels: list[int], seeds=(0, 1, 2), *, smoke: bool) -> list[dict]:
    """Smoke reserves 4 of the SAME 20/class for development; no extra labels."""
    if any(type(label) is not int or label not in range(10) for label in labels):
        raise ValueError("Labels must use the official integer CIFAR-10 mapping")
    groups = [[i for i, label in enumerate(labels) if label == c] for c in range(10)]
    if any(len(group) < 20 for group in groups):
        raise ValueError("At least 20 official training examples/class are required")
    records = []
    for seed in seeds:
        for class_id, indices in enumerate(groups):
            ranked = sorted(indices, key=lambda i: (stable_seed("split-v1", seed, class_id, i), i))[:20]
            for rank, index in enumerate(ranked):
                records.append({"schema_version": 1, "split_seed": seed,
                                "image_id": f"cifar10_train_{index:05d}", "original_index": index,
                                "official_partition": "train", "class_id": class_id,
                                "class_name": CLASSES[class_id],
                                "split": "dev" if smoke and rank >= 16 else "train"})
    return records


def validate_real_manifest(manifest: dict, config: dict, *, smoke: bool) -> None:
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "success"
            or manifest.get("config_hash") != config_hash(config)
            or manifest.get("official_test_loaded") is not False
            or manifest.get("mode") != ("smoke" if smoke else "main")):
        raise ValueError("Real manifest has incompatible provenance")
    images = manifest["images"]
    by_id = {row["image_id"]: row for row in images}
    if len(by_id) != len(images):
        raise ValueError("Duplicate real image IDs")
    for row in images:
        if (row.get("source") != "real" or row.get("official_partition") != "train"
                or type(row.get("class_id")) is not int or row["class_id"] not in range(10)
                or row.get("class_name") != CLASSES[row["class_id"]]
                or type(row.get("original_index")) is not int or not 0 <= row["original_index"] < 50000
                or row["image_id"] != f"cifar10_train_{row['original_index']:05d}"):
            raise ValueError("Real image mapping/source is invalid")
    seen, counts = set(), Counter()
    for row in manifest["splits"]:
        item = by_id.get(row["image_id"])
        key = (row["split_seed"], row["image_id"])
        if (item is None or key in seen or row["split_seed"] not in config["dataset"]["split_seeds"]
                or row["split"] not in ({"train", "dev"} if smoke else {"train"})
                or any(row.get(field) != item.get(field) for field in
                       ("class_id", "class_name", "original_index", "official_partition"))):
            raise ValueError("Duplicate/leaking split or mismatched labels/IDs")
        seen.add(key)
        counts[(row["split_seed"], row["class_id"], row["split"])] += 1
    parts = (("train", 16), ("dev", 4)) if smoke else (("train", 20),)
    expected = Counter({(seed, c, part): count for seed in config["dataset"]["split_seeds"]
                        for c in range(10) for part, count in parts})
    if counts != expected or {image_id for _, image_id in seen} != set(by_id):
        raise ValueError("Real train/dev counts or selected image coverage differ")


def validate_selection(rows: list[dict], candidates: list[dict], *, pool_id: str) -> dict[str, list[str]]:
    """Validate C's exact counts, class IDs, unique IDs, ranks and nested K sets."""
    by_id = {row["image_id"]: row for row in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Duplicate candidate image IDs")
    selected = {name: [] for name, count, _ in CONDITIONS if count}
    ranks = set()
    for row in rows:
        condition = row["condition_id"]
        item = by_id.get(row["image_id"])
        if (row.get("schema_version") != 1 or row.get("pool_id") != pool_id
                or condition not in selected or item is None
                or item.get("source") != "synthetic" or item.get("status") != "success"
                or type(row.get("class_id")) is not int or row["class_id"] != item["class_id"]
                or type(row.get("rank")) is not int or row["rank"] < 1):
            raise ValueError("Invalid C selection row or candidate/class/pool mismatch")
        key = (condition, row["class_id"], row["rank"])
        if key in ranks or row["image_id"] in selected[condition]:
            raise ValueError("Duplicate selection ID/rank")
        ranks.add(key)
        selected[condition].append(row["image_id"])
    for name, count, _ in CONDITIONS:
        if not count:
            continue
        counts = Counter(by_id[i]["class_id"] for i in selected[name])
        if counts != Counter({i: count for i in range(10)}):
            raise ValueError(f"{name} requires exactly {count}/class")
        for c in range(10):
            actual = {rank for condition, class_id, rank in ranks if condition == name and class_id == c}
            if actual != set(range(1, count + 1)):
                raise ValueError("Selection ranks must be contiguous top-K ranks")
    for method in ("random", "clip"):
        if not set(selected[f"{method}_50"]) <= set(selected[f"{method}_150"]):
            raise ValueError("K=50 must be nested in K=150")
    return selected
