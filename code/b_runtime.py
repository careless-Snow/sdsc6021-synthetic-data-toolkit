"""B real-data preparation, frozen features and linear-head development baseline.

Only official CIFAR-10 TRAIN batches are opened. No official-test evaluator here.
"""
from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from a_protocol import ROOT, CLASSES, atomic_json, config_hash, file_hash, inside, require_g2
from b_protocol import paired_splits, stable_seed, validate_real_manifest


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def b_settings(config: dict) -> dict:
    classifier = config["classifier"]
    params = classifier.get("hyperparameters")
    views = classifier.get("view_parameters")
    if not params or not views:
        raise ValueError("B classifier/view parameters are missing from the shared config")
    if (classifier["weights"] != "IMAGENET1K_V1" or params["optimizer"] != "SGD"
            or params["steps"] < 1 or params["learning_rate"] <= 0
            or params["weight_decay"] < 0 or params["momentum"] != 0
            or classifier["feature_normalization"] not in {"none", "l2"}
            or views["base_resize"] != 256 or views["crop_size"] != 224
            or len(views["augmentation_seeds"]) != 3):
        raise ValueError("Unsupported B configuration")
    return classifier


def sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def context(config: dict, *, smoke: bool) -> Path:
    b_settings(config)
    if not smoke:
        require_g2(config)
    fingerprint = config_hash(config)
    relative = f"b_smoke/{fingerprint[:12]}" if smoke else config["run_id"]
    root = inside(ROOT / "runs", relative)
    root.mkdir(parents=True, exist_ok=True)
    packages = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "pillow")}
    contract = {"schema_version": 1, "mode": "smoke" if smoke else "main",
                "config_hash": fingerprint, "config": config,
                "b_source_sha256": {name: file_hash(ROOT / "code" / name)
                                    for name in ("b_protocol.py", "b_runtime.py", "toolkit.py")},
                "packages": packages, "python": platform.python_version()}
    path = root / "b_protocol.json"
    if path.exists() and read_json(path) != contract:
        raise ValueError("B code/config/environment changed: preserve this run and use a new config run_id")
    if not path.exists():
        atomic_json(path, contract)
    environment = root / "b_environment.json"
    if not environment.exists():
        atomic_json(environment, {"schema_version": 1, "python": sys.version,
                                  "platform": platform.platform(), "packages": packages,
                                  "executable": sys.executable, "official_test_loaded": False})
    return root


@contextlib.contextmanager
def stage(root: Path, name: str, device="cpu"):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    protocol = read_json(root / "b_protocol.json")
    record = {"schema_version": 1, "run_id": root.relative_to(ROOT / "runs").as_posix(),
              "mode": protocol["mode"], "stage": name, "device": device,
              "config_hash": protocol["config_hash"], "b_source_sha256": protocol["b_source_sha256"],
              "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    path = root / "logs" / f"b_{name}_{stamp}.json"
    atomic_json(path, record)
    sync(device)
    start = time.perf_counter()
    try:
        yield record
        sync(device)
        record["status"] = "success"
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record.update(ended_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - start)
        atomic_json(path, record)


def verified_json(path: Path) -> dict:
    record = read_json(path)
    if record.get("status") != "success":
        raise ValueError(f"Incomplete output: {path}")
    return record


def prepare(config: dict, *, smoke: bool, download: bool) -> dict:
    import torchvision
    from torchvision.datasets import CIFAR10
    root = context(config, smoke=smoke)
    path = root / "manifests/b_real.json"
    with stage(root, "prepare") as log:
        if path.exists():
            saved = verified_json(path)
            validate_real_manifest(saved, config, smoke=smoke)
            for row in saved["images"]:
                if file_hash(inside(root, row["relative_path"])) != row["sha256"]:
                    raise ValueError("Existing real image is corrupt")
            log["cache_hit"] = True
            return {"status": "success", "stage": "prepare", "manifest": str(path.relative_to(ROOT)),
                    "images": len(saved["images"]), "cache_hit": True}
        # torchvision verifies the official archive checksums. train=True reads only train_list.
        dataset = CIFAR10(str(ROOT / "runs/_cache/cifar10"), train=True, download=download)
        if tuple(dataset.classes) != CLASSES:
            raise ValueError("Official CIFAR-10 class mapping differs")
        splits = paired_splits(dataset.targets, config["dataset"]["split_seeds"], smoke=smoke)
        images = []
        for index in sorted({row["original_index"] for row in splits}):
            image_id = f"cifar10_train_{index:05d}"
            relative = f"images/real/{image_id}.png"
            image_path = inside(root, relative)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image, label = dataset[index]
            temporary = image_path.with_suffix(".tmp")
            image.save(temporary, format="PNG")
            temporary.replace(image_path)
            images.append({"schema_version": 1, "image_id": image_id, "original_index": index,
                           "official_partition": "train", "class_id": label, "class_name": CLASSES[label],
                           "relative_path": relative, "sha256": file_hash(image_path),
                           "source": "real", "status": "success"})
        source = Path(dataset.root) / dataset.base_folder
        hashes = {name: file_hash(source / name) for name, _ in dataset.train_list}
        hashes[dataset.meta["filename"]] = file_hash(source / dataset.meta["filename"])
        record = {"schema_version": 1, "status": "success", "mode": "smoke" if smoke else "main",
                  "config_hash": config_hash(config), "class_mapping": dict(enumerate(CLASSES)),
                  "source": "CIFAR10 official training partition", "source_sha256": hashes,
                  "official_test_loaded": False, "images": images, "splits": splits,
                  "torchvision": torchvision.__version__,
                  "train_per_class": 16 if smoke else 20, "dev_per_class": 4 if smoke else 0}
        validate_real_manifest(record, config, smoke=smoke)
        atomic_json(path, record)
        log.update(cache_hit=False, images=len(images), output_sha256=file_hash(path))
    return {"status": "success", "stage": "prepare", "manifest": str(path.relative_to(ROOT)),
            "images": len(images), "official_test_loaded": False}


def build_backbone(config: dict, device: str, *, download: bool):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    weights = ResNet18_Weights.IMAGENET1K_V1
    torch.hub.set_dir(str(ROOT / "runs/_cache/torch"))
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[1]
    if not weight_path.exists() and not download:
        raise FileNotFoundError("ResNet-18 weights absent; explicitly use --download to fetch official weights")
    state = torch.hub.load_state_dict_from_url(weights.url, progress=True, check_hash=True, map_location="cpu")
    digest = file_hash(weight_path)
    if not digest.startswith(weight_path.stem.rsplit("-", 1)[1]):
        raise ValueError("Pretrained weight checksum mismatch")
    backbone = resnet18(weights=None)
    backbone.load_state_dict(state)
    backbone.fc = torch.nn.Identity()
    backbone.requires_grad_(False)
    backbone.eval()
    backbone.to(device)
    return backbone, {"architecture": "resnet18", "weights": "IMAGENET1K_V1",
                      "weights_url": weights.url, "weights_sha256": digest}


def image_views(image, image_id: str, config: dict):
    import torch
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    params = b_settings(config)["view_parameters"]
    normalization = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base = T.Compose([T.Resize(params["base_resize"], interpolation=InterpolationMode.BILINEAR, antialias=True),
                      T.CenterCrop(params["crop_size"]), T.ToTensor(), normalization])
    augment = T.Compose([T.RandomResizedCrop(params["crop_size"], scale=tuple(params["scale"]),
                                           ratio=tuple(params["ratio"]),
                                           interpolation=InterpolationMode.BILINEAR, antialias=True),
                         T.RandomHorizontalFlip(params["horizontal_flip_probability"]), T.ToTensor(), normalization])
    result = [base(image)]
    for seed in params["augmentation_seeds"]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(stable_seed("view-v1", image_id, seed))
            result.append(augment(image))
    return torch.stack(result)


def cache_key(config: dict, manifest: dict, backbone: dict) -> str:
    return config_hash({"schema_version": 1, "images": manifest["images"], "splits": manifest["splits"],
                        "backbone": backbone, "classifier": config["classifier"],
                        "packages": {n: importlib.metadata.version(n) for n in ("torch", "torchvision", "pillow")},
                        "implementation": file_hash(Path(__file__))})


def load_features(root: Path, config: dict):
    import torch
    record = verified_json(root / "features/b_real.json")
    manifest = verified_json(root / "manifests/b_real.json")
    validate_real_manifest(manifest, config, smoke=read_json(root / "b_protocol.json")["mode"] == "smoke")
    if record["cache_key"] != cache_key(config, manifest, record["backbone"]):
        raise ValueError("Feature cache key is stale")
    for row in manifest["images"]:
        if file_hash(inside(root, row["relative_path"])) != row["sha256"]:
            raise ValueError("Source image changed after feature extraction")
    path = inside(root, record["relative_path"])
    if file_hash(path) != record["sha256"]:
        raise ValueError("Feature cache failed checksum")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if (tuple(tensor.shape) != (len(record["image_ids"]), 4, 512)
            or not torch.isfinite(tensor).all()
            or record["image_ids"] != [row["image_id"] for row in manifest["images"]]):
        raise ValueError("Feature shape, finiteness or image-to-row mapping is invalid")
    return tensor, record, manifest


def features(config: dict, *, smoke: bool, download: bool, device: str, batch_size: int) -> dict:
    import torch
    from PIL import Image
    root = context(config, smoke=smoke)
    with stage(root, "features", device) as log:
        manifest = verified_json(root / "manifests/b_real.json")
        validate_real_manifest(manifest, config, smoke=smoke)
        path = root / "features/b_real.json"
        if path.exists():
            tensor, record, _ = load_features(root, config)
            log.update(cache_hit=True, output_sha256=record["sha256"])
            return {"status": "success", "stage": "features", "cache_hit": True, "shape": list(tensor.shape)}
        if (root / "features/b_real.pt").exists():
            raise ValueError("Incomplete feature output without success metadata: inspect before retrying")
        sync(device)
        loading = time.perf_counter()
        backbone, info = build_backbone(config, device, download=download)
        sync(device)
        log["download_load_seconds"] = time.perf_counter() - loading
        bn_before = {name: value.detach().cpu().clone() for name, value in backbone.named_buffers()}
        if any(p.requires_grad for p in backbone.parameters()) or any(m.training for m in backbone.modules()):
            raise ValueError("Backbone/BN is not completely frozen")
        sync(device)
        warmup = time.perf_counter()
        with torch.inference_mode():
            backbone(torch.zeros((1, 3, 224, 224), device=device))
        sync(device)
        log["warmup_seconds"] = time.perf_counter() - warmup
        rows, results = manifest["images"], []
        extraction = time.perf_counter()
        for offset in range(0, len(rows), batch_size):
            batch = []
            for row in rows[offset:offset + batch_size]:
                image_path = inside(root, row["relative_path"])
                if file_hash(image_path) != row["sha256"]:
                    raise ValueError("Real image checksum mismatch")
                with Image.open(image_path) as image:
                    batch.append(image_views(image.convert("RGB"), row["image_id"], config))
            inputs = torch.stack(batch).flatten(0, 1).to(device)
            with torch.inference_mode():
                output = backbone(inputs).reshape(len(batch), 4, 512)
                if config["classifier"]["feature_normalization"] == "l2":
                    output = torch.nn.functional.normalize(output, dim=-1)
            results.append(output.cpu())
            print(f"[B features] {min(offset + batch_size, len(rows))}/{len(rows)} images", flush=True)
        sync(device)
        log["extraction_seconds"] = time.perf_counter() - extraction
        if any(not torch.equal(value.detach().cpu(), bn_before[name]) for name, value in backbone.named_buffers()):
            raise ValueError("BatchNorm/backbone buffers changed")
        tensor = torch.cat(results)
        if not torch.isfinite(tensor).all():
            raise ValueError("Non-finite features")
        target = root / "features/b_real.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        torch.save(tensor, temporary)
        temporary.replace(target)
        record = {"schema_version": 1, "status": "success", "config_hash": config_hash(config),
                  "relative_path": target.relative_to(root).as_posix(), "sha256": file_hash(target),
                  "cache_key": cache_key(config, manifest, info), "backbone": info,
                  "image_ids": [row["image_id"] for row in rows], "view_ids": [0, 1, 2, 3],
                  "shape": list(tensor.shape), "feature_dimension": 512,
                  "classifier": config["classifier"], "backbone_and_bn_frozen": True,
                  "official_test_loaded": False}
        atomic_json(path, record)
        log.update(cache_hit=False, backbone_and_bn_frozen=True, output_sha256=record["sha256"])
    return {"status": "success", "stage": "features", "shape": list(tensor.shape),
            "manifest": str(path.relative_to(ROOT)), "backbone_and_bn_frozen": True}


def group_loss(head, x, labels):
    import torch.nn.functional as F
    if x.ndim != 3 or labels.ndim != 1 or len(x) != len(labels) or len(x) == 0:
        raise ValueError("Expected nonempty [images, views, features] and one label/image")
    logits = head(x)
    targets = labels[:, None].expand(-1, x.shape[1]).reshape(-1)
    return F.cross_entropy(logits.flatten(0, 1), targets, reduction="none").reshape(x.shape[:2]).mean(1).mean()


def objective(head, real_x, real_y, synthetic_x=None, synthetic_y=None):
    real = group_loss(head, real_x, real_y)
    if synthetic_x is None:
        if synthetic_y is not None:
            raise ValueError("Synthetic features and labels must be supplied together")
        return real
    return 0.5 * real + 0.5 * group_loss(head, synthetic_x, synthetic_y)


def fit_head(real_x, real_y, *, seed: int, params: dict, device: str,
             synthetic_x=None, synthetic_y=None):
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = torch.nn.Linear(real_x.shape[-1], 10)
    initial = {key: value.clone() for key, value in head.state_dict().items()}
    head.to(device)
    real_x, real_y = real_x.to(device), real_y.to(device)
    if synthetic_x is not None:
        synthetic_x, synthetic_y = synthetic_x.to(device), synthetic_y.to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=params["learning_rate"],
                                weight_decay=params["weight_decay"], momentum=0)
    history = []
    for _ in range(params["steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(head, real_x, real_y, synthetic_x, synthetic_y)
        if not torch.isfinite(loss):
            raise ValueError("Classifier loss became non-finite")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return head.cpu().eval(), initial, history


def train(config: dict, *, smoke: bool, device: str) -> dict:
    import torch
    root = context(config, smoke=smoke)
    with stage(root, "train_real_only", device) as log:
        tensor, feature_record, manifest = load_features(root, config)
        by_id = {row["image_id"]: i for i, row in enumerate(manifest["images"])}
        results = []
        log["model_cache_hits"] = 0
        for seed in config["dataset"]["split_seeds"]:
            rows = [row for row in manifest["splits"] if row["split_seed"] == seed]
            training = [row for row in rows if row["split"] == "train"]
            development = [row for row in rows if row["split"] == "dev"]
            if {r["image_id"] for r in training} & {r["image_id"] for r in development}:
                raise ValueError("Train/dev leakage")
            x = tensor[[by_id[row["image_id"]] for row in training]]
            y = torch.tensor([row["class_id"] for row in training])
            target = root / f"models/real_only_seed{seed}.pt"
            metadata_path = target.with_suffix(".json")
            if metadata_path.exists():
                metadata = verified_json(metadata_path)
                for output in metadata["outputs"]:
                    if file_hash(inside(root, output["relative_path"])) != output["sha256"]:
                        raise ValueError("Existing model/predictions failed integrity check")
                if metadata["feature_cache_key"] != feature_record["cache_key"]:
                    raise ValueError("Model uses a stale feature cache")
                results.append(metadata["result"])
                log["model_cache_hits"] += 1
                continue
            if target.exists():
                raise ValueError("Incomplete prior model: inspect it before retrying")
            sync(device)
            started = time.perf_counter()
            head, initial, history = fit_head(x, y, seed=seed,
                                             params=config["classifier"]["hyperparameters"], device=device)
            sync(device)
            elapsed = time.perf_counter() - started
            checkpoint = {"schema_version": 1, "head_state_dict": head.state_dict(),
                          "initial_state_dict": initial, "condition_id": "real_only", "seed": seed,
                          "mode": "smoke" if smoke else "main", "config_hash": config_hash(config),
                          "backbone": feature_record["backbone"], "feature_dimension": 512,
                          "classifier": config["classifier"], "class_mapping": list(CLASSES),
                          "training_image_ids": [row["image_id"] for row in training],
                          "feature_cache_key": feature_record["cache_key"]}
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(target)
            reloaded = torch.load(target, map_location="cpu", weights_only=True)
            loaded_head = torch.nn.Linear(512, 10)
            loaded_head.load_state_dict(reloaded["head_state_dict"])
            loaded_head.eval()
            with torch.inference_mode():
                if not torch.equal(head(x[:, 0]), loaded_head(x[:, 0])):
                    raise ValueError("Reloaded classifier predictions changed")
            outputs = [{"relative_path": target.relative_to(root).as_posix(), "sha256": file_hash(target)}]
            result = {"seed": seed, "condition_id": "real_only", "train_images": len(training),
                      "dev_images": len(development), "optimizer_steps": len(history),
                      "initial_loss": history[0], "last_loss": history[-1],
                      "training_seconds": elapsed, "reload_verified": True,
                      "official_test_loaded": False}
            if development:
                dev_x = tensor[[by_id[row["image_id"]] for row in development], 0]
                with torch.inference_mode():
                    predictions = loaded_head(dev_x).argmax(-1).tolist()
                prediction_rows = [{"schema_version": 1, "sample_id": row["image_id"],
                                    "condition_id": "real_only", "seed": seed, "split": "dev",
                                    "mode": "smoke", "predicted_label": predicted,
                                    "true_label": row["class_id"], "config_hash": config_hash(config)}
                                   for row, predicted in zip(development, predictions)]
                prediction_path = root / f"metrics/dev_real_only_seed{seed}.json"
                atomic_json(prediction_path, prediction_rows)
                outputs.append({"relative_path": prediction_path.relative_to(root).as_posix(),
                                "sha256": file_hash(prediction_path)})
                result["dev_accuracy"] = sum(r["predicted_label"] == r["true_label"] for r in prediction_rows) / len(prediction_rows)
            atomic_json(metadata_path, {"schema_version": 1, "status": "success", "result": result,
                                        "config_hash": config_hash(config), "feature_cache_key": feature_record["cache_key"],
                                        "loss_history": history, "outputs": outputs})
            results.append(result)
        summary = {"schema_version": 1, "status": "success", "stage": "B real-only development baseline",
                   "mode": "smoke" if smoke else "main", "config_hash": config_hash(config), "results": results,
                   "official_test_loaded": False, "g1_complete": False, "g2_complete": False,
                   "formal_results": False if smoke else True,
                   "remaining": ["A model checks and candidate pool", "C selection manifests",
                                 "full five-condition smoke and protocol freeze", "formal 15-model matrix"]}
        path = root / "metrics/b_summary.json"
        atomic_json(path, summary)
        log["output_sha256"] = file_hash(path)
    return summary | {"summary": str(path.relative_to(ROOT))}


def run_b(config: dict, args) -> dict:
    import torch
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("--threads and --batch-size must be positive")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; choose the device explicitly")
    os.environ.setdefault("TORCH_HOME", str(ROOT / "runs/_cache/torch"))
    if args.command == "prepare":
        return prepare(config, smoke=args.smoke, download=args.download)
    if args.command == "features":
        return features(config, smoke=args.smoke, download=args.download,
                        device=args.device, batch_size=args.batch_size)
    if args.command == "train":
        return train(config, smoke=args.smoke, device=args.device)
    if args.command == "smoke" and args.component == "b":
        prepare(config, smoke=True, download=args.download)
        features(config, smoke=True, download=args.download, device=args.device, batch_size=args.batch_size)
        return train(config, smoke=True, device=args.device)
    raise ValueError("Use smoke --component b for the B-only development check")
