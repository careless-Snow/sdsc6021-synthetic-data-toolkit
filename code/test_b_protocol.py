"""Scientific checks for B; local tensors only, no model/data downloads."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from a_protocol import ROOT, CLASSES, atomic_json, config_hash, file_hash, load_config
from b_protocol import paired_splits, validate_real_manifest, validate_selection
from b_runtime import cache_key, context, fit_head, image_views, load_features, objective


class BProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = load_config("configs/study.json")
        cls.labels = [c for c in range(10) for _ in range(30)]
        cls.temp_root = ROOT / "runs/_cache/tmp/b_tests"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def manifest(self):
        splits = paired_splits(self.labels, smoke=True)
        by_id = {}
        for row in splits:
            by_id[row["image_id"]] = {key: row[key] for key in
                                     ("image_id", "class_id", "class_name", "original_index", "official_partition")}
        images = [row | {"source": "real"} for row in by_id.values()]
        return {"schema_version": 1, "status": "success", "config_hash": config_hash(self.config),
                "mode": "smoke", "official_test_loaded": False, "images": images, "splits": splits}

    def test_splits_are_balanced_reproducible_and_dev_uses_same_twenty(self):
        smoke = paired_splits(self.labels, smoke=True)
        main = paired_splits(self.labels, smoke=False)
        self.assertEqual(smoke, paired_splits(self.labels, smoke=True))
        self.assertEqual(len(smoke), 600)
        for seed in (0, 1, 2):
            for c in range(10):
                group = [r for r in smoke if r["split_seed"] == seed and r["class_id"] == c]
                self.assertEqual(sum(r["split"] == "train" for r in group), 16)
                self.assertEqual(sum(r["split"] == "dev" for r in group), 4)
                self.assertEqual({r["image_id"] for r in group},
                                 {r["image_id"] for r in main if r["split_seed"] == seed and r["class_id"] == c})
        self.assertNotEqual([r["image_id"] for r in main if r["split_seed"] == 0],
                            [r["image_id"] for r in main if r["split_seed"] == 1])

    def test_invalid_labels_and_short_classes_are_rejected(self):
        for labels in ([0] * 100, self.labels[:-11], self.labels + [10]):
            with self.assertRaises(ValueError):
                paired_splits(labels, smoke=True)

    def test_real_manifest_rejects_leakage_label_errors_and_wrong_mode(self):
        manifest = self.manifest()
        validate_real_manifest(manifest, self.config, smoke=True)
        for mutation in ("duplicate", "label", "test", "count"):
            changed = copy.deepcopy(manifest)
            if mutation == "duplicate":
                changed["splits"].append(changed["splits"][0])
            elif mutation == "label":
                changed["splits"][0]["class_id"] = 9
            elif mutation == "test":
                changed["images"][0]["official_partition"] = "test"
            else:
                changed["splits"].pop()
            with self.assertRaises(ValueError):
                validate_real_manifest(changed, self.config, smoke=True)
        with self.assertRaises(ValueError):
            validate_real_manifest(manifest, self.config, smoke=False)

    def test_group_weight_does_not_depend_on_synthetic_count_and_matches_gradient(self):
        torch.manual_seed(1)
        head = torch.nn.Linear(5, 10)
        x, sx = torch.randn(4, 4, 5), torch.randn(7, 4, 5)
        y, sy = torch.tensor([1, 2, 3, 4]), torch.arange(7)
        loss = objective(head, x, y, sx, sy)
        repeated = objective(head, x, y, sx.repeat(3, 1, 1), sy.repeat(3))
        self.assertTrue(torch.allclose(loss, repeated))
        reference = 0.5 * F.cross_entropy(head(x).flatten(0, 1), y.repeat_interleave(4))
        reference += 0.5 * F.cross_entropy(head(sx).flatten(0, 1), sy.repeat_interleave(4))
        actual_grad = torch.autograd.grad(loss, head.weight, retain_graph=True)[0]
        reference_grad = torch.autograd.grad(reference, head.weight)[0]
        self.assertTrue(torch.allclose(actual_grad, reference_grad))
        with self.assertRaises(ValueError):
            objective(head, x, y, sx[:0], sy[:0])

    def test_paired_initialization_fixed_update_budget_and_reload(self):
        torch.manual_seed(3)
        x, y = torch.randn(20, 4, 512), torch.arange(10).repeat(2)
        params = {"steps": 5, "learning_rate": 0.05, "weight_decay": 0.0001}
        head, initial, history = fit_head(x, y, seed=0, params=params, device="cpu")
        mixed, mixed_initial, mixed_history = fit_head(x, y, seed=0, params=params, device="cpu",
                                                       synthetic_x=x[:10], synthetic_y=y[:10])
        self.assertEqual(len(history), len(mixed_history))
        self.assertEqual(len(history), 5)
        self.assertTrue(all(torch.equal(initial[k], mixed_initial[k]) for k in initial))
        self.assertFalse(torch.equal(head.weight, mixed.weight))
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            path = Path(folder) / "head.pt"
            torch.save(head.state_dict(), path)
            reloaded = torch.nn.Linear(512, 10)
            reloaded.load_state_dict(torch.load(path, weights_only=True))
            self.assertTrue(torch.equal(head(x[:, 0]), reloaded(x[:, 0])))

    def test_views_are_reproducible_and_restore_rng_state(self):
        image = Image.new("RGB", (32, 32), (20, 100, 200))
        before = torch.get_rng_state()
        first = image_views(image, "same-id", self.config)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        second = image_views(image, "same-id", self.config)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (4, 3, 224, 224))

    def test_cache_invalidates_image_preprocessing_split_and_backbone_changes(self):
        manifest = self.manifest()
        info = {"weights_sha256": "original"}
        key = cache_key(self.config, manifest, info)
        changed = copy.deepcopy(self.config)
        changed["classifier"]["view_parameters"]["augmentation_seeds"][0] += 1
        self.assertNotEqual(key, cache_key(changed, manifest, info))
        self.assertNotEqual(key, cache_key(self.config, manifest, {"weights_sha256": "changed"}))
        changed_manifest = copy.deepcopy(manifest)
        changed_manifest["images"][0]["sha256"] = "changed"
        self.assertNotEqual(key, cache_key(self.config, changed_manifest, info))
        changed_manifest = copy.deepcopy(manifest)
        changed_manifest["splits"][0]["split"] = "dev"
        self.assertNotEqual(key, cache_key(self.config, changed_manifest, info))

    def test_feature_checksum_failure_is_not_a_cache_hit(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            manifest = self.manifest()
            for row in manifest["images"]:
                row.update(relative_path=f"images/{row['image_id']}.png")
                path = root / row["relative_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"source-image")
                row["sha256"] = file_hash(path)
            atomic_json(root / "manifests/b_real.json", manifest)
            atomic_json(root / "b_protocol.json", {"mode": "smoke"})
            path = root / "features/b_real.pt"
            path.parent.mkdir(parents=True)
            torch.save(torch.zeros(len(manifest["images"]), 4, 512), path)
            info = {"weights_sha256": "test-fixture"}
            record = {"status": "success", "relative_path": "features/b_real.pt", "sha256": file_hash(path),
                      "image_ids": [r["image_id"] for r in manifest["images"]], "backbone": info,
                      "cache_key": cache_key(self.config, manifest, info)}
            atomic_json(root / "features/b_real.json", record)
            load_features(root, self.config)
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_features(root, self.config)

    def test_formal_stage_is_blocked_until_g2(self):
        with self.assertRaises(ValueError):
            context(self.config, smoke=False)

    def test_c_selection_requires_correct_counts_labels_ranks_and_nesting(self):
        candidates = [{"image_id": f"s{c}_{i}", "class_id": c, "source": "synthetic", "status": "success"}
                      for c in range(10) for i in range(200)]
        rows = [{"schema_version": 1, "pool_id": "pool-v1", "condition_id": f"{method}_{k}",
                 "image_id": f"s{c}_{i}", "class_id": c, "rank": i + 1}
                for method in ("random", "clip") for k in (50, 150) for c in range(10) for i in range(k)]
        validate_selection(rows, candidates, pool_id="pool-v1")
        for mutation in ("missing", "label", "duplicate", "nesting", "pool"):
            changed = copy.deepcopy(rows)
            if mutation == "missing":
                changed.pop()
            elif mutation == "label":
                changed[0]["class_id"] = 9
            elif mutation == "duplicate":
                changed[1]["image_id"] = changed[0]["image_id"]
            elif mutation == "nesting":
                changed[0]["image_id"] = "s0_199"
            else:
                changed[0]["pool_id"] = "wrong-pool"
            with self.assertRaises(ValueError):
                validate_selection(changed, candidates, pool_id="pool-v1")


if __name__ == "__main__":
    unittest.main()
