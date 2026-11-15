"""Scientific-protocol and output-integrity checks, requiring no model downloads."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a_protocol import (ROOT, atomic_json, candidate_plan, config_hash, file_hash,
                        inside, load_config, require_g2, training_plan, verified_item)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = ROOT / "runs/_cache/tmp/tests"
        temporary.mkdir(parents=True, exist_ok=True)
        self.temporary = str(temporary)
        self.config = load_config(ROOT / "configs/study.json")

    def test_plan_preserves_paired_matrix_and_unique_candidate_rng(self):
        candidates = candidate_plan(self.config)
        self.assertEqual(len(candidates), 2000)
        self.assertEqual(len({row["image_id"] for row in candidates}), 2000)
        self.assertEqual(len({row["generation_seed"] for row in candidates}), 2000)
        self.assertEqual([sum(row["class_id"] == i for row in candidates) for i in range(10)], [200] * 10)
        runs = training_plan(self.config)
        self.assertEqual(len(runs), 15)
        for seed in (0, 1, 2):
            subset = [row for row in runs if row["seed"] == seed]
            self.assertEqual(len(subset), 5)
            self.assertEqual(sum(row["synthetic_per_class"] == 0 for row in subset), 1)
            self.assertEqual({row["real_per_class"] for row in subset}, {20})

    def test_full_generation_rejects_draft_and_unfrozen_classifier(self):
        with self.assertRaises(ValueError):
            require_g2(self.config)
        altered = copy.deepcopy(self.config)
        altered.update(protocol_status="frozen")
        altered["gates"]["g2_passed"] = True
        altered["classifier"]["hyperparameters"] = None
        with self.assertRaises(ValueError):
            require_g2(altered)

    def test_cache_fingerprint_is_order_independent_but_parameter_sensitive(self):
        reordered = dict(reversed(list(self.config.items())))
        self.assertEqual(config_hash(self.config), config_hash(reordered))
        altered = copy.deepcopy(self.config)
        altered["generation"]["guidance_scale"] += 1
        self.assertNotEqual(config_hash(self.config), config_hash(altered))

    def test_resume_rejects_tampered_image_and_mismatched_config(self):
        item = candidate_plan(self.config)[0]
        fingerprint = config_hash(self.config)
        with tempfile.TemporaryDirectory(dir=self.temporary) as folder:
            root = Path(folder)
            self.assertIsNone(verified_item(root, item, fingerprint))
            path = inside(root, item["relative_path"])
            path.parent.mkdir(parents=True)
            path.write_bytes(b"original-output")
            record = dict(item, status="success", config_hash=fingerprint, sha256=file_hash(path))
            atomic_json(root / f"manifests/items/{item['image_id']}.json", record)
            self.assertEqual(verified_item(root, item, fingerprint)["sha256"], file_hash(path))
            with self.assertRaises(ValueError):
                verified_item(root, item, "another-config")
            path.write_bytes(b"tampered-output")
            with self.assertRaises(ValueError):
                verified_item(root, item, fingerprint)

    def test_paths_cannot_escape_their_run_root(self):
        with tempfile.TemporaryDirectory(dir=self.temporary) as folder:
            with self.assertRaises(ValueError):
                inside(Path(folder), "../other-run/image.png")

    def test_config_rejects_changed_class_order_and_unfrozen_bn(self):
        for change in ("classes", "bn"):
            altered = copy.deepcopy(self.config)
            if change == "classes":
                altered["dataset"]["classes"].reverse()
            else:
                altered["classifier"]["freeze_batchnorm"] = False
            with tempfile.TemporaryDirectory(dir=self.temporary) as folder:
                path = Path(folder) / "study.json"
                path.write_text(json.dumps(altered), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
