"""Single project CLI: A diagnostics/generation and B real-only development baseline."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from a_protocol import ROOT, atomic_json, candidate_plan, config_hash, load_config, training_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "plan", "prepare", "generate", "smoke", "select", "features", "train", "evaluate", "summarize"])
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--models", action="store_true", help="doctor: download/load SD1.5, CLIP and ResNet, and test inference")
    parser.add_argument("--smoke", action="store_true", help="isolated development mode for generate/prepare/features/train, not full G2")
    parser.add_argument("--limit", type=int, default=1, help="generate --smoke: maximum 10")
    parser.add_argument("--final", action="store_true", help="reserved for the final evaluation gate")
    parser.add_argument("--component", choices=["b"], help="smoke: B-only check, not full G2")
    parser.add_argument("--download", action="store_true", help="B: allow public CIFAR-10/ResNet weight downloads")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="B: explicit compute device")
    parser.add_argument("--threads", type=int, default=4, help="B: CPU threads")
    parser.add_argument("--batch-size", type=int, default=8, help="B: original images/batch (4 views each)")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "plan":
            fingerprint = config_hash(config)
            plan = {"schema_version": 1, "config_hash": fingerprint, "status": "planned",
                    "protocol_status": config["protocol_status"], "config": config,
                    "candidates": candidate_plan(config), "training_runs": training_plan(config)}
            path = ROOT / "runs/plans" / fingerprint / "plan.json"
            atomic_json(path, plan)
            result = {"status": "planned", "candidate_images": len(plan["candidates"]),
                      "classifier_runs": len(plan["training_runs"]), "config_hash": fingerprint,
                      "plan": str(path.relative_to(ROOT)), "gpu_used": False, "models_downloaded": False}
        elif args.command in {"prepare", "features", "train"} or (args.command == "smoke" and args.component == "b"):
            from b_runtime import run_b
            result = run_b(config, args)
        elif args.command in {"doctor", "generate"}:
            from a_runtime import doctor, generate
            result = doctor(config, models=args.models) if args.command == "doctor" else generate(config, smoke=args.smoke, limit=args.limit)
        else:
            parser.error(f"{args.command} is not implemented yet. G1 exposes doctor/plan/generate; G2 remains pending.")
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        atomic_json(ROOT / "runs/failures" / f"{stamp}.json", {
            "status": "failed", "command": args.command, "config": args.config,
            "error_type": type(exc).__name__, "error": str(exc)})
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
