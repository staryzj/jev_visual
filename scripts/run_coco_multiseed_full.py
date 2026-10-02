#!/usr/bin/env python3
"""Run full COCO hard-negative Visual-JEV training for multiple random seeds.

The data split and frozen features stay fixed.  Seeds affect training order and
optimizer stochasticity only, so variation estimates training sensitivity
rather than split-selection variance.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_independent_datasets as runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260928, 20260929, 20260930])
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("experiments/results/coco_multiseed_full"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage-a-epochs", type=int, default=2)
    parser.add_argument("--stage-b-epochs", type=int, default=16)
    parser.add_argument("--stage-c-epochs", type=int, default=6)
    parser.add_argument(
        "--existing-base-result",
        type=Path,
        default=Path("experiments/results/independent_datasets_full/coco/result.json"),
        help="reuse the completed base-seed full COCO result when its config seed matches",
    )
    args = parser.parse_args()

    root = args.result_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    context = runner.Context(
        args.device,
        manifest_root=Path("data/benchmark_v1/manifests").resolve(),
        experiment_root=Path("experiments/benchmark_v1_controlled").resolve(),
    )
    bundles, _ = runner.build_executed_bundles(
        context,
        full_manifest_root=Path("data/benchmark_v1_full/manifests").resolve(),
        full_experiment_root=Path("experiments/benchmark_v1_full").resolve(),
        train_cap=0,
        manifest_root=Path("data/independent_datasets").resolve(),
    )
    bundle = bundles["coco_hard_negative"]
    checkpoint = Path("experiments/visual_jev_v3_paper/checkpoints/v3_full.pt").resolve()

    results = []
    for seed in args.seeds:
        runner.SEED = seed
        runner.set_seed(seed)
        probe = runner.fresh_model(checkpoint, bundle.features["train"][0], args.device)
        smoke = {"initial_state_sha256": runner._state_sha256(probe)}
        del probe
        runner.set_seed(seed)
        output = root / f"seed_{seed}"
        result_path = output / "result.json"
        existing = args.existing_base_result.resolve()
        existing_config = existing.parent / "config.json"
        if (
            not result_path.is_file()
            and existing.is_file()
            and existing_config.is_file()
            and json.loads(existing_config.read_text(encoding="utf-8")).get("seed") == seed
        ):
            result_path = existing
            result = json.loads(existing.read_text(encoding="utf-8"))
        elif result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = runner.train_one(
                "coco_hard_negative",
                bundle,
                context,
                output=output,
                reference_checkpoint=checkpoint,
                device=args.device,
                stage_a_epochs=args.stage_a_epochs,
                stage_b_epochs=args.stage_b_epochs,
                stage_c_epochs=args.stage_c_epochs,
                smoke=smoke,
            )
        pair = result["visual_dependency"]["semantic_counterfactual"]
        results.append(
            {
                "seed": seed,
                "accuracy": result["metrics"]["accuracy"],
                "macro_f1": result["metrics"]["macro_f1"],
                "nll": result["metrics"]["nll"],
                "ece": result["metrics"]["ece"],
                "pair_both": pair["both_directions_accuracy"],
                "pair_flip": pair["prediction_flip_rate"],
                "training_time_seconds": result["training_time_seconds"],
                "result": str(result_path),
            }
        )

    fields = ("accuracy", "macro_f1", "nll", "ece", "pair_both", "pair_flip")
    aggregate = {}
    for field in fields:
        values = [row[field] for row in results]
        aggregate[field] = {
            "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values),
            "max": max(values),
        }
    payload = {
        "seeds": args.seeds,
        "fixed_split": True,
        "frozen_feature_cache": True,
        "stage_epochs": {
            "A_alignment": args.stage_a_epochs,
            "B_candidate": args.stage_b_epochs,
            "C_gated": args.stage_c_epochs,
        },
        "runs": results,
        "aggregate": aggregate,
    }
    (root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
