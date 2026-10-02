#!/usr/bin/env python3
"""Run full independent Visual-JEV training for several datasets and seeds.

Run this file from the Open-Jev repository root.  It reuses a completed base
seed when compatible, trains every other dataset/seed independently, and
writes per-dataset mean, sample standard deviation, minimum, and maximum.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "run_independent_datasets.py").is_file():
    raise SystemExit("run this script from the Open-Jev repository root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_independent_datasets as runner  # noqa: E402


DEFAULT_DATASETS = ("aokvqa", "scienceqa_image_only", "iconqa_choice")
METRIC_FIELDS = ("accuracy", "macro_f1", "nll", "brier", "ece")


def aggregate(rows: list[dict[str, object]], field: str) -> dict[str, float | None]:
    values = [float(row[field]) for row in rows]
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def compatible_base_result(path: Path, dataset: str, seed: int) -> bool:
    config_path = path.parent / "config.json"
    if not path.is_file() or not config_path.is_file():
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return (
        config.get("dataset") == dataset
        and config.get("seed") == seed
        and config.get("mixed_training") is False
        and config.get("profile") == "full_independent"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=runner.DATASETS, default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260928, 20260929, 20260930])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("experiments/results/independent_multiseed_full"),
    )
    parser.add_argument(
        "--base-result-root",
        type=Path,
        default=Path("experiments/results/independent_datasets_full"),
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=Path("experiments/visual_jev_v3_paper/checkpoints/v3_full.pt"),
    )
    parser.add_argument("--stage-a-epochs", type=int, default=2)
    parser.add_argument("--stage-b-epochs", type=int, default=16)
    parser.add_argument("--stage-c-epochs", type=int, default=6)
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    base_root = args.base_result_root.resolve()
    checkpoint = args.reference_checkpoint.resolve()
    result_root.mkdir(parents=True, exist_ok=True)

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

    complete: dict[str, object] = {
        "datasets": args.datasets,
        "seeds": args.seeds,
        "mixed_training": False,
        "fixed_splits_and_frozen_features": True,
        "results": {},
    }
    for dataset in args.datasets:
        bundle = bundles[dataset]
        output_name = runner.OUTPUT_NAMES[dataset]
        rows: list[dict[str, object]] = []
        for seed in args.seeds:
            runner.SEED = seed
            runner.set_seed(seed)
            probe = runner.fresh_model(checkpoint, bundle.features["train"][0], args.device)
            smoke = {"initial_state_sha256": runner._state_sha256(probe)}
            del probe
            runner.set_seed(seed)

            output = result_root / output_name / f"seed_{seed}"
            result_path = output / "result.json"
            base_path = base_root / output_name / "result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                source = result_path
            elif compatible_base_result(base_path, dataset, seed):
                result = json.loads(base_path.read_text(encoding="utf-8"))
                source = base_path
            else:
                result = runner.train_one(
                    dataset,
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
                source = result_path

            calibrated = result["calibration"]["methods"]["global_temperature_scaling"]
            row: dict[str, object] = {
                "dataset": dataset,
                "seed": seed,
                "test_n": int(result["metrics"]["count"]),
                **{field: float(result["metrics"][field]) for field in METRIC_FIELDS},
                "calibrated_nll": float(calibrated["nll"]),
                "calibrated_brier": float(calibrated["brier"]),
                "calibrated_ece": float(calibrated["ece"]),
                "training_time_seconds": float(result["training_time_seconds"]),
                "result_path": str(source.resolve()),
            }
            rows.append(row)
            del result
            gc.collect()
            if runner.torch.cuda.is_available():
                runner.torch.cuda.empty_cache()

        fields = (*METRIC_FIELDS, "calibrated_nll", "calibrated_brier", "calibrated_ece")
        dataset_summary = {
            "dataset": dataset,
            "display_name": runner.DISPLAY_NAMES[dataset],
            "seeds": args.seeds,
            "fixed_split": True,
            "mixed_training": False,
            "runs": rows,
            "aggregate": {field: aggregate(rows, field) for field in fields},
        }
        summary_path = result_root / output_name / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(dataset_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        complete["results"][dataset] = dataset_summary

    (result_root / "summary.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(complete, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
