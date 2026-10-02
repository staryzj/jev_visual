#!/usr/bin/env python3
"""Run isolated Visual-JEV V3 controlled experiments and aggregate real metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


METRIC_PATHS = {
    "accuracy": ("classification", "accuracy"),
    "macro_f1": ("classification", "macro_f1"),
    "nll": ("classification", "nll"),
    "brier": ("classification", "brier"),
    "ece": ("classification", "ece"),
    "pair_both_accuracy": ("semantic_counterfactual", "both_directions_accuracy"),
    "pair_flip_rate": ("semantic_counterfactual", "prediction_flip_rate"),
}


def nested(report: dict, keys: tuple[str, ...]) -> float:
    value = report
    for key in keys:
        value = value[key]
    return float(value)


def summarize(values: list[float]) -> dict[str, float | int | list[float]]:
    return {
        "n": len(values),
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "sem": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=Path("configs/visual_jev_v3_paper.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260928, 20260929, 20260930])
    parser.add_argument("--root", type=Path, default=Path("experiments/results/controlled_multiseed"))
    parser.add_argument("--reuse-seed", type=int, default=20260928)
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    output_root = args.root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shared_features = (repo / base["paper_root"] / "features").resolve()
    if not shared_features.exists():
        raise FileNotFoundError(f"Missing shared feature cache: {shared_features}")

    seed_metrics: dict[str, dict] = {}
    for seed in args.seeds:
        seed_root = output_root / f"seed-{seed}"
        result_root = seed_root / "results"
        paper_root = seed_root / "paper"
        config_path = seed_root / "config.json"
        metrics_path = result_root / "metrics.json"
        seed_root.mkdir(parents=True, exist_ok=True)

        if seed == args.reuse_seed and not metrics_path.exists():
            source_metrics = repo / "experiments/results/metrics.json"
            source_provenance = repo / "experiments/results/visual_jev_v3_provenance.json"
            if source_metrics.exists():
                result_root.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_metrics, metrics_path)
                if source_provenance.exists():
                    shutil.copy2(source_provenance, result_root / source_provenance.name)

        if not metrics_path.exists():
            paper_root.mkdir(parents=True, exist_ok=True)
            feature_link = paper_root / "features"
            if not feature_link.exists():
                os.symlink(shared_features, feature_link, target_is_directory=True)
            config = dict(base)
            config["seed"] = seed
            config["paper_root"] = str(paper_root)
            config["results_root"] = str(result_root)
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            log_path = seed_root / "run.log"
            with log_path.open("a", encoding="utf-8") as log:
                subprocess.run(
                    [sys.executable, "train_visual_jev_v3.py", "--config", str(config_path), "--skip-cache"],
                    cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True,
                )
                subprocess.run(
                    [sys.executable, "eval_visual_jev_v3.py", "--config", str(config_path)],
                    cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True,
                )

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        seed_metrics[str(seed)] = payload

    aggregate: dict[str, object] = {
        "protocol": "controlled 128-example test split; 48 semantic counterfactual pairs",
        "seeds": args.seeds,
        "seed_metrics_files": {
            seed: str((output_root / f"seed-{seed}" / "results" / "metrics.json").resolve())
            for seed in args.seeds
        },
        "models": {},
    }
    for model_name in ("v3_without_stage_c", "v3_full"):
        per_metric = {}
        for label, path in METRIC_PATHS.items():
            values = [nested(seed_metrics[str(seed)]["models"][model_name], path) for seed in args.seeds]
            per_metric[label] = summarize(values)
        aggregate["models"][model_name] = per_metric

    aggregate_path = output_root / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
