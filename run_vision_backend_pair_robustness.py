"""Run the semantic-pair backend proof across fixed seeds and aggregate means."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from jev.vision_backends import BACKEND_NAMES
from run_vision_backend_compatibility import load_manifests, prepare_backend_features, write_json
from run_vision_backend_pair_proof import (
    feature_index,
    load_pair_rows,
    pair_tensors,
    train_pair_model,
)
from scripts.run_visual_jev_v2_experiment import DiskFeatureDataset

SEEDS = (20260928, 20260929, 20260930)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path(
            "experiments/results/benchmark_v1/recent_methods_fix/"
            "pipeline_regression/manifests"
        ),
    )
    parser.add_argument(
        "--compatibility-root",
        type=Path,
        default=Path("experiments/results/benchmark_v1/vision_backend_compatibility"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", default=",".join(str(value) for value in SEEDS))
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("robustness proof requires at least three distinct seeds")

    root = args.compatibility_root.resolve()
    output = root / "semantic_pair_robustness"
    output.mkdir(parents=True, exist_ok=True)
    examples = load_manifests(args.manifest_root)
    train_rows = load_pair_rows(Path("data/visual-jev-v3/pairs/train.jsonl"))
    validation_rows = load_pair_rows(Path("data/visual-jev-v3/pairs/validation.jsonl"))
    train_text = DiskFeatureDataset(
        Path("data/visual-jev-v3/features/train-pair-text-shards"), len(train_rows)
    )
    validation_text = DiskFeatureDataset(
        Path("data/visual-jev-v3/features/validation-pair-text-shards"),
        len(validation_rows),
    )
    validation_indices = [
        index for index, row in enumerate(validation_rows) if int(row["source_index"]) % 2 == 0
    ]
    test_indices = [
        index for index, row in enumerate(validation_rows) if int(row["source_index"]) % 2 == 1
    ]

    all_results = []
    backend_aggregates = []
    for backend_name in BACKEND_NAMES:
        features, controls, metadata = prepare_backend_features(
            backend_name, examples, args.manifest_root, root, args.device, 16
        )
        train_visual, validation_visual = feature_index(examples, features)
        train = pair_tensors(train_rows, train_text, train_visual)
        validation = pair_tensors(
            validation_rows, validation_text, validation_visual, validation_indices
        )
        test = pair_tensors(validation_rows, validation_text, validation_visual, test_indices)
        initial = root / "checkpoints" / f"{backend_name}.pt"
        backend_results = []
        for seed in seeds:
            result = train_pair_model(
                backend_name,
                initial,
                train,
                validation,
                test,
                controls,
                metadata,
                output,
                args.device,
                args.epochs,
                args.batch_size,
                seed=seed,
            )
            write_json(output / "seed_results" / f"{backend_name}_seed{seed}.json", result)
            backend_results.append(result)
            all_results.append(result)
        names = (
            "source_accuracy",
            "counterfactual_accuracy",
            "both_directions_accuracy",
            "prediction_flip_rate",
        )
        aggregate = {"backend": backend_name, "seeds": list(seeds), "runs": backend_results}
        for name in names:
            values = [result["test"]["original"][name] for result in backend_results]
            aggregate[f"mean_{name}"] = float(np.mean(values))
            aggregate[f"std_{name}"] = float(np.std(values))
        aggregate["acceptance"] = {
            "mean_source_accuracy_at_least_0_60": aggregate["mean_source_accuracy"] >= 0.60,
            "mean_counterfactual_accuracy_at_least_0_60": aggregate[
                "mean_counterfactual_accuracy"
            ]
            >= 0.60,
            "mean_pair_both_at_least_0_40": aggregate["mean_both_directions_accuracy"]
            >= 0.40,
            "mean_flip_at_least_0_40": aggregate["mean_prediction_flip_rate"] >= 0.40,
            "all_seeds_original_exceeds_invalid": all(
                result["test"]["original"]["both_directions_accuracy"]
                > max(
                    result["test"]["blank"]["both_directions_accuracy"],
                    result["test"]["noise"]["both_directions_accuracy"],
                )
                for result in backend_results
            ),
        }
        aggregate["acceptance"]["passed"] = all(aggregate["acceptance"].values())
        write_json(output / "backend_aggregates" / f"{backend_name}.json", aggregate)
        backend_aggregates.append(aggregate)
        del features, controls, train, validation, test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "objective": "three-seed robustness proof for a shared Visual-JEV training loop across two encoder families",
        "seeds": list(seeds),
        "selection_policy": "per-seed checkpoint chosen on validation pair-both then mean directional accuracy",
        "test_policy": "same fixed odd-source-index held-out semantic pairs; no seed selected on test",
        "acceptance": "pre-declared single-run thresholds applied to the arithmetic mean across all fixed seeds",
        "backends": backend_aggregates,
        "overall_passed": all(row["acceptance"]["passed"] for row in backend_aggregates),
        "claim_boundary": "two frozen encoder families on COCO semantic counterfactual pairs",
    }
    write_json(output / "summary.json", summary)
    rows = [
        {
            "backend": row["backend"],
            "mean_source_accuracy": row["mean_source_accuracy"],
            "std_source_accuracy": row["std_source_accuracy"],
            "mean_counterfactual_accuracy": row["mean_counterfactual_accuracy"],
            "std_counterfactual_accuracy": row["std_counterfactual_accuracy"],
            "mean_pair_both": row["mean_both_directions_accuracy"],
            "std_pair_both": row["std_both_directions_accuracy"],
            "mean_pair_flip": row["mean_prediction_flip_rate"],
            "std_pair_flip": row["std_prediction_flip_rate"],
            "passed": row["acceptance"]["passed"],
        }
        for row in backend_aggregates
    ]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"overall_passed": summary["overall_passed"], "backends": rows}, indent=2))


if __name__ == "__main__":
    main()
