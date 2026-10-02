"""Audit Visual-JEV visual dependence from the paper evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_visual_jev_v3 import load_config, open_shards


def pooled_distance(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a = a.float().mean(0)
    b = b.float().mean(0)
    return float((a - b).norm()), float(1.0 - F.cosine_similarity(a, b, dim=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/visual_jev_v3_paper.json"))
    parser.add_argument("--metrics", type=Path, default=Path("experiments/results/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/visual_dependency.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    full = metrics["models"]["v3_full"]
    no_c = metrics["models"]["v3_without_stage_c"]
    no_a = metrics["models"]["v3_without_stage_a"]
    data = full["visual_dependency"]

    validation_count = metrics["test_records"] * 2
    paper_root = Path(config["paper_root"])
    original = open_shards(paper_root / "features" / "validation-shards", validation_count)
    controls = open_shards(paper_root / "features" / "validation_controls-shards", validation_count)
    if original is None or controls is None:
        raise RuntimeError("pre-merger diagnostic caches are missing")
    rows = {"blank_l2": [], "blank_cosine": [], "noise_l2": [], "noise_cosine": []}
    for index in range(min(32, validation_count)):
        source = original[index]["pre_tokens"]
        blank_l2, blank_cos = pooled_distance(source, controls[index]["blank_pre"])
        noise_l2, noise_cos = pooled_distance(source, controls[index]["noise_pre"])
        rows["blank_l2"].append(blank_l2)
        rows["blank_cosine"].append(blank_cos)
        rows["noise_l2"].append(noise_l2)
        rows["noise_cosine"].append(noise_cos)
    cache_audit = {
        name: {"mean": float(np.mean(values)), "min": float(np.min(values))}
        for name, values in rows.items()
    }
    cache_audit["passed"] = cache_audit["blank_l2"]["min"] > 1e-6 and cache_audit["noise_l2"]["min"] > 1e-6

    checks = {
        "cache_inputs_are_distinct": cache_audit["passed"],
        "blank_is_near_uniform": data["blank"]["mean_uniform_kl"] < 0.01,
        "noise_is_near_uniform": data["noise"]["mean_uniform_kl"] < 0.01,
        "invalid_images_reduce_confidence": data["visual_sensitivity"]["invalid_confidence_drop"] > 0.10,
        "semantic_pairs_flip_above_chance": full["semantic_counterfactual"]["prediction_flip_rate"] > 0.50,
        "both_directions_above_without_stage_c": full["semantic_counterfactual"]["both_directions_accuracy"] > no_c["semantic_counterfactual"]["both_directions_accuracy"],
        "stage_a_improves_both_directions": full["semantic_counterfactual"]["both_directions_accuracy"] > no_a["semantic_counterfactual"]["both_directions_accuracy"],
    }
    report = {
        "model": "v3_full",
        "checks": checks,
        "passed": all(checks.values()),
        "cache_audit": cache_audit,
        "conditions": data,
        "semantic_counterfactual": full["semantic_counterfactual"],
        "ablations": {
            "without_stage_a": no_a["semantic_counterfactual"],
            "without_stage_c": no_c["semantic_counterfactual"],
        },
        "definition": {
            "visual_sensitivity": "mean Jensen-Shannon divergence between original and perturbed candidate distributions",
            "uniformity": "KL(p || Uniform); lower is more balanced",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
