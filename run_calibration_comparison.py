"""Compare post-hoc calibration methods on the frozen benchmark_v1 V3 model.

No model or calibrator parameter is fitted on the test partition.  The script
uses the frozen feature shards/checkpoint produced by
``run_benchmark_v1_experiment.py`` and writes a self-contained paper audit to
``experiments/results/benchmark_v1/calibration``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from jev.benchmark_v1 import BenchmarkExample, read_jsonl
from jev.calibration import AdaptiveTemperatureScaler, DirichletCalibrator, TemperatureScaler
from scripts.run_visual_jev_v2_experiment import DiskFeatureDataset, file_sha256, open_cached_dataset
from scripts.run_visual_jev_v3_experiment import load_pairs
from train_visual_jev_v2 import load_records
from train_visual_jev_v3 import open_shards, subset_pairs
from visual_jev_v3_pipeline import ThreeStageVisualJEV
from run_benchmark_v1_experiment import _js, metrics_by_dataset, variable_metrics


SEED = 20260928
METHODS = ("uncalibrated", "global_ts", "multi_domain_ts", "visual_adaptive_ts", "dirichlet")
FEATURE_NAMES = (
    "normalised_entropy",
    "top1_top2_probability_gap",
    "blank_visual_sensitivity_js",
    "mean_visual_gate_weight",
    "attention_concentration",
    "log_candidate_count",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def feature_dataset(root: Path, split: str, count: int) -> DiskFeatureDataset:
    directory = root / "features" / f"{split}-shards"
    if not (directory / "manifest.json").is_file():
        raise FileNotFoundError(f"missing feature cache: {directory}")
    return DiskFeatureDataset(directory, count)


def adaptive_features(output: Any, blank_output: Any) -> torch.Tensor:
    logits = output.scores.detach().double().cpu()
    probabilities = logits.softmax(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    normalised_entropy = entropy / math.log(logits.numel())
    top = probabilities.topk(2).values
    blank_probabilities = blank_output.scores.detach().double().cpu().softmax(-1)
    visual_sensitivity = _js(probabilities, blank_probabilities)
    visual_gate_weight = float((1.0 - output.gates.detach().double().cpu()).mean().item())
    attention = output.attention.detach().double().cpu().clamp_min(1e-12)
    attention_entropy = -(attention * attention.log()).sum(-1)
    if attention.shape[-1] > 1:
        concentration = 1.0 - attention_entropy.mean() / math.log(attention.shape[-1])
    else:
        concentration = torch.tensor(1.0, dtype=torch.float64)
    return torch.tensor(
        [
            float(normalised_entropy.item()),
            float((top[0] - top[1]).item()),
            visual_sensitivity,
            visual_gate_weight,
            float(concentration.item()),
            math.log(logits.numel()),
        ],
        dtype=torch.float64,
    )


@torch.no_grad()
def score_one(
    model: ThreeStageVisualJEV,
    pre_tokens: torch.Tensor,
    text_features: torch.Tensor,
    blank_pre: torch.Tensor,
    blank_output: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    output = model(pre_tokens, text_features)
    if blank_output is None:
        blank_output = model(blank_pre, text_features)
    return output.scores.detach().float().cpu(), adaptive_features(output, blank_output), blank_output


def deterministic_folds(count: int, fold_count: int = 4) -> list[list[int]]:
    order = sorted(range(count), key=lambda i: hashlib.sha256(f"{SEED}:{i}".encode()).digest())
    return [order[fold::fold_count] for fold in range(fold_count)]


def select_adaptive_regularization(
    logits: Sequence[torch.Tensor], labels: Sequence[int], features: torch.Tensor
) -> tuple[float, dict[str, float]]:
    folds = deterministic_folds(len(logits))
    scores: dict[str, float] = {}
    for regularization in (0.01, 0.1, 1.0, 10.0):
        losses = []
        for held_out in folds:
            held = set(held_out)
            train = [i for i in range(len(logits)) if i not in held]
            global_scaler = TemperatureScaler()
            global_fit = global_scaler.fit([logits[i] for i in train], [labels[i] for i in train])
            scaler = AdaptiveTemperatureScaler(features.shape[1], initial_temperature=global_fit.temperature)
            scaler.fit(
                [logits[i] for i in train], [labels[i] for i in train], features[train],
                regularization=regularization,
            )
            transformed = scaler.apply_rows([logits[i] for i in held_out], features[held_out])
            losses.extend(
                -float(row.softmax(-1)[labels[index]].clamp_min(1e-12).log().item())
                for row, index in zip(transformed, held_out)
            )
        scores[str(regularization)] = float(np.mean(losses))
    selected = min((float(key) for key in scores), key=lambda value: (scores[str(value)], -value))
    return selected, scores


def fit_methods(
    logits: Sequence[torch.Tensor], labels: Sequence[int], features: torch.Tensor,
    examples: Sequence[BenchmarkExample], min_domain_samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    global_scaler = TemperatureScaler()
    global_fit = global_scaler.fit(logits, labels)

    domain_scalers: dict[str, TemperatureScaler] = {}
    domain_fits: dict[str, Any] = {}
    by_domain: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_domain[example.dataset].append(index)
    for domain, indices in sorted(by_domain.items()):
        if len(indices) < min_domain_samples:
            domain_fits[domain] = {"fallback": "global_ts", "sample_count": len(indices)}
            continue
        scaler = TemperatureScaler()
        fit = scaler.fit([logits[i] for i in indices], [labels[i] for i in indices])
        domain_scalers[domain] = scaler
        domain_fits[domain] = asdict(fit)

    adaptive_regularization, adaptive_cv = select_adaptive_regularization(logits, labels, features)
    adaptive = AdaptiveTemperatureScaler(features.shape[1], initial_temperature=global_fit.temperature)
    adaptive_fit = adaptive.fit(
        logits, labels, features, regularization=adaptive_regularization
    )

    dirichlet: dict[int, DirichletCalibrator] = {}
    dirichlet_fits: dict[str, Any] = {}
    by_count: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(logits):
        by_count[row.numel()].append(index)
    for class_count, indices in sorted(by_count.items()):
        minimum = max(8, 2 * class_count)
        if len(indices) < minimum:
            dirichlet_fits[str(class_count)] = {
                "fallback": "global_ts", "sample_count": len(indices), "minimum": minimum,
            }
            continue
        calibrator = DirichletCalibrator(class_count)
        matrix = torch.stack([logits[i].double() for i in indices])
        fit = calibrator.fit(matrix, [labels[i] for i in indices], regularization=0.1)
        dirichlet[class_count] = calibrator
        dirichlet_fits[str(class_count)] = calibrator.state_payload(fit)

    state = {
        "global_scaler": global_scaler,
        "domain_scalers": domain_scalers,
        "adaptive": adaptive,
        "dirichlet": dirichlet,
    }
    report = {
        "global_ts": asdict(global_fit),
        "multi_domain_ts": {
            "minimum_domain_samples": min_domain_samples,
            "fallback_temperature": global_fit.temperature,
            "domains": domain_fits,
        },
        "visual_adaptive_ts": {
            **asdict(adaptive_fit),
            "selected_regularization": adaptive_regularization,
            "calibration_only_cross_validation_nll": adaptive_cv,
            "feature_names": FEATURE_NAMES,
            "feature_mean": adaptive.feature_mean.detach().cpu().tolist(),
            "feature_std": adaptive.feature_std.detach().cpu().tolist(),
            "linear_weight": adaptive.linear.weight.detach().cpu().flatten().tolist(),
            "linear_bias": float(adaptive.linear.bias.detach().cpu().item()),
        },
        "dirichlet": {
            "grouping": "candidate_count",
            "regularization": 0.1,
            "fallback_temperature": global_fit.temperature,
            "groups": dirichlet_fits,
        },
    }
    return state, report


def apply_method(
    method: str, logits: Sequence[torch.Tensor], features: torch.Tensor,
    domains: Sequence[str], state: dict[str, Any],
) -> list[torch.Tensor]:
    if method == "uncalibrated":
        return [row.clone() for row in logits]
    if method == "global_ts":
        temperature = float(state["global_scaler"].temperature.detach().cpu().item())
        return [row / temperature for row in logits]
    if method == "multi_domain_ts":
        fallback = float(state["global_scaler"].temperature.detach().cpu().item())
        return [
            row / float(state["domain_scalers"][domain].temperature.detach().cpu().item())
            if domain in state["domain_scalers"] else row / fallback
            for row, domain in zip(logits, domains)
        ]
    if method == "visual_adaptive_ts":
        return state["adaptive"].apply_rows(logits, features)
    if method == "dirichlet":
        fallback = float(state["global_scaler"].temperature.detach().cpu().item())
        transformed = []
        for row in logits:
            calibrator = state["dirichlet"].get(row.numel())
            transformed.append(
                calibrator(row.double()).squeeze(0).detach().float().cpu()
                if calibrator is not None else row / fallback
            )
        return transformed
    raise KeyError(method)


@torch.no_grad()
def collect_original(
    model: ThreeStageVisualJEV, features: DiskFeatureDataset,
    examples: Sequence[BenchmarkExample], blank_pre: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    logits, adaptive = [], []
    model.eval()
    for index, _example in tqdm(
        enumerate(examples), total=len(examples), desc="score original", unit="sample", dynamic_ncols=True
    ):
        item = features[index]
        row, feature, _ = score_one(model, item["pre_tokens"], item["text_features"], blank_pre)
        logits.append(row)
        adaptive.append(feature)
    return logits, torch.stack(adaptive)


@torch.no_grad()
def collect_visual_conditions(
    model: ThreeStageVisualJEV, features: DiskFeatureDataset,
    examples: Sequence[BenchmarkExample], controls: dict[str, torch.Tensor],
) -> tuple[dict[str, list[torch.Tensor]], dict[str, torch.Tensor]]:
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_dataset[example.dataset].append(index)
    same_dataset_next: dict[int, int] = {}
    for indices in by_dataset.values():
        for position, index in enumerate(indices):
            same_dataset_next[index] = indices[(position + 1) % len(indices)]
    all_next = {index: (index + 1) % len(examples) for index in range(len(examples))}
    names = ("original", "blank", "noise", "wrong", "swap")
    logits = {name: [] for name in names}
    adaptive: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    for index, _example in tqdm(
        enumerate(examples), total=len(examples), desc="score visual conditions", unit="sample", dynamic_ncols=True
    ):
        item = features[index]
        text = item["text_features"]
        blank_output = model(controls["blank_pre"], text)
        pre_by_condition = {
            "original": item["pre_tokens"],
            "blank": controls["blank_pre"],
            "noise": controls["noise_pre"],
            "wrong": features[same_dataset_next[index]]["pre_tokens"],
            "swap": features[all_next[index]]["pre_tokens"],
        }
        for name, pre_tokens in pre_by_condition.items():
            row, feature, _ = score_one(model, pre_tokens, text, controls["blank_pre"], blank_output)
            logits[name].append(row)
            adaptive[name].append(feature)
    return logits, {name: torch.stack(rows) for name, rows in adaptive.items()}


def visual_metrics_from_logits(
    conditions: dict[str, list[torch.Tensor]], labels: Sequence[int]
) -> dict[str, Any]:
    original_probabilities = [row.softmax(-1) for row in conditions["original"]]
    result: dict[str, Any] = {}
    for name, rows in conditions.items():
        correct_probabilities, margins, uniform_kls, js_values, flips = [], [], [], [], []
        for index, (row, label) in enumerate(zip(rows, labels)):
            probability = row.softmax(-1)
            other = torch.cat((row[:label], row[label + 1 :])).max()
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
            correct_probabilities.append(float(probability[label].item()))
            margins.append(float((row[label] - other).item()))
            uniform_kls.append(float((math.log(row.numel()) - entropy).item()))
            js_values.append(_js(original_probabilities[index], probability))
            flips.append(int(probability.argmax().item() != original_probabilities[index].argmax().item()))
        result[name] = {
            "mean_correct_probability": float(np.mean(correct_probabilities)),
            "mean_margin": float(np.mean(margins)),
            "mean_uniform_kl": float(np.mean(uniform_kls)),
            "mean_js_from_original": float(np.mean(js_values)),
            "prediction_flip_rate_vs_original": float(np.mean(flips)),
        }
    result["summary"] = {
        "invalid_confidence_drop": result["original"]["mean_correct_probability"]
        - float(np.mean([result["blank"]["mean_correct_probability"], result["noise"]["mean_correct_probability"]])),
        "invalid_mean_kl_u": float(np.mean([result["blank"]["mean_uniform_kl"], result["noise"]["mean_uniform_kl"]])),
        "wrong_js": result["wrong"]["mean_js_from_original"],
        "swap_js": result["swap"]["mean_js_from_original"],
    }
    return result


@torch.no_grad()
def collect_semantic_pairs(
    model: ThreeStageVisualJEV, alignment: Sequence[Any], pairs: dict[int, dict[str, Any]],
    pair_text: Sequence[Any], blank_pre: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    for source, pair in tqdm(
        pairs.items(), total=len(pairs), desc="score semantic pairs", unit="pair", dynamic_ncols=True
    ):
        partner = int(pair["counterfactual_index"])
        target = int(pair["counterfactual_target_index"])
        text = pair_text[int(pair["_cache_index"])]["text_features"]
        blank_output = model(blank_pre, text)
        a, a_feature, _ = score_one(model, alignment[source]["pre_tokens"], text, blank_pre, blank_output)
        b, b_feature, _ = score_one(model, alignment[partner]["pre_tokens"], text, blank_pre, blank_output)
        rows.append({"a": a, "b": b, "a_feature": a_feature, "b_feature": b_feature, "target": target})
    return rows


def semantic_metrics_for_method(
    rows: list[dict[str, Any]], method: str, state: dict[str, Any]
) -> dict[str, float]:
    logits = [row[key] for row in rows for key in ("a", "b")]
    features = torch.stack([row[f"{key}_feature"] for row in rows for key in ("a", "b")])
    calibrated = apply_method(method, logits, features, ["coco_hard_negative"] * len(logits), state)
    source_correct = partner_correct = both = flips = 0
    source_margins, partner_margins = [], []
    for index, row in enumerate(rows):
        a_scores, b_scores = calibrated[2 * index], calibrated[2 * index + 1]
        target = row["target"]
        a, b = int(a_scores.argmax().item()), int(b_scores.argmax().item())
        a_ok, b_ok = a == 0, b == target
        source_correct += int(a_ok); partner_correct += int(b_ok)
        both += int(a_ok and b_ok); flips += int(a != b)
        source_margins.append(float(a_scores[0] - a_scores[1:].max()))
        partner_margins.append(float(b_scores[target] - torch.cat((b_scores[:target], b_scores[target + 1 :])).max()))
    count = max(1, len(rows))
    return {
        "pair_count": len(rows), "source_accuracy": source_correct / count,
        "counterfactual_accuracy": partner_correct / count,
        "both_directions_accuracy": both / count, "prediction_flip_rate": flips / count,
        "source_target_margin": float(np.mean(source_margins)),
        "counterfactual_target_margin": float(np.mean(partner_margins)),
    }


def write_summary(path: Path, comparison: dict[str, Any], visual: dict[str, Any]) -> None:
    baseline = comparison["methods"]["uncalibrated"]["test"]["overall"]
    best = {metric: min(METHODS, key=lambda name: comparison["methods"][name]["test"]["overall"][metric]) for metric in ("nll", "brier", "ece")}
    lines = [
        "# Visual-JEV V3 calibration comparison",
        "",
        "All calibrators were fitted only on the 64-example calibration split. The 64-example test split was used only once for reporting.",
        "",
        "| Method | Accuracy | Macro-F1 | NLL | Brier | ECE-10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        metrics = comparison["methods"][method]["test"]["overall"]
        lines.append(f"| {method} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['nll']:.4f} | {metrics['brier']:.4f} | {metrics['ece']:.4f} |")
    lines.extend(["", "Best by metric:"])
    for metric in ("nll", "brier", "ece"):
        method = best[metric]
        value = comparison["methods"][method]["test"]["overall"][metric]
        lines.append(f"- {metric.upper()}: {method} ({value:.4f}; improvement {baseline[metric] - value:.4f})")
    adaptive = comparison["methods"]["visual_adaptive_ts"]["test"]["overall"]
    global_ts = comparison["methods"]["global_ts"]["test"]["overall"]
    wins = sum(adaptive[m] < global_ts[m] for m in ("nll", "brier", "ece"))
    recommendation = (
        "worth advancing to a larger multi-seed paper ablation, but not yet replacing Multi-Domain TS as the headline method"
        if wins >= 2 else "not yet justified as the paper method on this controlled run"
    )
    lines.extend([
        "",
        f"Visual-Adaptive TS beats Global TS on {wins}/3 probability metrics and is therefore **{recommendation}**.",
        "",
        "Argmax visual-dependency checks:",
    ])
    for method in METHODS:
        semantic = visual["methods"][method]["semantic_counterfactual"]
        lines.append(f"- {method}: Pair-both={semantic['both_directions_accuracy']:.4f}, Pair-flip={semantic['prediction_flip_rate']:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path("data/benchmark_v1/manifests"))
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments/benchmark_v1_controlled"))
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/benchmark_v1_controlled/checkpoints/v3_full.pt"))
    parser.add_argument("--results-root", type=Path, default=Path("experiments/results/benchmark_v1/calibration"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-domain-samples", type=int, default=8)
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    manifest_root, experiment_root = args.manifest_root.resolve(), args.experiment_root.resolve()
    results_root = args.results_root.resolve(); results_root.mkdir(parents=True, exist_ok=True)
    manifest_metadata = json.loads((manifest_root / "manifest.json").read_text(encoding="utf-8"))
    manifest_profile = manifest_metadata.get("provenance", {}).get("profile", "unknown")
    examples = {split: list(read_jsonl(manifest_root / f"{split}.jsonl")) for split in ("calibration", "test")}
    groups = {split: {row.group_id for row in values} for split, values in examples.items()}
    overlap = len(groups["calibration"] & groups["test"])
    if overlap:
        raise RuntimeError(f"calibration/test leakage: {overlap} groups")
    features = {split: feature_dataset(experiment_root, split, len(rows)) for split, rows in examples.items()}
    controls = torch.load(experiment_root / "features" / "fixed_controls.pt", map_location="cpu", weights_only=True)
    model, checkpoint_payload = ThreeStageVisualJEV.from_checkpoint(args.checkpoint, map_location=args.device)
    model = model.to(args.device).eval()

    calibration_logits, calibration_features = collect_original(model, features["calibration"], examples["calibration"], controls["blank_pre"])
    test_logits, test_features = collect_original(model, features["test"], examples["test"], controls["blank_pre"])
    calibration_labels = [row.label for row in examples["calibration"]]
    test_labels = [row.label for row in examples["test"]]
    calibration_domains = [row.dataset for row in examples["calibration"]]
    test_domains = [row.dataset for row in examples["test"]]
    state, fits = fit_methods(
        calibration_logits, calibration_labels, calibration_features,
        examples["calibration"], args.min_domain_samples,
    )

    method_reports: dict[str, Any] = {}
    transformed_test: dict[str, list[torch.Tensor]] = {}
    for method in METHODS:
        calibrated_cal = apply_method(method, calibration_logits, calibration_features, calibration_domains, state)
        calibrated_test = apply_method(method, test_logits, test_features, test_domains, state)
        transformed_test[method] = calibrated_test
        method_reports[method] = {
            "fit": fits.get(method, {"fitted": False}),
            "calibration": metrics_by_dataset(calibrated_cal, calibration_labels, examples["calibration"]),
            "test": metrics_by_dataset(calibrated_test, test_labels, examples["test"]),
            "argmax_invariant_vs_uncalibrated": [int(row.argmax()) for row in calibrated_test]
            == [int(row.argmax()) for row in test_logits],
        }
    baseline = method_reports["uncalibrated"]["test"]["overall"]
    for method in METHODS:
        metrics = method_reports[method]["test"]["overall"]
        method_reports[method]["test_improvement_vs_uncalibrated"] = {
            key: baseline[key] - metrics[key] for key in ("nll", "brier", "ece")
        }

    condition_logits, condition_features = collect_visual_conditions(model, features["test"], examples["test"], controls)
    visual_methods: dict[str, Any] = {}
    legacy_data = Path("data/visual-jev-v2")
    legacy_root = Path("experiments/visual_jev_v3_paper")
    legacy_pair_root = Path("data/visual-jev-v3")
    legacy_validation_records = load_records(
        legacy_data / "splits" / "validation.jsonl", legacy_data,
        require_images=False,
    )
    legacy_alignment = open_shards(legacy_root / "features" / "validation-shards", len(legacy_validation_records))
    pair_path = legacy_pair_root / "pairs" / "validation.jsonl"
    pairs_all = load_pairs(pair_path, len(legacy_validation_records))
    pair_text = open_cached_dataset(
        legacy_pair_root / "features" / "validation-pair-text.pt",
        source_sha256=file_sha256(pair_path), item_key="pair_text",
    )
    if legacy_alignment is None or pair_text is None:
        raise RuntimeError("legacy semantic-pair caches are incomplete")
    semantic_test = subset_pairs(pairs_all, {i for i in range(len(legacy_validation_records)) if i % 2 == 1})
    semantic_rows = collect_semantic_pairs(model, legacy_alignment, semantic_test, pair_text, controls["blank_pre"])
    for method in METHODS:
        calibrated_conditions = {
            name: apply_method(method, rows, condition_features[name], test_domains, state)
            for name, rows in condition_logits.items()
        }
        visual_methods[method] = {
            "conditions": visual_metrics_from_logits(calibrated_conditions, test_labels),
            "semantic_counterfactual": semantic_metrics_for_method(semantic_rows, method, state),
        }

    comparison = {
        "experiment": "Visual-JEV V3 post-hoc calibration comparison",
        "seed": SEED,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_stage": checkpoint_payload.get("stage"),
        "fit_policy": "all calibrator parameters and calibration-only CV use calibration split; test is reporting only",
        "calibration_split": {
            "count": len(examples["calibration"]),
            "sha256": file_sha256(manifest_root / "calibration.jsonl"),
            "datasets": dict(Counter(calibration_domains)),
            "candidate_counts": dict(Counter(row.numel() for row in calibration_logits)),
        },
        "test_split": {
            "count": len(examples["test"]),
            "sha256": file_sha256(manifest_root / "test.jsonl"),
            "datasets": dict(Counter(test_domains)),
            "candidate_counts": dict(Counter(row.numel() for row in test_logits)),
        },
        "calibration_test_group_overlap": overlap,
        "adaptive_feature_definitions": {
            "normalised_entropy": "H(softmax(logits)) / log(candidate_count)",
            "top1_top2_probability_gap": "largest minus second-largest predicted probability",
            "blank_visual_sensitivity_js": "Jensen-Shannon divergence between original and fixed-blank distributions",
            "mean_visual_gate_weight": "mean(1 - text_gate) in the Visual-JEV fusion layer",
            "attention_concentration": "1 - mean candidate attention entropy / log(visual_token_count)",
            "log_candidate_count": "log(number of answer candidates)",
        },
        "methods": method_reports,
        "runtime_seconds": time.time() - started,
        "device": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else str(args.device),
    }
    visual = {
        "test_conditions": ("original", "blank", "noise", "wrong", "swap"),
        "semantic_pair_count": len(semantic_rows),
        "methods": visual_methods,
    }
    write_json(results_root / "calibration_comparison.json", comparison)
    write_json(results_root / "visual_dependency_comparison.json", visual)
    write_json(results_root / "calibrator_fits.json", fits)
    write_json(results_root / "run_provenance.json", {
        "seed": SEED,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_root": str(manifest_root),
        "calibration_manifest_sha256": comparison["calibration_split"]["sha256"],
        "test_manifest_sha256": comparison["test_split"]["sha256"],
        "calibration_test_group_overlap": overlap,
        "device": comparison["device"],
        "runtime_seconds": comparison["runtime_seconds"],
        "torch_version": torch.__version__,
        "full_scale_pending": manifest_metadata.get("provenance", {}).get("full_scale_pending", True),
        "profile": manifest_profile,
    })
    torch.save(
        {
            "calibration_logits": calibration_logits, "calibration_labels": calibration_labels,
            "calibration_features": calibration_features, "calibration_domains": calibration_domains,
            "test_logits": test_logits, "test_labels": test_labels,
            "test_features": test_features, "test_domains": test_domains,
            "feature_names": FEATURE_NAMES,
        },
        results_root / "logits_features_cache.pt",
    )
    with (results_root / "calibration_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "scope", "count", "accuracy", "macro_f1", "nll", "brier", "ece", "mean_correct_probability", "mean_margin", "argmax_invariant"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for method in METHODS:
            for scope, metrics in method_reports[method]["test"].items():
                writer.writerow({"method": method, "scope": scope, **{key: metrics.get(key) for key in fields[2:-1]}, "argmax_invariant": method_reports[method]["argmax_invariant_vs_uncalibrated"]})
    with (results_root / "visual_dependency_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "condition", "mean_correct_probability", "mean_margin", "mean_uniform_kl", "mean_js_from_original", "prediction_flip_rate_vs_original"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for method in METHODS:
            for condition in ("original", "blank", "noise", "wrong", "swap"):
                writer.writerow({"method": method, "condition": condition, **visual_methods[method]["conditions"][condition]})
    write_summary(results_root / "summary.md", comparison, visual)
    print(json.dumps({
        "results_root": str(results_root),
        "test": {method: method_reports[method]["test"]["overall"] for method in METHODS},
        "argmax_invariant": {method: method_reports[method]["argmax_invariant_vs_uncalibrated"] for method in METHODS},
        "semantic": {method: visual_methods[method]["semantic_counterfactual"] for method in METHODS},
        "runtime_seconds": comparison["runtime_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
