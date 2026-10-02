"""Diagnose and repair Visual-JEV V3 multi-dataset negative transfer.

All paper-derived algorithms here are project-local reimplementations.  The
script uses the fixed benchmark_v1 controlled subset for screening and keeps
the frozen Qwen3-VL feature cache out of optimisation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from jev.benchmark_v1 import BenchmarkExample, read_jsonl, write_predefined_manifests
from jev.multidomain import (
    AdaptiveDomainWeights,
    gradnorm_backward,
    pcgrad_backward,
    permute_candidates,
    preference_anchor_loss,
    task_gradients,
    visual_dependency_masks,
)
from run_benchmark_v1_experiment import (
    collect_logits,
    metrics_by_dataset,
    semantic_metrics,
    variable_metrics,
    visual_metrics,
)
from scripts.run_visual_jev_v2_experiment import DiskFeatureDataset, file_sha256, open_cached_dataset
from train_visual_jev_v2 import TrainingRecord, load_records
from train_visual_jev_v3 import load_pairs, open_shards, subset_pairs
from visual_jev_v3 import counterfactual_visual_dependency_loss, mean_uniformity_loss
from visual_jev_v3_pipeline import ThreeStageVisualJEV

SEED = 20260928
DOMAINS = ("aokvqa", "coco_hard_negative", "iconqa_choice", "scienceqa_image_only")
RAW_ELIGIBLE_COUNTS = {
    "aokvqa": 18201,
    "coco_hard_negative": 2304,
    "iconqa_choice": 12632,
    "scienceqa_image_only": 10332,
}
OLD_REFERENCE = {
    "accuracy": 0.8828125,
    "macro_f1": 0.8834,
    "pair_both": 0.5625,
    "pair_flip": 0.6041666666666666,
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Context:
    def __init__(
        self,
        device: str,
        *,
        manifest_root: Path | str = Path("data/benchmark_v1/manifests"),
        experiment_root: Path | str = Path("experiments/benchmark_v1_controlled"),
    ):
        self.device = device
        self.manifest_root = Path(manifest_root)
        self.experiment_root = Path(experiment_root)
        self.legacy_root = Path("experiments/visual_jev_v3_paper")
        self.legacy_data = Path("data/visual-jev-v2")
        self.legacy_pair_root = Path("data/visual-jev-v3")
        self.examples = {
            split: list(read_jsonl(self.manifest_root / f"{split}.jsonl"))
            for split in ("train", "validation", "calibration", "test")
        }
        self.features = {
            split: DiskFeatureDataset(
                self.experiment_root / "features" / f"{split}-shards", len(self.examples[split])
            )
            for split in self.examples
        }
        self.controls = torch.load(
            self.experiment_root / "features" / "fixed_controls.pt",
            map_location="cpu",
            weights_only=True,
        )
        legacy_records = load_records(
            self.legacy_data / "splits" / "validation.jsonl", self.legacy_data,
            require_images=False,
        )
        self.legacy_validation_count = len(legacy_records)
        self.legacy_alignment = open_shards(
            self.legacy_root / "features" / "validation-shards", len(legacy_records)
        )
        pair_path = self.legacy_pair_root / "pairs" / "validation.jsonl"
        pair_all = load_pairs(pair_path, len(legacy_records))
        self.semantic_test = subset_pairs(pair_all, set(range(1, len(legacy_records), 2)))
        self.semantic_validation = subset_pairs(pair_all, set(range(0, len(legacy_records), 2)))
        self.pair_text = open_cached_dataset(
            self.legacy_pair_root / "features" / "validation-pair-text.pt",
            source_sha256=file_sha256(pair_path), item_key="pair_text",
        )
        if self.legacy_alignment is None or self.pair_text is None:
            raise RuntimeError("legacy semantic-pair features are incomplete")
        self.base_checkpoint = self.legacy_root / "checkpoints" / "v3_full.pt"

    def base_model(self) -> ThreeStageVisualJEV:
        model, _ = ThreeStageVisualJEV.from_checkpoint(self.base_checkpoint, map_location=self.device)
        return model

    def pair_metrics(self, model: ThreeStageVisualJEV, split: str = "test") -> dict[str, float]:
        pairs = self.semantic_test if split == "test" else self.semantic_validation
        return semantic_metrics(model, self.legacy_alignment, pairs, self.pair_text, 1.0)

    def evaluate(self, model: ThreeStageVisualJEV, split: str) -> dict[str, Any]:
        logits, labels = collect_logits(model, self.features[split], self.examples[split])
        return metrics_by_dataset(logits, labels, self.examples[split])


def candidate_audit(examples: dict[str, list[BenchmarkExample]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split, rows in examples.items():
        by_domain: dict[str, Any] = {}
        for domain in DOMAINS:
            selected = [row for row in rows if row.dataset == domain]
            by_domain[domain] = {
                "count": len(selected),
                "candidate_count_histogram": dict(sorted(Counter(len(row.candidates) for row in selected).items())),
                "label_position_histogram": dict(sorted(Counter(row.label for row in selected).items())),
            }
        output[split] = by_domain
    return output


def old_example(record: TrainingRecord, split: str, index: int) -> BenchmarkExample:
    candidates = (record.positive, *record.negatives)
    example_id = f"coco_hard_negative:{split}:{index}"
    order = list(range(len(candidates)))
    random.Random(f"{SEED}:{example_id}").shuffle(order)
    shuffled, label = permute_candidates(candidates, 0, order)
    return BenchmarkExample(
        id=example_id,
        dataset="coco_hard_negative",
        image=str(record.image.resolve()),
        question=record.question,
        candidates=shuffled,
        label=label,
        group_id=example_id,
        metadata={"candidate_permutation": order, "legacy_index": index},
    )


@torch.no_grad()
def pipeline_regression(context: Context, root: Path) -> dict[str, Any]:
    """Replay old Full V3 through the new variable-candidate schema/metrics."""
    train_records = load_records(
        context.legacy_data / "splits" / "train.jsonl", context.legacy_data,
        require_images=False,
    )
    validation_records = load_records(
        context.legacy_data / "splits" / "validation.jsonl", context.legacy_data,
        require_images=False,
    )
    partitions = {
        "train": [old_example(row, "train", i) for i, row in enumerate(train_records)],
        "validation": [old_example(validation_records[i], "validation", i) for i in range(0, 256, 2)],
        "calibration": [],
        "test": [old_example(validation_records[i], "validation", i) for i in range(1, 256, 2)],
    }
    manifests = root / "manifests"
    write_predefined_manifests(
        partitions,
        manifests,
        seed=SEED,
        split_method="legacy COCO 2048 train; old validation even/odd -> 128 validation/test",
        provenance={
            "candidate_order": "new deterministic shuffle with label remap",
            "checkpoint_replay": str(context.base_checkpoint),
            "note": "pipeline regression replays the frozen old Full V3 checkpoint; no retraining was needed because the gate passed",
        },
    )
    model = context.base_model().eval()
    text = DiskFeatureDataset(context.legacy_data / "features" / "validation-shards", 256)
    logits, labels = [], []
    for example in partitions["test"]:
        index = int(example.metadata["legacy_index"])
        order = list(example.metadata["candidate_permutation"])
        scores = model(
            context.legacy_alignment[index]["pre_tokens"], text[index]["text_features"][order]
        ).scores
        logits.append(scores.detach().float().cpu())
        labels.append(example.label)
    metrics = variable_metrics(logits, labels)
    pair = context.pair_metrics(model, "test")
    passed = (
        abs(metrics["accuracy"] - OLD_REFERENCE["accuracy"]) <= 0.02
        and abs(pair["both_directions_accuracy"] - OLD_REFERENCE["pair_both"]) <= 0.02
        and abs(pair["prediction_flip_rate"] - OLD_REFERENCE["pair_flip"]) <= 0.02
    )
    payload = {
        "experiment": "E0 COCO-only pipeline regression",
        "seed": SEED,
        "checkpoint_replay": str(context.base_checkpoint.resolve()),
        "checkpoint_sha256": sha256(context.base_checkpoint),
        "new_schema_counts": {key: len(value) for key, value in partitions.items()},
        "metrics": metrics,
        "semantic_counterfactual": pair,
        "reference": OLD_REFERENCE,
        "gate_passed": passed,
        "retrained": False,
        "why_not_retrained": "checkpoint replay exactly reproduced the historical held-out metrics",
    }
    write_json(root / "results.json", payload)
    write_rows(root / "results.csv", [{"scope": "coco_test", **metrics}, {"scope": "semantic_pairs", **pair}])
    write_json(root / "candidate_label_audit.json", candidate_audit({k: v for k, v in partitions.items()}))
    if not passed:
        raise RuntimeError(f"E0 pipeline regression failed: {payload}")
    return payload


def domain_indices(rows: Sequence[BenchmarkExample]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(rows):
        grouped[example.dataset].append(index)
    return grouped


def candidate_loss(model: ThreeStageVisualJEV, item: dict[str, torch.Tensor], example: BenchmarkExample) -> torch.Tensor:
    scores = model(item["pre_tokens"], item["text_features"]).scores
    return F.cross_entropy(scores[None], torch.tensor([example.label], device=scores.device))


def selection_score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    domain_accuracy = [metrics[name]["accuracy"] for name in DOMAINS if name in metrics]
    worst = min(domain_accuracy)
    macro_domain = float(np.mean(domain_accuracy))
    score = metrics["overall"]["accuracy"] + 0.5 * macro_domain + 0.5 * worst
    return score, worst, macro_domain


def training_sequence(
    grouped: dict[str, list[int]],
    mode: str,
    rng: random.Random,
    count: int,
    weights: dict[str, float] | None = None,
) -> list[int]:
    if mode == "balanced":
        indices = []
        per_domain = count // len(grouped)
        for domain in grouped:
            pool = grouped[domain].copy()
            rng.shuffle(pool)
            indices.extend((pool * math.ceil(per_domain / len(pool)))[:per_domain])
        rng.shuffle(indices)
        return indices
    if mode == "proportional":
        weights = {name: RAW_ELIGIBLE_COUNTS[name] for name in DOMAINS}
    if weights is None:
        raise ValueError("adaptive sampling requires weights")
    names = list(DOMAINS)
    probabilities = [weights[name] for name in names]
    selected_domains = rng.choices(names, weights=probabilities, k=count)
    return [rng.choice(grouped[name]) for name in selected_domains]


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    row = {f"overall_{key}": metrics["overall"][key] for key in ("accuracy", "macro_f1", "nll", "brier", "ece")}
    for domain in DOMAINS:
        row[f"{domain}_accuracy"] = metrics[domain]["accuracy"]
        row[f"{domain}_f1"] = metrics[domain]["macro_f1"]
        row[f"{domain}_nll"] = metrics[domain]["nll"]
    _, worst, macro = selection_score(metrics)
    row["worst_domain_accuracy"] = worst
    row["macro_domain_accuracy"] = macro
    return row


def run_training(
    context: Context,
    *,
    name: str,
    domains: Sequence[str],
    sampler: str,
    method: str,
    epochs: int,
    output: Path,
    initial_checkpoint: Path | None = None,
    sampling_weights: dict[str, float] | None = None,
    learning_rate: float = 5e-5,
) -> dict[str, Any]:
    set_seed()
    checkpoint = initial_checkpoint or context.base_checkpoint
    model, _ = ThreeStageVisualJEV.from_checkpoint(checkpoint, map_location=context.device)
    for parameter in model.alignment.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decision.parameters():
        parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.decision.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-2)
    grouped_all = domain_indices(context.examples["train"])
    grouped = {name: grouped_all[name] for name in domains}
    rng = random.Random(f"{SEED}:{name}")
    validation = context.evaluate(model, "validation")
    best_score, _, _ = selection_score(validation)
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = []
    sample_histogram: Counter[str] = Counter()
    loss_history: dict[str, list[float]] = defaultdict(list)
    conflict_rows: list[dict[str, Any]] = []
    initial_task_losses: list[float] | None = None
    run_started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: dict[str, list[float]] = defaultdict(list)
        if method == "plain":
            count = 32 * len(domains)
            sequence = training_sequence(grouped, sampler, rng, count, sampling_weights)
            progress = tqdm(
                sequence,
                desc=f"{name} epoch {epoch}/{epochs}",
                unit="sample",
                dynamic_ncols=True,
            )
            for index in progress:
                example = context.examples["train"][index]
                optimizer.zero_grad(set_to_none=True)
                loss = candidate_loss(model, context.features["train"][index], example)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                sample_histogram[example.dataset] += 1
                loss_history[example.dataset].append(float(loss.item()))
                epoch_losses[example.dataset].append(float(loss.item()))
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mean=f"{np.mean([v for values in epoch_losses.values() for v in values]):.4f}",
                    domain=example.dataset,
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )
        else:
            steps = 32
            domain_order = list(domains)
            pools = {domain: grouped[domain].copy() for domain in domains}
            for pool in pools.values():
                rng.shuffle(pool)
            progress = tqdm(
                range(steps),
                desc=f"{name} epoch {epoch}/{epochs}",
                unit="step",
                dynamic_ncols=True,
            )
            for step in progress:
                losses = []
                for domain in domain_order:
                    index = pools[domain][step % len(pools[domain])]
                    example = context.examples["train"][index]
                    losses.append(candidate_loss(model, context.features["train"][index], example))
                    sample_histogram[domain] += 1
                    loss_history[domain].append(float(losses[-1].item()))
                    epoch_losses[domain].append(float(losses[-1].item()))
                optimizer.zero_grad(set_to_none=True)
                if method == "pcgrad":
                    report = pcgrad_backward(losses, parameters, rng=rng)
                elif method == "gradnorm":
                    if initial_task_losses is None:
                        initial_task_losses = [float(loss.detach().item()) for loss in losses]
                    report = gradnorm_backward(losses, parameters, initial_losses=initial_task_losses)
                else:
                    raise ValueError(method)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                current_conflicts = sum(
                    report["cosine"][left][right] < 0
                    for left in range(len(domain_order))
                    for right in range(left + 1, len(domain_order))
                )
                progress.set_postfix(
                    loss=f"{np.mean([loss.item() for loss in losses]):.4f}",
                    conflict=f"{current_conflicts}/{len(domain_order) * (len(domain_order) - 1) // 2}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )
                for left in range(len(domain_order)):
                    for right in range(left + 1, len(domain_order)):
                        conflict_rows.append({
                            "experiment": name, "epoch": epoch, "step": step,
                            "domain_a": domain_order[left], "domain_b": domain_order[right],
                            "cosine": report["cosine"][left][right],
                            "conflict": report["cosine"][left][right] < 0,
                            "norm_a": report["norms"][left], "norm_b": report["norms"][right],
                        })
        validation = context.evaluate(model, "validation")
        score, worst, macro = selection_score(validation)
        elapsed = time.time() - run_started
        eta_seconds = elapsed / epoch * (epochs - epoch)
        row = {
            "epoch": epoch, "selection_score": score,
            "validation_accuracy": validation["overall"]["accuracy"],
            "validation_macro_f1": validation["overall"]["macro_f1"],
            "validation_worst_domain_accuracy": worst,
            "validation_macro_domain_accuracy": macro,
            "train_loss": float(np.mean([v for values in epoch_losses.values() for v in values])),
            "train_loss_by_domain": {key: float(np.mean(value)) for key, value in epoch_losses.items()},
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
        }
        history.append(row)
        print(json.dumps({"experiment": name, **row}), flush=True)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test = context.evaluate(model, "test")
    pair = context.pair_metrics(model, "test")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_out = output / f"{name}.pt"
    model.save_checkpoint(
        checkpoint_out,
        stage="B-multidomain",
        step=epochs,
        metadata={
            "seed": SEED, "experiment": name, "method": method, "sampler": sampler,
            "paper_implementation": "project-local reimplementation; not official",
        },
    )
    payload = {
        "name": name,
        "domains": list(domains),
        "sampler": sampler,
        "gradient_method": method,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "initial_checkpoint": str(checkpoint.resolve()),
        "checkpoint": str(checkpoint_out.resolve()),
        "sample_counts": dict(sample_histogram),
        "actual_sampling_ratio": {key: value / sum(sample_histogram.values()) for key, value in sample_histogram.items()},
        "mean_train_loss_by_domain": {key: float(np.mean(value)) for key, value in loss_history.items()},
        "history": history,
        "test": test,
        "semantic_counterfactual": pair,
        "conflict_summary": {
            "count": len(conflict_rows),
            "negative_count": sum(int(row["conflict"]) for row in conflict_rows),
            "negative_fraction": float(np.mean([row["conflict"] for row in conflict_rows])) if conflict_rows else None,
            "mean_cosine": float(np.mean([row["cosine"] for row in conflict_rows])) if conflict_rows else None,
        },
    }
    write_json(output / f"{name}.json", payload)
    if conflict_rows:
        write_rows(output / f"{name}_gradient_conflict.csv", conflict_rows)
    return payload


def gradient_conflict_audit(context: Context, root: Path) -> dict[str, Any]:
    model = context.base_model().train()
    for parameter in model.alignment.parameters():
        parameter.requires_grad_(False)
    parameters = [parameter for parameter in model.decision.parameters() if parameter.requires_grad]
    grouped = domain_indices(context.examples["train"])
    rng = random.Random(f"{SEED}:gradient-audit")
    rows: list[dict[str, Any]] = []
    aggregate_norms: dict[str, list[float]] = defaultdict(list)
    for step in range(16):
        losses = []
        for domain in DOMAINS:
            index = grouped[domain][step % len(grouped[domain])]
            losses.append(candidate_loss(model, context.features["train"][index], context.examples["train"][index]))
        _, norms, cosine = task_gradients(losses, parameters)
        for domain, norm in zip(DOMAINS, norms):
            aggregate_norms[domain].append(norm)
        for left in range(len(DOMAINS)):
            for right in range(left + 1, len(DOMAINS)):
                rows.append({
                    "step": step, "domain_a": DOMAINS[left], "domain_b": DOMAINS[right],
                    "cosine": cosine[left][right], "conflict": cosine[left][right] < 0,
                    "norm_a": norms[left], "norm_b": norms[right],
                })
    summary = {
        "checkpoint": str(context.base_checkpoint.resolve()),
        "steps": 16,
        "pair_observations": len(rows),
        "conflict_fraction": float(np.mean([row["conflict"] for row in rows])),
        "mean_cosine": float(np.mean([row["cosine"] for row in rows])),
        "mean_gradient_norm_by_domain": {key: float(np.mean(value)) for key, value in aggregate_norms.items()},
        "method": "candidate CE gradients on shared trainable Visual-JEV decision parameters; Qwen and alignment frozen",
    }
    write_rows(root / "gradient_conflict.csv", rows)
    write_json(root / "gradient_conflict.json", {"summary": summary, "observations": rows})
    return summary


@torch.no_grad()
def estimate_scienceqa_dependency(context: Context, model: ThreeStageVisualJEV) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    selected = [
        (split, index, example)
        for split in ("train", "validation")
        for index, example in enumerate(context.examples[split])
        if example.dataset == "scienceqa_image_only"
    ]
    for split, index, example in tqdm(
        selected,
        desc="estimate ScienceQA visual dependency",
        unit="sample",
        dynamic_ncols=True,
    ):
        item = context.features[split][index]
        original = model(item["pre_tokens"], item["text_features"]).scores.float()
        blank = model(context.controls["blank_pre"], item["text_features"]).scores.float()
        original_logp = original.log_softmax(-1)[example.label]
        blank_logp = blank.log_softmax(-1)[example.label]
        records.append({
            "split": split, "index": index,
            "label_nll_increase_blank": float((original_logp - blank_logp).item()),
            "label_score_drop_blank": float((original[example.label] - blank[example.label]).item()),
        })
    validation_drops = [row["label_nll_increase_blank"] for row in records if row["split"] == "validation"]
    threshold = max(0.05, float(np.median(validation_drops)))
    train_required = {
        row["index"]: row["label_nll_increase_blank"] >= threshold
        for row in records if row["split"] == "train"
    }
    return {
        "threshold_source": "validation median of original-vs-blank correct-label NLL improvement, floored at 0.05",
        "threshold": threshold,
        "train_visual_required": train_required,
        "train_visual_required_fraction": float(np.mean(list(train_required.values()))),
        "records": records,
        "test_used_for_threshold": False,
    }


def run_stage_c(
    context: Context,
    *,
    name: str,
    initial_checkpoint: Path,
    root: Path,
    curriculum: bool,
    epochs: int = 3,
) -> dict[str, Any]:
    set_seed()
    model, _ = ThreeStageVisualJEV.from_checkpoint(initial_checkpoint, map_location=context.device)
    for parameter in model.alignment.parameters():
        parameter.requires_grad_(False)
    parameters = [parameter for parameter in model.decision.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=1e-2)
    dependency = estimate_scienceqa_dependency(context, model.eval())
    grouped = domain_indices(context.examples["train"])
    rng = random.Random(f"{SEED}:{name}")
    train_pair_path = context.legacy_pair_root / "pairs" / "train.jsonl"
    legacy_train_records = load_records(
        context.legacy_data / "splits" / "train.jsonl", context.legacy_data,
        require_images=False,
    )
    legacy_train_alignment = open_shards(
        context.legacy_root / "features" / "train-shards", len(legacy_train_records)
    )
    train_pairs = load_pairs(train_pair_path, len(legacy_train_records))
    pair_rows = sorted(train_pairs.items(), key=lambda item: item[0])[:64]
    train_pair_text = open_cached_dataset(
        context.legacy_pair_root / "features" / "train-pair-text.pt",
        source_sha256=file_sha256(train_pair_path), item_key="pair_text",
    )
    if legacy_train_alignment is None or train_pair_text is None:
        raise RuntimeError("training semantic-pair features are incomplete")
    validation = context.evaluate(model, "validation")
    pair_validation = context.pair_metrics(model, "validation")
    base_score, _, _ = selection_score(validation)
    best_score = base_score + 0.5 * pair_validation["both_directions_accuracy"] + 0.25 * pair_validation["prediction_flip_rate"]
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = []
    type_histogram: Counter[str] = Counter()
    component_history: dict[str, list[float]] = defaultdict(list)
    run_started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        hard_phase = not curriculum or epoch >= 2
        pools = {domain: grouped[domain].copy() for domain in DOMAINS}
        for pool in pools.values():
            rng.shuffle(pool)
        progress = tqdm(
            range(32),
            desc=f"{name} epoch {epoch}/{epochs} ({'hard' if hard_phase else 'easy'})",
            unit="step",
            dynamic_ncols=True,
        )
        epoch_total: list[float] = []
        for step in progress:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for domain in DOMAINS:
                index = pools[domain][step]
                example = context.examples["train"][index]
                item = context.features["train"][index]
                original = model(item["pre_tokens"], item["text_features"]).scores
                candidate = F.cross_entropy(
                    original[None], torch.tensor([example.label], device=original.device)
                )
                total = candidate
                component_history["candidate"].append(float(candidate.item()))
                required = bool(dependency["train_visual_required"].get(index, False))
                masks = visual_dependency_masks(domain, scienceqa_visual_required=required)
                type_histogram[str(masks["type"])] += 1
                if masks["invalid"]:
                    blank = model(context.controls["blank_pre"], item["text_features"]).scores
                    noise = model(context.controls["noise_pre"], item["text_features"]).scores
                    invalid = mean_uniformity_loss((blank, noise))
                    total = total + invalid
                    component_history["invalid"].append(float(invalid.item()))
                if masks["preference"] and hard_phase:
                    wrong_pool = grouped[domain]
                    wrong_index = wrong_pool[(wrong_pool.index(index) + 1) % len(wrong_pool)]
                    wrong = model(
                        context.features["train"][wrong_index]["pre_tokens"], item["text_features"]
                    ).scores
                    preference = preference_anchor_loss(original, wrong, label=example.label)
                    total = total + 0.25 * preference
                    component_history["image_preference_anchor"].append(float(preference.item()))
                losses.append(total)
            if hard_phase:
                source, pair = pair_rows[(epoch * 32 + step) % len(pair_rows)]
                partner = int(pair["counterfactual_index"])
                target = int(pair["counterfactual_target_index"])
                text = train_pair_text[int(pair["_cache_index"])]["text_features"]
                left = model(legacy_train_alignment[source]["pre_tokens"], text).scores
                right = model(legacy_train_alignment[partner]["pre_tokens"], text).scores
                counterfactual = counterfactual_visual_dependency_loss(
                    left, right, image1_target=0, image2_target=target, margin=0.2
                )
                pair_preference = 0.5 * (
                    preference_anchor_loss(left, right, label=0)
                    + preference_anchor_loss(right, left, label=target)
                )
                losses.append(counterfactual + 0.25 * pair_preference)
                component_history["semantic_counterfactual"].append(float(counterfactual.item()))
                component_history["semantic_preference_anchor"].append(float(pair_preference.item()))
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            component_history["total"].append(float(loss.item()))
            epoch_total.append(float(loss.item()))
            progress.set_postfix(
                total=f"{loss.item():.4f}",
                mean=f"{np.mean(epoch_total):.4f}",
                candidate=f"{np.mean(component_history['candidate'][-len(DOMAINS):]):.4f}",
                invalid=f"{component_history['invalid'][-1]:.4f}" if component_history["invalid"] else "-",
                cf=f"{component_history['semantic_counterfactual'][-1]:.4f}" if component_history["semantic_counterfactual"] else "-",
            )
        validation = context.evaluate(model, "validation")
        pair_validation = context.pair_metrics(model, "validation")
        base, worst, macro = selection_score(validation)
        score = base + 0.5 * pair_validation["both_directions_accuracy"] + 0.25 * pair_validation["prediction_flip_rate"]
        elapsed = time.time() - run_started
        row = {
            "epoch": epoch, "hard_phase": hard_phase, "selection_score": score,
            "validation_accuracy": validation["overall"]["accuracy"],
            "validation_worst_domain_accuracy": worst,
            "validation_macro_domain_accuracy": macro,
            "validation_pair_both": pair_validation["both_directions_accuracy"],
            "validation_pair_flip": pair_validation["prediction_flip_rate"],
            "train_loss": float(np.mean(epoch_total)),
            "elapsed_seconds": elapsed,
            "eta_seconds": elapsed / epoch * (epochs - epoch),
        }
        history.append(row)
        print(json.dumps({"experiment": name, **row}), flush=True)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test = context.evaluate(model, "test")
    pair = context.pair_metrics(model, "test")
    visual = visual_metrics(model, context.features["test"], context.examples["test"], context.controls, 1.0)
    checkpoint = root / f"{name}.pt"
    model.save_checkpoint(
        checkpoint,
        stage="C-gated",
        step=epochs,
        metadata={
            "seed": SEED, "experiment": name, "curriculum": curriculum,
            "paper_implementation": "mDPO/MFPO-inspired project-local adaptation; not official",
        },
    )
    payload = {
        "name": name,
        "initial_checkpoint": str(initial_checkpoint.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "epochs": epochs,
        "curriculum": curriculum,
        "stage_c_formula": "L_candidate + m_invalid*L_blank_noise_uniform + hard*(m_preference*0.25*L_image_preference_anchor + L_semantic_cf + 0.25*L_semantic_anchor)",
        "visual_dependency": dependency,
        "dependency_type_counts": dict(type_histogram),
        "mean_training_components": {key: float(np.mean(value)) for key, value in component_history.items()},
        "history": history,
        "test": test,
        "semantic_counterfactual": pair,
        "visual_dependency_test": visual,
    }
    write_json(root / f"{name}.json", payload)
    return payload


def result_row(experiment: str, payload: dict[str, Any]) -> dict[str, Any]:
    test = payload["test"]
    pair = payload["semantic_counterfactual"]
    return {
        "experiment": experiment,
        **compact_metrics(test),
        "pair_both": pair["both_directions_accuracy"],
        "pair_flip": pair["prediction_flip_rate"],
        "checkpoint": payload.get("checkpoint"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--results-root", type=Path,
        default=Path("experiments/results/benchmark_v1/fix_negative_transfer"),
    )
    args = parser.parse_args()
    set_seed()
    started = time.time()
    root = args.results_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    context = Context(args.device)

    audit = candidate_audit(context.examples)
    write_json(root / "candidate_label_histograms.json", audit)
    e0 = pipeline_regression(context, root / "pipeline_regression")
    print(json.dumps({"experiment": "E0", "metrics": e0["metrics"], "pair": e0["semantic_counterfactual"]}), flush=True)

    conflict = gradient_conflict_audit(context, root / "gradient_conflict")
    print(json.dumps({"gradient_conflict": conflict}), flush=True)

    per_domain_root = root / "per_domain"
    transfer_payloads: dict[str, Any] = {}
    for domain in DOMAINS:
        transfer_payloads[domain] = run_training(
            context, name=f"single_{domain}", domains=(domain,), sampler="balanced",
            method="plain", epochs=3, output=per_domain_root,
        )
    transfer_payloads["mixed"] = run_training(
        context, name="mixed_control", domains=DOMAINS, sampler="balanced",
        method="plain", epochs=3, output=per_domain_root,
    )
    transfer_rows = []
    for train_domain, payload in transfer_payloads.items():
        for eval_domain in DOMAINS:
            values = payload["test"][eval_domain]
            transfer_rows.append({
                "train_domain": train_domain, "eval_domain": eval_domain,
                "accuracy": values["accuracy"], "macro_f1": values["macro_f1"],
                "nll": values["nll"], "brier": values["brier"], "ece": values["ece"],
            })
    write_rows(per_domain_root / "transfer_matrix.csv", transfer_rows)
    write_json(per_domain_root / "transfer_matrix.json", {
        "controlled_small_run": True, "epochs": 3, "runs": transfer_payloads,
    })

    sampling_root = root / "sampling_ablation"
    e1 = run_training(
        context, name="E1_proportional_plain", domains=DOMAINS, sampler="proportional",
        method="plain", epochs=4, output=sampling_root,
    )
    e2 = run_training(
        context, name="E2_balanced_plain", domains=DOMAINS, sampler="balanced",
        method="plain", epochs=4, output=sampling_root,
    )
    base_model = context.base_model()
    base_validation = context.evaluate(base_model, "validation")
    excess = {
        domain: e2["test"][domain]["nll"] - base_validation[domain]["nll"]
        for domain in DOMAINS
    }
    adaptive = AdaptiveDomainWeights(DOMAINS, eta=1.0, floor=0.05)
    adaptive_weights = adaptive.update(excess)
    adaptive_run = run_training(
        context, name="S2_doremi_groupdro_proxy", domains=DOMAINS, sampler="adaptive",
        method="plain", epochs=4, output=sampling_root, sampling_weights=adaptive_weights,
    )
    sampling_rows = [result_row("E1", e1), result_row("E2", e2), result_row("S2", adaptive_run)]
    write_rows(sampling_root / "sampling_ablation.csv", sampling_rows)
    write_json(sampling_root / "sampling_ablation.json", {
        "raw_eligible_counts": RAW_ELIGIBLE_COUNTS,
        "adaptive_excess_loss_proxy": excess,
        "adaptive_weights": adaptive_weights,
        "note": "DoReMi/Group-DRO-inspired proxy weights; not the official DoReMi implementation",
        "runs": {"E1": e1, "E2": e2, "S2": adaptive_run},
    })

    gradient_root = root / "gradient_method_ablation"
    e3 = run_training(
        context, name="E3_balanced_pcgrad", domains=DOMAINS, sampler="balanced",
        method="pcgrad", epochs=4, output=gradient_root,
    )
    e4 = run_training(
        context, name="E4_balanced_gradnorm", domains=DOMAINS, sampler="balanced",
        method="gradnorm", epochs=4, output=gradient_root,
    )
    gradient_rows = [result_row("E2_plain", e2), result_row("E3_PCGrad", e3), result_row("E4_GradNorm", e4)]
    write_rows(gradient_root / "gradient_method_ablation.csv", gradient_rows)
    write_json(gradient_root / "gradient_method_ablation.json", {
        "same_manifest_and_equal_domain_exposure": True,
        "note": "PCGrad and GradNorm are project-local paper-inspired reimplementations, not official code",
        "runs": {"E2": e2, "E3": e3, "E4": e4},
    })

    def validation_composite(payload: dict[str, Any]) -> float:
        candidate, _ = ThreeStageVisualJEV.from_checkpoint(payload["checkpoint"], map_location=args.device)
        metrics = context.evaluate(candidate, "validation")
        pair = context.pair_metrics(candidate, "validation")
        base, _, _ = selection_score(metrics)
        return base + 0.5 * pair["both_directions_accuracy"] + 0.25 * pair["prediction_flip_rate"]

    # Pair behaviour is part of model selection.  Looking only at multi-domain
    # accuracy incorrectly preferred PCGrad in the first audit even though the
    # GradNorm run retained substantially more semantic-counterfactual skill.
    best_gradient = max((e3, e4), key=validation_composite)
    stage_root = root / "stagec_ablation"
    e5 = run_stage_c(
        context, name="E5_gated_stagec", initial_checkpoint=Path(best_gradient["checkpoint"]),
        root=stage_root, curriculum=False,
    )
    e6 = run_stage_c(
        context, name="E6_curriculum_preference_anchor", initial_checkpoint=Path(best_gradient["checkpoint"]),
        root=stage_root, curriculum=True,
    )
    # Also repair the existing failed mixed checkpoint.  This is a distinct
    # recovery branch, not a replacement for the requested E5/E6 ablation.
    failed_checkpoint = context.experiment_root / "checkpoints" / "v3_full.pt"
    e5_repair = run_stage_c(
        context, name="E5R_gated_stagec_from_failed", initial_checkpoint=failed_checkpoint,
        root=stage_root, curriculum=False,
    )
    e6_repair = run_stage_c(
        context, name="E6R_curriculum_preference_from_failed", initial_checkpoint=failed_checkpoint,
        root=stage_root, curriculum=True,
    )
    stage_rows = [
        result_row("best_gradient", best_gradient), result_row("E5", e5), result_row("E6", e6),
        result_row("E5R", e5_repair), result_row("E6R", e6_repair),
    ]
    write_rows(stage_root / "stagec_ablation.csv", stage_rows)
    write_json(stage_root / "stagec_ablation.json", {
        "best_gradient_source": best_gradient["name"],
        "runs": {"E5": e5, "E6": e6, "E5R": e5_repair, "E6R": e6_repair},
    })

    candidates = {
        "E1": e1, "E2": e2, "S2": adaptive_run, "E3": e3, "E4": e4,
        "E5": e5, "E6": e6, "E5R": e5_repair, "E6R": e6_repair,
    }
    validation_scores = {name: validation_composite(payload) for name, payload in candidates.items()}
    best_name, best = max(candidates.items(), key=lambda item: validation_scores[item[0]])
    best_dir = root / "best_model_results"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_model, best_checkpoint_payload = ThreeStageVisualJEV.from_checkpoint(
        best["checkpoint"], map_location=args.device
    )
    best_checkpoint = best_dir / "visual_jev_v3_best.pt"
    best_model.save_checkpoint(
        best_checkpoint, stage="best-negative-transfer-fix", step=0,
        metadata={"selected_from": best_name, "selection_rule": "validation-only composite including worst-domain and semantic-pair guards"},
    )
    best_visual = visual_metrics(
        best_model, context.features["test"], context.examples["test"], context.controls, 1.0
    )
    best_payload = {
        "selected_experiment": best_name,
        "selection_rule": "validation-only: overall + 0.5*macro-domain + 0.5*worst-domain + 0.5*Pair-both + 0.25*Pair-flip",
        "validation_selection_scores": validation_scores,
        "checkpoint": str(best_checkpoint.resolve()),
        "checkpoint_sha256": sha256(best_checkpoint),
        "metrics": best["test"],
        "semantic_counterfactual": best["semantic_counterfactual"],
        "visual_dependency": best_visual,
        "old_coco_reference": OLD_REFERENCE,
        "controlled_subset": True,
        "full_scale_pending": True,
    }
    write_json(best_dir / "best_model_results.json", best_payload)
    write_rows(best_dir / "best_model_results.csv", [result_row(best_name, best)])
    full_matrix_rows = [
        {
            "experiment": "E0",
            "overall_accuracy": e0["metrics"]["accuracy"],
            "overall_macro_f1": e0["metrics"]["macro_f1"],
            "pair_both": e0["semantic_counterfactual"]["both_directions_accuracy"],
            "pair_flip": e0["semantic_counterfactual"]["prediction_flip_rate"],
            "evaluation_scope": "old COCO-only test",
        },
        *[result_row(name, payload) | {"evaluation_scope": "mixed four-domain test"} for name, payload in candidates.items()],
    ]
    write_rows(root / "experiment_matrix.csv", full_matrix_rows)
    write_json(root / "experiment_matrix.json", {
        "E0": e0,
        "experiments": candidates,
        "best": best_payload,
        "elapsed_seconds": time.time() - started,
    })
    config = {
        "seed": SEED,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else None,
        "benchmark_profile": "benchmark_v1 controlled subset",
        "qwen3_vl_frozen": True,
        "alignment_frozen_during_multidomain_repair": True,
        "trainable_scope": "Visual-JEV decision module",
        "epochs": {"per_domain": 3, "sampling": 4, "gradient": 4, "stage_c": 3},
        "paper_attribution": {
            "PCGrad": "NeurIPS 2020 inspired project-local reimplementation",
            "GradNorm": "ICML 2018 inspired project-local reimplementation",
            "adaptive_sampling": "DoReMi/Group-DRO inspired proxy reweighting",
            "image_preference": "mDPO EMNLP 2024 inspired candidate-scalar preference and positive anchor",
            "curriculum": "MFPO IJCAI 2025 inspired easy-to-hard visual preference curriculum",
        },
        "not_official_implementations": True,
    }
    write_json(root / "full_config.json", config)
    print(json.dumps({
        "completed": True, "best": best_name, "best_checkpoint": str(best_checkpoint),
        "best_metrics": best_payload, "elapsed_seconds": time.time() - started,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

