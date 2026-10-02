"""Independent Visual-JEV adaptation on four visual-choice datasets.

This runner is intentionally not a mixed-data experiment. It creates one
fresh, identically initialised Visual-JEV model per dataset, keeps Qwen3-VL
frozen through cached features, fits calibration on that dataset's calibration
partition only, and writes one self-contained result directory per dataset.

The default executable profile is a fixed subset suitable for a 16 GB GPU:
32 training examples and 16 validation/calibration/test examples per dataset.
Full isolated manifests are also emitted, but full-scale training is never
claimed unless train-cap 0 is used with complete feature caches.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from jev.benchmark_v1 import BenchmarkExample, read_jsonl
from jev.calibration import TemperatureScaler
from jev.multidomain import preference_anchor_loss
from run_benchmark_v1_experiment import _js
from run_fix_negative_transfer import Context, SEED, pipeline_regression, set_seed
from scripts.run_visual_jev_v2_experiment import DiskFeatureDataset, file_sha256, open_cached_dataset
from train_visual_jev_v2 import load_records
from train_visual_jev_v3 import alignment_metrics, load_pairs, open_shards
from visual_jev_v3 import counterfactual_visual_dependency_loss, mean_uniformity_loss
from visual_jev_v3_pipeline import ThreeStageVisualJEV, alignment_loss


DATASETS = (
    "coco_hard_negative",
    "aokvqa",
    "scienceqa_image_only",
    "iconqa_choice",
)
DISPLAY_NAMES = {
    "coco_hard_negative": "COCO hard-negative",
    "aokvqa": "A-OKVQA",
    "scienceqa_image_only": "ScienceQA visual-only",
    "iconqa_choice": "IconQA choice",
}
OUTPUT_NAMES = {
    "coco_hard_negative": "coco",
    "aokvqa": "aokvqa",
    "scienceqa_image_only": "scienceqa",
    "iconqa_choice": "iconqa",
}
SPLITS = ("train", "validation", "calibration", "test")
DEFAULT_RESULT_ROOT = Path("experiments/results/independent_datasets")
DEFAULT_DATA_ROOT = Path("data/independent_datasets")


def passes_regression_floor(observed: float, target: float, tolerance: float) -> bool:
    """Accept a metric when it has not regressed beyond a one-sided tolerance."""
    return observed >= target - tolerance


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _row(example: BenchmarkExample) -> dict[str, Any]:
    metadata = dict(example.metadata)
    metadata["group_id"] = example.group_id or example.id
    return {
        "uid": example.id,
        "dataset": example.dataset,
        "image": example.image,
        "question": example.question,
        "candidates": list(example.candidates),
        "label": example.label,
        "source_split": metadata.get("source_split"),
        "metadata": metadata,
    }


def _example(payload: Mapping[str, Any]) -> BenchmarkExample:
    metadata = dict(payload.get("metadata", {}))
    return BenchmarkExample(
        id=str(payload["uid"]),
        dataset=str(payload["dataset"]),
        image=str(payload["image"]),
        question=str(payload["question"]),
        candidates=tuple(str(value) for value in payload["candidates"]),
        label=int(payload["label"]),
        group_id=str(metadata.get("group_id") or payload["uid"]),
        metadata={**metadata, "source_split": payload.get("source_split")},
    )


def _write_schema_manifest(path: Path, rows: Sequence[BenchmarkExample]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in rows:
            encoded = (json.dumps(_row(example), ensure_ascii=False, sort_keys=True) + "\n").encode()
            handle.write(encoded.decode())
            digest.update(encoded)
    return {"path": path.name, "count": len(rows), "sha256": digest.hexdigest()}


def _audit_partitions(partitions: Mapping[str, Sequence[BenchmarkExample]]) -> dict[str, Any]:
    audit: dict[str, Any] = {"splits": {}, "total": 0}
    seen_uids: set[str] = set()
    split_groups: dict[str, str] = {}
    for split in SPLITS:
        rows = list(partitions.get(split, ()))
        missing = sum(not Path(example.image).is_file() for example in rows)
        candidate_hist = Counter(len(example.candidates) for example in rows)
        label_hist = Counter(example.label for example in rows)
        invalid_candidate_rows = 0
        empty_candidate_count = 0
        permutation_failures = 0
        for example in rows:
            if example.id in seen_uids:
                raise ValueError(f"duplicate uid across splits: {example.id}")
            seen_uids.add(example.id)
            group = example.group_id or example.id
            prior = split_groups.setdefault(group, split)
            if prior != split:
                raise ValueError(f"group leakage for {group}: {prior} vs {split}")
            order = list(example.metadata.get("candidate_permutation", ()))
            if order and sorted(order) != list(range(len(example.candidates))):
                permutation_failures += 1
            if order and not 0 <= example.label < len(order):
                permutation_failures += 1
            empty_candidate_count += sum(not str(value).strip() for value in example.candidates)
            if len(example.candidates) < 2 or not 0 <= example.label < len(example.candidates):
                invalid_candidate_rows += 1
        audit["splits"][split] = {
            "count": len(rows),
            "candidate_count_histogram": dict(sorted(candidate_hist.items())),
            "label_position_histogram": dict(sorted(label_hist.items())),
            "image_missing_count": missing,
            "invalid_candidate_row_count": invalid_candidate_rows,
            "empty_candidate_count": empty_candidate_count,
            "candidate_permutation_failures": permutation_failures,
            "variable_k": len(candidate_hist) > 1,
            "variable_k_policy": "native unpadded per-example scoring; no padded candidate can enter softmax",
        }
        audit["total"] += len(rows)
    audit["uid_overlap_count"] = audit["total"] - len(seen_uids)
    audit["group_split_leakage_count"] = 0
    return audit


def isolate_existing_manifests(
    source_root: Path,
    output_root: Path,
    *,
    profile: str,
) -> dict[str, Any]:
    source = {split: list(read_jsonl(source_root / f"{split}.jsonl")) for split in SPLITS}
    report: dict[str, Any] = {"profile": profile, "source": str(source_root.resolve()), "datasets": {}}
    for dataset in DATASETS:
        partitions = {
            split: [example for example in source[split] if example.dataset == dataset]
            for split in SPLITS
        }
        directory = output_root / profile / dataset
        files = {
            split: _write_schema_manifest(directory / f"{split}.jsonl", partitions[split])
            for split in SPLITS
        }
        audit = _audit_partitions(partitions)
        manifest = {
            "schema": "visual-jev-independent-v1",
            "seed": SEED,
            "dataset": dataset,
            "mixed_training": False,
            "candidate_permutation": "inherited deterministic seed=20260928 permutation with remapped label",
            "split_policy": (
                "official/source train retained; labelled held-out source partitioned "
                "deterministically without using test for calibration"
            ),
            "files": files,
            "audit": audit,
        }
        write_json(directory / "manifest.json", manifest)
        report["datasets"][dataset] = manifest
    return report


class IndexedFeatures(Sequence[dict[str, torch.Tensor]]):
    def __init__(self, source: Sequence[dict[str, torch.Tensor]], indices: Sequence[int]):
        self.source = source
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.source[self.indices[index]]


class DatasetBundle:
    def __init__(
        self,
        dataset: str,
        examples: Mapping[str, Sequence[BenchmarkExample]],
        features: Mapping[str, Sequence[dict[str, torch.Tensor]]],
        profile: str,
    ):
        self.dataset = dataset
        self.examples = {split: list(examples[split]) for split in SPLITS}
        self.features = dict(features)
        self.profile = profile


def _open_complete_feature_cache(
    examples: Mapping[str, Sequence[BenchmarkExample]],
    manifest_root: Path,
    experiment_root: Path,
) -> dict[str, DiskFeatureDataset]:
    output: dict[str, DiskFeatureDataset] = {}
    problems: list[str] = []
    for split in SPLITS:
        directory = experiment_root / "features" / f"{split}-shards"
        manifest = directory / "manifest.json"
        count = len(examples[split])
        expected_sha = file_sha256(manifest_root / f"{split}.jsonl")
        if not manifest.is_file():
            problems.append(f"{split}: missing manifest")
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("source_sha256") != expected_sha or int(payload.get("count", -1)) != count:
            problems.append(f"{split}: stale manifest")
            continue
        missing = next((index for index in range(count) if not (directory / f"{index:06d}.pt").is_file()), None)
        if missing is not None:
            problems.append(f"{split}: first missing shard {missing}/{count}")
            continue
        output[split] = DiskFeatureDataset(directory, count)
    if problems:
        raise RuntimeError(
            "full frozen-backbone feature cache is incomplete: " + "; ".join(problems)
        )
    return output


def build_executed_bundles(
    context: Context,
    *,
    full_manifest_root: Path,
    full_experiment_root: Path,
    train_cap: int,
    manifest_root: Path,
) -> tuple[dict[str, DatasetBundle], dict[str, Any]]:
    bundles: dict[str, DatasetBundle] = {}
    if train_cap <= 0:
        profile = "full_independent"
        source_examples = {
            split: list(read_jsonl(full_manifest_root / f"{split}.jsonl"))
            for split in SPLITS
        }
        source_features = _open_complete_feature_cache(
            source_examples, full_manifest_root, full_experiment_root
        )
    else:
        profile = "fixed_controlled_subset"
        source_examples = context.examples
        source_features = context.features
    report: dict[str, Any] = {
        "profile": profile,
        "train_cap_per_dataset": train_cap,
        "datasets": {},
    }
    for dataset in DATASETS:
        candidates = [
            index for index, example in enumerate(source_examples["train"])
            if example.dataset == dataset
        ]
        candidates.sort(
            key=lambda index: hashlib.sha256(
                f"{SEED}\0executed\0{source_examples['train'][index].id}".encode()
            ).digest()
        )
        train_indices = candidates if train_cap <= 0 else candidates[:train_cap]
        examples: dict[str, list[BenchmarkExample]] = {
            "train": [source_examples["train"][index] for index in train_indices]
        }
        features: dict[str, Sequence[dict[str, torch.Tensor]]] = {
            "train": IndexedFeatures(source_features["train"], train_indices)
        }
        for split in ("validation", "calibration", "test"):
            indices = [
                index for index, example in enumerate(source_examples[split])
                if example.dataset == dataset
            ]
            examples[split] = [source_examples[split][index] for index in indices]
            features[split] = IndexedFeatures(source_features[split], indices)
        partitions = {split: examples[split] for split in SPLITS}
        directory = manifest_root / ("full" if train_cap <= 0 else "executed_subset") / dataset
        files = {
            split: _write_schema_manifest(directory / f"{split}.jsonl", partitions[split])
            for split in SPLITS
        }
        audit = _audit_partitions(partitions)
        manifest = {
            "schema": "visual-jev-independent-v1",
            "seed": SEED,
            "dataset": dataset,
            "profile": profile,
            "paper_full_scale": train_cap <= 0,
            "mixed_training": False,
            "files": files,
            "audit": audit,
        }
        write_json(directory / "manifest.json", manifest)
        report["datasets"][dataset] = manifest
        bundles[dataset] = DatasetBundle(dataset, examples, features, profile)
    return bundles, report


def fresh_model(reference_checkpoint: Path, first_item: Mapping[str, torch.Tensor], device: str) -> ThreeStageVisualJEV:
    set_seed(SEED)
    reference, _ = ThreeStageVisualJEV.from_checkpoint(reference_checkpoint, map_location="cpu")
    if reference.alignment.config.pre_merger_dim != first_item["pre_tokens"].shape[-1]:
        raise ValueError("cached pre-merger width differs from the reference architecture")
    if reference.alignment.config.teacher_dim != first_item["teacher_tokens"].shape[-1]:
        raise ValueError("cached teacher width differs from the reference architecture")
    model = ThreeStageVisualJEV(reference.alignment.config, reference.decision.config)
    del reference
    return model.to(device)


def _macro_f1(predictions: Sequence[int], labels: Sequence[int]) -> float:
    present_classes = sorted(set(labels) | set(predictions))
    values = []
    for class_index in present_classes:
        tp = sum(p == class_index and y == class_index for p, y in zip(predictions, labels))
        fp = sum(p == class_index and y != class_index for p, y in zip(predictions, labels))
        fn = sum(p != class_index and y == class_index for p, y in zip(predictions, labels))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(values))


def extended_metrics(logits: Sequence[torch.Tensor], labels: Sequence[int]) -> dict[str, float]:
    if not logits:
        raise ValueError("metrics require at least one row")
    probabilities = [row.detach().float().cpu().softmax(-1) for row in logits]
    predictions = [int(row.argmax()) for row in probabilities]
    accuracy = float(np.mean([prediction == label for prediction, label in zip(predictions, labels)]))
    nll = float(np.mean([
        -math.log(max(float(probability[label]), 1e-12))
        for probability, label in zip(probabilities, labels)
    ]))
    brier = float(np.mean([
        float(((probability - F.one_hot(torch.tensor(label), probability.numel()).float()) ** 2).sum())
        for probability, label in zip(probabilities, labels)
    ]))
    confidence = np.asarray([float(row.max()) for row in probabilities])
    correct = np.asarray([float(p == y) for p, y in zip(predictions, labels)])
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.mean() * abs(confidence[mask].mean() - correct[mask].mean()))
    entropy = [float(-(row * row.clamp_min(1e-12).log()).sum()) for row in probabilities]
    normalised_entropy = [
        value / math.log(row.numel()) for value, row in zip(entropy, probabilities)
    ]
    margins = []
    for row, label in zip(logits, labels):
        other = torch.cat((row[:label], row[label + 1 :])).max()
        margins.append(float((row[label] - other).detach().cpu()))
    return {
        "accuracy": accuracy,
        "macro_f1": _macro_f1(predictions, labels),
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "mean_correct_probability": float(np.mean([
            float(row[label]) for row, label in zip(probabilities, labels)
        ])),
        "mean_margin": float(np.mean(margins)),
        "entropy": float(np.mean(entropy)),
        "normalized_entropy": float(np.mean(normalised_entropy)),
        "count": len(labels),
    }


@torch.no_grad()
def collect_logits(
    model: ThreeStageVisualJEV,
    bundle: DatasetBundle,
    split: str,
) -> tuple[list[torch.Tensor], list[int]]:
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[int] = []
    for index, example in enumerate(bundle.examples[split]):
        item = bundle.features[split][index]
        logits.append(model(item["pre_tokens"], item["text_features"]).scores.detach().float().cpu())
        labels.append(example.label)
    return logits, labels


def failure_case_summary(
    logits: Sequence[torch.Tensor],
    bundle: DatasetBundle,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for row, example in zip(logits, bundle.examples["test"]):
        probability = row.float().softmax(-1)
        prediction = int(probability.argmax())
        if prediction == example.label:
            continue
        failures.append({
            "uid": example.id,
            "question": example.question,
            "candidates": list(example.candidates),
            "label": example.label,
            "prediction": prediction,
            "predicted_probability": float(probability[prediction]),
            "correct_probability": float(probability[example.label]),
            "source_split": example.metadata.get("source_split"),
        })
    failures.sort(key=lambda row: row["predicted_probability"], reverse=True)
    return {
        "test_count": len(bundle.examples["test"]),
        "error_count": len(failures),
        "error_rate": len(failures) / max(1, len(bundle.examples["test"])),
        "high_confidence_examples": failures[:limit],
    }


def train_stage_a(model: ThreeStageVisualJEV, bundle: DatasetBundle, *, epochs: int) -> list[dict[str, Any]]:
    for parameter in model.decision.parameters():
        parameter.requires_grad_(False)
    for parameter in model.alignment.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.alignment.parameters(), lr=5e-4, weight_decay=1e-2)
    rng = random.Random(f"{SEED}:{bundle.dataset}:A")
    best_state = copy.deepcopy(model.alignment.state_dict())
    best_loss = float("inf")
    history = []
    validation_indices = list(range(len(bundle.examples["validation"])))
    for epoch in range(1, epochs + 1):
        order = list(range(len(bundle.examples["train"])))
        rng.shuffle(order)
        values = []
        model.alignment.train()
        for index in tqdm(order, desc=f"{bundle.dataset} Stage A {epoch}/{epochs}", leave=False):
            item = bundle.features["train"][index]
            optimizer.zero_grad(set_to_none=True)
            output = alignment_loss(model.alignment(item["pre_tokens"]), item["teacher_tokens"])
            output.total.backward()
            torch.nn.utils.clip_grad_norm_(model.alignment.parameters(), 1.0)
            optimizer.step()
            values.append(float(output.total.detach()))
        validation = alignment_metrics(model, bundle.features["validation"], validation_indices)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(values)),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_state = copy.deepcopy(model.alignment.state_dict())
    model.alignment.load_state_dict(best_state)
    return history


def train_stage_b(model: ThreeStageVisualJEV, bundle: DatasetBundle, *, epochs: int) -> list[dict[str, Any]]:
    for parameter in model.alignment.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decision.parameters():
        parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.decision.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-2)
    rng = random.Random(f"{SEED}:{bundle.dataset}:B")
    best_state = copy.deepcopy(model.state_dict())
    best_key = (-float("inf"), -float("inf"))
    history = []
    for epoch in range(1, epochs + 1):
        order = list(range(len(bundle.examples["train"])))
        rng.shuffle(order)
        values = []
        model.train()
        for index in tqdm(order, desc=f"{bundle.dataset} Stage B {epoch}/{epochs}", leave=False):
            item = bundle.features["train"][index]
            example = bundle.examples["train"][index]
            optimizer.zero_grad(set_to_none=True)
            scores = model(item["pre_tokens"], item["text_features"]).scores
            loss = F.cross_entropy(scores[None], torch.tensor([example.label], device=scores.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            values.append(float(loss.detach()))
        logits, labels = collect_logits(model, bundle, "validation")
        metrics = extended_metrics(logits, labels)
        row = {"epoch": epoch, "train_loss": float(np.mean(values)), **{f"validation_{k}": v for k, v in metrics.items()}}
        history.append(row)
        key = (metrics["accuracy"], -metrics["nll"])
        if key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def estimate_visual_required(
    model: ThreeStageVisualJEV,
    bundle: DatasetBundle,
    blank_pre: torch.Tensor,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        for index, example in enumerate(bundle.examples[split]):
            item = bundle.features[split][index]
            original = model(item["pre_tokens"], item["text_features"]).scores.float()
            blank = model(blank_pre, item["text_features"]).scores.float()
            drop = float((original.log_softmax(-1)[example.label] - blank.log_softmax(-1)[example.label]).cpu())
            score_drop = float((original[example.label] - blank[example.label]).cpu())
            records.append({
                "uid": example.id,
                "split": split,
                "nll_improvement_original_vs_blank": drop,
                "correct_score_drop": score_drop,
            })
    validation = [
        row["nll_improvement_original_vs_blank"]
        for row in records if row["split"] == "validation"
    ]
    floor = 0.05 if bundle.dataset == "scienceqa_image_only" else 0.02
    threshold = max(floor, float(np.median(validation)))
    if bundle.dataset in {"coco_hard_negative", "iconqa_choice"}:
        required = {row["uid"]: True for row in records if row["split"] == "train"}
        rule = "all training examples; dataset is explicitly visual"
    else:
        required = {
            row["uid"]: row["nll_improvement_original_vs_blank"] >= threshold
            for row in records if row["split"] == "train"
        }
        rule = "train/validation original-vs-blank correct-label NLL improvement threshold"
    return {
        "threshold": threshold,
        "threshold_rule": rule,
        "threshold_uses_test": False,
        "train_visual_required": required,
        "train_visual_required_fraction": float(np.mean(list(required.values()))),
        "records": records,
    }


def _coco_pair_training(context: Context) -> tuple[Any, list[tuple[int, dict[str, Any]]], Any]:
    pair_path = context.legacy_pair_root / "pairs" / "train.jsonl"
    records = load_records(
        context.legacy_data / "splits" / "train.jsonl",
        context.legacy_data,
        require_images=False,
    )
    alignment = open_shards(context.legacy_root / "features" / "train-shards", len(records))
    pairs = sorted(load_pairs(pair_path, len(records)).items(), key=lambda value: value[0])
    pair_text = open_cached_dataset(
        context.legacy_pair_root / "features" / "train-pair-text.pt",
        source_sha256=file_sha256(pair_path),
        item_key="pair_text",
    )
    if alignment is None or pair_text is None or not pairs:
        raise RuntimeError("COCO semantic-pair training features are incomplete")
    return alignment, pairs, pair_text


def train_stage_c(
    model: ThreeStageVisualJEV,
    bundle: DatasetBundle,
    context: Context,
    dependency: Mapping[str, Any],
    *,
    epochs: int,
) -> list[dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=1e-2)
    rng = random.Random(f"{SEED}:{bundle.dataset}:C")
    required = dependency["train_visual_required"]
    pair_alignment = pair_rows = pair_text = None
    if bundle.dataset == "coco_hard_negative":
        pair_alignment, pair_rows, pair_text = _coco_pair_training(context)
    best_state = copy.deepcopy(model.state_dict())
    best_key = (-float("inf"), -float("inf"))
    history = []
    for epoch in range(1, epochs + 1):
        hard_phase = bundle.dataset == "coco_hard_negative" and epoch >= 2
        order = list(range(len(bundle.examples["train"])))
        rng.shuffle(order)
        totals: list[float] = []
        component: dict[str, list[float]] = {"candidate": [], "invalid": [], "semantic_pair": [], "pair_anchor": []}
        model.train()
        for position, index in enumerate(tqdm(order, desc=f"{bundle.dataset} Stage C {epoch}/{epochs}", leave=False)):
            item = bundle.features["train"][index]
            example = bundle.examples["train"][index]
            optimizer.zero_grad(set_to_none=True)
            original = model(item["pre_tokens"], item["text_features"]).scores
            candidate = F.cross_entropy(original[None], torch.tensor([example.label], device=original.device))
            total = candidate
            component["candidate"].append(float(candidate.detach()))
            if required.get(example.id, False):
                blank = model(context.controls["blank_pre"], item["text_features"]).scores
                noise = model(context.controls["noise_pre"], item["text_features"]).scores
                invalid = mean_uniformity_loss((blank, noise))
                total = total + 0.5 * invalid
                component["invalid"].append(float(invalid.detach()))
            if hard_phase and pair_rows is not None and pair_alignment is not None and pair_text is not None:
                source, pair = pair_rows[(epoch * len(order) + position) % len(pair_rows)]
                partner = int(pair["counterfactual_index"])
                target = int(pair["counterfactual_target_index"])
                text = pair_text[int(pair["_cache_index"])]["text_features"]
                left = model(pair_alignment[source]["pre_tokens"], text).scores
                right = model(pair_alignment[partner]["pre_tokens"], text).scores
                semantic = counterfactual_visual_dependency_loss(
                    left, right, image1_target=0, image2_target=target, margin=0.2
                )
                anchor = 0.5 * (
                    preference_anchor_loss(left, right, label=0)
                    + preference_anchor_loss(right, left, label=target)
                )
                total = total + semantic + 0.25 * anchor
                component["semantic_pair"].append(float(semantic.detach()))
                component["pair_anchor"].append(float(anchor.detach()))
            total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            totals.append(float(total.detach()))
        logits, labels = collect_logits(model, bundle, "validation")
        metrics = extended_metrics(logits, labels)
        pair_validation = (
            context.pair_metrics(model, "validation")
            if bundle.dataset == "coco_hard_negative"
            else None
        )
        row = {
            "epoch": epoch,
            "hard_semantic_pair_phase": hard_phase,
            "train_total_loss": float(np.mean(totals)),
            "train_components": {
                name: (float(np.mean(values)) if values else None)
                for name, values in component.items()
            },
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        if pair_validation is not None:
            row["validation_pair_both"] = pair_validation["both_directions_accuracy"]
            row["validation_pair_flip"] = pair_validation["prediction_flip_rate"]
        history.append(row)
        if pair_validation is None:
            key = (metrics["accuracy"], -metrics["nll"])
        else:
            key = (
                metrics["accuracy"]
                + 0.5 * pair_validation["both_directions_accuracy"]
                + 0.25 * pair_validation["prediction_flip_rate"],
                -metrics["nll"],
            )
        if key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def visual_dependency_report(
    model: ThreeStageVisualJEV,
    bundle: DatasetBundle,
    context: Context,
    dependency: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for index, example in enumerate(bundle.examples["test"]):
        item = bundle.features["test"][index]
        wrong_item = bundle.features["test"][(index + 1) % len(bundle.examples["test"])]
        logits = {
            "original": model(item["pre_tokens"], item["text_features"]).scores.detach().float().cpu(),
            "blank": model(context.controls["blank_pre"], item["text_features"]).scores.detach().float().cpu(),
            "noise": model(context.controls["noise_pre"], item["text_features"]).scores.detach().float().cpu(),
            "wrong": model(wrong_item["pre_tokens"], item["text_features"]).scores.detach().float().cpu(),
        }
        probabilities = {name: value.softmax(-1) for name, value in logits.items()}
        original_logp = logits["original"].log_softmax(-1)[example.label]
        blank_logp = logits["blank"].log_softmax(-1)[example.label]
        row = {
            "uid": example.id,
            "label": example.label,
            "visual_required_by_trainval_rule": float(original_logp - blank_logp) >= dependency["threshold"],
        }
        for name in ("original", "blank", "noise", "wrong"):
            probability = probabilities[name]
            entropy = float(-(probability * probability.clamp_min(1e-12).log()).sum())
            row[f"{name}_correct_probability"] = float(probability[example.label])
            row[f"{name}_kl_uniform"] = math.log(probability.numel()) - entropy
            row[f"{name}_prediction"] = int(probability.argmax())
            row[f"{name}_js_from_original"] = _js(probabilities["original"], probability)
        rows.append(row)

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"count": 0}
        result: dict[str, Any] = {
            "count": len(selected),
            "original": {
                "accuracy": float(np.mean([
                    row["original_prediction"] == row["label"] for row in selected
                ])),
                "mean_correct_probability": float(np.mean([
                    row["original_correct_probability"] for row in selected
                ])),
                "kl_uniform": float(np.mean([row["original_kl_uniform"] for row in selected])),
            },
        }
        for condition in ("blank", "noise", "wrong"):
            result[condition] = {
                "accuracy": float(np.mean([
                    row[f"{condition}_prediction"] == row["label"] for row in selected
                ])),
                "mean_correct_probability": float(np.mean([
                    row[f"{condition}_correct_probability"] for row in selected
                ])),
                "confidence_drop": float(np.mean([
                    row["original_correct_probability"] - row[f"{condition}_correct_probability"]
                    for row in selected
                ])),
                "flip_rate": float(np.mean([
                    row["original_prediction"] != row[f"{condition}_prediction"]
                    for row in selected
                ])),
                "kl_uniform": float(np.mean([row[f"{condition}_kl_uniform"] for row in selected])),
                "js_from_original": float(np.mean([row[f"{condition}_js_from_original"] for row in selected])),
            }
        return result

    report: dict[str, Any] = {
        "dataset": bundle.dataset,
        "overall": aggregate(rows),
        "records": rows,
        "test_used_for_threshold_fit": False,
    }
    if bundle.dataset == "scienceqa_image_only":
        report["visual_required_subset"] = aggregate([
            row for row in rows if row["visual_required_by_trainval_rule"]
        ])
    if bundle.dataset == "coco_hard_negative":
        report["semantic_counterfactual"] = context.pair_metrics(model, "test")
        report["swap"] = report["semantic_counterfactual"]
    else:
        report["semantic_counterfactual"] = {
            "reported": False,
            "reason": "no verified semantic image pair in this dataset manifest",
        }
    return report


def calibrate(model: ThreeStageVisualJEV, bundle: DatasetBundle) -> dict[str, Any]:
    calibration_logits, calibration_labels = collect_logits(model, bundle, "calibration")
    test_logits, test_labels = collect_logits(model, bundle, "test")
    before = extended_metrics(test_logits, test_labels)
    scaler = TemperatureScaler()
    fit = scaler.fit(calibration_logits, calibration_labels)
    transformed = [row / fit.temperature for row in test_logits]
    after = extended_metrics(transformed, test_labels)
    before_argmax = [int(row.argmax()) for row in test_logits]
    after_argmax = [int(row.argmax()) for row in transformed]
    if before_argmax != after_argmax:
        raise AssertionError("global temperature scaling changed argmax")
    if before["accuracy"] != after["accuracy"] or before["macro_f1"] != after["macro_f1"]:
        raise AssertionError("global temperature scaling changed Accuracy/Macro-F1")
    return {
        "calibration_partition": "this dataset only",
        "test_used_for_fit": False,
        "methods": {"none": before, "global_temperature_scaling": after},
        "temperature_fit": asdict(fit),
        "argmax_invariant": True,
        "accuracy_invariant": True,
        "macro_f1_invariant": True,
    }


def gpu_info() -> dict[str, Any]:
    result = {
        "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        name, memory = [value.strip() for value in text.rsplit(",", 1)]
        result.update({"name": name, "memory_mib": int(memory)})
    except Exception:
        pass
    return result


def smoke_test(
    bundles: Mapping[str, DatasetBundle],
    reference_checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    rows = []
    initial_hashes = []
    for dataset in DATASETS:
        bundle = bundles[dataset]
        model = fresh_model(reference_checkpoint, bundle.features["train"][0], device)
        initial_hashes.append(_state_sha256(model))
        example = bundle.examples["train"][0]
        item = bundle.features["train"][0]
        scores = model(item["pre_tokens"], item["text_features"]).scores
        loss = F.cross_entropy(scores[None], torch.tensor([example.label], device=scores.device))
        if scores.numel() != len(example.candidates) or not torch.isfinite(loss):
            raise AssertionError(f"smoke failure for {dataset}")
        model.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        finite_gradients = bool(gradients) and all(torch.isfinite(value).all() for value in gradients)
        if not finite_gradients:
            raise AssertionError(f"backward smoke failure for {dataset}")

        representative: dict[int, int] = {}
        for index, candidate_example in enumerate(bundle.examples["train"]):
            representative.setdefault(len(candidate_example.candidates), index)
        native_rows = []
        native_predictions = []
        for index in representative.values():
            candidate_item = bundle.features["train"][index]
            candidate_scores = model(
                candidate_item["pre_tokens"], candidate_item["text_features"]
            ).scores.detach()
            native_rows.append(candidate_scores)
            native_predictions.append(int(candidate_scores.argmax()))
        width = max(value.numel() for value in native_rows)
        padded = torch.full(
            (len(native_rows), width), -torch.inf, device=native_rows[0].device
        )
        mask = torch.zeros_like(padded, dtype=torch.bool)
        for row_index, value in enumerate(native_rows):
            padded[row_index, : value.numel()] = value
            mask[row_index, : value.numel()] = True
        masked_predictions = padded.masked_fill(~mask, -torch.inf).argmax(-1).tolist()
        padding_mask_invariant = masked_predictions == native_predictions
        if not padding_mask_invariant:
            raise AssertionError(f"variable-K padding mask failure for {dataset}")
        rows.append({
            "dataset": dataset,
            "candidate_count": len(example.candidates),
            "label": example.label,
            "score_count": scores.numel(),
            "finite_loss": bool(torch.isfinite(loss)),
            "finite_gradients": finite_gradients,
            "one_batch_forward_backward": True,
            "candidate_counts_checked": sorted(representative),
            "padding_mask_invariant": padding_mask_invariant,
        })
        del model
    if len(set(initial_hashes)) != 1:
        raise AssertionError("fresh initialisation differs across datasets")
    return {
        "passed": True,
        "same_fresh_initial_state": True,
        "initial_state_sha256": initial_hashes[0],
        "rows": rows,
    }


def train_one(
    dataset: str,
    bundle: DatasetBundle,
    context: Context,
    *,
    output: Path,
    reference_checkpoint: Path,
    device: str,
    stage_a_epochs: int,
    stage_b_epochs: int,
    stage_c_epochs: int,
    smoke: Mapping[str, Any],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = fresh_model(reference_checkpoint, bundle.features["train"][0], device)
    initial_hash = _state_sha256(model)
    if initial_hash != smoke["initial_state_sha256"]:
        raise AssertionError("dataset did not start from the shared fresh initial state")
    architecture_parameters = sum(parameter.numel() for parameter in model.parameters())
    stage_a = train_stage_a(model, bundle, epochs=stage_a_epochs)
    stage_b = train_stage_b(model, bundle, epochs=stage_b_epochs)
    dependency = estimate_visual_required(model.eval(), bundle, context.controls["blank_pre"])
    stage_c = train_stage_c(model, bundle, context, dependency, epochs=stage_c_epochs)
    model.eval()
    test_logits, test_labels = collect_logits(model, bundle, "test")
    metrics = extended_metrics(test_logits, test_labels)
    failures = failure_case_summary(test_logits, bundle)
    calibration = calibrate(model, bundle)
    visual = visual_dependency_report(model, bundle, context, dependency)
    checkpoint = output / "checkpoint" / "visual_jev_v3.pt"
    model.save_checkpoint(
        checkpoint,
        stage="C-gated-independent",
        step=stage_a_epochs + stage_b_epochs + stage_c_epochs,
        metadata={
            "seed": SEED,
            "dataset": dataset,
            "mixed_training": False,
            "fresh_initial_state_sha256": initial_hash,
            "qwen3_vl_backbone_frozen": True,
            "profile": bundle.profile,
        },
    )
    runtime = time.time() - started
    config = {
        "seed": SEED,
        "dataset": dataset,
        "display_name": DISPLAY_NAMES[dataset],
        "profile": bundle.profile,
        "pending_full_scale": bundle.profile != "full_independent",
        "mixed_training": False,
        "backbone": "Qwen3-VL-4B-Instruct",
        "backbone_frozen": True,
        "feature_mode": "cached frozen-backbone features",
        "fresh_visual_jev_initialization": True,
        "fresh_initial_state_sha256": initial_hash,
        "architecture_trainable_parameters": architecture_parameters,
        "stage_epochs": {
            "A_alignment": stage_a_epochs,
            "B_candidate": stage_b_epochs,
            "C_gated": stage_c_epochs,
        },
        "stage_c": {
            "invalid_visual": "blank/noise only",
            "wrong_image_to_uniform": False,
            "semantic_pair_loss": dataset == "coco_hard_negative",
            "scienceqa_threshold_from_train_validation_only": dataset == "scienceqa_image_only",
        },
        "split_sizes": {split: len(bundle.examples[split]) for split in SPLITS},
    }
    write_json(output / "config.json", config)
    write_json(output / "training_history.json", {"stage_a": stage_a, "stage_b": stage_b, "stage_c": stage_c})
    write_json(output / "metrics.json", metrics)
    write_json(output / "calibration.json", calibration)
    write_json(output / "visual_dependency.json", visual)
    write_json(output / "visual_required_assignment.json", dependency)
    write_json(output / "failure_cases.json", failures)
    result = {
        "dataset": dataset,
        "display_name": DISPLAY_NAMES[dataset],
        "profile": bundle.profile,
        "pending_full_scale": bundle.profile != "full_independent",
        "split_sizes": config["split_sizes"],
        "trainable_parameters": architecture_parameters,
        "metrics": metrics,
        "calibration": calibration,
        "visual_dependency": visual,
        "failure_cases": failures,
        "training_time_seconds": runtime,
        "gpu": gpu_info(),
        "checkpoint": str(checkpoint.resolve()),
        "initial_state_sha256": initial_hash,
    }
    write_json(output / "result.json", result)
    return result


def write_paper_results(root: Path, results: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]) -> None:
    hardest = min(results, key=lambda row: row["metrics"]["accuracy"])
    full_scale = all(row["profile"] == "full_independent" for row in results)
    lines = [
        "# Visual-JEV independent-dataset results",
        "",
        f"Seed: {SEED}. Qwen3-VL-4B-Instruct was frozen in every run. "
        "Each dataset started from the same fresh Visual-JEV initialization and used only its own checkpoint.",
        "",
        "## Final results",
        "",
        "| Dataset | Train/Val/Cal/Test | Params | Accuracy | Macro-F1 | NLL | Brier | ECE | TS NLL | TS Brier | TS ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        raw = result["calibration"]["methods"]["none"]
        calibrated = result["calibration"]["methods"]["global_temperature_scaling"]
        sizes = result["split_sizes"]
        lines.append(
            f"| {result['display_name']} | {sizes['train']}/{sizes['validation']}/{sizes['calibration']}/{sizes['test']} "
            f"| {result['trainable_parameters']:,} | {raw['accuracy']:.4f} | {raw['macro_f1']:.4f} "
            f"| {raw['nll']:.4f} | {raw['brier']:.4f} | {raw['ece']:.4f} "
            f"| {calibrated['nll']:.4f} | {calibrated['brier']:.4f} | {calibrated['ece']:.4f} |"
        )
    lines.extend([
        "",
        "## COCO reproduction",
        "",
        (
            "The historical Full V3 regression gate passed before non-COCO training. "
            f"Gate status: `{bool(gate.get('gate_passed'))}`. "
            "The independently trained COCO row above is the fresh-adapter result; "
            "the regression checkpoint was not used to initialize it."
        ),
        "",
        "## Difficulty and failure cases",
        "",
        f"The lowest held-out accuracy was on **{hardest['display_name']}** ({hardest['metrics']['accuracy']:.2%}).",
        "",
    ])
    for result in results:
        failure = result["failure_cases"]
        visual = result["visual_dependency"]["overall"]
        lines.append(
            f"- **{result['display_name']}**: {failure['error_count']}/{failure['test_count']} test errors; "
            f"blank/noise confidence drop {visual['blank']['confidence_drop']:.4f}/"
            f"{visual['noise']['confidence_drop']:.4f}. Detailed high-confidence errors are in "
            f"`{OUTPUT_NAMES[result['dataset']]}/failure_cases.json`."
        )
    scale_text = (
        "All four rows are full-manifest independent adaptations."
        if full_scale else
        "At least one row is a fixed real subset and is explicitly marked pending full scale."
    )
    lines.extend([
        "",
        "## Claim assessment",
        "",
        scale_text,
        "The experiment tests **independent adaptation generality**: one architecture is freshly trained "
        "and selected separately on each task. It does not test **zero-shot cross-dataset generalization**, "
        "because no checkpoint trained on one dataset is evaluated as-is on another.",
        "",
        "The architecture/task generality claim should be conditioned on the per-dataset held-out results "
        "and visual-dependence diagnostics above; high task accuracy alone is not evidence of strong image use.",
        "",
    ])
    (root / "PAPER_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def load_completed_result(
    root: Path,
    dataset: str,
    bundle: DatasetBundle,
    *,
    stage_a_epochs: int,
    stage_b_epochs: int,
    stage_c_epochs: int,
    initial_state_sha256: str,
) -> dict[str, Any] | None:
    """Return a fully compatible result, or None when this dataset must rerun."""
    output = root / OUTPUT_NAMES[dataset]
    result_path = output / "result.json"
    config_path = output / "config.json"
    checkpoint_path = output / "checkpoint" / "visual_jev_v3.pt"
    required = (
        result_path,
        config_path,
        checkpoint_path,
        output / "training_history.json",
        output / "metrics.json",
        output / "calibration.json",
        output / "visual_dependency.json",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    expected_sizes = {split: len(bundle.examples[split]) for split in SPLITS}
    expected_epochs = {
        "A_alignment": stage_a_epochs,
        "B_candidate": stage_b_epochs,
        "C_gated": stage_c_epochs,
    }
    if (
        result.get("dataset") != dataset
        or result.get("profile") != bundle.profile
        or result.get("split_sizes") != expected_sizes
        or result.get("initial_state_sha256") != initial_state_sha256
        or config.get("stage_epochs") != expected_epochs
        or config.get("mixed_training") is not False
        or config.get("backbone_frozen") is not True
    ):
        return None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--full-manifest-root", type=Path, default=Path("data/benchmark_v1_full/manifests"))
    parser.add_argument("--controlled-manifest-root", type=Path, default=Path("data/benchmark_v1/manifests"))
    parser.add_argument("--full-experiment-root", type=Path, default=Path("experiments/benchmark_v1_full"))
    parser.add_argument("--controlled-experiment-root", type=Path, default=Path("experiments/benchmark_v1_controlled"))
    parser.add_argument("--reference-checkpoint", type=Path, default=Path("experiments/visual_jev_v3_paper/checkpoints/v3_full.pt"))
    parser.add_argument("--train-cap", type=int, default=32)
    parser.add_argument("--stage-a-epochs", type=int, default=2)
    parser.add_argument("--stage-b-epochs", type=int, default=16)
    parser.add_argument("--stage-c-epochs", type=int, default=6)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip datasets whose complete result/checkpoint matches this exact full-run configuration",
    )
    args = parser.parse_args()
    set_seed(SEED)
    root = args.result_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.resolve()

    full_audit = isolate_existing_manifests(args.full_manifest_root.resolve(), data_root, profile="full")
    context = Context(
        args.device,
        manifest_root=args.controlled_manifest_root.resolve(),
        experiment_root=args.controlled_experiment_root.resolve(),
    )
    bundles, executed_audit = build_executed_bundles(
        context,
        full_manifest_root=args.full_manifest_root.resolve(),
        full_experiment_root=args.full_experiment_root.resolve(),
        train_cap=args.train_cap,
        manifest_root=data_root,
    )
    raw_audit_path = args.full_manifest_root.resolve() / "data_audit.json"
    raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
    audit = {
        "seed": SEED,
        "raw_data_root": str(Path("data/visual_jev_raw").resolve()),
        "raw_rows": raw_audit.get("raw_rows", {}),
        "excluded": raw_audit.get("excluded", {}),
        "eligibility": {
            "scienceqa_image_only": "image != None",
            "iconqa_choice": "ques_type == choose_txt and answer in choices",
            "aokvqa": "labelled correct_choice_idx only",
            "coco_hard_negative": "existing verified hard-negative corpus",
        },
        "full_independent_manifests": full_audit,
        "executed_subset_manifests": executed_audit,
    }
    write_json(root / "data_audit" / "data_audit.json", audit)
    audit_rows = []
    for profile_name, profile in (("full", full_audit), ("executed_subset", executed_audit)):
        for dataset, payload in profile["datasets"].items():
            for split, values in payload["audit"]["splits"].items():
                audit_rows.append({"profile": profile_name, "dataset": dataset, "split": split, **values})
    write_csv(root / "data_audit" / "split_audit.csv", audit_rows)

    smoke = smoke_test(bundles, args.reference_checkpoint.resolve(), args.device)
    write_json(root / "smoke_test.json", smoke)
    if args.smoke_only:
        print(json.dumps(smoke, ensure_ascii=False, indent=2))
        return

    coco_root = root / OUTPUT_NAMES["coco_hard_negative"]
    gate_path = coco_root / "pipeline_regression" / "results.json"
    if args.resume and gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not gate.get("gate_passed"):
            gate = pipeline_regression(context, coco_root / "pipeline_regression")
        else:
            print(json.dumps({"resume": "pipeline_regression", "status": "skipped"}), flush=True)
    else:
        gate = pipeline_regression(context, coco_root / "pipeline_regression")
    if not gate["gate_passed"]:
        raise RuntimeError("COCO Full V3 regression gate failed")

    run_manifest = {
        "status": "running",
        "profile": next(iter({bundle.profile for bundle in bundles.values()})),
        "mixed_training": False,
        "train_cap": args.train_cap,
        "stage_epochs": {
            "A_alignment": args.stage_a_epochs,
            "B_candidate": args.stage_b_epochs,
            "C_gated": args.stage_c_epochs,
        },
        "completed_datasets": [],
    }
    write_json(root / "run_manifest.json", run_manifest)

    results = []
    for dataset in DATASETS:
        result = None
        if args.resume:
            result = load_completed_result(
                root,
                dataset,
                bundles[dataset],
                stage_a_epochs=args.stage_a_epochs,
                stage_b_epochs=args.stage_b_epochs,
                stage_c_epochs=args.stage_c_epochs,
                initial_state_sha256=smoke["initial_state_sha256"],
            )
        if result is None:
            result = train_one(
                dataset,
                bundles[dataset],
                context,
                output=root / OUTPUT_NAMES[dataset],
                reference_checkpoint=args.reference_checkpoint.resolve(),
                device=args.device,
                stage_a_epochs=args.stage_a_epochs,
                stage_b_epochs=args.stage_b_epochs,
                stage_c_epochs=args.stage_c_epochs,
                smoke=smoke,
            )
        else:
            print(json.dumps({"resume": dataset, "status": "skipped_complete"}), flush=True)
        results.append(result)
        if dataset == "coco_hard_negative":
            pair = result["visual_dependency"]["semantic_counterfactual"]
            independent_gate = {
                "gate_type": "one_sided_regression_floor",
                "accuracy_target": 0.8828,
                "accuracy_tolerance": 0.02,
                "pair_both_target": 0.5625,
                "pair_flip_target": 0.6042,
                "pair_tolerance": 0.15,
                "observed": {
                    "accuracy": result["metrics"]["accuracy"],
                    "pair_both": pair["both_directions_accuracy"],
                    "pair_flip": pair["prediction_flip_rate"],
                },
            }
            independent_gate["passed"] = (
                passes_regression_floor(
                    independent_gate["observed"]["accuracy"],
                    independent_gate["accuracy_target"],
                    independent_gate["accuracy_tolerance"],
                )
                and passes_regression_floor(
                    independent_gate["observed"]["pair_both"],
                    independent_gate["pair_both_target"],
                    independent_gate["pair_tolerance"],
                )
                and passes_regression_floor(
                    independent_gate["observed"]["pair_flip"],
                    independent_gate["pair_flip_target"],
                    independent_gate["pair_tolerance"],
                )
            )
            write_json(root / OUTPUT_NAMES[dataset] / "independent_reproduction_gate.json", independent_gate)
            if not independent_gate["passed"]:
                raise RuntimeError(
                    "fresh COCO independent reproduction gate failed; stop before other datasets"
                )
        print(json.dumps({
            "completed": dataset,
            "accuracy": result["metrics"]["accuracy"],
            "ece": result["metrics"]["ece"],
            "seconds": result["training_time_seconds"],
        }), flush=True)
        run_manifest["completed_datasets"].append(dataset)
        write_json(root / "run_manifest.json", run_manifest)

    initial_hashes = {result["initial_state_sha256"] for result in results}
    if len(initial_hashes) != 1:
        raise AssertionError("independent runs did not share the same fresh initial state")
    rows = []
    for result in results:
        dataset = result["dataset"]
        visual = result["visual_dependency"]
        if dataset == "coco_hard_negative":
            pair = visual["semantic_counterfactual"]
            visual_metric = f"Pair-both={pair['both_directions_accuracy']:.6f};Pair-flip={pair['prediction_flip_rate']:.6f}"
        elif dataset == "scienceqa_image_only":
            subset = visual["visual_required_subset"]
            visual_metric = (
                f"overall blank drop={visual['overall']['blank']['confidence_drop']:.6f};"
                f"visual-required n={subset['count']}"
            )
        else:
            visual_metric = (
                f"blank drop={visual['overall']['blank']['confidence_drop']:.6f};"
                f"noise drop={visual['overall']['noise']['confidence_drop']:.6f}"
            )
        none = result["calibration"]["methods"]["none"]
        calibrated = result["calibration"]["methods"]["global_temperature_scaling"]
        sizes = result["split_sizes"]
        rows.append({
            "Dataset": result["display_name"],
            "Profile": result["profile"],
            "Train size": sizes["train"],
            "Val size": sizes["validation"],
            "Cal size": sizes["calibration"],
            "Test size": sizes["test"],
            "Trainable Params": result["trainable_parameters"],
            "Accuracy": none["accuracy"],
            "Macro-F1": none["macro_f1"],
            "NLL": none["nll"],
            "Brier": none["brier"],
            "ECE": none["ece"],
            "Calibrated NLL": calibrated["nll"],
            "Calibrated Brier": calibrated["brier"],
            "Calibrated ECE": calibrated["ece"],
            "Visual Dependency Metric": visual_metric,
            "Training Time": result["training_time_seconds"],
            "GPU": result["gpu"].get("name", result["gpu"]["torch_device"]),
            "Checkpoint": result["checkpoint"],
            "pending_full_scale": result["pending_full_scale"],
        })
    write_csv(root / "main_generality.csv", rows)
    profiles = {result["profile"] for result in results}
    profile = next(iter(profiles)) if len(profiles) == 1 else "mixed_execution_profiles"
    full_scale_complete = profile == "full_independent"
    summary = {
        "seed": SEED,
        "mixed_training": False,
        "profile": profile,
        "full_scale_training_completed": full_scale_complete,
        "full_scale_pending_reason": None if full_scale_complete else (
            "fixed real subsets were used; see each result's pending_full_scale flag"
        ),
        "qwen3_vl_backbone_frozen": True,
        "same_fresh_initial_state": True,
        "fresh_initial_state_sha256": next(iter(initial_hashes)),
        "coco_pipeline_gate": gate,
        "results": results,
        "table": rows,
    }
    write_json(root / "main_generality.json", summary)
    write_paper_results(root, results, gate)
    run_manifest["status"] = "complete"
    run_manifest["main_generality"] = str((root / "main_generality.json").resolve())
    write_json(root / "run_manifest.json", run_manifest)
    print(json.dumps({
        "complete": True,
        "profile": summary["profile"],
        "main_generality": str((root / "main_generality.json").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
