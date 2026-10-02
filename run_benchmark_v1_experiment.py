"""Run the controlled multi-dataset Visual-JEV V3 training/calibration study."""

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
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from jev.benchmark_v1 import BenchmarkExample, read_jsonl
from jev.calibration import TemperatureScaler
from jev.serving import load_predictor
from scripts.run_visual_jev_v2_experiment import DiskFeatureDataset, file_sha256, open_cached_dataset
from scripts.run_visual_jev_v3_experiment import load_pairs
from train_visual_jev_v2 import format_candidate, load_records
from train_visual_jev_v3 import encode_image_stages, open_shards, subset_pairs
from visual_jev_v2 import Qwen3VLFeatureExtractor, VisualJEVV2
from visual_jev_v3 import (
    VisualJEVV3LossWeights,
    combine_v3_losses,
    counterfactual_visual_dependency_loss,
    cross_image_ranking_loss,
    mean_uniformity_loss,
    text_null_features,
    uniformity_loss,
)
from visual_jev_v3_pipeline import AlignmentAdapterConfig, ThreeStageVisualJEV, alignment_loss

SEED = 20260928
SPLITS = ("train", "validation", "calibration", "test")
OLD_REFERENCE = {
    "accuracy": 0.8828, "nll": 0.670, "brier": 0.369, "ece": 0.336,
    "blank_correct_probability": 0.3339, "noise_correct_probability": 0.3333,
    "semantic_pair_both": 0.5625, "semantic_pair_flip": 0.6042,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _feature_dir(root: Path, split: str) -> Path:
    return root / "features" / f"{split}-shards"


def _cache_valid(directory: Path, source_sha: str, count: int) -> bool:
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return (
        payload.get("format") == "benchmark-v1-feature-shards"
        and payload.get("source_sha256") == source_sha
        and int(payload.get("count", -1)) == count
        and all(
            (directory / f"{index:06d}.pt").is_file()
            and (directory / f"{index:06d}.pt").stat().st_size > 0
            for index in range(count)
        )
    )


def _write_feature_manifest(directory: Path, source_sha: str, count: int) -> None:
    _write_json(directory / "manifest.json", {
        "format": "benchmark-v1-feature-shards", "source_sha256": source_sha, "count": count,
        "fields": ["pre_tokens", "teacher_tokens", "text_features"],
    })


def prepare_features(
    examples: dict[str, list[BenchmarkExample]], manifest_root: Path, experiment_root: Path,
    model_id: str, device: str,
) -> tuple[dict[str, DiskFeatureDataset], dict[str, torch.Tensor], float]:
    started = time.time()
    ready: dict[str, DiskFeatureDataset] = {}
    missing = []
    for split in SPLITS:
        directory = _feature_dir(experiment_root, split)
        sha = file_sha256(manifest_root / f"{split}.jsonl")
        if _cache_valid(directory, sha, len(examples[split])):
            ready[split] = DiskFeatureDataset(directory, len(examples[split]))
        else:
            missing.append(split)
    control_path = experiment_root / "features" / "fixed_controls.pt"
    if not missing and control_path.is_file():
        return ready, torch.load(control_path, map_location="cpu", weights_only=True), 0.0

    predictor = load_predictor(
        model_id=model_id, device=device, max_length=512, batch_size=8,
        vision=True, image_root=manifest_root.parent,
    )
    backbone_model = predictor.scorer.model
    extractor = Qwen3VLFeatureExtractor(backbone_model)
    for split in missing:
        directory = _feature_dir(experiment_root, split)
        directory.mkdir(parents=True, exist_ok=True)
        split_started = time.time()
        progress = tqdm(
            enumerate(examples[split]),
            total=len(examples[split]),
            desc=f"features {split}",
            unit="sample",
            dynamic_ncols=True,
        )
        for index, example in progress:
            shard = directory / f"{index:06d}.pt"
            if shard.is_file():
                try:
                    cached = torch.load(shard, map_location="cpu", weights_only=True)
                    if not {"pre_tokens", "teacher_tokens", "text_features"} <= set(cached):
                        raise ValueError("incomplete feature shard")
                    progress.set_postfix(status="cached")
                    continue
                except Exception:
                    # A previously interrupted torch.save can leave a partial
                    # file. It is safe to discard because the shard is derived
                    # entirely from the immutable manifest row and frozen model.
                    shard.unlink()
            with Image.open(example.image) as source:
                image = source.convert("RGB")
                image.load()
            pre, teacher = encode_image_stages(backbone_model, image)
            prompts = [format_candidate(example.question, candidate) for candidate in example.candidates]
            text = extractor.encode_candidates(prompts).cpu()
            temporary = shard.with_suffix(".pt.partial")
            torch.save({"pre_tokens": pre, "teacher_tokens": teacher, "text_features": text}, temporary)
            temporary.replace(shard)
            elapsed = max(time.time() - split_started, 1e-6)
            progress.set_postfix(
                status="extracting",
                rate=f"{(index + 1) / elapsed:.2f}/s",
            )
        sha = file_sha256(manifest_root / f"{split}.jsonl")
        _write_feature_manifest(directory, sha, len(examples[split]))
        ready[split] = DiskFeatureDataset(directory, len(examples[split]))

    if control_path.is_file():
        controls = torch.load(control_path, map_location="cpu", weights_only=True)
    else:
        blank = Image.new("RGB", (512, 512), (255, 255, 255))
        rng = np.random.default_rng(SEED)
        noise = Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8), mode="RGB")
        blank_pre, _ = encode_image_stages(backbone_model, blank)
        noise_pre, _ = encode_image_stages(backbone_model, noise)
        controls = {"blank_pre": blank_pre, "noise_pre": noise_pre}
        control_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(controls, control_path)
    del extractor, backbone_model, predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ready, controls, time.time() - started


def _model_logits(model: ThreeStageVisualJEV, item: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(item["pre_tokens"], item["text_features"]).scores


@torch.no_grad()
def collect_logits(
    model: ThreeStageVisualJEV, features: Sequence[Any], examples: Sequence[BenchmarkExample]
) -> tuple[list[torch.Tensor], list[int]]:
    model.eval()
    logits, labels = [], []
    for index, example in enumerate(examples):
        logits.append(_model_logits(model, features[index]).detach().float().cpu())
        labels.append(example.label)
    return logits, labels


def variable_metrics(logits: Sequence[torch.Tensor], labels: Sequence[int], bins: int = 10) -> dict[str, float]:
    probabilities = [row.float().softmax(-1) for row in logits]
    predictions = [int(row.argmax().item()) for row in probabilities]
    accuracy = sum(int(a == b) for a, b in zip(predictions, labels)) / max(1, len(labels))
    max_classes = max(row.numel() for row in logits)
    f1 = []
    for class_index in range(max_classes):
        tp = sum(p == class_index and y == class_index for p, y in zip(predictions, labels))
        fp = sum(p == class_index and y != class_index for p, y in zip(predictions, labels))
        fn = sum(p != class_index and y == class_index for p, y in zip(predictions, labels))
        denominator = 2 * tp + fp + fn
        if tp + fp + fn:
            f1.append(0.0 if denominator == 0 else 2 * tp / denominator)
    nll = np.mean([-float(row[label].clamp_min(1e-12).log().item()) for row, label in zip(probabilities, labels)])
    brier = np.mean([
        float(((row - F.one_hot(torch.tensor(label), row.numel()).float()) ** 2).sum().item())
        for row, label in zip(probabilities, labels)
    ])
    confidence = np.array([float(row.max().item()) for row in probabilities])
    correct = np.array([float(p == y) for p, y in zip(predictions, labels)])
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.mean() * abs(confidence[mask].mean() - correct[mask].mean()))
    correct_probability = np.mean([float(row[label].item()) for row, label in zip(probabilities, labels)])
    margins = []
    entropies = []
    uniform_kls = []
    for row, prob, label in zip(logits, probabilities, labels):
        other = torch.cat((row[:label], row[label + 1 :])).max()
        margins.append(float((row[label] - other).item()))
        entropy = float(-(prob * prob.clamp_min(1e-12).log()).sum().item())
        entropies.append(entropy)
        uniform_kls.append(math.log(row.numel()) - entropy)
    return {
        "accuracy": float(accuracy), "macro_f1": float(np.mean(f1)), "nll": float(nll),
        "brier": float(brier), "ece": float(ece),
        "mean_correct_probability": float(correct_probability), "mean_margin": float(np.mean(margins)),
        "mean_entropy": float(np.mean(entropies)), "mean_uniform_kl": float(np.mean(uniform_kls)),
    }


def metrics_by_dataset(
    logits: Sequence[torch.Tensor], labels: Sequence[int], examples: Sequence[BenchmarkExample]
) -> dict[str, Any]:
    output = {"overall": variable_metrics(logits, labels)}
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example.dataset].append(index)
    for dataset, indices in sorted(grouped.items()):
        output[dataset] = variable_metrics([logits[i] for i in indices], [labels[i] for i in indices])
        output[dataset]["count"] = len(indices)
    output["overall"]["count"] = len(examples)
    return output


def train_model(
    features: dict[str, DiskFeatureDataset], examples: dict[str, list[BenchmarkExample]],
    controls: dict[str, torch.Tensor], config: dict[str, Any], device: str,
) -> tuple[ThreeStageVisualJEV, dict[str, Any]]:
    started = time.time()
    old_v2, old_payload = VisualJEVV2.from_checkpoint(config["v2_checkpoint"], map_location="cpu")
    first = features["train"][0]
    alignment_config = AlignmentAdapterConfig(
        pre_merger_dim=first["pre_tokens"].shape[-1], teacher_dim=first["teacher_tokens"].shape[-1], hidden_dim=1024
    )
    model = ThreeStageVisualJEV(alignment_config, old_v2.config).to(device)
    model.decision.load_state_dict(old_v2.state_dict())
    legacy_full_checkpoint = Path(config["legacy_paper_root"]) / "checkpoints" / "v3_full.pt"
    initialization = "legacy_v2_decision"
    if legacy_full_checkpoint.is_file():
        legacy_full, _ = ThreeStageVisualJEV.from_checkpoint(legacy_full_checkpoint, map_location="cpu")
        if (
            legacy_full.alignment.config == alignment_config
            and legacy_full.decision.config.__dict__ == old_v2.config.__dict__
        ):
            model.load_state_dict(legacy_full.state_dict())
            initialization = "legacy_full_v3"
        del legacy_full
    stage_a_checkpoint = Path(config["legacy_paper_root"]) / "checkpoints" / "stage_a_alignment.pt"
    stage_a_reused = False
    if stage_a_checkpoint.is_file() and initialization != "legacy_full_v3":
        payload = torch.load(stage_a_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("config") == alignment_config.__dict__:
            model.alignment.load_state_dict(payload["state_dict"])
            stage_a_reused = True
    history: dict[str, list[dict[str, Any]]] = {"stage_a": [], "stage_b": [], "stage_c": []}

    # Rehearse a fixed semantic-pair subset throughout all three stages.  This
    # prevents cross-domain alignment/decision updates from erasing the visual
    # flip behaviour learned by the legacy Full V3 checkpoint.
    legacy_root = Path(config["legacy_paper_root"])
    legacy_data = Path(config["legacy_data_root"])
    legacy_pair_root = Path(config["legacy_pair_root"])
    legacy_train_records = load_records(legacy_data / "splits" / "train.jsonl", legacy_data)
    legacy_validation_records = load_records(legacy_data / "splits" / "validation.jsonl", legacy_data)
    legacy_train_alignment = open_shards(legacy_root / "features" / "train-shards", len(legacy_train_records))
    legacy_validation_alignment = open_shards(legacy_root / "features" / "validation-shards", len(legacy_validation_records))
    train_pair_path = legacy_pair_root / "pairs" / "train.jsonl"
    validation_pair_path = legacy_pair_root / "pairs" / "validation.jsonl"
    train_pairs_all = load_pairs(train_pair_path, len(legacy_train_records))
    validation_pairs_all = load_pairs(validation_pair_path, len(legacy_validation_records))
    train_pair_text = open_cached_dataset(legacy_pair_root / "features" / "train-pair-text.pt", source_sha256=file_sha256(train_pair_path), item_key="pair_text")
    validation_pair_text = open_cached_dataset(legacy_pair_root / "features" / "validation-pair-text.pt", source_sha256=file_sha256(validation_pair_path), item_key="pair_text")
    if any(x is None for x in (legacy_train_alignment, legacy_validation_alignment, train_pair_text, validation_pair_text)):
        raise RuntimeError("legacy semantic-pair caches required for rehearsal are incomplete")
    semantic_train = sorted(train_pairs_all.items(), key=lambda kv: hashlib.sha256(f"{SEED}:{kv[0]}".encode()).digest())[:64]
    semantic_validation = subset_pairs(validation_pairs_all, {i for i in range(len(legacy_validation_records)) if i % 2 == 0})

    for parameter in model.decision.parameters():
        parameter.requires_grad_(False)
    for parameter in model.alignment.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.alignment.parameters(), lr=1e-4, weight_decay=1e-2)
    model.alignment.eval(); validation_losses = []
    with torch.no_grad():
        for index in range(len(features["validation"])):
            item = features["validation"][index]
            validation_losses.append(float(alignment_loss(model.alignment(item["pre_tokens"]), item["teacher_tokens"]).total.item()))
    pair = semantic_metrics(model, legacy_validation_alignment, semantic_validation, validation_pair_text, temperature=1.0)
    baseline_a = {"epoch": 0, "train_loss": None, "validation_loss": float(np.mean(validation_losses)),
                  "validation_pair_both": pair["both_directions_accuracy"], "validation_pair_flip": pair["prediction_flip_rate"]}
    history["stage_a"].append(baseline_a); print(json.dumps({"stage": "A", **baseline_a}), flush=True)
    best_loss = baseline_a["validation_loss"] - 0.5 * baseline_a["validation_pair_both"] - 0.25 * baseline_a["validation_pair_flip"]
    best_state = copy.deepcopy(model.alignment.state_dict())
    rng = random.Random(SEED)
    for epoch in range(1, 4):
        model.alignment.train(); order = list(range(len(features["train"]))); rng.shuffle(order); losses = []
        for position, index in enumerate(order):
            item = features["train"][index]
            sem_source, sem_pair = semantic_train[position % len(semantic_train)]
            sem_partner = int(sem_pair["counterfactual_index"])
            sem_text = train_pair_text[int(sem_pair["_cache_index"])]["text_features"]
            optimizer.zero_grad(set_to_none=True)
            alignment = alignment_loss(model.alignment(item["pre_tokens"]), item["teacher_tokens"])
            sem_a = model(legacy_train_alignment[sem_source]["pre_tokens"], sem_text).scores
            sem_b = model(legacy_train_alignment[sem_partner]["pre_tokens"], sem_text).scores
            rehearsal = counterfactual_visual_dependency_loss(
                sem_a, sem_b, image1_target=0, image2_target=int(sem_pair["counterfactual_target_index"])
            )
            loss = alignment.total + rehearsal
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.alignment.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.item()))
        model.alignment.eval(); validation_losses = []
        with torch.no_grad():
            for index in range(len(features["validation"])):
                item = features["validation"][index]
                validation_losses.append(float(alignment_loss(model.alignment(item["pre_tokens"]), item["teacher_tokens"]).total.item()))
        pair = semantic_metrics(model, legacy_validation_alignment, semantic_validation, validation_pair_text, temperature=1.0)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": float(np.mean(validation_losses)),
               "validation_pair_both": pair["both_directions_accuracy"], "validation_pair_flip": pair["prediction_flip_rate"]}
        history["stage_a"].append(row); print(json.dumps({"stage": "A", **row}), flush=True)
        selection_loss = row["validation_loss"] - 0.5 * row["validation_pair_both"] - 0.25 * row["validation_pair_flip"]
        if selection_loss < best_loss:
            best_loss, best_state = selection_loss, copy.deepcopy(model.alignment.state_dict())
    model.alignment.load_state_dict(best_state)

    for parameter in model.alignment.parameters(): parameter.requires_grad_(False)
    for parameter in model.decision.parameters(): parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.decision.parameters(), lr=2e-5, weight_decay=1e-2)
    logits, labels = collect_logits(model, features["validation"], examples["validation"])
    accuracy = variable_metrics(logits, labels)["accuracy"]
    pair = semantic_metrics(model, legacy_validation_alignment, semantic_validation, validation_pair_text, temperature=1.0)
    best_accuracy = accuracy + 0.5 * pair["both_directions_accuracy"] + 0.25 * pair["prediction_flip_rate"]
    best_state = copy.deepcopy(model.state_dict())
    baseline_b = {"epoch": 0, "train_loss": None, "validation_accuracy": accuracy,
                  "validation_pair_both": pair["both_directions_accuracy"], "validation_pair_flip": pair["prediction_flip_rate"],
                  "selection": best_accuracy}
    history["stage_b"].append(baseline_b); print(json.dumps({"stage": "B", **baseline_b}), flush=True)
    for epoch in range(1, 5):
        model.train(); order = list(range(len(features["train"]))); rng.shuffle(order); losses = []
        for position, index in enumerate(order):
            item, example = features["train"][index], examples["train"][index]
            sem_source, sem_pair = semantic_train[position % len(semantic_train)]
            sem_partner = int(sem_pair["counterfactual_index"])
            sem_text = train_pair_text[int(sem_pair["_cache_index"])]["text_features"]
            optimizer.zero_grad(set_to_none=True)
            scores = _model_logits(model, item)
            candidate = F.cross_entropy(scores[None], torch.tensor([example.label], device=scores.device))
            sem_a = model(legacy_train_alignment[sem_source]["pre_tokens"], sem_text).scores
            sem_b = model(legacy_train_alignment[sem_partner]["pre_tokens"], sem_text).scores
            rehearsal = counterfactual_visual_dependency_loss(
                sem_a, sem_b, image1_target=0, image2_target=int(sem_pair["counterfactual_target_index"])
            )
            loss = candidate + rehearsal
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.decision.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.item()))
        logits, labels = collect_logits(model, features["validation"], examples["validation"])
        accuracy = variable_metrics(logits, labels)["accuracy"]
        pair = semantic_metrics(model, legacy_validation_alignment, semantic_validation, validation_pair_text, temperature=1.0)
        selection = accuracy + 0.5 * pair["both_directions_accuracy"] + 0.25 * pair["prediction_flip_rate"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_accuracy": accuracy,
               "validation_pair_both": pair["both_directions_accuracy"], "validation_pair_flip": pair["prediction_flip_rate"],
               "selection": selection}
        history["stage_b"].append(row); print(json.dumps({"stage": "B", **row}), flush=True)
        if selection > best_accuracy:
            best_accuracy, best_state = selection, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    for parameter in model.parameters(): parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    weights = VisualJEVV3LossWeights(counterfactual=1.0, invalid_image=5.0, text_null=0.25, cross_image_rank=0.5)
    logits, labels = collect_logits(model, features["validation"], examples["validation"])
    validation = variable_metrics(logits, labels)
    pair = semantic_metrics(model, legacy_validation_alignment, semantic_validation, validation_pair_text, temperature=1.0)
    best_selection = validation["accuracy"] + 0.5 * pair["both_directions_accuracy"] + 0.25 * pair["prediction_flip_rate"]
    best_state = copy.deepcopy(model.state_dict())
    baseline_c = {"epoch": 0, "validation_accuracy": validation["accuracy"],
                  "validation_pair_both": pair["both_directions_accuracy"], "validation_pair_flip": pair["prediction_flip_rate"],
                  "selection": best_selection}
    history["stage_c"].append(baseline_c); print(json.dumps({"stage": "C", **baseline_c}), flush=True)
    for epoch in range(1, 5):
        model.train(); order = list(range(len(features["train"]))); rng.shuffle(order); sums = defaultdict(list)
        for position, index in enumerate(order):
            item, example = features["train"][index], examples["train"][index]
            wrong_index = order[(position + 1) % len(order)]
            wrong_item = features["train"][wrong_index]
            sem_source, sem_pair = semantic_train[position % len(semantic_train)]
            sem_partner = int(sem_pair["counterfactual_index"])
            sem_text = train_pair_text[int(sem_pair["_cache_index"])]["text_features"]
            optimizer.zero_grad(set_to_none=True)
            original = _model_logits(model, item)
            wrong = model(wrong_item["pre_tokens"], item["text_features"]).scores
            blank = model(controls["blank_pre"], item["text_features"]).scores
            noise = model(controls["noise_pre"], item["text_features"]).scores
            null = model(item["pre_tokens"], text_null_features(item["text_features"], mode="mean")).scores
            sem_a = model(legacy_train_alignment[sem_source]["pre_tokens"], sem_text).scores
            sem_b = model(legacy_train_alignment[sem_partner]["pre_tokens"], sem_text).scores
            candidate = F.cross_entropy(original[None], torch.tensor([example.label], device=original.device))
            counterfactual = counterfactual_visual_dependency_loss(
                sem_a, sem_b, image1_target=0, image2_target=int(sem_pair["counterfactual_target_index"]), margin=0.2
            )
            invalid = mean_uniformity_loss((blank, noise, wrong))
            null_loss = uniformity_loss(null)
            rank = cross_image_ranking_loss(original, wrong, positive_index=example.label, margin=0.2)
            losses = combine_v3_losses(candidate=candidate, counterfactual=counterfactual, invalid_image=invalid, text_null=null_loss, cross_image_rank=rank, weights=weights)
            losses.total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            for name in ("total", "candidate", "counterfactual", "invalid_image", "text_null", "cross_image_rank"):
                sums[name].append(float(getattr(losses, name).item()))
        logits, labels = collect_logits(model, features["validation"], examples["validation"])
        validation = variable_metrics(logits, labels)
        pair = semantic_metrics(model, legacy_validation_alignment, semantic_validation, validation_pair_text, temperature=1.0)
        selection = validation["accuracy"] + 0.5 * pair["both_directions_accuracy"] + 0.25 * pair["prediction_flip_rate"]
        row = {"epoch": epoch, **{f"train_{k}_loss": float(np.mean(v)) for k, v in sums.items()},
               "validation_accuracy": validation["accuracy"], "validation_pair_both": pair["both_directions_accuracy"],
               "validation_pair_flip": pair["prediction_flip_rate"], "selection": selection}
        history["stage_c"].append(row); print(json.dumps({"stage": "C", **row}), flush=True)
        if selection > best_selection:
            best_selection, best_state = selection, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    training = {
        "history": history, "training_seconds": time.time() - started,
        "initialization": initialization, "stage_a_checkpoint_reused_as_initialization": stage_a_reused,
        "parameters": {"alignment": model.alignment.trainable_parameter_count, "decision": model.decision.trainable_parameter_count,
                       "full": model.trainable_parameter_count},
        "semantic_stage_c_pairs": len(semantic_train), "legacy_v2_checkpoint_step": old_payload.get("step"),
    }
    return model, training


def _js(p: torch.Tensor, q: torch.Tensor) -> float:
    middle = 0.5 * (p + q)
    value = 0.5 * ((p * (p.clamp_min(1e-12).log() - middle.log())).sum() + (q * (q.clamp_min(1e-12).log() - middle.log())).sum())
    return float(value.item())


@torch.no_grad()
def visual_metrics(
    model: ThreeStageVisualJEV, features: Sequence[Any], examples: Sequence[BenchmarkExample],
    controls: dict[str, torch.Tensor], temperature: float,
) -> dict[str, Any]:
    model.eval()
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples): by_dataset[example.dataset].append(index)
    same_dataset_next = {}
    for indices in by_dataset.values():
        for position, index in enumerate(indices): same_dataset_next[index] = indices[(position + 1) % len(indices)]
    all_next = {index: (index + 1) % len(examples) for index in range(len(examples))}
    conditions = {name: [] for name in ("original", "blank", "noise", "wrong", "swap")}
    for index, example in enumerate(examples):
        item = features[index]; text = item["text_features"]
        pre_by_condition = {
            "original": item["pre_tokens"], "blank": controls["blank_pre"], "noise": controls["noise_pre"],
            "wrong": features[same_dataset_next[index]]["pre_tokens"], "swap": features[all_next[index]]["pre_tokens"],
        }
        for name, pre in pre_by_condition.items():
            conditions[name].append((model(pre, text).scores.detach().float().cpu() / temperature, example.label))
    result: dict[str, Any] = {}
    original_probabilities = [row.softmax(-1) for row, _ in conditions["original"]]
    for name, rows in conditions.items():
        correct_probabilities, margins, uniform_kls, js_values, flips = [], [], [], [], []
        for index, (logit, label) in enumerate(rows):
            prob = logit.softmax(-1); other = torch.cat((logit[:label], logit[label + 1 :])).max()
            entropy = -(prob * prob.clamp_min(1e-12).log()).sum()
            correct_probabilities.append(float(prob[label].item())); margins.append(float((logit[label] - other).item()))
            uniform_kls.append(float((math.log(logit.numel()) - entropy).item()))
            js_values.append(_js(original_probabilities[index], prob))
            flips.append(int(prob.argmax().item() != original_probabilities[index].argmax().item()))
        result[name] = {
            "mean_correct_probability": float(np.mean(correct_probabilities)), "mean_margin": float(np.mean(margins)),
            "mean_uniform_kl": float(np.mean(uniform_kls)), "mean_js_from_original": float(np.mean(js_values)),
            "prediction_flip_rate_vs_original": float(np.mean(flips)),
        }
    result["summary"] = {
        "invalid_confidence_drop": result["original"]["mean_correct_probability"] - np.mean([result["blank"]["mean_correct_probability"], result["noise"]["mean_correct_probability"]]),
        "invalid_mean_kl_u": np.mean([result["blank"]["mean_uniform_kl"], result["noise"]["mean_uniform_kl"]]),
        "wrong_js": result["wrong"]["mean_js_from_original"], "swap_js": result["swap"]["mean_js_from_original"],
    }
    return result


@torch.no_grad()
def semantic_metrics(
    model: ThreeStageVisualJEV, alignment: Sequence[Any], pairs: dict[int, dict[str, Any]],
    pair_text: Sequence[Any], temperature: float,
) -> dict[str, float]:
    model.eval(); source_correct = partner_correct = both = flips = 0; source_margins = []; partner_margins = []
    for source, pair in pairs.items():
        partner = int(pair["counterfactual_index"]); target = int(pair["counterfactual_target_index"])
        text = pair_text[int(pair["_cache_index"])]["text_features"]
        a_scores = model(alignment[source]["pre_tokens"], text).scores.detach().float().cpu() / temperature
        b_scores = model(alignment[partner]["pre_tokens"], text).scores.detach().float().cpu() / temperature
        a, b = int(a_scores.argmax().item()), int(b_scores.argmax().item())
        a_ok, b_ok = a == 0, b == target
        source_correct += int(a_ok); partner_correct += int(b_ok); both += int(a_ok and b_ok); flips += int(a != b)
        source_margins.append(float(a_scores[0] - a_scores[1:].max()))
        partner_margins.append(float(b_scores[target] - torch.cat((b_scores[:target], b_scores[target + 1 :])).max()))
    count = max(1, len(pairs))
    return {"pair_count": len(pairs), "source_accuracy": source_correct / count,
            "counterfactual_accuracy": partner_correct / count, "both_directions_accuracy": both / count,
            "prediction_flip_rate": flips / count, "source_target_margin": float(np.mean(source_margins)),
            "counterfactual_target_margin": float(np.mean(partner_margins))}


def _csv_metrics(path: Path, before: dict[str, Any], after: dict[str, Any]) -> None:
    fields = ["scope", "temperature_state", "count", "accuracy", "macro_f1", "nll", "brier", "ece", "mean_correct_probability", "mean_margin"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for state, payload in (("before", before), ("after", after)):
            for scope, values in payload.items():
                writer.writerow({"scope": scope, "temperature_state": state, **{key: values.get(key) for key in fields[2:]}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path("data/benchmark_v1/manifests"))
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments/benchmark_v1_controlled"))
    parser.add_argument("--results-root", type=Path, default=Path("experiments/results/benchmark_v1"))
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build/validate feature shards and controls, then exit without running the legacy trainer.",
    )
    args = parser.parse_args()
    set_seed(SEED)
    overall_started = time.time()
    manifest_root = args.manifest_root.resolve(); experiment_root = args.experiment_root.resolve(); results_root = args.results_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True); (experiment_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    examples = {split: list(read_jsonl(manifest_root / f"{split}.jsonl")) for split in SPLITS}
    split_groups = {split: {item.group_id for item in rows} for split, rows in examples.items()}
    overlap = {f"{a}/{b}": len(split_groups[a] & split_groups[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1 :]}
    if any(overlap.values()): raise RuntimeError(f"group leakage: {overlap}")
    features, controls, feature_seconds = prepare_features(examples, manifest_root, experiment_root, args.model, args.device)
    if args.prepare_only:
        print(json.dumps({
            "status": "features_ready",
            "manifest_root": str(manifest_root),
            "experiment_root": str(experiment_root),
            "counts": {split: len(rows) for split, rows in examples.items()},
            "feature_seconds": feature_seconds,
        }), flush=True)
        return
    config = {
        "v2_checkpoint": str(Path("checkpoints/visual-jev-v2-coco-2k.pt").resolve()),
        "legacy_paper_root": str(Path("experiments/visual_jev_v3_paper").resolve()),
        "legacy_data_root": str(Path("data/visual-jev-v2").resolve()),
        "legacy_pair_root": str(Path("data/visual-jev-v3").resolve()),
    }
    model, training = train_model(features, examples, controls, config, args.device)
    checkpoint = experiment_root / "checkpoints" / "v3_full.pt"
    model.save_checkpoint(checkpoint, stage="C", step=sum(len(row) for row in training["history"].values()), metadata={"seed": SEED, "profile": "controlled_subset"})

    calibration_logits, calibration_labels = collect_logits(model, features["calibration"], examples["calibration"])
    test_logits, test_labels = collect_logits(model, features["test"], examples["test"])
    scaler = TemperatureScaler(); fit = scaler.fit(calibration_logits, calibration_labels, max_iter=100)
    temperature = fit.temperature
    scaled_calibration = [row / temperature for row in calibration_logits]
    scaled_test = [row / temperature for row in test_logits]
    calibration_before = metrics_by_dataset(calibration_logits, calibration_labels, examples["calibration"])
    calibration_after = metrics_by_dataset(scaled_calibration, calibration_labels, examples["calibration"])
    test_before = metrics_by_dataset(test_logits, test_labels, examples["test"])
    test_after = metrics_by_dataset(scaled_test, test_labels, examples["test"])
    if test_before["overall"]["accuracy"] != test_after["overall"]["accuracy"] or test_before["overall"]["macro_f1"] != test_after["overall"]["macro_f1"]:
        raise RuntimeError("temperature scaling changed argmax metrics")

    legacy_data = Path(config["legacy_data_root"]); legacy_root = Path(config["legacy_paper_root"]); legacy_pair_root = Path(config["legacy_pair_root"])
    legacy_validation_records = load_records(legacy_data / "splits" / "validation.jsonl", legacy_data)
    legacy_alignment = open_shards(legacy_root / "features" / "validation-shards", len(legacy_validation_records))
    pair_path = legacy_pair_root / "pairs" / "validation.jsonl"
    pairs_all = load_pairs(pair_path, len(legacy_validation_records))
    pair_text = open_cached_dataset(legacy_pair_root / "features" / "validation-pair-text.pt", source_sha256=file_sha256(pair_path), item_key="pair_text")
    semantic_test = subset_pairs(pairs_all, {i for i in range(len(legacy_validation_records)) if i % 2 == 1})
    visual_before = visual_metrics(model, features["test"], examples["test"], controls, 1.0)
    visual_after = visual_metrics(model, features["test"], examples["test"], controls, temperature)
    semantic_before = semantic_metrics(model, legacy_alignment, semantic_test, pair_text, 1.0)
    semantic_after = semantic_metrics(model, legacy_alignment, semantic_test, pair_text, temperature)

    old_delta_before = {key: test_before["overall"][key] - OLD_REFERENCE[key] for key in ("accuracy", "nll", "brier", "ece")}
    old_delta_after = {key: test_after["overall"][key] - OLD_REFERENCE[key] for key in ("accuracy", "nll", "brier", "ece")}
    calibration_payload = {
        "method": "single scalar temperature scaling", "temperature": temperature, "fit": scaler.state_payload(fit),
        "calibration_split_count": len(examples["calibration"]), "test_split_count": len(examples["test"]),
        "calibration_test_group_overlap": len(split_groups["calibration"] & split_groups["test"]),
        "calibration": {"before": calibration_before, "after": calibration_after},
        "test": {"before": test_before, "after": test_after},
        "argmax_invariant": True, "old_reference": OLD_REFERENCE,
        "new_minus_old": {"before": old_delta_before, "after": old_delta_after},
        "calibration_improvement": {key: test_before["overall"][key] - test_after["overall"][key] for key in ("nll", "brier", "ece")},
    }
    main_payload = {
        "profile": "controlled_subset", "full_scale_pending": True, "seed": SEED,
        "checkpoint": str(checkpoint), "candidate_order": "fixed deterministic shuffle stored in manifest",
        "splits": {split: {"count": len(rows), "datasets": dict(Counter(x.dataset for x in rows)), "sha256": file_sha256(manifest_root / f"{split}.jsonl")} for split, rows in examples.items()},
        "group_overlap": overlap, "temperature": temperature, "before": test_before, "after": test_after,
        "training": training,
    }
    visual_payload = {
        "temperature": temperature, "before": visual_before, "after": visual_after,
        "semantic_counterfactual": {"before": semantic_before, "after": semantic_after, "argmax_invariant": semantic_before["both_directions_accuracy"] == semantic_after["both_directions_accuracy"] and semantic_before["prediction_flip_rate"] == semantic_after["prediction_flip_rate"]},
        "old_reference": {"blank_correct_probability": OLD_REFERENCE["blank_correct_probability"], "noise_correct_probability": OLD_REFERENCE["noise_correct_probability"], "semantic_pair_both": OLD_REFERENCE["semantic_pair_both"], "semantic_pair_flip": OLD_REFERENCE["semantic_pair_flip"]},
    }
    provenance = {
        "seed": SEED, "profile": "controlled_subset", "full_scale_pending": True,
        "data_audit": str(manifest_root / "data_audit.json"), "manifest": str(manifest_root / "manifest.json"),
        "feature_extraction_seconds_this_run": feature_seconds, "training_seconds": training["training_seconds"],
        "total_seconds": time.time() - overall_started,
        "gpu": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else None,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "trainable_parameters": training["parameters"],
    }
    _write_json(results_root / "calibration_before_after.json", calibration_payload)
    _csv_metrics(results_root / "calibration_before_after.csv", test_before, test_after)
    _write_json(results_root / "main_results.json", main_payload)
    _csv_metrics(results_root / "main_results.csv", test_before, test_after)
    _write_json(results_root / "visual_dependency.json", visual_payload)
    with (results_root / "visual_dependency.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["temperature_state", "condition", "mean_correct_probability", "mean_margin", "mean_uniform_kl", "mean_js_from_original", "prediction_flip_rate_vs_original"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for state, payload in (("before", visual_before), ("after", visual_after)):
            for condition in ("original", "blank", "noise", "wrong", "swap"):
                writer.writerow({"temperature_state": state, "condition": condition, **payload[condition]})
    _write_json(results_root / "run_provenance.json", provenance)
    summary = {
        "temperature": temperature, "test_before": test_before["overall"], "test_after": test_after["overall"],
        "calibration_improvement": calibration_payload["calibration_improvement"],
        "visual_before": visual_before["summary"], "visual_after": visual_after["summary"],
        "semantic_before": semantic_before, "semantic_after": semantic_after, "provenance": provenance,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
