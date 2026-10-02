"""Reproducible three-stage training for the Visual-JEV V3 paper experiment.

The script preserves all legacy V1/V2/V3 entry points.  It reuses the current
COCO hard-negative data and frozen text/post-merger caches, adds a resumable
pre-merger cache, and produces Stage-A, Stage-B, Stage-C and ablation
checkpoints under ``experiments/visual_jev_v3_paper``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from jev.serving import load_predictor
from scripts.run_visual_jev_v2_experiment import (
    DiskFeatureDataset,
    file_sha256,
    load_image,
    open_cached_dataset,
)
from scripts.run_visual_jev_v3_experiment import load_pairs
from train_visual_jev_v2 import load_records
from visual_jev_v2 import VisualJEVV2, candidate_pair_loss
from visual_jev_v3 import (
    VisualJEVV3LossWeights,
    combine_v3_losses,
    counterfactual_visual_dependency_loss,
    cross_image_ranking_loss,
    mean_uniformity_loss,
    text_null_features,
    uniformity_loss,
)
from visual_jev_v3_pipeline import (
    AlignmentAdapterConfig,
    MeanPoolJEV,
    ThreeStageVisualJEV,
    alignment_loss,
    grouped_pre_merger_tokens,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def write_manifest(directory: Path, *, count: int, source_sha256: str, item_key: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": "visual-jev-disk-shards-v1",
                "source_sha256": source_sha256,
                "count": count,
                "item_key": item_key,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def open_shards(directory: Path, count: int) -> DiskFeatureDataset | None:
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if int(payload.get("count", -1)) != count:
        return None
    return DiskFeatureDataset(directory, count)


@torch.no_grad()
def encode_image_stages(decision_model: Any, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Extract visual evidence."},
        ],
    }]
    encoded = decision_model.processor.apply_chat_template(
        [conversation],
        tokenize=True,
        add_generation_prompt=True,
        processor_kwargs={"padding": True},
        return_dict=True,
        return_tensors="pt",
    )
    visual = decision_model.backbone.visual
    parameter = next(visual.parameters())
    pixel_values = encoded["pixel_values"].to(parameter.device, parameter.dtype)
    grid_thw = encoded["image_grid_thw"].to(parameter.device)
    output = visual(pixel_values, grid_thw=grid_thw, return_dict=True)
    pre = output.last_hidden_state
    teacher = output.pooler_output
    if pre.ndim == 3:
        pre = pre[0]
    if teacher.ndim == 3:
        teacher = teacher[0]
    grouped = grouped_pre_merger_tokens(pre, teacher.shape[0])
    return grouped.cpu(), teacher.cpu()


def prepare_alignment_caches(
    config: dict[str, Any],
    train_records: Sequence[Any],
    validation_records: Sequence[Any],
    train_sha: str,
    validation_sha: str,
) -> dict[str, DiskFeatureDataset]:
    paper_root = Path(config["paper_root"])
    feature_root = paper_root / "features"
    specifications = {
        "train": (train_records, train_sha, False),
        "validation": (validation_records, validation_sha, False),
        "train_controls": (train_records, train_sha, True),
        "validation_controls": (validation_records, validation_sha, True),
    }
    cached: dict[str, DiskFeatureDataset] = {}
    missing = []
    for name, (records, _, _) in specifications.items():
        dataset = open_shards(feature_root / f"{name}-shards", len(records))
        if dataset is None:
            missing.append(name)
        else:
            cached[name] = dataset
    if not missing:
        return cached

    predictor = load_predictor(
        model_id=config["model"],
        device=config["device"],
        max_length=512,
        batch_size=1,
        vision=True,
        image_root=Path(config["data_root"]),
    )
    decision_model = predictor.scorer.model
    decision_model.backbone.eval()
    for parameter in decision_model.backbone.parameters():
        parameter.requires_grad_(False)
    rng = np.random.default_rng(config["seed"])
    for name, (records, source_sha, controls) in specifications.items():
        directory = feature_root / f"{name}-shards"
        if name not in missing:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        started = time.time()
        for index, record in enumerate(records):
            shard = directory / f"{index:06d}.pt"
            if shard.is_file():
                continue
            original = load_image(record.image)
            if controls:
                blank = Image.new("RGB", original.size, (255, 255, 255))
                noise_array = rng.integers(
                    0, 256, size=(original.height, original.width, 3), dtype=np.uint8
                )
                noise = Image.fromarray(noise_array, mode="RGB")
                blank_pre, _ = encode_image_stages(decision_model, blank)
                noise_pre, _ = encode_image_stages(decision_model, noise)
                item = {"blank_pre": blank_pre, "noise_pre": noise_pre}
            else:
                pre, teacher = encode_image_stages(decision_model, original)
                item = {"pre_tokens": pre, "teacher_tokens": teacher}
            torch.save(item, shard)
            if (index + 1) % 25 == 0 or index + 1 == len(records):
                print(
                    json.dumps(
                        {
                            "cache": name,
                            "completed": index + 1,
                            "total": len(records),
                            "elapsed_seconds": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
        write_manifest(
            directory,
            count=len(records),
            source_sha256=source_sha,
            item_key=name,
        )
        cached[name] = DiskFeatureDataset(directory, len(records))
    del predictor, decision_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cached


def alignment_metrics(model: ThreeStageVisualJEV, dataset: Sequence[Any], indices: Sequence[int]) -> dict[str, float]:
    model.alignment.eval()
    totals = {"loss": 0.0, "cosine_similarity": 0.0, "mse": 0.0, "norm_ratio": 0.0}
    student_norms: list[float] = []
    teacher_norms: list[float] = []
    with torch.no_grad():
        for index in indices:
            item = dataset[index]
            student = model.alignment(item["pre_tokens"])
            teacher = item["teacher_tokens"].to(student.device)
            loss = alignment_loss(student, teacher)
            totals["loss"] += float(loss.total.item())
            totals["cosine_similarity"] += float(
                F.cosine_similarity(student.float(), teacher.float(), dim=-1).mean().item()
            )
            totals["mse"] += float(F.mse_loss(student.float(), teacher.float()).item())
            sn = student.float().norm(dim=-1).mean().item()
            tn = teacher.float().norm(dim=-1).mean().item()
            totals["norm_ratio"] += sn / max(tn, 1e-8)
            student_norms.append(sn)
            teacher_norms.append(tn)
    count = max(1, len(indices))
    result = {name: value / count for name, value in totals.items()}
    result.update(
        {
            "student_norm_mean": float(np.mean(student_norms)),
            "student_norm_std": float(np.std(student_norms)),
            "teacher_norm_mean": float(np.mean(teacher_norms)),
            "teacher_norm_std": float(np.std(teacher_norms)),
        }
    )
    return result


def train_stage_a(
    model: ThreeStageVisualJEV,
    train_data: Sequence[Any],
    validation_data: Sequence[Any],
    validation_indices: Sequence[int],
    settings: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    for parameter in model.decision.parameters():
        parameter.requires_grad_(False)
    for parameter in model.alignment.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.alignment.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    history = []
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    rng = random.Random(seed)
    for epoch in range(1, settings["epochs"] + 1):
        model.alignment.train()
        order = list(range(len(train_data)))
        rng.shuffle(order)
        sums = {"total": 0.0, "cosine": 0.0, "mse": 0.0, "norm": 0.0}
        for index in order:
            item = train_data[index]
            optimizer.zero_grad(set_to_none=True)
            student = model.alignment(item["pre_tokens"])
            loss = alignment_loss(
                student,
                item["teacher_tokens"],
                cosine_weight=settings["cosine_weight"],
                mse_weight=settings["mse_weight"],
                norm_weight=settings["norm_weight"],
            )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.alignment.parameters(), 1.0)
            optimizer.step()
            for name in sums:
                sums[name] += float(getattr(loss, name).item())
        validation = alignment_metrics(model, validation_data, validation_indices)
        row = {
            "stage": "A",
            "epoch": epoch,
            **{f"train_{name}": value / len(order) for name, value in sums.items()},
            **{f"validation_{name}": value for name, value in validation.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.alignment.state_dict())
    if best_state is None:
        raise RuntimeError("Stage A did not produce a checkpoint")
    model.alignment.load_state_dict(best_state)
    return history, best_epoch


@torch.no_grad()
def decision_accuracy(
    model: ThreeStageVisualJEV,
    alignment_data: Sequence[Any],
    text_data: Sequence[Any],
    indices: Sequence[int],
) -> float:
    model.eval()
    correct = 0
    for index in indices:
        scores = model(
            alignment_data[index]["pre_tokens"], text_data[index]["text_features"]
        ).scores
        correct += int(int(scores.argmax().item()) == 0)
    return correct / max(1, len(indices))


def decision_optimizer(model: ThreeStageVisualJEV, settings: dict[str, Any], train_alignment: bool):
    for parameter in model.decision.parameters():
        parameter.requires_grad_(True)
    for parameter in model.alignment.parameters():
        parameter.requires_grad_(train_alignment)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        parameters,
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )


def train_stage_b(
    model: ThreeStageVisualJEV,
    train_alignment_data: Sequence[Any],
    train_text_data: Sequence[Any],
    validation_alignment_data: Sequence[Any],
    validation_text_data: Sequence[Any],
    validation_indices: Sequence[int],
    settings: dict[str, Any],
    seed: int,
    *,
    train_alignment: bool,
    label: str,
) -> tuple[list[dict[str, Any]], int]:
    optimizer = decision_optimizer(model, settings, train_alignment)
    rng = random.Random(seed)
    history = []
    best_state = None
    best_accuracy = -1.0
    best_epoch = 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = list(range(len(train_alignment_data)))
        rng.shuffle(order)
        losses = []
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            scores = model(
                train_alignment_data[index]["pre_tokens"],
                train_text_data[index]["text_features"],
            ).scores
            loss = candidate_pair_loss(scores[:1], scores[1:].unsqueeze(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(float(loss.item()))
        accuracy = decision_accuracy(
            model,
            validation_alignment_data,
            validation_text_data,
            validation_indices,
        )
        row = {
            "stage": label,
            "epoch": epoch,
            "train_candidate_loss": float(np.mean(losses)),
            "validation_accuracy": accuracy,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Stage B did not produce a checkpoint")
    model.load_state_dict(best_state)
    return history, best_epoch


def subset_pairs(pairs: dict[int, dict[str, Any]], allowed: set[int]) -> dict[int, dict[str, Any]]:
    return {
        source: pair
        for source, pair in pairs.items()
        if source in allowed and int(pair["counterfactual_index"]) in allowed
    }


@torch.no_grad()
def stage_c_selection(
    model: ThreeStageVisualJEV,
    alignment_data: Sequence[Any],
    controls: Sequence[Any],
    text_data: Sequence[Any],
    indices: Sequence[int],
    pairs: dict[int, dict[str, Any]],
    pair_text: Sequence[Any],
) -> dict[str, float]:
    model.eval()
    correct = 0
    invalid_kl = []
    flips = 0
    pair_both = 0
    for index in indices:
        text = text_data[index]["text_features"]
        original = model(alignment_data[index]["pre_tokens"], text).scores
        correct += int(int(original.argmax().item()) == 0)
        for key in ("blank_pre", "noise_pre"):
            scores = model(controls[index][key], text).scores
            invalid_kl.append(float(uniformity_loss(scores).item()))
    for source, pair in pairs.items():
        text = pair_text[int(pair["_cache_index"])] ["text_features"]
        partner = int(pair["counterfactual_index"])
        target = int(pair["counterfactual_target_index"])
        source_scores = model(alignment_data[source]["pre_tokens"], text).scores
        partner_scores = model(alignment_data[partner]["pre_tokens"], text).scores
        a = int(source_scores.argmax().item())
        b = int(partner_scores.argmax().item())
        flips += int(a != b)
        pair_both += int(a == 0 and b == target)
    return {
        "accuracy": correct / max(1, len(indices)),
        "invalid_uniform_kl": float(np.mean(invalid_kl)),
        "counterfactual_both_accuracy": pair_both / max(1, len(pairs)),
        "counterfactual_flip_rate": flips / max(1, len(pairs)),
    }


def train_stage_c(
    model: ThreeStageVisualJEV,
    train_alignment_data: Sequence[Any],
    train_controls: Sequence[Any],
    train_text_data: Sequence[Any],
    train_pairs: dict[int, dict[str, Any]],
    train_pair_text: Sequence[Any],
    validation_alignment_data: Sequence[Any],
    validation_controls: Sequence[Any],
    validation_text_data: Sequence[Any],
    validation_indices: Sequence[int],
    validation_pairs: dict[int, dict[str, Any]],
    validation_pair_text: Sequence[Any],
    settings: dict[str, Any],
    seed: int,
    *,
    train_alignment: bool,
    label: str,
) -> tuple[list[dict[str, Any]], int]:
    optimizer = decision_optimizer(model, settings, train_alignment)
    weights = VisualJEVV3LossWeights(
        counterfactual=settings["lambda_counterfactual"],
        invalid_image=settings["lambda_invalid_image"],
        text_null=settings["lambda_text_null"],
        cross_image_rank=settings["lambda_cross_image_rank"],
    )
    rng = random.Random(seed)
    history = []
    best_state = None
    best_score = -float("inf")
    best_epoch = 0
    allowed = set(validation_indices)
    val_pairs = subset_pairs(validation_pairs, allowed)
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = list(range(len(train_alignment_data)))
        rng.shuffle(order)
        sums = {"total": [], "candidate": [], "counterfactual": [], "invalid_image": [], "text_null": [], "cross_image_rank": []}
        for position, index in enumerate(order):
            item = train_alignment_data[index]
            text = train_text_data[index]["text_features"]
            wrong_index = order[(position + 1) % len(order)]
            pair = train_pairs.get(index)
            if pair is not None and wrong_index == int(pair["counterfactual_index"]):
                wrong_index = order[(position + 2) % len(order)]
            optimizer.zero_grad(set_to_none=True)
            original = model(item["pre_tokens"], text).scores
            wrong = model(train_alignment_data[wrong_index]["pre_tokens"], text).scores
            blank = model(train_controls[index]["blank_pre"], text).scores
            noise = model(train_controls[index]["noise_pre"], text).scores
            null = model(item["pre_tokens"], text_null_features(text, mode="mean")).scores
            candidate = candidate_pair_loss(original[:1], original[1:].unsqueeze(0))
            if pair is not None:
                partner_index = int(pair["counterfactual_index"])
                pair_candidates = train_pair_text[int(pair["_cache_index"])] ["text_features"]
                source_scores = model(item["pre_tokens"], pair_candidates).scores
                partner_scores = model(
                    train_alignment_data[partner_index]["pre_tokens"], pair_candidates
                ).scores
                counterfactual = counterfactual_visual_dependency_loss(
                    source_scores,
                    partner_scores,
                    image1_target=0,
                    image2_target=int(pair["counterfactual_target_index"]),
                    margin=settings["counterfactual_margin"],
                )
            else:
                counterfactual = original.sum() * 0.0
            invalid = mean_uniformity_loss((blank, noise, wrong))
            null_loss = uniformity_loss(null)
            rank = cross_image_ranking_loss(
                original, wrong, margin=settings["rank_margin"]
            )
            losses = combine_v3_losses(
                candidate=candidate,
                counterfactual=counterfactual,
                invalid_image=invalid,
                text_null=null_loss,
                cross_image_rank=rank,
                weights=weights,
            )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            for name in sums:
                sums[name].append(float(getattr(losses, name).item()))
        metrics = stage_c_selection(
            model,
            validation_alignment_data,
            validation_controls,
            validation_text_data,
            validation_indices,
            val_pairs,
            validation_pair_text,
        )
        selection = (
            metrics["accuracy"]
            + 0.5 * metrics["counterfactual_both_accuracy"]
            + 0.25 * metrics["counterfactual_flip_rate"]
            - 0.25 * metrics["invalid_uniform_kl"]
        )
        row = {
            "stage": label,
            "epoch": epoch,
            **{f"train_{name}_loss": float(np.mean(values)) for name, values in sums.items()},
            **{f"validation_{name}": value for name, value in metrics.items()},
            "validation_selection_score": selection,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if selection > best_score:
            best_score = selection
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Stage C did not produce a checkpoint")
    model.load_state_dict(best_state)
    return history, best_epoch


def train_meanpool(
    model: MeanPoolJEV,
    train_data: Sequence[Any],
    validation_data: Sequence[Any],
    validation_indices: Sequence[int],
    settings: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    rng = random.Random(seed)
    history = []
    best_state = None
    best_accuracy = -1.0
    best_epoch = 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = list(range(len(train_data)))
        rng.shuffle(order)
        losses = []
        for index in order:
            item = train_data[index]
            optimizer.zero_grad(set_to_none=True)
            scores = model(item["visual_tokens"], item["text_features"])
            loss = candidate_pair_loss(scores[:1], scores[1:].unsqueeze(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        correct = 0
        with torch.no_grad():
            for index in validation_indices:
                item = validation_data[index]
                correct += int(int(model(item["visual_tokens"], item["text_features"]).argmax().item()) == 0)
        accuracy = correct / max(1, len(validation_indices))
        row = {"stage": "meanpool", "epoch": epoch, "train_loss": float(np.mean(losses)), "validation_accuracy": accuracy}
        history.append(row)
        print(json.dumps(row), flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history, best_epoch


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("data_root", "pair_root", "paper_root", "v2_checkpoint"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/visual_jev_v3_paper.json"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    started = time.time()
    paper_root = Path(config["paper_root"])
    checkpoint_root = paper_root / "checkpoints"
    results_root = Path(config.get("results_root", "experiments/results")).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(config["data_root"])
    pair_root = Path(config["pair_root"])
    train_path = data_root / "splits" / "train.jsonl"
    validation_path = data_root / "splits" / "validation.jsonl"
    train_records = load_records(train_path, data_root)
    validation_records = load_records(validation_path, data_root)
    train_sha = file_sha256(train_path)
    validation_sha = file_sha256(validation_path)
    if args.skip_cache:
        caches = {
            name: open_shards(paper_root / "features" / f"{name}-shards", count)
            for name, count in (
                ("train", len(train_records)),
                ("validation", len(validation_records)),
                ("train_controls", len(train_records)),
                ("validation_controls", len(validation_records)),
            )
        }
        if any(value is None for value in caches.values()):
            raise RuntimeError("--skip-cache requested but a paper cache is missing")
    else:
        caches = prepare_alignment_caches(
            config, train_records, validation_records, train_sha, validation_sha
        )
    if args.prepare_only:
        print(json.dumps({"prepared": True, "paper_root": str(paper_root)}))
        return
    train_features = open_cached_dataset(
        data_root / "features" / "train.pt", source_sha256=train_sha, item_key="features"
    )
    validation_features = open_cached_dataset(
        data_root / "features" / "validation.pt", source_sha256=validation_sha, item_key="features"
    )
    train_pair_text = open_cached_dataset(
        pair_root / "features" / "train-pair-text.pt",
        source_sha256=file_sha256(pair_root / "pairs" / "train.jsonl"),
        item_key="pair_text",
    )
    validation_pair_text = open_cached_dataset(
        pair_root / "features" / "validation-pair-text.pt",
        source_sha256=file_sha256(pair_root / "pairs" / "validation.jsonl"),
        item_key="pair_text",
    )
    if any(x is None for x in (train_features, validation_features, train_pair_text, validation_pair_text)):
        raise RuntimeError("existing V2/V3 feature caches are incomplete")
    train_pairs = load_pairs(pair_root / "pairs" / "train.jsonl", len(train_records))
    validation_pairs = load_pairs(pair_root / "pairs" / "validation.jsonl", len(validation_records))
    modulus = config["split"]["validation_modulus"]
    validation_indices = [i for i in range(len(validation_records)) if i % modulus == config["split"]["validation_remainder"]]
    test_indices = [i for i in range(len(validation_records)) if i % modulus == config["split"]["test_remainder"]]

    v2_model, v2_payload = VisualJEVV2.from_checkpoint(config["v2_checkpoint"], map_location="cpu")
    alignment_config = AlignmentAdapterConfig(
        pre_merger_dim=caches["train"][0]["pre_tokens"].shape[-1],
        teacher_dim=caches["train"][0]["teacher_tokens"].shape[-1],
        hidden_dim=1024,
    )
    model = ThreeStageVisualJEV(alignment_config, v2_model.config).to(config["device"])
    model.decision.load_state_dict(v2_model.state_dict())
    del v2_model

    histories: dict[str, Any] = {}
    stage_a_history, stage_a_epoch = train_stage_a(
        model, caches["train"], caches["validation"], validation_indices, config["stage_a"], config["seed"]
    )
    histories["stage_a"] = stage_a_history
    stage_a_state = copy.deepcopy(model.alignment.state_dict())
    torch.save(
        {
            "model_type": "visual-jev-v3-stage-a-alignment",
            "config": asdict(alignment_config),
            "state_dict": stage_a_state,
            "best_epoch": stage_a_epoch,
        },
        checkpoint_root / "stage_a_alignment.pt",
    )

    stage_b_history, stage_b_epoch = train_stage_b(
        model,
        caches["train"], train_features,
        caches["validation"], validation_features,
        validation_indices, config["stage_b"], config["seed"] + 1,
        train_alignment=False, label="B",
    )
    histories["stage_b"] = stage_b_history
    model.save_checkpoint(
        checkpoint_root / "v3_without_stage_c.pt",
        stage="B", step=config["stage_b"]["epochs"] * len(train_records),
        metadata={"best_epoch": stage_b_epoch, "stage_a_best_epoch": stage_a_epoch},
    )
    stage_b_state = copy.deepcopy(model.state_dict())

    stage_c_history, stage_c_epoch = train_stage_c(
        model,
        caches["train"], caches["train_controls"], train_features,
        train_pairs, train_pair_text,
        caches["validation"], caches["validation_controls"], validation_features,
        validation_indices, validation_pairs, validation_pair_text,
        config["stage_c"], config["seed"] + 2,
        train_alignment=bool(config["stage_c"].get("finetune_alignment", True)), label="C",
    )
    histories["stage_c"] = stage_c_history
    model.save_checkpoint(
        checkpoint_root / "v3_full.pt",
        stage="C", step=(config["stage_b"]["epochs"] + config["stage_c"]["epochs"]) * len(train_records),
        metadata={"best_epoch": stage_c_epoch, "stage_a_best_epoch": stage_a_epoch, "stage_b_best_epoch": stage_b_epoch},
    )

    no_a = ThreeStageVisualJEV(alignment_config, model.decision.config).to(config["device"])
    old_v2, _ = VisualJEVV2.from_checkpoint(config["v2_checkpoint"], map_location=config["device"])
    no_a.decision.load_state_dict(old_v2.state_dict())
    del old_v2
    no_a_b_history, no_a_b_epoch = train_stage_b(
        no_a,
        caches["train"], train_features,
        caches["validation"], validation_features,
        validation_indices, config["stage_b"], config["seed"] + 3,
        train_alignment=True, label="B_without_A",
    )
    no_a_c_history, no_a_c_epoch = train_stage_c(
        no_a,
        caches["train"], caches["train_controls"], train_features,
        train_pairs, train_pair_text,
        caches["validation"], caches["validation_controls"], validation_features,
        validation_indices, validation_pairs, validation_pair_text,
        config["stage_c"], config["seed"] + 4,
        train_alignment=True, label="C_without_A",
    )
    histories["without_stage_a"] = no_a_b_history + no_a_c_history
    no_a.save_checkpoint(
        checkpoint_root / "v3_without_stage_a.pt",
        stage="C_without_A",
        step=(config["stage_b"]["epochs"] + config["stage_c"]["epochs"]) * len(train_records),
        metadata={"stage_b_best_epoch": no_a_b_epoch, "stage_c_best_epoch": no_a_c_epoch},
    )

    meanpool = MeanPoolJEV(
        train_features[0]["visual_tokens"].shape[-1],
        train_features[0]["text_features"].shape[-1],
        config["adapter_dim"],
    ).to(config["device"])
    meanpool_history, meanpool_epoch = train_meanpool(
        meanpool, train_features, validation_features, validation_indices,
        config["stage_b"], config["seed"] + 5,
    )
    histories["meanpool"] = meanpool_history
    torch.save(
        {
            "model_type": "meanpool-mlp-jev",
            "vision_dim": train_features[0]["visual_tokens"].shape[-1],
            "text_dim": train_features[0]["text_features"].shape[-1],
            "hidden_dim": config["adapter_dim"],
            "state_dict": meanpool.state_dict(),
            "best_epoch": meanpool_epoch,
        },
        checkpoint_root / "meanpool_jev.pt",
    )

    provenance = {
        "config": config,
        "data": {
            "train_records": len(train_records),
            "validation_records": len(validation_indices),
            "test_records": len(test_indices),
            "train_pairs": len(train_pairs),
            "validation_pairs": len(subset_pairs(validation_pairs, set(validation_indices))),
            "test_pairs": len(subset_pairs(validation_pairs, set(test_indices))),
            "train_sha256": train_sha,
            "validation_sha256": validation_sha,
        },
        "parameters": {
            "alignment": model.alignment.trainable_parameter_count,
            "decision": model.decision.trainable_parameter_count,
            "full": model.trainable_parameter_count,
            "meanpool": sum(p.numel() for p in meanpool.parameters()),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(config["device"]) if torch.cuda.is_available() else None,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        },
        "training_seconds": time.time() - started,
        "v2_checkpoint_step": v2_payload.get("step"),
        "best_epochs": {
            "stage_a": stage_a_epoch,
            "stage_b": stage_b_epoch,
            "stage_c": stage_c_epoch,
            "without_stage_a_b": no_a_b_epoch,
            "without_stage_a_c": no_a_c_epoch,
            "meanpool": meanpool_epoch,
        },
    }
    (results_root / "visual_jev_v3_training_history.json").write_text(
        json.dumps(histories, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (results_root / "visual_jev_v3_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
