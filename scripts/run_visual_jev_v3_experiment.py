"""Train and diagnose Visual-JEV V3 from frozen Visual-JEV V2 feature caches."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from jev.serving import load_predictor
from scripts.run_visual_jev_v2_experiment import (
    DiskFeatureDataset,
    encode_controls,
    file_sha256,
    open_cached_dataset,
)
from train_visual_jev_v2 import format_candidate, load_records
from visual_jev_v2 import Qwen3VLFeatureExtractor, VisualJEVV2, candidate_pair_loss
from visual_jev_v3 import (
    VisualJEVV3,
    VisualJEVV3LossWeights,
    combine_v3_losses,
    counterfactual_visual_dependency_loss,
    cross_image_ranking_loss,
    mean_uniformity_loss,
    text_null_features,
    uniformity_loss,
)

FeatureItem = dict[str, torch.Tensor]


def load_pairs(path: Path, record_count: int) -> dict[int, dict[str, Any]]:
    pairs: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            pair = json.loads(line)
            source = int(pair["source_index"])
            partner = int(pair["counterfactual_index"])
            target = int(pair["counterfactual_target_index"])
            if not 0 <= source < record_count or not 0 <= partner < record_count:
                raise ValueError(f"{path}:{line_number}: pair index out of range")
            if source == partner:
                raise ValueError(f"{path}:{line_number}: pair reuses one record")
            if target <= 0:
                raise ValueError(f"{path}:{line_number}: target did not flip")
            pair["_cache_index"] = len(pairs)
            pairs[source] = pair
    if not pairs:
        raise ValueError(f"counterfactual pair file is empty: {path}")
    return pairs


def encode_pair_texts(
    pairs: dict[int, dict[str, Any]],
    extractor: Qwen3VLFeatureExtractor,
    *,
    cache_path: Path,
    source_sha256: str,
) -> DiskFeatureDataset:
    cached = open_cached_dataset(
        cache_path, source_sha256=source_sha256, item_key="pair_text"
    )
    if cached is not None:
        return cached
    directory = cache_path.parent / f"{cache_path.stem}-shards"
    directory.mkdir(parents=True, exist_ok=True)
    ordered = sorted(pairs.values(), key=lambda pair: pair["_cache_index"])
    for index, pair in enumerate(ordered):
        candidates = pair["candidate_set"]
        prompts = [
            format_candidate("Which object is present in the image?", candidate)
            for candidate in candidates
        ]
        item = {"text_features": extractor.encode_candidates(prompts).cpu()}
        torch.save(item, directory / f"{index:06d}.pt")
        if (index + 1) % 128 == 0 or index + 1 == len(ordered):
            print(f"encoded counterfactual text features: {index + 1}/{len(ordered)}")
    manifest = {
        "format": "visual-jev-disk-shards-v1",
        "source_sha256": source_sha256,
        "count": len(ordered),
        "item_key": "pair_text",
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return DiskFeatureDataset(directory, len(ordered))


def pooled_visual(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError("visual tokens must have shape [tokens, dim]")
    return tokens.float().mean(dim=0)


def feature_distance(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a_pool = pooled_visual(a)
    b_pool = pooled_visual(b)
    l2 = (a_pool - b_pool).norm().item()
    cosine_distance = (1.0 - F.cosine_similarity(a_pool, b_pool, dim=0)).item()
    return l2, cosine_distance


def audit_control_cache(
    features: Sequence[FeatureItem],
    controls: Sequence[FeatureItem],
    *,
    sample_count: int = 32,
    fail_threshold: float = 1e-6,
) -> dict[str, Any]:
    """Prove controls are not aliases of original or neighboring cache shards."""
    count = min(sample_count, len(features), len(controls))
    if count < 2:
        raise ValueError("at least two cached samples are required for cache audit")
    rows: dict[str, list[float]] = {
        "blank_l2": [],
        "blank_cosine_distance": [],
        "noise_l2": [],
        "noise_cosine_distance": [],
        "image_swap_l2": [],
        "image_swap_cosine_distance": [],
    }
    for index in range(count):
        original = features[index]["visual_tokens"]
        blank_l2, blank_cos = feature_distance(original, controls[index]["blank"])
        noise_l2, noise_cos = feature_distance(original, controls[index]["noise"])
        swap = features[(index + 1) % len(features)]["visual_tokens"]
        swap_l2, swap_cos = feature_distance(original, swap)
        rows["blank_l2"].append(blank_l2)
        rows["blank_cosine_distance"].append(blank_cos)
        rows["noise_l2"].append(noise_l2)
        rows["noise_cosine_distance"].append(noise_cos)
        rows["image_swap_l2"].append(swap_l2)
        rows["image_swap_cosine_distance"].append(swap_cos)
    report: dict[str, Any] = {"samples_checked": count, "passed": True}
    for name, values in rows.items():
        report[f"mean_{name}"] = float(np.mean(values))
        report[f"min_{name}"] = float(np.min(values))
    aliases = [
        name
        for name in ("blank_l2", "noise_l2", "image_swap_l2")
        if report[f"min_{name}"] <= fail_threshold
    ]
    report["alias_failures"] = aliases
    report["passed"] = not aliases
    if aliases:
        raise RuntimeError(
            "control cache may reuse original features; zero-distance controls: "
            + ", ".join(aliases)
        )
    return report


def _condition_inputs(
    features: Sequence[FeatureItem],
    controls: Sequence[FeatureItem],
    index: int,
    condition: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    item = features[index]
    visual = item["visual_tokens"]
    text = item["text_features"]
    if condition == "blank":
        visual = controls[index]["blank"]
    elif condition == "noise":
        visual = controls[index]["noise"]
    elif condition == "image_swap":
        visual = features[(index + 1) % len(features)]["visual_tokens"]
    elif condition == "text_null":
        text = text_null_features(text, mode="zero")
    elif condition != "original":
        raise ValueError(f"unknown condition: {condition}")
    return visual, text


@torch.no_grad()
def collect_condition(
    model: VisualJEVV2,
    features: Sequence[FeatureItem],
    controls: Sequence[FeatureItem],
    condition: str,
) -> dict[str, Any]:
    model.eval()
    logits = []
    predictions = []
    visual_l2 = []
    visual_cosine = []
    for index in range(len(features)):
        item = features[index]
        visual, text = _condition_inputs(features, controls, index, condition)
        output = model(visual, text)
        logits.append(output.scores.detach().float().cpu())
        predictions.append(int(output.scores.argmax().item()))
        l2, cosine = feature_distance(item["visual_tokens"], visual)
        visual_l2.append(l2)
        visual_cosine.append(cosine)
        del item, visual, text, output
    score_tensor = torch.stack(logits)
    probabilities = score_tensor.softmax(dim=-1)
    maxima = score_tensor.max(dim=-1, keepdim=True).values
    ties = torch.isclose(score_tensor, maxima, atol=1e-6, rtol=1e-6)
    tie_aware_correct = ties[:, 0].float() / ties.sum(dim=-1).float()
    hard_argmax_accuracy = float(
        (score_tensor.argmax(-1) == 0).float().mean().item()
    )
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    uniform_kl = (
        probabilities
        * (
            probabilities.clamp_min(1e-12).log()
            + math.log(score_tensor.shape[1])
        )
    ).sum(-1)
    margins = score_tensor[:, 0] - score_tensor[:, 1:].max(dim=1).values
    return {
        "condition": condition,
        "accuracy": float(tie_aware_correct.mean().item()),
        "hard_argmax_accuracy": hard_argmax_accuracy,
        "stochastic_accuracy": float(probabilities[:, 0].mean().item()),
        "mean_positive_probability": float(probabilities[:, 0].mean().item()),
        "mean_entropy": float(entropy.mean().item()),
        "normalized_entropy": float(
            (entropy / math.log(score_tensor.shape[1])).mean().item()
        ),
        "mean_positive_score": float(score_tensor[:, 0].mean().item()),
        "mean_margin": float(margins.mean().item()),
        "mean_uniform_kl": float(uniform_kl.mean().item()),
        "mean_visual_feature_l2": float(np.mean(visual_l2)),
        "mean_visual_feature_cosine_distance": float(np.mean(visual_cosine)),
        "_logits": score_tensor,
        "_predictions": torch.tensor(predictions),
    }


@torch.no_grad()
def evaluate_controls(
    model: VisualJEVV2,
    features: Sequence[FeatureItem],
    controls: Sequence[FeatureItem],
) -> dict[str, dict[str, Any]]:
    results = {
        condition: collect_condition(model, features, controls, condition)
        for condition in ("original", "blank", "noise", "image_swap", "text_null")
    }
    original_logits = results["original"]["_logits"]
    original_predictions = results["original"]["_predictions"]
    for metrics in results.values():
        logits = metrics.pop("_logits")
        predictions = metrics.pop("_predictions")
        delta = logits - original_logits
        metrics["prediction_flip_rate_vs_original"] = float(
            (predictions != original_predictions).float().mean().item()
        )
        metrics["mean_logit_delta_l2_vs_original"] = float(
            delta.norm(dim=-1).mean().item()
        )
        metrics["mean_abs_logit_delta_vs_original"] = float(delta.abs().mean().item())
    return results


@torch.no_grad()
def evaluate_counterfactual_pairs(
    model: VisualJEVV2,
    features: Sequence[FeatureItem],
    pairs: dict[int, dict[str, Any]],
    pair_text_features: Sequence[FeatureItem],
) -> dict[str, float]:
    model.eval()
    source_correct = 0
    counterfactual_correct = 0
    both_correct = 0
    prediction_flips = 0
    losses = []
    for source_index, pair in pairs.items():
        source = features[source_index]
        partner = features[int(pair["counterfactual_index"])]
        target = int(pair["counterfactual_target_index"])
        text = pair_text_features[int(pair["_cache_index"])]["text_features"]
        source_scores = model(
            source["visual_tokens"], text
        ).scores
        partner_scores = model(
            partner["visual_tokens"], text
        ).scores
        source_prediction = int(source_scores.argmax().item())
        partner_prediction = int(partner_scores.argmax().item())
        source_ok = source_prediction == 0
        partner_ok = partner_prediction == target
        source_correct += int(source_ok)
        counterfactual_correct += int(partner_ok)
        both_correct += int(source_ok and partner_ok)
        prediction_flips += int(source_prediction != partner_prediction)
        losses.append(
            counterfactual_visual_dependency_loss(
                source_scores,
                partner_scores,
                image1_target=0,
                image2_target=target,
            ).item()
        )
    count = len(pairs)
    return {
        "pair_count": count,
        "source_accuracy": source_correct / count,
        "counterfactual_accuracy": counterfactual_correct / count,
        "both_directions_accuracy": both_correct / count,
        "prediction_flip_rate": prediction_flips / count,
        "mean_loss": float(np.mean(losses)),
    }


def validation_selection_score(metrics: dict[str, dict[str, Any]]) -> float:
    invalid_kl = np.mean(
        [metrics[name]["mean_uniform_kl"] for name in ("blank", "noise", "image_swap")]
    )
    return float(metrics["original"]["accuracy"] - 0.25 * invalid_kl)


def train_adapter(
    model: VisualJEVV3,
    train_features: Sequence[FeatureItem],
    train_controls: Sequence[FeatureItem],
    train_pairs: dict[int, dict[str, Any]],
    train_pair_text_features: Sequence[FeatureItem],
    validation_features: Sequence[FeatureItem],
    validation_controls: Sequence[FeatureItem],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    weights: VisualJEVV3LossWeights,
    uniform_divergence: str,
    counterfactual_loss_type: str,
    counterfactual_margin: float,
    rank_margin: float,
    max_grad_norm: float,
    seed: int,
    max_steps: int | None,
) -> tuple[torch.optim.Optimizer, list[dict[str, float]], int, int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    rng = random.Random(seed)
    history: list[dict[str, float]] = []
    best_epoch = 0
    best_state = None
    best_optimizer_state = None
    best_key = float("-inf")
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_features)))
        rng.shuffle(order)
        epoch_values: dict[str, list[float]] = {
            name: []
            for name in (
                "total",
                "candidate",
                "counterfactual",
                "invalid_image",
                "text_null",
                "cross_image_rank",
            )
        }
        for position, index in enumerate(order):
            item = train_features[index]
            pair = train_pairs.get(index)
            if pair is not None:
                partner_index = int(pair["counterfactual_index"])
                counterfactual_target = int(pair["counterfactual_target_index"])
                partner = train_features[partner_index]
                pair_text = train_pair_text_features[
                    int(pair["_cache_index"])
                ]["text_features"]
            else:
                partner = None
                pair_text = None
                counterfactual_target = 1
            wrong_index = order[(position + 1) % len(order)]
            if wrong_index == index or (
                pair is not None and wrong_index == int(pair["counterfactual_index"])
            ):
                wrong_index = order[(position + 2) % len(order)]
            wrong = train_features[wrong_index]
            invalid = train_controls[index]

            optimizer.zero_grad(set_to_none=True)
            original_scores = model(
                item["visual_tokens"], item["text_features"]
            ).scores
            wrong_scores = model(
                wrong["visual_tokens"], item["text_features"]
            ).scores
            blank_scores = model(invalid["blank"], item["text_features"]).scores
            noise_scores = model(invalid["noise"], item["text_features"]).scores
            null_scores = model(
                item["visual_tokens"],
                text_null_features(item["text_features"], mode="mean"),
            ).scores

            candidate = candidate_pair_loss(
                original_scores[:1], original_scores[1:].unsqueeze(0)
            )
            source_pair_scores = None
            partner_scores = None
            if pair is not None and partner is not None and pair_text is not None:
                source_pair_scores = model(
                    item["visual_tokens"], pair_text
                ).scores
                partner_scores = model(
                    partner["visual_tokens"], pair_text
                ).scores
                counterfactual = counterfactual_visual_dependency_loss(
                    source_pair_scores,
                    partner_scores,
                    image1_target=0,
                    image2_target=counterfactual_target,
                    loss_type=counterfactual_loss_type,
                    margin=counterfactual_margin,
                )
            else:
                counterfactual = original_scores.sum() * 0.0
            invalid_image = mean_uniformity_loss(
                (blank_scores, noise_scores, wrong_scores),
                divergence=uniform_divergence,
            )
            text_null = uniformity_loss(
                null_scores, divergence=uniform_divergence
            )
            cross_rank = cross_image_ranking_loss(
                original_scores, wrong_scores, margin=rank_margin
            )
            losses = combine_v3_losses(
                candidate=candidate,
                counterfactual=counterfactual,
                invalid_image=invalid_image,
                text_null=text_null,
                cross_image_rank=cross_rank,
                weights=weights,
            )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            global_step += 1
            for name, values in epoch_values.items():
                values.append(float(getattr(losses, name).item()))
            del (
                item,
                pair,
                partner,
                pair_text,
                wrong,
                invalid,
                original_scores,
                wrong_scores,
                source_pair_scores,
                partner_scores,
                blank_scores,
                noise_scores,
                null_scores,
                losses,
            )
            if max_steps is not None and global_step >= max_steps:
                break

        validation = evaluate_controls(
            model, validation_features, validation_controls
        )
        selection_score = validation_selection_score(validation)
        row: dict[str, float] = {
            "epoch": float(epoch),
            "global_step": float(global_step),
            **{
                f"train_{name}_loss": float(np.mean(values))
                for name, values in epoch_values.items()
            },
            "validation_selection_score": selection_score,
            "validation_original_accuracy": validation["original"]["accuracy"],
            "validation_original_margin": validation["original"]["mean_margin"],
            "validation_blank_kl": validation["blank"]["mean_uniform_kl"],
            "validation_noise_kl": validation["noise"]["mean_uniform_kl"],
            "validation_swap_kl": validation["image_swap"]["mean_uniform_kl"],
            "validation_swap_flip_rate": validation["image_swap"][
                "prediction_flip_rate_vs_original"
            ],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if selection_score > best_key:
            best_key = selection_score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        if max_steps is not None and global_step >= max_steps:
            break
    if best_state is None or best_optimizer_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    optimizer.load_state_dict(best_optimizer_state)
    return optimizer, history, best_epoch, global_step


def save_curve(history: list[dict[str, float]], path: Path) -> None:
    from PIL import Image, ImageDraw

    if not history:
        raise ValueError("history must not be empty")
    image = Image.new("RGB", (1400, 560), "white")
    draw = ImageDraw.Draw(image)
    panels = ((70, 70, 670, 480), (760, 70, 1360, 480))
    series_groups = (
        (
            "Visual-JEV V3 training loss",
            (
                ("total", "train_total_loss", "#2563eb"),
                ("candidate", "train_candidate_loss", "#16a34a"),
                ("counterfactual", "train_counterfactual_loss", "#dc2626"),
            ),
        ),
        (
            "Validation dependence controls",
            (
                ("original accuracy", "validation_original_accuracy", "#2563eb"),
                ("swap flip rate", "validation_swap_flip_rate", "#7c3aed"),
                ("blank KL", "validation_blank_kl", "#ea580c"),
                ("noise KL", "validation_noise_kl", "#0891b2"),
            ),
        ),
    )
    for panel, (title, series) in zip(panels, series_groups):
        left, top, right, bottom = panel
        values = [float(row[key]) for _, key, _ in series for row in history]
        low = min(0.0, min(values))
        high = max(values)
        if high <= low:
            high = low + 1.0
        draw.rectangle(panel, outline="#334155", width=2)
        draw.text((left, 28), title, fill="#0f172a")
        draw.text((left - 5, bottom + 12), "epoch", fill="#475569")
        draw.text((left - 55, top - 5), f"{high:.3f}", fill="#475569")
        draw.text((left - 55, bottom - 8), f"{low:.3f}", fill="#475569")
        for series_index, (label, key, color) in enumerate(series):
            points = []
            for index, row in enumerate(history):
                x = left + (right - left) * (
                    index / max(1, len(history) - 1)
                )
                value = float(row[key])
                y = bottom - (bottom - top) * ((value - low) / (high - low))
                points.append((x, y))
            if len(points) == 1:
                x, y = points[0]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            else:
                draw.line(points, fill=color, width=4)
            legend_y = top + 12 + 24 * series_index
            draw.line((left + 12, legend_y + 6, left + 42, legend_y + 6), fill=color, width=4)
            draw.text((left + 50, legend_y), label, fill="#0f172a")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, default=Path("data/visual-jev-v2"))
    parser.add_argument("--v3-root", type=Path, default=Path("data/visual-jev-v3"))
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lambda-cf", type=float, default=1.0)
    parser.add_argument("--lambda-img", type=float, default=5.0)
    parser.add_argument("--lambda-txt", type=float, default=0.25)
    parser.add_argument("--lambda-rank", type=float, default=0.5)
    parser.add_argument("--uniform-divergence", choices=("kl", "js"), default="kl")
    parser.add_argument(
        "--counterfactual-loss", choices=("cross_entropy", "ranking"), default="cross_entropy"
    )
    parser.add_argument("--counterfactual-margin", type=float, default=0.2)
    parser.add_argument("--rank-margin", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path("checkpoints/visual-jev-v2-coco-2k.pt"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/visual-jev-v3-coco-2k.pt"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/visual-jev-v3-coco-2k.json")
    )
    parser.add_argument(
        "--curve", type=Path, default=Path("reports/figures/visual-jev-v3-training.png")
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"{args.device} requested but CUDA is unavailable")
    for name in ("lambda_cf", "lambda_img", "lambda_txt", "lambda_rank"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    v2_root = args.v2_root.expanduser().resolve()
    v3_root = args.v3_root.expanduser().resolve()
    train_path = v2_root / "splits" / "train.jsonl"
    validation_path = v2_root / "splits" / "validation.jsonl"
    train_records = load_records(train_path, v2_root)
    validation_records = load_records(validation_path, v2_root)
    train_pairs = load_pairs(v3_root / "pairs" / "train.jsonl", len(train_records))
    validation_pairs = load_pairs(
        v3_root / "pairs" / "validation.jsonl", len(validation_records)
    )
    train_pair_path = v3_root / "pairs" / "train.jsonl"
    validation_pair_path = v3_root / "pairs" / "validation.jsonl"

    train_sha = file_sha256(train_path)
    validation_sha = file_sha256(validation_path)
    train_features = open_cached_dataset(
        v2_root / "features" / "train.pt",
        source_sha256=train_sha,
        item_key="features",
    )
    validation_features = open_cached_dataset(
        v2_root / "features" / "validation.pt",
        source_sha256=validation_sha,
        item_key="features",
    )
    validation_controls = open_cached_dataset(
        v2_root / "features" / "validation-controls.pt",
        source_sha256=validation_sha,
        item_key="controls",
    )
    train_control_cache = v3_root / "features" / "train-controls.pt"
    train_controls = open_cached_dataset(
        train_control_cache, source_sha256=train_sha, item_key="controls"
    )
    train_pair_text_cache = v3_root / "features" / "train-pair-text.pt"
    validation_pair_text_cache = v3_root / "features" / "validation-pair-text.pt"
    train_pair_text_features = open_cached_dataset(
        train_pair_text_cache,
        source_sha256=file_sha256(train_pair_path),
        item_key="pair_text",
    )
    validation_pair_text_features = open_cached_dataset(
        validation_pair_text_cache,
        source_sha256=file_sha256(validation_pair_path),
        item_key="pair_text",
    )
    backbone_frozen = True
    if any(
        value is None
        for value in (
            train_controls,
            train_pair_text_features,
            validation_pair_text_features,
        )
    ):
        predictor = load_predictor(
            model_id=args.model,
            device=args.device,
            max_length=512,
            batch_size=3,
            vision=True,
            image_root=v2_root,
        )
        extractor = Qwen3VLFeatureExtractor(predictor.scorer.model)
        if train_controls is None:
            train_controls = encode_controls(
                train_records,
                extractor,
                cache_path=train_control_cache,
                source_sha256=train_sha,
            )
        if train_pair_text_features is None:
            train_pair_text_features = encode_pair_texts(
                train_pairs,
                extractor,
                cache_path=train_pair_text_cache,
                source_sha256=file_sha256(train_pair_path),
            )
        if validation_pair_text_features is None:
            validation_pair_text_features = encode_pair_texts(
                validation_pairs,
                extractor,
                cache_path=validation_pair_text_cache,
                source_sha256=file_sha256(validation_pair_path),
            )
        backbone_frozen = extractor.backbone_is_frozen
        del extractor, predictor
        gc.collect()
        torch.cuda.empty_cache()
    if any(
        value is None
        for value in (
            train_features,
            validation_features,
            validation_controls,
            train_controls,
            train_pair_text_features,
            validation_pair_text_features,
        )
    ):
        raise RuntimeError("required V2/V3 feature cache is unavailable")

    cache_audit = {
        "train": audit_control_cache(train_features, train_controls),
        "validation": audit_control_cache(validation_features, validation_controls),
    }
    print(json.dumps({"cache_audit": cache_audit}, indent=2), flush=True)

    baseline_model, baseline_payload = VisualJEVV2.from_checkpoint(
        args.init_checkpoint.expanduser().resolve(), map_location=args.device
    )
    baseline_model = baseline_model.to(args.device)
    baseline_controls = evaluate_controls(
        baseline_model, validation_features, validation_controls
    )
    baseline_counterfactual = evaluate_counterfactual_pairs(
        baseline_model,
        validation_features,
        validation_pairs,
        validation_pair_text_features,
    )
    model, init_payload = VisualJEVV3.from_checkpoint(
        args.init_checkpoint.expanduser().resolve(), map_location=args.device
    )
    model = model.to(args.device)
    del baseline_model
    gc.collect()

    weights = VisualJEVV3LossWeights(
        counterfactual=args.lambda_cf,
        invalid_image=args.lambda_img,
        text_null=args.lambda_txt,
        cross_image_rank=args.lambda_rank,
    )
    optimizer, history, best_epoch, global_step = train_adapter(
        model,
        train_features,
        train_controls,
        train_pairs,
        train_pair_text_features,
        validation_features,
        validation_controls,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        weights=weights,
        uniform_divergence=args.uniform_divergence,
        counterfactual_loss_type=args.counterfactual_loss,
        counterfactual_margin=args.counterfactual_margin,
        rank_margin=args.rank_margin,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    v3_controls = evaluate_controls(model, validation_features, validation_controls)
    v3_counterfactual = evaluate_counterfactual_pairs(
        model,
        validation_features,
        validation_pairs,
        validation_pair_text_features,
    )

    checkpoint_path = args.checkpoint.expanduser().resolve()
    model.save_checkpoint(
        checkpoint_path,
        optimizer=optimizer,
        step=global_step,
        metadata={
            "best_epoch": best_epoch,
            "initialized_from": str(args.init_checkpoint.expanduser().resolve()),
            "initialized_model_type": init_payload.get("model_type"),
            "backbone": args.model,
            "backbone_frozen": backbone_frozen,
            "loss_weights": asdict(weights),
            "uniform_divergence": args.uniform_divergence,
            "counterfactual_loss": args.counterfactual_loss,
            "counterfactual_margin": args.counterfactual_margin,
            "rank_margin": args.rank_margin,
        },
    )
    report: dict[str, Any] = {
        "model": "Visual-JEV V3",
        "config": asdict(model.config),
        "training": {
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "global_step": global_step,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss_formula": (
                "L_candidate + lambda_cf*L_counterfactual + "
                "lambda_img*L_invalid_image + lambda_txt*L_text_null + "
                "lambda_rank*L_cross_image"
            ),
            "loss_weights": asdict(weights),
            "uniform_divergence": args.uniform_divergence,
            "counterfactual_loss": args.counterfactual_loss,
        },
        "data": {
            "train_records": len(train_features),
            "validation_records": len(validation_features),
            "train_counterfactual_pairs": len(train_pairs),
            "validation_counterfactual_pairs": len(validation_pairs),
            "pair_manifest": str(v3_root / "manifest.json"),
            "strict_image_id_isolation": True,
        },
        "feature_cache": {
            "backbone_frozen": backbone_frozen,
            "reused_v2_train_validation": True,
            "incremental_train_controls": str(train_control_cache),
            "incremental_pair_text": {
                "train": str(train_pair_text_cache),
                "validation": str(validation_pair_text_cache),
            },
            "audit": cache_audit,
        },
        "v2_baseline": {
            "checkpoint": str(args.init_checkpoint.expanduser().resolve()),
            "checkpoint_step": baseline_payload.get("step"),
            "controls": baseline_controls,
            "counterfactual": baseline_counterfactual,
        },
        "v3": {
            "checkpoint": str(checkpoint_path),
            "controls": v3_controls,
            "counterfactual": v3_counterfactual,
        },
        "history": history,
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    curve_path = args.curve.expanduser().resolve()
    save_curve(history, curve_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote checkpoint: {checkpoint_path}")
    print(f"wrote report: {report_path}")
    print(f"wrote curve: {curve_path}")


if __name__ == "__main__":
    main()
