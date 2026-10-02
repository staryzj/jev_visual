"""Full-manifest Visual-JEV V3 training with balanced domains and gated Stage C.

This is the production/full-data counterpart of ``run_fix_negative_transfer.py``.
Paper-inspired losses are project-local reimplementations, not official code.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from jev.multidomain import preference_anchor_loss, visual_dependency_masks
from run_fix_negative_transfer import (
    DOMAINS,
    SEED,
    Context,
    candidate_loss,
    domain_indices,
    estimate_scienceqa_dependency,
    selection_score,
    set_seed,
    write_json,
)
from scripts.run_visual_jev_v2_experiment import file_sha256, open_cached_dataset
from train_visual_jev_v2 import load_records
from train_visual_jev_v3 import load_pairs, open_shards
from visual_jev_v3 import counterfactual_visual_dependency_loss, mean_uniformity_loss
from visual_jev_v3_pipeline import ThreeStageVisualJEV


def _load_pair_training(context: Context) -> tuple[Any, list[tuple[int, dict[str, Any]]], Any]:
    pair_path = context.legacy_pair_root / "pairs" / "train.jsonl"
    records = load_records(
        context.legacy_data / "splits" / "train.jsonl",
        context.legacy_data,
        require_images=False,
    )
    alignment = open_shards(context.legacy_root / "features" / "train-shards", len(records))
    pairs = sorted(load_pairs(pair_path, len(records)).items(), key=lambda row: row[0])
    text = open_cached_dataset(
        context.legacy_pair_root / "features" / "train-pair-text.pt",
        source_sha256=file_sha256(pair_path),
        item_key="pair_text",
    )
    if alignment is None or text is None or not pairs:
        raise RuntimeError("semantic-pair training features are incomplete")
    return alignment, pairs, text


def _balanced_sample_weights(context: Context) -> dict[str, float]:
    counts = Counter(row.dataset for row in context.examples["train"])
    missing = [domain for domain in DOMAINS if counts[domain] == 0]
    if missing:
        raise RuntimeError(f"full train manifest is missing domains: {missing}")
    total = sum(counts.values())
    return {domain: total / (len(DOMAINS) * counts[domain]) for domain in DOMAINS}


def _ordered_indices(context: Context, rng: random.Random, maximum: int) -> list[int]:
    indices = list(range(len(context.examples["train"])))
    rng.shuffle(indices)
    return indices if maximum <= 0 else indices[:maximum]


def _step_optimizer(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
) -> None:
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _save_model(
    model: ThreeStageVisualJEV,
    path: Path,
    *,
    epoch: int,
    stage: str,
    initial_checkpoint: Path,
) -> None:
    model.save_checkpoint(
        path,
        stage=stage,
        step=epoch,
        metadata={
            "seed": SEED,
            "profile": "full_manifest",
            "initial_checkpoint": str(initial_checkpoint.resolve()),
            "domain_weighting": "all samples once per epoch; inverse-frequency loss weights",
            "paper_implementation": "mDPO/MFPO-inspired project-local adaptation; not official",
        },
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = Context(
        args.device,
        manifest_root=args.manifest_root.resolve(),
        experiment_root=args.experiment_root.resolve(),
    )
    model, _ = ThreeStageVisualJEV.from_checkpoint(
        args.initial_checkpoint.resolve(), map_location="cpu"
    )
    model = model.to(args.device)
    for parameter in model.alignment.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decision.parameters():
        parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.decision.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-2)
    weights = _balanced_sample_weights(context)
    grouped = domain_indices(context.examples["train"])
    rng = random.Random(f"{SEED}:full")
    dependency = estimate_scienceqa_dependency(context, model.eval())
    pair_alignment, pair_rows, pair_text = _load_pair_training(context)

    validation = context.evaluate(model, "validation")
    pair_validation = context.pair_metrics(model, "validation")
    base, _, _ = selection_score(validation)
    best_score = base + 0.5 * pair_validation["both_directions_accuracy"] + 0.25 * pair_validation["prediction_flip_rate"]
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = []
    sampled = Counter()
    components: dict[str, list[float]] = defaultdict(list)
    started = time.time()
    total_epochs = args.candidate_epochs + args.stagec_epochs

    for global_epoch in range(1, total_epochs + 1):
        stage_c = global_epoch > args.candidate_epochs
        stage_epoch = global_epoch - args.candidate_epochs if stage_c else global_epoch
        hard_phase = stage_c and stage_epoch > args.easy_stagec_epochs
        stage_name = "stageC-hard" if hard_phase else "stageC-easy" if stage_c else "candidate"
        order = _ordered_indices(context, rng, args.max_train_samples)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss: list[float] = []
        epoch_components: dict[str, list[float]] = defaultdict(list)
        progress = tqdm(
            enumerate(order, start=1),
            total=len(order),
            desc=f"full {stage_name} {global_epoch}/{total_epochs}",
            unit="sample",
            dynamic_ncols=True,
        )
        for position, index in progress:
            example = context.examples["train"][index]
            item = context.features["train"][index]
            raw_candidate = candidate_loss(model, item, example)
            total = raw_candidate
            epoch_components["candidate"].append(float(raw_candidate.detach().item()))

            if stage_c:
                required = bool(dependency["train_visual_required"].get(index, False))
                masks = visual_dependency_masks(example.dataset, scienceqa_visual_required=required)
                original = model(item["pre_tokens"], item["text_features"]).scores
                if masks["invalid"]:
                    blank = model(context.controls["blank_pre"], item["text_features"]).scores
                    noise = model(context.controls["noise_pre"], item["text_features"]).scores
                    invalid = mean_uniformity_loss((blank, noise))
                    total = total + args.lambda_invalid * invalid
                    epoch_components["invalid"].append(float(invalid.detach().item()))
                if hard_phase and masks["preference"] and len(grouped[example.dataset]) > 1:
                    wrong_index = rng.choice(grouped[example.dataset])
                    if wrong_index == index:
                        wrong_index = grouped[example.dataset][
                            (grouped[example.dataset].index(index) + 1) % len(grouped[example.dataset])
                        ]
                    wrong = model(
                        context.features["train"][wrong_index]["pre_tokens"],
                        item["text_features"],
                    ).scores
                    preference = preference_anchor_loss(original, wrong, label=example.label)
                    total = total + args.lambda_preference * preference
                    epoch_components["image_preference"].append(float(preference.detach().item()))

                if hard_phase and position % args.pair_interval == 0:
                    source, pair = pair_rows[((global_epoch - 1) * len(order) + position) // args.pair_interval % len(pair_rows)]
                    partner = int(pair["counterfactual_index"])
                    target = int(pair["counterfactual_target_index"])
                    text = pair_text[int(pair["_cache_index"])]["text_features"]
                    left = model(pair_alignment[source]["pre_tokens"], text).scores
                    right = model(pair_alignment[partner]["pre_tokens"], text).scores
                    counterfactual = counterfactual_visual_dependency_loss(
                        left, right, image1_target=0, image2_target=target, margin=0.2
                    )
                    anchor = 0.5 * (
                        preference_anchor_loss(left, right, label=0)
                        + preference_anchor_loss(right, left, label=target)
                    )
                    total = total + args.lambda_counterfactual * counterfactual + args.lambda_anchor * anchor
                    epoch_components["semantic_counterfactual"].append(float(counterfactual.detach().item()))
                    epoch_components["semantic_anchor"].append(float(anchor.detach().item()))

            weighted = total * weights[example.dataset]
            (weighted / args.grad_accumulation).backward()
            if position % args.grad_accumulation == 0 or position == len(order):
                _step_optimizer(optimizer, parameters)
            value = float(total.detach().item())
            epoch_loss.append(value)
            sampled[example.dataset] += 1
            progress.set_postfix(
                loss=f"{value:.4f}",
                mean=f"{np.mean(epoch_loss[-200:]):.4f}",
                domain=example.dataset,
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )

        for key, values in epoch_components.items():
            components[key].extend(values)
        model.eval()
        validation = context.evaluate(model, "validation")
        pair_validation = context.pair_metrics(model, "validation")
        base, worst, macro = selection_score(validation)
        score = base + 0.5 * pair_validation["both_directions_accuracy"] + 0.25 * pair_validation["prediction_flip_rate"]
        elapsed = time.time() - started
        row = {
            "epoch": global_epoch,
            "stage": stage_name,
            "train_samples": len(order),
            "train_loss": float(np.mean(epoch_loss)),
            "train_components": {key: float(np.mean(values)) for key, values in epoch_components.items()},
            "validation_accuracy": validation["overall"]["accuracy"],
            "validation_macro_f1": validation["overall"]["macro_f1"],
            "validation_worst_domain_accuracy": worst,
            "validation_macro_domain_accuracy": macro,
            "validation_pair_both": pair_validation["both_directions_accuracy"],
            "validation_pair_flip": pair_validation["prediction_flip_rate"],
            "selection_score": score,
            "elapsed_seconds": elapsed,
            "eta_seconds": elapsed / global_epoch * (total_epochs - global_epoch),
        }
        history.append(row)
        print(json.dumps({"full_training": row}, ensure_ascii=False), flush=True)
        _save_model(
            model, output / "last.pt", epoch=global_epoch, stage=stage_name,
            initial_checkpoint=args.initial_checkpoint,
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            _save_model(
                model, output / "best.pt", epoch=global_epoch, stage=stage_name,
                initial_checkpoint=args.initial_checkpoint,
            )
        write_json(output / "training_history.json", {"history": history})

    model.load_state_dict(best_state)
    if not (output / "best.pt").is_file():
        _save_model(
            model, output / "best.pt", epoch=0, stage="initial-best",
            initial_checkpoint=args.initial_checkpoint,
        )
    test = context.evaluate(model, "test")
    pair_test = context.pair_metrics(model, "test")
    result = {
        "profile": "full_manifest",
        "manifest_root": str(args.manifest_root.resolve()),
        "experiment_root": str(args.experiment_root.resolve()),
        "initial_checkpoint": str(args.initial_checkpoint.resolve()),
        "best_checkpoint": str((output / "best.pt").resolve()),
        "seed": SEED,
        "candidate_epochs": args.candidate_epochs,
        "stagec_epochs": args.stagec_epochs,
        "max_train_samples": args.max_train_samples,
        "domain_weights": weights,
        "actual_sample_counts": dict(sampled),
        "mean_loss_components": {key: float(np.mean(values)) for key, values in components.items()},
        "history": history,
        "test": test,
        "semantic_counterfactual": pair_test,
        "runtime_seconds": time.time() - started,
        "paper_implementation": "mDPO/MFPO-inspired project-local adaptation; not official",
    }
    write_json(output / "full_training_results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path("data/benchmark_v1_full/manifests"))
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments/benchmark_v1_full"))
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path("experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/best_model_results/visual_jev_v3_best.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/results/benchmark_v1/fix_negative_transfer/full_run"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-epochs", type=int, default=2)
    parser.add_argument("--stagec-epochs", type=int, default=3)
    parser.add_argument("--easy-stagec-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--pair-interval", type=int, default=8)
    parser.add_argument("--lambda-invalid", type=float, default=1.0)
    parser.add_argument("--lambda-preference", type=float, default=0.25)
    parser.add_argument("--lambda-counterfactual", type=float, default=1.0)
    parser.add_argument("--lambda-anchor", type=float, default=0.25)
    parser.add_argument(
        "--max-train-samples", type=int, default=0,
        help="0 means every training sample; positive values are smoke-test only.",
    )
    args = parser.parse_args()
    if args.candidate_epochs < 0 or args.stagec_epochs < 0 or args.candidate_epochs + args.stagec_epochs == 0:
        parser.error("at least one training epoch is required")
    if args.grad_accumulation <= 0 or args.pair_interval <= 0:
        parser.error("grad accumulation and pair interval must be positive")
    result = train(args)
    print(json.dumps({
        "status": "complete",
        "best_checkpoint": result["best_checkpoint"],
        "test_overall": result["test"]["overall"],
        "pair": result["semantic_counterfactual"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
