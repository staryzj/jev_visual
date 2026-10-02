"""Semantic counterfactual proof for swappable Visual-JEV vision backends.

Each pair presents exactly the same two candidate texts with two images and
opposite labels.  A deterministic text-only scorer cannot get both directions
correct, so pair-both directly tests whether the backend supplies usable visual
information.  Validation pairs select checkpoints; held-out pairs report once.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from jev.vision_backends import BACKEND_NAMES, batched_candidate_scores
from run_vision_backend_compatibility import (
    load_manifests,
    prepare_backend_features,
    set_seed,
    sha256,
    write_json,
)
from scripts.run_visual_jev_v2_experiment import DiskFeatureDataset
from visual_jev_v3 import VisualJEVV3, VisualJEVV3Config, uniformity_loss

SEED = 20260928


def load_pair_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def feature_index(
    examples: dict[str, list[Any]], features: dict[str, torch.Tensor]
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    train = {
        int(row.metadata["legacy_index"]): features["train"][position]
        for position, row in enumerate(examples["train"])
    }
    validation = {}
    for split in ("validation", "test"):
        for position, row in enumerate(examples[split]):
            validation[int(row.metadata["legacy_index"])] = features[split][position]
    if set(train) != set(range(2048)) or set(validation) != set(range(256)):
        raise RuntimeError("backend cache does not cover the full legacy COCO indices")
    return train, validation


def pair_tensors(
    rows: list[dict[str, Any]],
    text_cache: DiskFeatureDataset,
    visual: dict[int, torch.Tensor],
    selected: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    indices = selected if selected is not None else list(range(len(rows)))
    return {
        "source": torch.stack([visual[int(rows[i]["source_index"])] for i in indices]),
        "counterfactual": torch.stack(
            [visual[int(rows[i]["counterfactual_index"])] for i in indices]
        ),
        "text": torch.stack([text_cache[i]["text_features"].float() for i in indices]),
    }


def pair_metrics(source: torch.Tensor, counterfactual: torch.Tensor) -> dict[str, float]:
    source_prediction = source.argmax(-1)
    counter_prediction = counterfactual.argmax(-1)
    source_correct = source_prediction.eq(0)
    counter_correct = counter_prediction.eq(1)
    source_probability = source.float().softmax(-1)[:, 0]
    counter_probability = counterfactual.float().softmax(-1)[:, 1]
    return {
        "pair_count": int(len(source)),
        "source_accuracy": float(source_correct.float().mean().item()),
        "counterfactual_accuracy": float(counter_correct.float().mean().item()),
        "both_directions_accuracy": float(
            (source_correct & counter_correct).float().mean().item()
        ),
        "prediction_flip_rate": float(
            source_prediction.ne(counter_prediction).float().mean().item()
        ),
        "source_correct_probability": float(source_probability.mean().item()),
        "counterfactual_correct_probability": float(counter_probability.mean().item()),
        "source_target_margin": float((source[:, 0] - source[:, 1]).mean().item()),
        "counterfactual_target_margin": float(
            (counterfactual[:, 1] - counterfactual[:, 0]).mean().item()
        ),
    }


@torch.no_grad()
def predict_pairs(
    model: VisualJEVV3,
    data: dict[str, torch.Tensor],
    device: str,
    batch_size: int,
    control: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    source_scores, counter_scores = [], []
    for start in range(0, len(data["text"]), batch_size):
        text = data["text"][start : start + batch_size].to(device)
        source = data["source"][start : start + batch_size]
        counter = data["counterfactual"][start : start + batch_size]
        if control is not None:
            source = control[None].expand(len(text), -1, -1)
            counter = source
        source_scores.append(
            batched_candidate_scores(model, source.to(device).float(), text).float().cpu()
        )
        counter_scores.append(
            batched_candidate_scores(model, counter.to(device).float(), text).float().cpu()
        )
    return torch.cat(source_scores), torch.cat(counter_scores)


def train_pair_model(
    backend_name: str,
    initial_checkpoint: Path,
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    test: dict[str, torch.Tensor],
    controls: dict[str, torch.Tensor],
    backend_metadata: dict[str, Any],
    output: Path,
    device: str,
    epochs: int,
    batch_size: int,
    seed: int = SEED,
) -> dict[str, Any]:
    set_seed(seed)
    payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
    model = VisualJEVV3(VisualJEVV3Config(**payload["decision_config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    best_state = None
    best_key = (-1.0, -1.0)
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train["text"]))
        epoch_values = []
        for start in tqdm(
            range(0, len(order), batch_size),
            desc=f"pair {backend_name} {epoch}/{epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        ):
            index = order[start : start + batch_size]
            text = train["text"][index].to(device)
            source_visual = train["source"][index].to(device).float()
            counter_visual = train["counterfactual"][index].to(device).float()
            optimizer.zero_grad(set_to_none=True)
            source = batched_candidate_scores(model, source_visual, text)
            counter = batched_candidate_scores(model, counter_visual, text)
            source_labels = torch.zeros(len(index), dtype=torch.long, device=device)
            counter_labels = torch.ones(len(index), dtype=torch.long, device=device)
            candidate = 0.5 * (
                F.cross_entropy(source, source_labels)
                + F.cross_entropy(counter, counter_labels)
            )
            preference = source.new_zeros(())
            anchor = source.new_zeros(())
            if epoch > epochs // 3:
                preference = 0.5 * (
                    F.relu(0.2 - source[:, 0] + counter[:, 0]).mean()
                    + F.relu(0.2 - counter[:, 1] + source[:, 1]).mean()
                )
                anchor = 0.5 * (
                    F.softplus(-source[:, 0]).mean()
                    + F.softplus(-counter[:, 1]).mean()
                )
            invalid = source.new_zeros(())
            if epoch > 2 * epochs // 3:
                invalid_losses = []
                for name in ("blank", "noise"):
                    visual = controls[name][None].expand(len(index), -1, -1).to(device).float()
                    invalid_scores = batched_candidate_scores(model, visual, text)
                    invalid_losses.append(uniformity_loss(invalid_scores))
                invalid = torch.stack(invalid_losses).mean()
            loss = candidate + 0.25 * preference + 0.05 * anchor + 0.1 * invalid
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_values.append(
                [
                    float(loss.item()),
                    float(candidate.item()),
                    float(preference.item()),
                    float(anchor.item()),
                    float(invalid.item()),
                ]
            )
        validation_scores = predict_pairs(model, validation, device, batch_size)
        metrics = pair_metrics(*validation_scores)
        means = np.mean(epoch_values, axis=0)
        row = {
            "epoch": epoch,
            "train_total": float(means[0]),
            "train_candidate": float(means[1]),
            "train_preference": float(means[2]),
            "train_anchor": float(means[3]),
            "train_invalid": float(means[4]),
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        print(json.dumps({"backend": backend_name, **row}), flush=True)
        key = (
            metrics["both_directions_accuracy"],
            0.5 * (metrics["source_accuracy"] + metrics["counterfactual_accuracy"]),
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    original_scores = predict_pairs(model, test, device, batch_size)
    blank_scores = predict_pairs(model, test, device, batch_size, controls["blank"])
    noise_scores = predict_pairs(model, test, device, batch_size, controls["noise"])
    original = pair_metrics(*original_scores)
    blank = pair_metrics(*blank_scores)
    noise = pair_metrics(*noise_scores)
    acceptance = {
        "text_only_pair_both_upper_bound_is_zero": True,
        "source_accuracy_at_least_0_60": original["source_accuracy"] >= 0.60,
        "counterfactual_accuracy_at_least_0_60": original["counterfactual_accuracy"] >= 0.60,
        "pair_both_at_least_0_40": original["both_directions_accuracy"] >= 0.40,
        "prediction_flip_at_least_0_40": original["prediction_flip_rate"] >= 0.40,
        "original_pair_both_exceeds_invalid": original["both_directions_accuracy"] > max(
            blank["both_directions_accuracy"], noise["both_directions_accuracy"]
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    checkpoint = output / "checkpoints" / f"{backend_name}_seed{seed}_pair_proof.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "backend-agnostic-visual-jev-v3-pair-proof",
            "backend": backend_metadata,
            "decision_config": model.config.__dict__,
            "state_dict": model.state_dict(),
            "seed": seed,
            "best_epoch": best_epoch,
            "initial_checkpoint": str(initial_checkpoint.resolve()),
        },
        checkpoint,
    )
    return {
        "backend": backend_name,
        "seed": seed,
        "backend_metadata": backend_metadata,
        "initial_checkpoint": str(initial_checkpoint.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "best_epoch": best_epoch,
        "validation_selection_key": list(best_key),
        "test": {"original": original, "blank": blank, "noise": noise},
        "acceptance": acceptance,
        "history": history,
    }


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
    parser.add_argument("--backends", default=",".join(BACKEND_NAMES))
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    root = args.compatibility_root.resolve()
    output = root / "semantic_pair_proof"
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
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
    if set(validation_indices) & set(test_indices) or not validation_indices or not test_indices:
        raise RuntimeError("semantic pair validation/test split is invalid")

    results = []
    for backend_name in [name.strip() for name in args.backends.split(",") if name.strip()]:
        features, controls, metadata = prepare_backend_features(
            backend_name,
            examples,
            args.manifest_root,
            root,
            args.device,
            16,
        )
        train_visual, validation_visual = feature_index(examples, features)
        train = pair_tensors(train_rows, train_text, train_visual)
        validation = pair_tensors(
            validation_rows, validation_text, validation_visual, validation_indices
        )
        test = pair_tensors(validation_rows, validation_text, validation_visual, test_indices)
        initial = root / "checkpoints" / f"{backend_name}.pt"
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
        )
        write_json(output / "backend_results" / f"{backend_name}.json", result)
        results.append(result)
        del features, controls, train, validation, test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "objective": "prove two structurally different frozen vision encoders provide usable visual evidence to the same JEV training code",
        "seed": SEED,
        "train_pair_count": len(train_rows),
        "validation_pair_count": len(validation_indices),
        "test_pair_count": len(test_indices),
        "split_policy": "legacy validation semantic pairs split by source_index parity; validation selects checkpoint; odd test reports only",
        "text_only_pair_both_upper_bound": 0.0,
        "acceptance_pre_registered": {
            "source_accuracy": 0.60,
            "counterfactual_accuracy": 0.60,
            "pair_both": 0.40,
            "prediction_flip": 0.40,
            "invalid_control": "original pair-both must exceed blank and noise",
        },
        "backends": results,
        "overall_passed": len(results) >= 2
        and all(result["acceptance"]["passed"] for result in results),
        "runtime_seconds": time.time() - started,
        "claim_boundary": "demonstrates trainability and visual dependence for two encoder families; does not promise equal accuracy for every future encoder",
    }
    write_json(output / "summary.json", summary)
    rows = []
    for result in results:
        original = result["test"]["original"]
        rows.append(
            {
                "backend": result["backend"],
                **original,
                "blank_pair_both": result["test"]["blank"]["both_directions_accuracy"],
                "noise_pair_both": result["test"]["noise"]["both_directions_accuracy"],
                "passed": result["acceptance"]["passed"],
                "checkpoint": result["checkpoint"],
            }
        )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "overall_passed": summary["overall_passed"],
                "backends": [
                    {
                        "name": row["backend"],
                        "original": row["test"]["original"],
                        "passed": row["acceptance"]["passed"],
                    }
                    for row in results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
