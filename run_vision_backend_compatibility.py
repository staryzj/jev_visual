"""Train the same Visual-JEV decision code with structurally different vision encoders.

This is a compatibility proof, not a claim that ImageNet encoders match a
Qwen3-VL vision tower.  Hyperparameters and acceptance thresholds are fixed
before test evaluation.  Test data never selects a checkpoint.
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
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from jev.benchmark_v1 import BenchmarkExample, read_jsonl
from jev.vision_backends import (
    BACKEND_NAMES,
    FrozenVisionBackend,
    TextOnlyCandidateScorer,
    batched_candidate_scores,
    cached_weight_sha256,
)
from run_benchmark_v1_experiment import variable_metrics
from scripts.run_visual_jev_v2_experiment import DiskFeatureDataset, file_sha256
from visual_jev_v3 import VisualJEVV3, VisualJEVV3Config, uniformity_loss

SEED = 20260928
SPLITS = ("train", "validation", "test")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def literature_basis() -> list[dict[str, Any]]:
    return [
        {
            "title": "Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs",
            "venue": "NeurIPS 2024",
            "url": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/9ee3a664ccfeabc0da16ac6f1f1cfe59-Abstract-Conference.html",
            "applied": "evaluate different frozen visual representations behind a shared decision interface",
        },
        {
            "title": "BRAVE: Broadening the visual encoding of vision-language models",
            "venue": "ECCV 2024",
            "url": "https://arxiv.org/abs/2404.07204",
            "applied": "treat the visual encoder as a replaceable frozen expert and train a small bridge",
        },
        {
            "title": "Honeybee: Locality-enhanced Projector for Multimodal LLM",
            "venue": "CVPR 2024",
            "url": "https://openaccess.thecvf.com/content/CVPR2024/html/Cha_Honeybee_Locality-enhanced_Projector_for_Multimodal_LLM_CVPR_2024_paper.html",
            "applied": "preserve local patch/grid tokens and avoid a fixed global-vector-only contract",
        },
        {
            "title": "MoVA: Adapting Mixture of Vision Experts to Multimodal Context",
            "venue": "NeurIPS 2024",
            "url": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/bb0fea29f7aa6ede17e906ac6a225f34-Abstract-Conference.html",
            "applied": "validate structurally different visual experts because no encoder dominates every task",
        },
        {
            "title": "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features",
            "venue": "2025",
            "url": "https://arxiv.org/abs/2502.14786",
            "applied": "future backend contract includes variable token counts and native-resolution dense features",
        },
    ]


def load_manifests(root: Path) -> dict[str, list[BenchmarkExample]]:
    result = {split: list(read_jsonl(root / f"{split}.jsonl")) for split in SPLITS}
    expected = {"train": 2048, "validation": 128, "test": 128}
    actual = {name: len(rows) for name, rows in result.items()}
    if actual != expected:
        raise RuntimeError(f"unexpected compatibility manifest counts: {actual}")
    if any(len(row.candidates) != 3 for rows in result.values() for row in rows):
        raise RuntimeError("COCO compatibility proof expects exactly three candidates")
    return result


def source_text_features(
    examples: dict[str, list[BenchmarkExample]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    train_cache = DiskFeatureDataset(Path("data/visual-jev-v2/features/train-shards"), 2048)
    validation_cache = DiskFeatureDataset(
        Path("data/visual-jev-v2/features/validation-shards"), 256
    )
    result = {}
    for split, rows in examples.items():
        features, labels = [], []
        source = train_cache if split == "train" else validation_cache
        for row in rows:
            index = int(row.metadata["legacy_index"])
            order = torch.tensor(row.metadata["candidate_permutation"], dtype=torch.long)
            features.append(source[index]["text_features"][order].float())
            labels.append(row.label)
        result[split] = (torch.stack(features), torch.tensor(labels, dtype=torch.long))
    return result


def cache_is_valid(
    path: Path,
    manifest_sha: str,
    backend: str,
    count: int,
    *,
    weight_id: str | None = None,
    weight_sha256: str | None = None,
) -> bool:
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return False
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    valid = (
        payload.get("manifest_sha256") == manifest_sha
        and payload.get("backend") == backend
        and payload.get("count") == count
        and payload.get("tensor_sha256") == sha256(path)
    )
    if weight_id is not None:
        valid = valid and payload.get("weight_id") == weight_id
    if weight_sha256 is not None:
        valid = valid and payload.get("weight_sha256") == weight_sha256
    return valid


def prepare_backend_features(
    backend_name: str,
    examples: dict[str, list[BenchmarkExample]],
    manifest_root: Path,
    output: Path,
    device: str,
    extraction_batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    feature_root = output / "features" / backend_name
    controls_path = feature_root / "controls.pt"
    metadata_path = feature_root / "backend.json"
    existing_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    cached_weight_id = existing_metadata.get("weight_id")
    cached_weight_sha = existing_metadata.get("weight_sha256")
    loaded: dict[str, torch.Tensor] = {}
    all_valid = bool(cached_weight_id and cached_weight_sha)
    for split, rows in examples.items():
        path = feature_root / f"{split}.pt"
        source_sha = file_sha256(manifest_root / f"{split}.jsonl")
        if cache_is_valid(
            path,
            source_sha,
            backend_name,
            len(rows),
            weight_id=cached_weight_id,
            weight_sha256=cached_weight_sha,
        ):
            loaded[split] = torch.load(path, map_location="cpu", weights_only=True)[
                "visual_tokens"
            ]
        else:
            all_valid = False
    if all_valid and controls_path.is_file() and metadata_path.is_file():
        return (
            loaded,
            torch.load(controls_path, map_location="cpu", weights_only=True),
            json.loads(metadata_path.read_text(encoding="utf-8")),
        )

    backend = FrozenVisionBackend(backend_name, device)
    feature_root.mkdir(parents=True, exist_ok=True)
    backend_metadata = backend.metadata()
    backend_metadata["weight_sha256"] = cached_weight_sha256(backend.spec.weight_url)
    for split, rows in examples.items():
        path = feature_root / f"{split}.pt"
        source_sha = file_sha256(manifest_root / f"{split}.jsonl")
        if cache_is_valid(
            path,
            source_sha,
            backend_name,
            len(rows),
            weight_id=backend_metadata["weight_id"],
            weight_sha256=backend_metadata["weight_sha256"],
        ):
            loaded[split] = torch.load(path, map_location="cpu", weights_only=True)[
                "visual_tokens"
            ]
            continue
        batches = []
        progress = tqdm(
            range(0, len(rows), extraction_batch_size),
            desc=f"extract {backend_name} {split}",
            unit="batch",
            dynamic_ncols=True,
        )
        for start in progress:
            images = []
            for row in rows[start : start + extraction_batch_size]:
                with Image.open(row.image) as source:
                    image = source.convert("RGB")
                    image.load()
                images.append(image)
            batches.append(backend.encode(images).to(torch.float16))
        visual_tokens = torch.cat(batches, dim=0).contiguous()
        torch.save({"visual_tokens": visual_tokens}, path)
        write_json(
            path.with_suffix(".json"),
            {
                "format": "open-jev-vision-backend-cache-v1",
                "backend": backend_name,
                "weight_id": backend_metadata["weight_id"],
                "weight_sha256": backend_metadata["weight_sha256"],
                "manifest_sha256": source_sha,
                "count": len(rows),
                "shape": list(visual_tokens.shape),
                "dtype": str(visual_tokens.dtype),
                "tensor_sha256": sha256(path),
            },
        )
        loaded[split] = visual_tokens

    blank = Image.new("RGB", (512, 512), (255, 255, 255))
    rng = np.random.default_rng(SEED)
    noise = Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8), "RGB")
    controls = {
        "blank": backend.encode([blank])[0].to(torch.float16),
        "noise": backend.encode([noise])[0].to(torch.float16),
    }
    torch.save(controls, controls_path)
    write_json(metadata_path, backend_metadata)
    return loaded, controls, backend_metadata


def scores_to_list(scores: torch.Tensor) -> list[torch.Tensor]:
    return [row.detach().float().cpu() for row in scores]


@torch.no_grad()
def predict_visual(
    model: VisualJEVV3,
    visual: torch.Tensor,
    text: torch.Tensor,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in range(0, len(text), batch_size):
        outputs.append(
            batched_candidate_scores(
                model,
                visual[start : start + batch_size].to(device).float(),
                text[start : start + batch_size].to(device),
            ).float().cpu()
        )
    return torch.cat(outputs)


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> float:
    p = left.float().softmax(-1)
    q = right.float().softmax(-1)
    middle = 0.5 * (p + q)
    value = 0.5 * (
        (p * (p.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(-1)
        + (q * (q.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(-1)
    )
    return float(value.mean().item())


def condition_report(
    original: torch.Tensor,
    condition: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    metrics = variable_metrics(scores_to_list(condition), labels.tolist())
    metrics["js_from_original"] = js_divergence(original, condition)
    metrics["flip_rate_from_original"] = float(
        (original.argmax(-1) != condition.argmax(-1)).float().mean().item()
    )
    return metrics


def train_text_only(
    text: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: str,
    epochs: int,
    batch_size: int,
) -> tuple[TextOnlyCandidateScorer, dict[str, Any]]:
    set_seed()
    width = text["train"][0].shape[-1]
    model = TextOnlyCandidateScorer(width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    best_state = None
    best_key = (-1.0, float("-inf"))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(text["train"][0]))
        losses = []
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            features = text["train"][0][index].to(device)
            labels = text["train"][1][index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_scores = model(text["validation"][0].to(device)).float().cpu()
        val = variable_metrics(scores_to_list(val_scores), text["validation"][1].tolist())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **val})
        key = (val["accuracy"], -val["nll"])
        if key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_scores = model(text["test"][0].to(device)).float().cpu()
    return model, {
        "validation_selection": {"accuracy": best_key[0], "negative_nll": best_key[1]},
        "test": variable_metrics(scores_to_list(test_scores), text["test"][1].tolist()),
        "history": history,
    }


def train_backend(
    backend_name: str,
    features: dict[str, torch.Tensor],
    controls: dict[str, torch.Tensor],
    text: dict[str, tuple[torch.Tensor, torch.Tensor]],
    backend_metadata: dict[str, Any],
    text_only: dict[str, Any],
    output: Path,
    device: str,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    set_seed()
    vision_dim = features["train"].shape[-1]
    text_dim = text["train"][0].shape[-1]
    config = VisualJEVV3Config(
        vision_dim=vision_dim,
        text_dim=text_dim,
        adapter_dim=256,
        num_heads=8,
        dropout=0.1,
    )
    model = VisualJEVV3(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    best_state = None
    best_key = (-1.0, float("-inf"))
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(features["train"]))
        losses, candidate_losses, invalid_losses = [], [], []
        for start in tqdm(
            range(0, len(order), batch_size),
            desc=f"train {backend_name} {epoch}/{epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        ):
            index = order[start : start + batch_size]
            visual = features["train"][index].to(device).float()
            candidate_text = text["train"][0][index].to(device)
            labels = text["train"][1][index].to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = batched_candidate_scores(model, visual, candidate_text)
            candidate = F.cross_entropy(scores, labels)
            invalid = scores.new_zeros(())
            # COCO captions require the image.  Easy-to-hard invalid-image
            # regularisation uses fixed blank first and adds fixed noise later;
            # random wrong images are never training targets.
            if epoch > epochs // 3:
                control_names = ["blank"] if epoch <= 2 * epochs // 3 else ["blank", "noise"]
                control_losses = []
                for name in control_names:
                    control = controls[name][None].expand(len(index), -1, -1).to(device).float()
                    control_scores = batched_candidate_scores(model, control, candidate_text)
                    control_losses.append(uniformity_loss(control_scores))
                invalid = torch.stack(control_losses).mean()
            loss = candidate + 0.1 * invalid
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            candidate_losses.append(float(candidate.item()))
            invalid_losses.append(float(invalid.item()))
        val_scores = predict_visual(
            model, features["validation"], text["validation"][0], device, batch_size
        )
        val = variable_metrics(scores_to_list(val_scores), text["validation"][1].tolist())
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_candidate_loss": float(np.mean(candidate_losses)),
            "train_invalid_loss": float(np.mean(invalid_losses)),
            **{f"validation_{key}": value for key, value in val.items()},
        }
        history.append(row)
        print(json.dumps({"backend": backend_name, **row}), flush=True)
        key = (val["accuracy"], -val["nll"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()

    test_visual = features["test"]
    test_text, test_labels = text["test"]
    original = predict_visual(model, test_visual, test_text, device, batch_size)
    blank_visual = controls["blank"][None].expand(len(test_visual), -1, -1)
    noise_visual = controls["noise"][None].expand(len(test_visual), -1, -1)
    blank = predict_visual(model, blank_visual, test_text, device, batch_size)
    noise = predict_visual(model, noise_visual, test_text, device, batch_size)
    wrong_index = torch.roll(torch.arange(len(test_visual)), shifts=1)
    wrong = predict_visual(model, test_visual[wrong_index], test_text, device, batch_size)
    conditions = {
        "original": condition_report(original, original, test_labels),
        "blank": condition_report(original, blank, test_labels),
        "noise": condition_report(original, noise, test_labels),
        "wrong_image": condition_report(original, wrong, test_labels),
    }
    original_correct = conditions["original"]["mean_correct_probability"]
    invalid_correct = 0.5 * (
        conditions["blank"]["mean_correct_probability"]
        + conditions["noise"]["mean_correct_probability"]
    )
    invalid_js = 0.5 * (
        conditions["blank"]["js_from_original"]
        + conditions["noise"]["js_from_original"]
    )
    first_loss = float(np.mean([row["train_loss"] for row in history[:2]]))
    last_loss = float(np.mean([row["train_loss"] for row in history[-2:]]))
    acceptance = {
        "same_training_code": True,
        "training_loss_decreased": last_loss < 0.9 * first_loss,
        "beats_text_only_by_5pt": (
            conditions["original"]["accuracy"] >= text_only["test"]["accuracy"] + 0.05
        ),
        "invalid_confidence_drop_at_least_3pt": original_correct - invalid_correct >= 0.03,
        "invalid_js_at_least_0_01": invalid_js >= 0.01,
    }
    acceptance["passed"] = all(acceptance.values())
    checkpoint = output / "checkpoints" / f"{backend_name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "backend-agnostic-visual-jev-v3",
            "backend": backend_metadata,
            "decision_config": config.__dict__,
            "state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "seed": SEED,
        },
        checkpoint,
    )
    return {
        "backend": backend_name,
        "backend_metadata": backend_metadata,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "best_epoch": best_epoch,
        "validation_selection": {"accuracy": best_key[0], "negative_nll": best_key[1]},
        "conditions": conditions,
        "text_only_test": text_only["test"],
        "visual_lift_vs_text_only_accuracy": (
            conditions["original"]["accuracy"] - text_only["test"]["accuracy"]
        ),
        "invalid_confidence_drop": original_correct - invalid_correct,
        "invalid_mean_js": invalid_js,
        "training_loss_first_two_epochs": first_loss,
        "training_loss_last_two_epochs": last_loss,
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
        "--output",
        type=Path,
        default=Path("experiments/results/benchmark_v1/vision_backend_compatibility"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backends", default=",".join(BACKEND_NAMES))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--extraction-batch-size", type=int, default=16)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    examples = load_manifests(args.manifest_root)
    text = source_text_features(examples)
    write_json(root / "literature_basis.json", literature_basis())
    text_model, text_only = train_text_only(
        text, args.device, args.epochs, args.batch_size
    )
    del text_model
    write_json(root / "text_only_baseline.json", text_only)

    results = []
    for backend_name in [name.strip() for name in args.backends.split(",") if name.strip()]:
        features, controls, backend_metadata = prepare_backend_features(
            backend_name,
            examples,
            args.manifest_root,
            root,
            args.device,
            args.extraction_batch_size,
        )
        result = train_backend(
            backend_name,
            features,
            controls,
            text,
            backend_metadata,
            text_only,
            root,
            args.device,
            args.epochs,
            args.batch_size,
        )
        write_json(root / "backend_results" / f"{backend_name}.json", result)
        results.append(result)
        del features, controls
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall_passed = len(results) >= 2 and all(row["acceptance"]["passed"] for row in results)
    summary = {
        "objective": "same Visual-JEV training code learns image+text decisions with structurally different frozen vision encoders",
        "seed": SEED,
        "manifest_counts": {name: len(rows) for name, rows in examples.items()},
        "selection_policy": "validation accuracy, then validation NLL; test reporting only",
        "acceptance_pre_registered": {
            "training_loss": "last two epochs < 90% of first two",
            "visual_lift": "test accuracy >= text-only + 5 percentage points",
            "invalid_confidence": "original correct probability exceeds blank/noise mean by >= 0.03",
            "invalid_js": "original-vs-blank/noise mean JS >= 0.01",
            "minimum_backends": 2,
        },
        "text_only": text_only["test"],
        "backends": results,
        "overall_passed": overall_passed,
        "runtime_seconds": time.time() - started,
        "claim_boundary": "compatibility proof on COCO hard-negative 2048/128/128; not a universal quality claim",
    }
    write_json(root / "summary.json", summary)
    rows = []
    for result in results:
        original = result["conditions"]["original"]
        rows.append({
            "backend": result["backend"],
            "test_accuracy": original["accuracy"],
            "test_macro_f1": original["macro_f1"],
            "test_nll": original["nll"],
            "text_only_accuracy": result["text_only_test"]["accuracy"],
            "visual_lift_accuracy": result["visual_lift_vs_text_only_accuracy"],
            "invalid_confidence_drop": result["invalid_confidence_drop"],
            "invalid_mean_js": result["invalid_mean_js"],
            "wrong_image_accuracy": result["conditions"]["wrong_image"]["accuracy"],
            "passed": result["acceptance"]["passed"],
            "checkpoint": result["checkpoint"],
        })
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "overall_passed": overall_passed,
        "text_only_accuracy": text_only["test"]["accuracy"],
        "backends": [
            {
                "name": row["backend"],
                "accuracy": row["conditions"]["original"]["accuracy"],
                "visual_lift": row["visual_lift_vs_text_only_accuracy"],
                "invalid_confidence_drop": row["invalid_confidence_drop"],
                "passed": row["acceptance"]["passed"],
            }
            for row in results
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
