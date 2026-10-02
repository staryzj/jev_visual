"""Train one backbone-specific adapter against a fixed Visual-JEV decision head.

The frozen visual backbone is used only to build a deterministic feature cache.
The Qwen/JEV text features follow the original training cache, while the
backbone-specific alignment adapter is the only trainable module.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jev.benchmark_v1 import BenchmarkExample, read_jsonl
from jev.hf_vision_backends import BACKBONE_NAMES, HFVisionTokenExtractor
from jev.vision_backends import batched_candidate_scores
from run_benchmark_v1_experiment import variable_metrics
from visual_jev_v3 import uniformity_loss
from visual_jev_v3_pipeline import (
    AlignmentAdapterConfig,
    PreMergerAlignmentAdapter,
    ThreeStageVisualJEV,
)


SPLITS = ("train", "validation", "calibration", "test")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(
    manifest_root: Path,
    dataset_substr: str,
    limits: dict[str, int],
) -> dict[str, list[tuple[int, BenchmarkExample]]]:
    result = {}
    for split in SPLITS:
        rows = [
            (index, row)
            for index, row in enumerate(read_jsonl(manifest_root / f"{split}.jsonl"))
            if dataset_substr.casefold() in row.dataset.casefold()
        ][: limits[split]]
        if len(rows) != limits[split]:
            raise RuntimeError(
                f"{split} provides {len(rows)} matching rows, expected {limits[split]}"
            )
        candidate_counts = {len(row.candidates) for _, row in rows}
        if len(candidate_counts) != 1:
            raise RuntimeError(f"{split} candidate counts are not batchable: {candidate_counts}")
        result[split] = rows
    return result


def load_text_features(
    examples: dict[str, list[tuple[int, BenchmarkExample]]],
    qwen_feature_root: Path,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    result = {}
    for split, rows in examples.items():
        features, labels = [], []
        root = qwen_feature_root / f"{split}-shards"
        for index, row in rows:
            payload = torch.load(
                root / f"{index:06d}.pt", map_location="cpu", weights_only=True
            )
            features.append(payload["text_features"].float())
            labels.append(row.label)
        result[split] = (
            torch.stack(features),
            torch.tensor(labels, dtype=torch.long),
        )
    return result


def build_visual_cache(
    args: argparse.Namespace,
    examples: dict[str, list[tuple[int, BenchmarkExample]]],
    output: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    cache_root = output / "features"
    expected = {split: cache_root / f"{split}.pt" for split in SPLITS}
    controls_path = cache_root / "controls.pt"
    metadata_path = cache_root / "metadata.json"
    if all(path.is_file() for path in expected.values()) and controls_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("backbone") == args.backbone
            and metadata.get("model_path") == str(args.model.expanduser().resolve())
            and metadata.get("token_count") == args.token_count
            and metadata.get("counts") == {key: len(value) for key, value in examples.items()}
        ):
            visual = {
                split: torch.load(path, map_location="cpu", weights_only=True)["tokens"]
                for split, path in expected.items()
            }
            controls = torch.load(controls_path, map_location="cpu", weights_only=True)
            return visual, controls, metadata

    cache_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            pass
    started = time.perf_counter()
    extractor = HFVisionTokenExtractor(args.backbone, args.model, device=device)
    extraction_times = []
    visual = {}
    native_shapes: set[tuple[int, int]] = set()
    try:
        for split, rows in examples.items():
            tensors = []
            for position, (_, row) in enumerate(rows, 1):
                with Image.open(row.image) as source:
                    image = source.convert("RGB")
                    sync(device)
                    tick = time.perf_counter()
                    native = extractor.encode(image)
                    sync(device)
                    extraction_times.append((time.perf_counter() - tick) * 1000.0)
                native_shapes.add(tuple(native.shape))
                from jev.hf_vision_backends import resample_visual_tokens
                tensors.append(
                    resample_visual_tokens(native, args.token_count).to(torch.float16).cpu()
                )
                if position % 100 == 0:
                    print(f"{args.backbone} extract {split}: {position}/{len(rows)}", flush=True)
            visual[split] = torch.stack(tensors)
            torch.save({"tokens": visual[split]}, expected[split])

        blank = Image.new("RGB", (512, 512), (255, 255, 255))
        rng = np.random.default_rng(args.seed)
        noise = Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8))
        controls = {
            "blank": extractor.encode(blank, args.token_count).to(torch.float16).cpu(),
            "noise": extractor.encode(noise, args.token_count).to(torch.float16).cpu(),
        }
        torch.save(controls, controls_path)
        arr = np.asarray(extraction_times, dtype=np.float64)
        try:
            peak_gpu_memory_gib = (
                torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda" else 0.0
            )
        except RuntimeError:
            peak_gpu_memory_gib = None
        metadata = {
            "backbone": args.backbone,
            "model_path": str(args.model.expanduser().resolve()),
            "parameter_count": extractor.parameter_count,
            "token_count": args.token_count,
            "hidden_dim": int(visual["train"].shape[-1]),
            "native_shapes": [list(shape) for shape in sorted(native_shapes)],
            "counts": {key: len(value) for key, value in examples.items()},
            "mean_extraction_ms": float(arr.mean()),
            "p50_extraction_ms": float(np.percentile(arr, 50)),
            "p95_extraction_ms": float(np.percentile(arr, 95)),
            "total_extraction_seconds": time.perf_counter() - started,
            "peak_gpu_memory_gib": peak_gpu_memory_gib,
        }
        write_json(metadata_path, metadata)
        return visual, controls, metadata
    finally:
        extractor.close()


def align_batch(adapter: PreMergerAlignmentAdapter, visual: torch.Tensor) -> torch.Tensor:
    batch, tokens, width = visual.shape
    aligned = adapter(visual.reshape(batch * tokens, width))
    return aligned.reshape(batch, tokens, -1)


def predict(
    adapter: PreMergerAlignmentAdapter,
    decision: torch.nn.Module,
    visual: torch.Tensor,
    text: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    adapter.eval()
    decision.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(visual), batch_size):
            image = visual[start : start + batch_size].to(device).float()
            candidates = text[start : start + batch_size].to(device)
            aligned = align_batch(adapter, image)
            outputs.append(batched_candidate_scores(decision, aligned, candidates).float().cpu())
    return torch.cat(outputs)


def fit_temperature(scores: torch.Tensor, labels: torch.Tensor) -> float:
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)
    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(scores / temperature, labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    return variable_metrics([row for row in scores], labels.tolist())


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> float:
    p, q = left.softmax(-1), right.softmax(-1)
    middle = 0.5 * (p + q)
    value = 0.5 * (
        (p * (p.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(-1)
        + (q * (q.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(-1)
    )
    return float(value.mean().item())


def condition_metrics(
    original: torch.Tensor,
    changed: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    result = metrics(changed, labels)
    result["js_from_original"] = js_divergence(original, changed)
    result["flip_rate_from_original"] = float(
        (original.argmax(-1) != changed.argmax(-1)).float().mean().item()
    )
    return result


def latency_report(
    adapter: PreMergerAlignmentAdapter,
    decision: torch.nn.Module,
    visual: torch.Tensor,
    text: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    adapter.eval()
    decision.eval()
    adapter_ms, decision_ms, total_ms = [], [], []
    with torch.inference_mode():
        for index in range(min(50, len(visual))):
            image = visual[index : index + 1].to(device).float()
            candidates = text[index : index + 1].to(device)
            sync(device)
            start = time.perf_counter()
            aligned = align_batch(adapter, image)
            sync(device)
            middle = time.perf_counter()
            batched_candidate_scores(decision, aligned, candidates)
            sync(device)
            end = time.perf_counter()
            adapter_ms.append((middle - start) * 1000.0)
            decision_ms.append((end - middle) * 1000.0)
            total_ms.append((end - start) * 1000.0)
    return {
        "adapter_mean_ms": float(np.mean(adapter_ms)),
        "adapter_p95_ms": float(np.percentile(adapter_ms, 95)),
        "jev_mean_ms": float(np.mean(decision_ms)),
        "jev_p95_ms": float(np.percentile(decision_ms, 95)),
        "adapter_plus_jev_mean_ms": float(np.mean(total_ms)),
        "adapter_plus_jev_p95_ms": float(np.percentile(total_ms, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONE_NAMES, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, default=Path("data/benchmark_v1_full/manifests"))
    parser.add_argument("--qwen-feature-root", type=Path, default=Path("experiments/benchmark_v1_full/features"))
    parser.add_argument("--dataset-substr", default="coco")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--validation-limit", type=int, default=128)
    parser.add_argument("--calibration-limit", type=int, default=128)
    parser.add_argument("--test-limit", type=int, default=128)
    parser.add_argument("--token-count", type=int, default=196)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.train_limit, args.validation_limit, args.calibration_limit, args.test_limit) < 1:
        parser.error("split limits must be positive")
    if min(args.token_count, args.epochs, args.batch_size, args.grad_accumulation) < 1:
        parser.error("token/training limits must be positive")

    set_seed(args.seed)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started = time.perf_counter()
    limits = {
        "train": args.train_limit,
        "validation": args.validation_limit,
        "calibration": args.calibration_limit,
        "test": args.test_limit,
    }
    examples = load_examples(args.manifest_root, args.dataset_substr, limits)
    text = load_text_features(examples, args.qwen_feature_root)
    visual, controls, backend_metadata = build_visual_cache(args, examples, output)

    base, payload = ThreeStageVisualJEV.from_checkpoint(args.base_checkpoint, map_location="cpu")
    decision = base.decision.to(device).eval()
    for parameter in decision.parameters():
        parameter.requires_grad_(False)
    adapter = PreMergerAlignmentAdapter(
        AlignmentAdapterConfig(
            pre_merger_dim=int(visual["train"].shape[-1]),
            teacher_dim=decision.config.vision_dim,
            hidden_dim=1024,
            dropout=0.0,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    best_state = None
    best_key = (-math.inf, -math.inf)
    best_epoch = 0
    history = []
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        order = torch.randperm(len(visual["train"]))
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, start in enumerate(range(0, len(order), args.batch_size), 1):
            index = order[start : start + args.batch_size]
            image = visual["train"][index].to(device).float()
            candidates = text["train"][0][index].to(device)
            labels = text["train"][1][index].to(device)
            aligned = align_batch(adapter, image)
            scores = batched_candidate_scores(decision, aligned, candidates)
            candidate_loss = F.cross_entropy(scores, labels)
            invalid_loss = scores.new_zeros(())
            if epoch > 1:
                invalid_scores = []
                for name in ("blank", "noise"):
                    control = controls[name][None].expand(len(index), -1, -1).to(device).float()
                    invalid_scores.append(
                        batched_candidate_scores(
                            decision, align_batch(adapter, control), candidates
                        )
                    )
                invalid_loss = torch.stack([uniformity_loss(value) for value in invalid_scores]).mean()
            loss = candidate_loss + 0.1 * invalid_loss
            (loss / args.grad_accumulation).backward()
            if step % args.grad_accumulation == 0 or start + args.batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        validation_scores = predict(
            adapter, decision, visual["validation"], text["validation"][0], device, args.batch_size
        )
        validation = metrics(validation_scores, text["validation"][1])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **validation}
        history.append(row)
        print(json.dumps({"backbone": args.backbone, **row}), flush=True)
        key = (validation["accuracy"], -validation["nll"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.state_dict())
    adapter.load_state_dict(best_state)

    calibration_scores = predict(
        adapter, decision, visual["calibration"], text["calibration"][0], device, args.batch_size
    )
    temperature = fit_temperature(calibration_scores, text["calibration"][1])
    test_scores = predict(adapter, decision, visual["test"], text["test"][0], device, args.batch_size)
    test_labels = text["test"][1]
    calibrated_scores = test_scores / temperature
    blank_visual = controls["blank"][None].expand(len(test_scores), -1, -1)
    noise_visual = controls["noise"][None].expand(len(test_scores), -1, -1)
    wrong_visual = torch.roll(visual["test"], shifts=1, dims=0)
    blank_scores = predict(adapter, decision, blank_visual, text["test"][0], device, args.batch_size)
    noise_scores = predict(adapter, decision, noise_visual, text["test"][0], device, args.batch_size)
    wrong_scores = predict(adapter, decision, wrong_visual, text["test"][0], device, args.batch_size)

    checkpoint = output / "adapter.pt"
    torch.save(
        {
            "format_version": 1,
            "model_type": "hf-backbone-visual-jev-alignment-adapter",
            "backbone": backend_metadata,
            "alignment_config": adapter.config.__dict__,
            "state_dict": adapter.state_dict(),
            "fixed_decision_checkpoint": str(args.base_checkpoint.resolve()),
            "fixed_decision_checkpoint_sha256": sha256(args.base_checkpoint),
            "best_epoch": best_epoch,
            "seed": args.seed,
        },
        checkpoint,
    )
    result = {
        "status": "success",
        "backbone": args.backbone,
        "seed": args.seed,
        "dataset_filter": args.dataset_substr,
        "split_counts": limits,
        "selection_policy": "validation accuracy, then validation NLL; test reporting only",
        "fixed_decision_head": True,
        "base_checkpoint_stage": payload.get("stage"),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "adapter_trainable_parameters": adapter.trainable_parameter_count,
        "best_epoch": best_epoch,
        "history": history,
        "temperature": temperature,
        "test_uncalibrated": metrics(test_scores, test_labels),
        "test_calibrated": metrics(calibrated_scores, test_labels),
        "visual_dependency": {
            "original": condition_metrics(test_scores, test_scores, test_labels),
            "blank": condition_metrics(test_scores, blank_scores, test_labels),
            "noise": condition_metrics(test_scores, noise_scores, test_labels),
            "wrong_image": condition_metrics(test_scores, wrong_scores, test_labels),
        },
        "latency": latency_report(adapter, decision, visual["test"], text["test"][0], device),
        "backbone_extraction": backend_metadata,
        "training_seconds": time.perf_counter() - training_started,
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "claim_boundary": (
            "fast controlled COCO-hard-negative subset; the visual backbone and JEV head are "
            "frozen, only the backbone-specific alignment adapter is trained"
        ),
    }
    write_json(output / "results.json", result)
    print(json.dumps({
        "backbone": args.backbone,
        "accuracy": result["test_uncalibrated"]["accuracy"],
        "macro_f1": result["test_uncalibrated"]["macro_f1"],
        "ece": result["test_uncalibrated"]["ece"],
        "calibrated_ece": result["test_calibrated"]["ece"],
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
