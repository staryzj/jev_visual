"""Cache frozen Qwen3-VL features, train Visual-JEV V2, and evaluate controls."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import random
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from jev.serving import load_predictor
from train_visual_jev_v2 import TrainingRecord, format_candidate, load_records
from visual_jev_v2 import (
    Qwen3VLFeatureExtractor,
    VisualJEVV2,
    VisualJEVV2Config,
    candidate_pair_loss,
)

FeatureItem = dict[str, torch.Tensor]


class DiskFeatureDataset(Sequence[FeatureItem]):
    """Read one feature shard at a time instead of retaining the split in RAM."""

    def __init__(self, directory: Path, count: int):
        self.directory = directory
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> FeatureItem:
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        path = self.directory / f"{index:06d}.pt"
        item = torch.load(path, map_location="cpu", weights_only=True)
        _drop_file_cache(path)
        return item


def _drop_file_cache(path: Path) -> None:
    """Hint that a consumed shard can leave the Linux page cache."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        with path.open("rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass


def _shard_directory(cache_path: Path) -> Path:
    return cache_path.parent / f"{cache_path.stem}-shards"


def _write_shard_manifest(
    directory: Path, *, source_sha256: str, count: int, item_key: str
) -> DiskFeatureDataset:
    manifest = {
        "format": "visual-jev-disk-shards-v1",
        "source_sha256": source_sha256,
        "count": count,
        "item_key": item_key,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return DiskFeatureDataset(directory, count)


def open_cached_dataset(
    cache_path: Path, *, source_sha256: str, item_key: str
) -> DiskFeatureDataset | None:
    """Open a shard cache, or convert a legacy monolith via memory mapping."""
    directory = _shard_directory(cache_path)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("format") == "visual-jev-disk-shards-v1"
            and manifest.get("source_sha256") == source_sha256
            and manifest.get("item_key") == item_key
        ):
            print(f"using disk-sharded feature cache: {directory}")
            return DiskFeatureDataset(directory, int(manifest["count"]))

    if not cache_path.is_file():
        return None
    payload = torch.load(
        cache_path, map_location="cpu", weights_only=False, mmap=True
    )
    if payload.get("source_sha256") != source_sha256:
        del payload
        return None
    items = payload[item_key]
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(len(items)):
        torch.save(items[index], directory / f"{index:06d}.pt")
        if (index + 1) % 128 == 0 or index + 1 == len(items):
            print(f"converted cache shards: {index + 1}/{len(items)}")
    dataset = _write_shard_manifest(
        directory,
        source_sha256=source_sha256,
        count=len(items),
        item_key=item_key,
    )
    del items, payload
    gc.collect()
    print(f"wrote disk-sharded feature cache: {directory}")
    return dataset


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.load()
    return image


def candidate_prompts(record: TrainingRecord) -> list[str]:
    return [
        format_candidate(record.question, candidate)
        for candidate in (record.positive, *record.negatives)
    ]


def load_sugarcrepe(
    root: Path, *, per_category: int, seed: int
) -> tuple[list[TrainingRecord], list[str], str]:
    data_root = root / "sugar-crepe" / "sugar-crepe-main" / "data"
    image_root = root / "coco" / "val2017"
    rng = random.Random(seed)
    records = []
    categories = []
    digests = []
    for path in sorted(data_root.glob("*.json")):
        digests.append(file_sha256(path))
        items = list(json.loads(path.read_text(encoding="utf-8")).values())
        rng.shuffle(items)
        for item in items[:per_category]:
            image_path = image_root / item["filename"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            records.append(
                TrainingRecord(
                    image=image_path,
                    positive=item["caption"],
                    negatives=(item["negative_caption"],),
                    question="Which description is supported by the image?",
                )
            )
            categories.append(path.stem)
    if not records:
        raise ValueError("SugarCrepe data was not found")
    source_sha256 = hashlib.sha256(
        ("".join(digests) + f":{per_category}:{seed}").encode()
    ).hexdigest()
    return records, categories, source_sha256


def encode_split(
    records: list[TrainingRecord],
    extractor: Qwen3VLFeatureExtractor,
    *,
    cache_path: Path,
    source_sha256: str,
    text_batch_size: int,
) -> DiskFeatureDataset:
    cached = open_cached_dataset(
        cache_path, source_sha256=source_sha256, item_key="features"
    )
    if cached is not None:
        return cached

    del text_batch_size
    directory = _shard_directory(cache_path)
    directory.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, start=1):
        item = {
            "visual_tokens": extractor.encode_image(load_image(record.image)).cpu(),
            "text_features": extractor.encode_candidates(
                candidate_prompts(record)
            ).cpu(),
        }
        torch.save(item, directory / f"{index - 1:06d}.pt")
        del item
        if index % 10 == 0 or index == len(records):
            print(f"encoded feature shards: {index}/{len(records)}")
    dataset = _write_shard_manifest(
        directory,
        source_sha256=source_sha256,
        count=len(records),
        item_key="features",
    )
    print(f"wrote disk-sharded feature cache: {directory}")
    return dataset


def encode_controls(
    records: list[TrainingRecord],
    extractor: Qwen3VLFeatureExtractor,
    *,
    cache_path: Path,
    source_sha256: str,
) -> DiskFeatureDataset:
    cached = open_cached_dataset(
        cache_path, source_sha256=source_sha256, item_key="controls"
    )
    if cached is not None:
        return cached

    rng = np.random.default_rng(20260927)
    directory = _shard_directory(cache_path)
    directory.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, start=1):
        original = load_image(record.image)
        blank = Image.new("RGB", original.size, (255, 255, 255))
        noise_array = rng.integers(
            0,
            256,
            size=(original.height, original.width, 3),
            dtype=np.uint8,
        )
        noise = Image.fromarray(noise_array, mode="RGB")
        item = {
            "blank": extractor.encode_image(blank).cpu(),
            "noise": extractor.encode_image(noise).cpu(),
        }
        torch.save(item, directory / f"{index - 1:06d}.pt")
        del item
        if index % 5 == 0 or index == len(records):
            print(f"encoded visual controls: {index}/{len(records)}")
    dataset = _write_shard_manifest(
        directory,
        source_sha256=source_sha256,
        count=len(records),
        item_key="controls",
    )
    print(f"wrote disk-sharded control cache: {directory}")
    return dataset


@torch.no_grad()
def evaluate(
    model: VisualJEVV2,
    features: Sequence[FeatureItem],
    *,
    alternate_features: Sequence[FeatureItem] | None = None,
    alternate_key: str | None = None,
    indices: Sequence[int] | None = None,
    swap_visual: bool = False,
) -> dict[str, float]:
    model.eval()
    correct = 0
    margins = []
    losses = []
    attention_entropies = []
    positive_scores = []
    hardest_negative_scores = []
    sample_indices = range(len(features)) if indices is None else indices
    for index in sample_indices:
        item = features[index]
        alternate_item = None
        if alternate_features is not None:
            if alternate_key is None:
                raise ValueError("alternate_key is required with alternate_features")
            alternate_item = alternate_features[index]
            visual_tokens = alternate_item[alternate_key]
        elif swap_visual:
            alternate_item = features[(index + 1) % len(features)]
            visual_tokens = alternate_item["visual_tokens"]
        else:
            visual_tokens = item["visual_tokens"]
        output = model(visual_tokens, item["text_features"])
        correct += int(output.scores.argmax().item() == 0)
        margins.append((output.scores[0] - output.scores[1:].max()).item())
        positive_scores.append(output.scores[0].item())
        hardest_negative_scores.append(output.scores[1:].max().item())
        losses.append(
            candidate_pair_loss(
                output.scores[:1], output.scores[1:].unsqueeze(0)
            ).item()
        )
        attention = output.attention.float().clamp_min(1e-12)
        attention_entropies.append(
            (-(attention * attention.log()).sum(-1).mean()).item()
        )
        del output, item, alternate_item
    sample_count = len(features) if indices is None else len(indices)
    return {
        "accuracy": correct / sample_count,
        "mean_margin": float(np.mean(margins)),
        "mean_loss": float(np.mean(losses)),
        "mean_positive_score": float(np.mean(positive_scores)),
        "mean_hardest_negative_score": float(np.mean(hardest_negative_scores)),
        "mean_attention_entropy": float(np.mean(attention_entropies)),
    }


def evaluate_by_category(
    model: VisualJEVV2,
    features: Sequence[FeatureItem],
    categories: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": evaluate(model, features)}
    for category in sorted(set(categories)):
        indices = [
            index for index, label in enumerate(categories) if label == category
        ]
        result[category] = evaluate(model, features, indices=indices)
    return result


def train_adapter(
    model: VisualJEVV2,
    train_features: Sequence[FeatureItem],
    validation_features: Sequence[FeatureItem],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    visual_loss_weight: float,
    visual_margin: float,
    seed: int,
) -> tuple[torch.optim.Optimizer, list[dict[str, float]], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    rng = random.Random(seed)
    history = []
    best_epoch = 0
    best_state = None
    best_optimizer_state = None
    best_key = (-1.0, float("-inf"))
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_features)))
        rng.shuffle(order)
        losses = []
        candidate_losses = []
        visual_losses = []
        for position, index in enumerate(order):
            item = train_features[index]
            swapped_item = train_features[order[(position + 1) % len(order)]]
            optimizer.zero_grad(set_to_none=True)
            output = model(item["visual_tokens"], item["text_features"])
            candidate_loss = candidate_pair_loss(
                output.scores[:1], output.scores[1:].unsqueeze(0)
            )
            swapped_output = model(swapped_item["visual_tokens"], item["text_features"])
            visual_loss = torch.relu(
                visual_margin - output.scores[0] + swapped_output.scores[0]
            )
            loss = candidate_loss + visual_loss_weight * visual_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            candidate_losses.append(candidate_loss.item())
            visual_losses.append(visual_loss.item())
            del item, swapped_item, output, swapped_output, loss
        validation = evaluate(model, validation_features)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "candidate_loss": float(np.mean(candidate_losses)),
            "visual_ranking_loss": float(np.mean(visual_losses)),
            "validation_accuracy": validation["accuracy"],
            "validation_margin": validation["mean_margin"],
            "validation_loss": validation["mean_loss"],
        }
        history.append(row)
        print(json.dumps(row))
        key = (validation["accuracy"], validation["mean_margin"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    if best_state is None or best_optimizer_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    optimizer.load_state_dict(best_optimizer_state)
    return optimizer, history, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/visual-jev-v2"))
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--visual-loss-weight", type=float, default=0.5)
    parser.add_argument("--visual-margin", type=float, default=0.5)
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--sugarcrepe-per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/visual-jev-v2-coco.pt")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/visual-jev-v2-coco.json")
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"{args.device} requested but CUDA is unavailable")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    root = args.root.expanduser().resolve()
    train_path = root / "splits" / "train.jsonl"
    validation_path = root / "splits" / "validation.jsonl"
    train_records = load_records(train_path, root)
    validation_records = load_records(validation_path, root)
    sugarcrepe_records, sugarcrepe_categories, sugarcrepe_sha256 = load_sugarcrepe(
        root, per_category=args.sugarcrepe_per_category, seed=args.seed
    )

    cache_dir = root / "features"
    train_sha256 = file_sha256(train_path)
    validation_sha256 = file_sha256(validation_path)
    train_cache = cache_dir / "train.pt"
    validation_cache = cache_dir / "validation.pt"
    control_cache = cache_dir / "validation-controls.pt"
    sugarcrepe_cache = cache_dir / "sugarcrepe.pt"
    train_features = open_cached_dataset(
        train_cache, source_sha256=train_sha256, item_key="features"
    )
    validation_features = open_cached_dataset(
        validation_cache,
        source_sha256=validation_sha256,
        item_key="features",
    )
    control_features = open_cached_dataset(
        control_cache,
        source_sha256=validation_sha256,
        item_key="controls",
    )
    sugarcrepe_features = open_cached_dataset(
        sugarcrepe_cache,
        source_sha256=sugarcrepe_sha256,
        item_key="features",
    )
    backbone_frozen = True
    if any(
        dataset is None
        for dataset in (
            train_features,
            validation_features,
            control_features,
            sugarcrepe_features,
        )
    ):
        predictor = load_predictor(
            model_id=args.model,
            device=args.device,
            max_length=512,
            batch_size=args.text_batch_size,
            vision=True,
            image_root=root,
        )
        extractor = Qwen3VLFeatureExtractor(predictor.scorer.model)
        if train_features is None:
            train_features = encode_split(
                train_records,
                extractor,
                cache_path=train_cache,
                source_sha256=train_sha256,
                text_batch_size=args.text_batch_size,
            )
        if validation_features is None:
            validation_features = encode_split(
                validation_records,
                extractor,
                cache_path=validation_cache,
                source_sha256=validation_sha256,
                text_batch_size=args.text_batch_size,
            )
        if control_features is None:
            control_features = encode_controls(
                validation_records,
                extractor,
                cache_path=control_cache,
                source_sha256=validation_sha256,
            )
        if sugarcrepe_features is None:
            sugarcrepe_features = encode_split(
                sugarcrepe_records,
                extractor,
                cache_path=sugarcrepe_cache,
                source_sha256=sugarcrepe_sha256,
                text_batch_size=args.text_batch_size,
            )
        backbone_frozen = extractor.backbone_is_frozen
        del extractor, predictor
        gc.collect()
        torch.cuda.empty_cache()
    if any(
        dataset is None
        for dataset in (
            train_features,
            validation_features,
            control_features,
            sugarcrepe_features,
        )
    ):
        raise RuntimeError("feature cache preparation failed")

    sample = train_features[0]
    model = VisualJEVV2(
        VisualJEVV2Config(
            vision_dim=sample["visual_tokens"].shape[-1],
            text_dim=sample["text_features"].shape[-1],
            adapter_dim=args.adapter_dim,
            num_heads=args.num_heads,
            dropout=0.0,
        )
    ).to(args.device)
    del sample
    before = evaluate(model, validation_features)
    sugarcrepe_before = evaluate_by_category(
        model, sugarcrepe_features, sugarcrepe_categories
    )
    optimizer, history, best_epoch = train_adapter(
        model,
        train_features,
        validation_features,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        visual_loss_weight=args.visual_loss_weight,
        visual_margin=args.visual_margin,
        seed=args.seed,
    )
    after_original = evaluate(model, validation_features)
    controls = {
        "original": after_original,
        "blank": evaluate(
            model,
            validation_features,
            alternate_features=control_features,
            alternate_key="blank",
        ),
        "noise": evaluate(
            model,
            validation_features,
            alternate_features=control_features,
            alternate_key="noise",
        ),
        "image_swap": evaluate(
            model, validation_features, swap_visual=True
        ),
    }
    original_margin = controls["original"]["mean_margin"]
    original_positive_score = controls["original"]["mean_positive_score"]
    original_accuracy = controls["original"]["accuracy"]
    for metrics in controls.values():
        metrics["margin_drop_vs_original"] = original_margin - metrics["mean_margin"]
        metrics["positive_score_drop_vs_original"] = (
            original_positive_score - metrics["mean_positive_score"]
        )
        metrics["accuracy_drop_vs_original"] = original_accuracy - metrics["accuracy"]
    sugarcrepe_after = evaluate_by_category(
        model, sugarcrepe_features, sugarcrepe_categories
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        args.checkpoint,
        optimizer=optimizer,
        step=best_epoch * len(train_features),
        metadata={
            "dataset_manifest": str(root / "manifest.json"),
            "best_epoch": best_epoch,
            "backbone": args.model,
            "backbone_frozen": backbone_frozen,
            "visual_loss_weight": args.visual_loss_weight,
            "visual_margin": args.visual_margin,
        },
    )
    report: dict[str, Any] = {
        "dataset": json.loads((root / "manifest.json").read_text(encoding="utf-8")),
        "config": asdict(model.config),
        "backbone": args.model,
        "backbone_frozen": backbone_frozen,
        "trainable_parameters": model.trainable_parameter_count,
        "feature_cache": {
            "mode": "disk_shards",
            "resident_examples": 2,
        },
        "before_training": before,
        "sugarcrepe_before_training": sugarcrepe_before,
        "best_epoch": best_epoch,
        "after_training": controls,
        "sugarcrepe_after_training": sugarcrepe_after,
        "history": history,
        "checkpoint": str(args.checkpoint.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
