"""Minimal trainer for the candidate-aware Visual-JEV V2 adapter.

JSONL input example::

    {"image": "0001.jpg", "positive": "a red bus", "negative": "a dog"}
    {"image": "0002.jpg", "positive": "a chart", "hard_negatives": ["a map", "a table"]}

Image paths are resolved relative to ``--image-root`` or the JSONL directory.
The Qwen3-VL backbone is frozen; only the V2 adapter and scalar head are saved.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from jev.serving import load_predictor
from visual_jev_v2 import (
    Qwen3VLFeatureExtractor,
    VisualJEVV2,
    VisualJEVV2Config,
    candidate_pair_loss,
)


@dataclass(frozen=True)
class TrainingRecord:
    image: Path
    positive: str
    negatives: tuple[str, ...]
    question: str | None = None


def _nonempty_text(value: Any, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {field} must be a nonempty string")
    return value.strip()


def load_records(
    path: Path,
    image_root: Path | None = None,
    *,
    require_images: bool = True,
) -> list[TrainingRecord]:
    """Read positive/negative or positive/hard-negative JSONL records."""
    base = image_root if image_root is not None else path.parent
    records: list[TrainingRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            image_value = _nonempty_text(raw.get("image"), "image", line_number)
            image_path = Path(image_value).expanduser()
            if not image_path.is_absolute():
                image_path = base / image_path
            positive = _nonempty_text(raw.get("positive"), "positive", line_number)
            negative_values: list[Any] = []
            if raw.get("negative") is not None:
                negative_values.append(raw["negative"])
            for key in ("negatives", "hard_negatives"):
                value = raw.get(key)
                if value is not None:
                    if not isinstance(value, list):
                        raise ValueError(f"line {line_number}: {key} must be a list")
                    negative_values.extend(value)
            negatives = tuple(
                _nonempty_text(value, "negative", line_number)
                for value in negative_values
            )
            if not negatives:
                raise ValueError(
                    f"line {line_number}: at least one negative is required"
                )
            question = raw.get("question")
            if question is not None:
                question = _nonempty_text(question, "question", line_number)
            records.append(TrainingRecord(image_path, positive, negatives, question))
    if not records:
        raise ValueError("training file contains no records")
    missing = [str(record.image) for record in records if not record.image.is_file()]
    if require_images and missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"missing training image(s): {preview}")
    return records


def format_candidate(question: str | None, candidate: str) -> str:
    if question is None:
        return candidate
    return (
        f"Question: {question}\n"
        f"Candidate: {candidate}\n"
        "Judge whether the candidate is supported by the image."
    )


def run_toy(args: argparse.Namespace) -> None:
    """Exercise forward, loss.backward, optimizer step, and checkpoint I/O."""
    torch.manual_seed(args.seed)
    config = VisualJEVV2Config(
        vision_dim=12,
        text_dim=16,
        adapter_dim=24,
        num_heads=4,
        dropout=0.0,
    )
    model = VisualJEVV2(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    visual_tokens = torch.randn(1, 7, config.vision_dim, device=args.device)
    text_features = torch.randn(3, config.text_dim, device=args.device)
    output = model(visual_tokens, text_features)
    loss = candidate_pair_loss(
        output.scores[:1],
        output.scores[1:].unsqueeze(0),
        loss_type=args.loss,
        margin=args.margin,
    )
    loss.backward()
    gradient_parameters = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    optimizer.step()
    checkpoint = model.save_checkpoint(
        args.output,
        optimizer=optimizer,
        step=1,
        metadata={"toy": True, "loss": loss.item()},
    )
    restored, payload = VisualJEVV2.from_checkpoint(checkpoint)
    print(
        json.dumps(
            {
                "mode": "toy",
                "scores": output.scores.detach().float().cpu().tolist(),
                "loss": loss.item(),
                "attention_shape": list(output.attention.shape),
                "attention_row_sums": output.attention.detach()
                .float()
                .sum(-1)
                .cpu()
                .tolist(),
                "parameters_with_finite_gradients": gradient_parameters,
                "trainable_parameters": model.trainable_parameter_count,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_step": payload["step"],
                "restored_parameters": restored.trainable_parameter_count,
            },
            indent=2,
        )
    )


def create_or_restore_model(
    args: argparse.Namespace,
    visual_tokens: torch.Tensor,
    text_features: torch.Tensor,
) -> tuple[VisualJEVV2, dict[str, Any] | None]:
    if args.resume is not None:
        model, payload = VisualJEVV2.from_checkpoint(args.resume)
        if model.config.vision_dim != visual_tokens.shape[-1]:
            raise ValueError("checkpoint vision_dim does not match Qwen3-VL features")
        if model.config.text_dim != text_features.shape[-1]:
            raise ValueError("checkpoint text_dim does not match Qwen3-VL features")
        return model.to(args.device), payload
    config = VisualJEVV2Config(
        vision_dim=visual_tokens.shape[-1],
        text_dim=text_features.shape[-1],
        adapter_dim=args.adapter_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    return VisualJEVV2(config).to(args.device), None


def train(args: argparse.Namespace) -> None:
    data_path = args.data.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve() if args.image_root else None
    records = load_records(data_path, image_root)
    predictor = load_predictor(
        model_id=args.model,
        device=args.device,
        max_length=args.max_length,
        batch_size=max(2, max(1 + len(record.negatives) for record in records)),
        vision=True,
        image_root=image_root or data_path.parent,
    )
    extractor = Qwen3VLFeatureExtractor(predictor.scorer.model)

    first = records[0]
    with Image.open(first.image) as source:
        first_image = source.convert("RGB")
        first_image.load()
    first_candidates = [first.positive, *first.negatives]
    first_tokens = extractor.encode_image(first_image)
    first_text = extractor.encode_candidates(
        format_candidate(first.question, candidate) for candidate in first_candidates
    )
    model, resume_payload = create_or_restore_model(args, first_tokens, first_text)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_step = 0
    if resume_payload is not None:
        start_step = int(resume_payload.get("step") or 0)
        optimizer_state = resume_payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    rng = random.Random(args.seed)
    global_step = start_step
    model.train()
    for epoch in range(args.epochs):
        order = list(range(len(records)))
        rng.shuffle(order)
        for record_index in order:
            record = records[record_index]
            with Image.open(record.image) as source:
                image = source.convert("RGB")
                image.load()
            candidates = [record.positive, *record.negatives]
            visual_tokens = extractor.encode_image(image)
            text_features = extractor.encode_candidates(
                format_candidate(record.question, candidate) for candidate in candidates
            )
            output = model(visual_tokens, text_features)
            loss = candidate_pair_loss(
                output.scores[:1],
                output.scores[1:].unsqueeze(0),
                loss_type=args.loss,
                margin=args.margin,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            global_step += 1
            score_margin = (
                (output.scores[0] - output.scores[1:].max()).detach().float().item()
            )
            print(
                f"epoch={epoch + 1} step={global_step} loss={loss.item():.6f} "
                f"margin={score_margin:.6f} grad_norm={float(gradient_norm):.6f}"
            )
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    output_path = model.save_checkpoint(
        args.output,
        optimizer=optimizer,
        step=global_step,
        metadata={
            "backbone": args.model,
            "backbone_frozen": extractor.backbone_is_frozen,
            "loss_type": args.loss,
            "records": len(records),
        },
    )
    print(f"saved Visual-JEV V2 adapter checkpoint: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="JSONL training records")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/visual-jev-v2.pt")
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--loss", choices=("cross_entropy", "ranking"), default="cross_entropy"
    )
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--adapter-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument(
        "--toy",
        action="store_true",
        help="run synthetic forward/backward/checkpoint validation without Qwen3-VL",
    )
    args = parser.parse_args()
    if not args.toy and args.data is None:
        parser.error("--data is required unless --toy is used")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"{args.device} requested but CUDA is unavailable")
    return args


def main() -> None:
    args = parse_args()
    if args.toy:
        run_toy(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
