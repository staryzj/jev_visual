"""Diagnose Visual-JEV V2 attention, margins, and visual dependence controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from jev.serving import load_predictor
from visual_jev_v2 import (
    Qwen3VLFeatureExtractor,
    VisualJEVV2,
    VisualJEVV2Output,
    attention_statistics,
)

DEFAULT_CANDIDATES = (
    "interface: A software interface or presentation frame is visible.",
    "animal: An animal is the main subject.",
)


def tensor_metadata(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def norm_list(value: torch.Tensor) -> list[float]:
    value = value.detach().float()
    return value.reshape(value.shape[0], -1).norm(dim=-1).cpu().tolist()


def score_control(
    model: VisualJEVV2,
    visual_tokens: torch.Tensor,
    text_features: torch.Tensor,
    candidates: tuple[str, ...],
) -> dict[str, Any]:
    output: VisualJEVV2Output = model(visual_tokens, text_features)
    scores = output.scores.detach().float()
    margin = scores[0] - scores[1:].max()
    return {
        "post_merger_tokens": {
            **tensor_metadata(visual_tokens),
            "global_norm": visual_tokens.detach().float().norm().item(),
            "mean_token_norm": visual_tokens.detach()
            .float()
            .norm(dim=-1)
            .mean()
            .item(),
        },
        "scores": {
            candidate: scores[index].item()
            for index, candidate in enumerate(candidates)
        },
        "score_margin_positive_minus_hardest_negative": margin.item(),
        "attention": {
            candidate: stats
            for candidate, stats in zip(
                candidates, attention_statistics(output.attention)
            )
        },
        "feature_norms": {
            "projected_text": norm_list(output.projected_text_features),
            "attended_visual": norm_list(output.attended_visual_features),
            "fused": norm_list(output.fused_features),
        },
        "gate": {
            "mean_per_candidate": output.gates.detach().float().mean(-1).cpu().tolist(),
            "min": output.gates.detach().float().min().item(),
            "max": output.gates.detach().float().max().item(),
        },
    }


def load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.load()
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image", type=Path, default=Path("test.jpg"))
    parser.add_argument(
        "--swap-image",
        type=Path,
        help="a different real image for the image-swap control",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/visual-jev-v2-diagnostic.json")
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"{args.device} requested but CUDA is unavailable")
    candidates = tuple(args.candidates or DEFAULT_CANDIDATES)
    if len(candidates) < 2:
        parser.error("at least two candidates are required; the first is positive")

    original = load_rgb(args.image.expanduser().resolve())
    controls: dict[str, Image.Image] = {
        "original": original,
        "blank_white": Image.new("RGB", original.size, (255, 255, 255)),
    }
    random_pixels = np.random.default_rng(20260927).integers(
        0, 256, size=(original.height, original.width, 3), dtype=np.uint8
    )
    controls["random_noise"] = Image.fromarray(random_pixels, mode="RGB")
    if args.swap_image is not None:
        controls["image_swap"] = load_rgb(args.swap_image.expanduser().resolve())

    predictor = load_predictor(
        model_id=args.model,
        device=args.device,
        max_length=args.max_length,
        batch_size=len(candidates),
        vision=True,
        image_root=args.image.expanduser().resolve().parent,
    )
    extractor = Qwen3VLFeatureExtractor(predictor.scorer.model)
    model, checkpoint = VisualJEVV2.from_checkpoint(
        args.checkpoint.expanduser().resolve(), map_location=args.device
    )
    model.eval()
    text_features = extractor.encode_candidates(candidates)
    if model.config.text_dim != text_features.shape[-1]:
        raise ValueError("checkpoint text_dim does not match the selected backbone")

    results: dict[str, Any] = {}
    with torch.inference_mode():
        for name, image in controls.items():
            visual_tokens = extractor.encode_image(image)
            if model.config.vision_dim != visual_tokens.shape[-1]:
                raise ValueError(
                    "checkpoint vision_dim does not match post-merger tokens"
                )
            results[name] = score_control(
                model, visual_tokens, text_features, candidates
            )

    original_margin = results["original"][
        "score_margin_positive_minus_hardest_negative"
    ]
    for result in results.values():
        result["margin_change_vs_original"] = (
            result["score_margin_positive_minus_hardest_negative"] - original_margin
        )
    report = {
        "model": "Visual-JEV V2",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_metadata": checkpoint.get("metadata", {}),
        "adapter_config": checkpoint["config"],
        "backbone": args.model,
        "backbone_frozen": extractor.backbone_is_frozen,
        "candidate_order": list(candidates),
        "positive_candidate_index": 0,
        "controls": results,
        "image_swap_status": (
            "included"
            if args.swap_image is not None
            else "not run; pass --swap-image with a different real image"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote diagnostic report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
