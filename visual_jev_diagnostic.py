"""Measure Visual-JEV V1 feature statistics and image sensitivity.

This is an isolated diagnostic.  It imports the existing implementation but
does not modify the vl_smoke/vision_test execution paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from jev.api import candidate_prompts, compile_request
from jev.serving import load_predictor
from visual_jev import VisualJEVBaseline


CANDIDATES = (
    ("interface", "interface: A software interface or presentation frame is visible."),
    ("animal", "animal: An animal is the main subject."),
)


def tensor_metadata(value: torch.Tensor) -> dict:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def vector_stats(value: torch.Tensor) -> dict:
    metadata = tensor_metadata(value)
    value = value.detach().float()
    return {
        **metadata,
        "norm": value.norm().item(),
        "mean": value.mean().item(),
        "variance": value.var(unbiased=False).item(),
        "std": value.std(unbiased=False).item(),
        "min": value.min().item(),
        "max": value.max().item(),
    }


def build_prompts(question: str) -> list[str]:
    criteria = {
        f"candidate_{index}": text
        for index, (_, text) in enumerate(CANDIDATES)
    }
    record = compile_request(
        "Judge each proposed answer using the supplied image.",
        {
            "visual_choice": {
                "type": "choice",
                "instructions": question,
                "criteria": criteria,
            }
        },
    )[0]
    return candidate_prompts(record)


def encode_image(baseline: VisualJEVBaseline, image: Image.Image) -> dict:
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Extract visual evidence."},
        ],
    }]
    encoded = baseline.processor.apply_chat_template(
        [conversation],
        tokenize=True,
        add_generation_prompt=True,
        processor_kwargs={"padding": True},
        return_dict=True,
        return_tensors="pt",
    )
    visual_parameter = next(baseline.visual_model.parameters())
    pixel_values = encoded["pixel_values"].to(visual_parameter.device)
    image_grid_thw = encoded["image_grid_thw"].to(visual_parameter.device)
    vision_output = baseline.visual_model(
        pixel_values.to(dtype=visual_parameter.dtype),
        grid_thw=image_grid_thw,
        return_dict=True,
    )
    pre_merger = vision_output.last_hidden_state
    post_merger = vision_output.pooler_output
    bridge_input = pre_merger.unsqueeze(0) if pre_merger.ndim == 2 else pre_merger
    image_feature = baseline.bridge.align_visual(bridge_input)
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "pre_merger": pre_merger,
        "post_merger": post_merger,
        "image_feature": image_feature,
    }


def score_image(
    baseline: VisualJEVBaseline,
    text_features: torch.Tensor,
    image: Image.Image,
) -> tuple[dict, dict[str, torch.Tensor]]:
    encoded = encode_image(baseline, image)
    image_feature = encoded["image_feature"]
    image_batch = image_feature.expand(text_features.shape[0], -1)
    fusion_parameter = next(baseline.bridge.fusion_norm.parameters())
    text_for_fusion = text_features.to(
        device=fusion_parameter.device, dtype=fusion_parameter.dtype)
    image_for_fusion = image_batch.to(
        device=fusion_parameter.device, dtype=fusion_parameter.dtype)
    pre_fusion_sum = text_for_fusion + image_for_fusion
    fused = baseline.bridge.fuse(text_features, image_batch)
    scores = baseline.bridge.score(fused)
    probabilities = torch.softmax(scores, dim=0)

    candidates = {}
    for index, (label, _) in enumerate(CANDIDATES):
        candidates[label] = {
            "cosine_image_text": F.cosine_similarity(
                image_feature.float(), text_features[index:index + 1].float(), dim=-1
            ).item(),
            "text_norm_before_fusion": text_features[index].float().norm().item(),
            "image_norm_before_fusion": image_feature[0].float().norm().item(),
            "sum_before_layernorm": vector_stats(pre_fusion_sum[index]),
            "fused_after_layernorm": vector_stats(fused[index]),
            "score": scores[index].float().item(),
            "probability": probabilities[index].float().item(),
        }

    result = {
        "pixel_values": tensor_metadata(encoded["pixel_values"]),
        "image_grid_thw": {
            **tensor_metadata(encoded["image_grid_thw"]),
            "value": encoded["image_grid_thw"].detach().cpu().tolist(),
        },
        "pre_merger_visual_tokens": {
            **tensor_metadata(encoded["pre_merger"]),
            "global_mean": encoded["pre_merger"].float().mean().item(),
            "global_variance": encoded["pre_merger"].float().var(unbiased=False).item(),
            "mean_pooled_feature": vector_stats(
                encoded["pre_merger"].float().mean(dim=0)),
        },
        "post_merger_visual_tokens": {
            **tensor_metadata(encoded["post_merger"]),
            "global_mean": encoded["post_merger"].float().mean().item(),
            "global_variance": encoded["post_merger"].float().var(unbiased=False).item(),
            "mean_pooled_feature": vector_stats(
                encoded["post_merger"].float().mean(dim=0)),
        },
        "image_feature": vector_stats(image_feature),
        "candidates": candidates,
        "score_margin_interface_minus_animal": (
            scores[0].float() - scores[1].float()).item(),
        "probability_sum": probabilities.float().sum().item(),
    }
    comparison_features = {
        "pre_merger_mean": encoded["pre_merger"].detach().float().mean(dim=0).cpu(),
        "post_merger_mean": encoded["post_merger"].detach().float().mean(dim=0).cpu(),
        "projected_image_feature": image_feature.detach().float().squeeze(0).cpu(),
    }
    return result, comparison_features


def parameter_stats(module: torch.nn.Module) -> dict:
    return {
        name: vector_stats(parameter)
        for name, parameter in module.named_parameters()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image", default="test.jpg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--question", default="Which description best matches the image?")
    parser.add_argument(
        "--output", default="reports/visual-jev-v1-diagnostic.json")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    predictor = load_predictor(
        model_id=args.model,
        device=args.device,
        max_length=args.max_length,
        batch_size=len(CANDIDATES),
        vision=True,
        image_root=image_path.parent,
    )
    decision_model = predictor.scorer.model
    baseline = VisualJEVBaseline(decision_model, debug=False)
    prompts = build_prompts(args.question)

    with Image.open(image_path) as source:
        original = source.convert("RGB")
        original.load()
    blank = Image.new("RGB", original.size, (255, 255, 255))
    random_pixels = np.random.default_rng(20260927).integers(
        0, 256, size=(original.height, original.width, 3), dtype=np.uint8)
    random_image = Image.fromarray(random_pixels, mode="RGB")

    with torch.inference_mode():
        text_features = baseline.extract_text_features(prompts)
        text_only_fused = baseline.bridge.fuse(
            text_features, torch.zeros_like(text_features))
        text_only_scores = baseline.bridge.score(text_only_fused)
        image_only_feature = torch.ones_like(text_features[:1])
        image_only_fused = baseline.bridge.fuse(
            torch.zeros_like(text_features),
            image_only_feature.expand(text_features.shape[0], -1),
        )
        image_only_scores = baseline.bridge.score(image_only_fused)

        image_results = {}
        comparison_features = {}
        for name, image in (
            ("original", original),
            ("blank_white", blank),
            ("random_noise", random_image),
        ):
            image_results[name], comparison_features[name] = score_image(
                baseline, text_features, image)

    head = decision_model.head
    generator = torch.Generator(device=head.weight.device).manual_seed(
        decision_model.head_seed)
    expected_head = torch.empty_like(head.weight)
    torch.nn.init.normal_(
        expected_head,
        mean=0.0,
        std=float(decision_model.backbone.config.text_config.initializer_range),
        generator=generator,
    )
    fusion_norm = baseline.bridge.fusion_norm

    text_stats = {
        label: vector_stats(text_features[index])
        for index, (label, _) in enumerate(CANDIDATES)
    }
    text_stats["cosine_interface_animal"] = F.cosine_similarity(
        text_features[0:1].float(), text_features[1:2].float(), dim=-1
    ).item()

    original_margin = image_results["original"][
        "score_margin_interface_minus_animal"]
    for name, result in image_results.items():
        result["margin_change_vs_original"] = (
            result["score_margin_interface_minus_animal"] - original_margin)
        result["representation_change_vs_original"] = {}
        for stage in (
            "pre_merger_mean", "post_merger_mean", "projected_image_feature"):
            original_feature = comparison_features["original"][stage]
            current_feature = comparison_features[name][stage]
            result["representation_change_vs_original"][stage] = {
                "cosine": F.cosine_similarity(
                    original_feature.unsqueeze(0), current_feature.unsqueeze(0), dim=-1
                ).item(),
                "l2_distance": (original_feature - current_feature).norm().item(),
            }

    report = {
        "run": {
            "model_argument": args.model,
            "image": str(image_path),
            "device": args.device,
            "checkpoint_argument": None,
            "loaded_from_vl_checkpoint": False,
            "torch_initial_seed_at_report_time": torch.initial_seed(),
        },
        "weight_provenance": {
            "visual_backbone": "pretrained base-model weights; frozen",
            "language_backbone": "pretrained base-model weights; frozen",
            "decision_head": {
                "status": decision_model.head_status,
                "method": decision_model.method,
                "seed": decision_model.head_seed,
                "matches_fresh_seeded_initialization": torch.equal(
                    head.weight, expected_head),
                "loaded_from_checkpoint": False,
                "vl_specific_training": False,
                "parameters": parameter_stats(head),
            },
            "visual_algorithm": {
                "status": "fresh random initialization in VisualJEVBaseline.__init__",
                "loaded_from_checkpoint": False,
                "trained": False,
                "parameters": parameter_stats(baseline.bridge.visual_algorithm),
            },
            "fusion_norm": {
                "status": "fresh default LayerNorm initialization",
                "loaded_from_checkpoint": False,
                "trained": False,
                "weight_all_ones": bool(torch.all(fusion_norm.weight == 1).item()),
                "bias_all_zeros": bool(torch.all(fusion_norm.bias == 0).item()),
                "parameters": parameter_stats(fusion_norm),
            },
        },
        "text_features": text_stats,
        "ablations": {
            "text_only_scores_after_layernorm": {
                label: text_only_scores[index].float().item()
                for index, (label, _) in enumerate(CANDIDATES)
            },
            "text_only_margin_interface_minus_animal": (
                text_only_scores[0].float() - text_only_scores[1].float()).item(),
            "image_only_control_scores_identical": bool(
                torch.equal(image_only_scores[0], image_only_scores[1])),
            "note": (
                "The image-only control uses one arbitrary shared image vector. "
                "Identical candidate scores demonstrate that V1 has no candidate-aware "
                "visual readout; image dependence can only enter indirectly through "
                "LayerNorm(text + shared_image)."
            ),
        },
        "images": image_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote diagnostic report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
