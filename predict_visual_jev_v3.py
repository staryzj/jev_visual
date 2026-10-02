"""Predict one image/question with a trained Visual-JEV V3 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image

from jev.serving import load_predictor
from train_visual_jev_v2 import format_candidate
from train_visual_jev_v3 import encode_image_stages
from visual_jev_v2 import Qwen3VLFeatureExtractor
from visual_jev_v3_pipeline import ThreeStageVisualJEV


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--candidate", action="append", required=True,
        help="Candidate answer; repeat this flag at least twice.",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/best_model_results/visual_jev_v3_best.pt"),
    )
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    candidates = [value.strip() for value in args.candidate]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if len(candidates) < 2 or any(not value for value in candidates):
        parser.error("provide at least two non-empty --candidate values")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    load_started = time.perf_counter()
    predictor = load_predictor(
        model_id=args.model,
        device=args.device,
        max_length=512,
        batch_size=max(1, len(candidates)),
        vision=True,
        image_root=image_path.parent,
    )
    backbone = predictor.scorer.model
    extractor = Qwen3VLFeatureExtractor(backbone)
    jev, checkpoint_payload = ThreeStageVisualJEV.from_checkpoint(checkpoint, map_location="cpu")
    jev = jev.to(args.device).eval()
    sync()
    load_ms = (time.perf_counter() - load_started) * 1000

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.load()

    sync(); started = time.perf_counter()
    pre_tokens, _ = encode_image_stages(backbone, image)
    sync(); image_ms = (time.perf_counter() - started) * 1000

    prompts = [format_candidate(args.question, candidate) for candidate in candidates]
    sync(); started = time.perf_counter()
    text_features = extractor.encode_candidates(prompts).cpu()
    sync(); text_ms = (time.perf_counter() - started) * 1000

    sync(); started = time.perf_counter()
    scores = jev(pre_tokens, text_features).scores.detach().float().cpu()
    sync(); jev_ms = (time.perf_counter() - started) * 1000
    probabilities = (scores / args.temperature).softmax(-1)
    prediction = int(probabilities.argmax().item())
    sorted_probabilities = probabilities.sort(descending=True).values
    margin = float(sorted_probabilities[0] - sorted_probabilities[1])
    entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())

    result = {
        "image": str(image_path),
        "question": args.question,
        "candidates": [
            {"index": index, "text": text, "score": float(scores[index]), "probability": float(probabilities[index])}
            for index, text in enumerate(candidates)
        ],
        "prediction_index": prediction,
        "prediction": candidates[prediction],
        "confidence": float(probabilities[prediction]),
        "top1_top2_margin": margin,
        "normalized_entropy": entropy / math.log(len(candidates)),
        "temperature": args.temperature,
        "checkpoint": str(checkpoint),
        "checkpoint_stage": checkpoint_payload.get("stage"),
        "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else args.device,
        "latency_ms": {
            "cold_load": load_ms,
            "image_encoding": image_ms,
            "candidate_text_encoding": text_ms,
            "jev_decision": jev_ms,
            "warm_total": image_ms + text_ms + jev_ms,
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
