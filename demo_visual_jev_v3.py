"""Run an end-to-end Visual-JEV V3 demo and latency benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from jev.benchmark_v1 import read_jsonl
from jev.serving import load_predictor
from train_visual_jev_v2 import format_candidate
from train_visual_jev_v3 import encode_image_stages
from visual_jev_v2 import Qwen3VLFeatureExtractor
from visual_jev_v3_pipeline import ThreeStageVisualJEV


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def select_examples(rows, count: int):
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row.dataset].append(row)
    selected = []
    while len(selected) < count:
        changed = False
        for domain in sorted(by_domain):
            if by_domain[domain] and len(selected) < count:
                selected.append(by_domain[domain].pop(0))
                changed = True
        if not changed:
            break
    return selected


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark_v1/manifests/test.jsonl"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/best_model_results/visual_jev_v3_best.pt"),
    )
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--head-repeats", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/benchmark_v1/live_demo/demo_results.json"))
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    rows = select_examples(list(read_jsonl(args.manifest)), args.examples)
    if not rows:
        raise RuntimeError("manifest contains no examples")

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    predictor = load_predictor(
        model_id=args.model,
        device=args.device,
        max_length=512,
        batch_size=8,
        vision=True,
        image_root=args.manifest.parent.parent,
    )
    backbone = predictor.scorer.model
    extractor = Qwen3VLFeatureExtractor(backbone)
    jev, checkpoint_payload = ThreeStageVisualJEV.from_checkpoint(args.checkpoint, map_location="cpu")
    jev = jev.to(args.device).eval()
    sync(args.device)
    load_seconds = time.perf_counter() - load_started

    # One untimed warmup exercises the real image, language and JEV paths.
    warm = rows[0]
    with Image.open(warm.image) as source:
        image = source.convert("RGB"); image.load()
    warm_pre, _ = encode_image_stages(backbone, image)
    warm_text = extractor.encode_candidates(
        [format_candidate(warm.question, candidate) for candidate in warm.candidates]
    ).cpu()
    jev(warm_pre, warm_text)
    sync(args.device)

    samples = []
    image_times: list[float] = []
    text_times: list[float] = []
    head_times: list[float] = []
    end_to_end_times: list[float] = []
    last_pre = last_text = None
    for row in rows:
        total_started = time.perf_counter()
        with Image.open(row.image) as source:
            image = source.convert("RGB"); image.load()

        sync(args.device); started = time.perf_counter()
        pre, _ = encode_image_stages(backbone, image)
        sync(args.device); image_seconds = time.perf_counter() - started

        prompts = [format_candidate(row.question, candidate) for candidate in row.candidates]
        sync(args.device); started = time.perf_counter()
        text = extractor.encode_candidates(prompts).cpu()
        sync(args.device); text_seconds = time.perf_counter() - started

        sync(args.device); started = time.perf_counter()
        output = jev(pre, text)
        sync(args.device); head_seconds = time.perf_counter() - started
        total_seconds = time.perf_counter() - total_started

        scores = output.scores.detach().float().cpu()
        probabilities = scores.softmax(-1)
        prediction = int(probabilities.argmax().item())
        samples.append({
            "id": row.id,
            "dataset": row.dataset,
            "image": row.image,
            "question": row.question,
            "candidates": list(row.candidates),
            "reference_index": row.label,
            "reference": row.candidates[row.label],
            "prediction_index": prediction,
            "prediction": row.candidates[prediction],
            "correct": prediction == row.label,
            "probabilities": [float(value) for value in probabilities],
            "latency_ms": {
                "image": image_seconds * 1000,
                "text": text_seconds * 1000,
                "jev_head": head_seconds * 1000,
                "end_to_end": total_seconds * 1000,
            },
        })
        image_times.append(image_seconds * 1000)
        text_times.append(text_seconds * 1000)
        head_times.append(head_seconds * 1000)
        end_to_end_times.append(total_seconds * 1000)
        last_pre, last_text = pre, text
        print(json.dumps(samples[-1], ensure_ascii=False), flush=True)

    # Stable head-only latency after all kernels and allocations are warm.
    head_micro = []
    for _ in range(args.head_repeats):
        sync(args.device); started = time.perf_counter()
        jev(last_pre, last_text)
        sync(args.device); head_micro.append((time.perf_counter() - started) * 1000)

    latency = {
        "image_mean_ms": statistics.mean(image_times),
        "text_mean_ms": statistics.mean(text_times),
        "jev_head_mean_ms": statistics.mean(head_times),
        "end_to_end_mean_ms": statistics.mean(end_to_end_times),
        "end_to_end_p50_ms": percentile(end_to_end_times, 0.50),
        "end_to_end_p95_ms": percentile(end_to_end_times, 0.95),
        "warm_jev_head_mean_ms": statistics.mean(head_micro),
        "warm_jev_head_p95_ms": percentile(head_micro, 0.95),
        "end_to_end_throughput_samples_per_second": 1000 / statistics.mean(end_to_end_times),
    }
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_stage": checkpoint_payload.get("stage"),
        "model": args.model,
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "load_seconds": load_seconds,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else 0,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if args.device.startswith("cuda") else 0,
        "sample_count": len(samples),
        "demo_accuracy": sum(int(row["correct"]) for row in samples) / len(samples),
        "latency": latency,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
