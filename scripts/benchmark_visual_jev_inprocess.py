#!/usr/bin/env python3
"""Warm, in-process, CUDA-synchronized Visual-JEV stage latency benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


STAGES = ("qwen_visual", "qwen_text", "alignment_adapter", "jev_decision", "image_to_decision_e2e")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position); upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark_v1_full/manifests/test.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/best_model_results/visual_jev_v3_best.pt"))
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/latency/inprocess_100.json"))
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    sys.path.insert(0, str(repo / "scripts"))
    import live_visual_jev_demo as live

    rows = [line for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.count > len(rows):
        raise ValueError(f"requested {args.count} examples from {len(rows)}")
    indices = [round(i * (len(rows) - 1) / max(1, args.count - 1)) for i in range(args.count)]
    live_args = live.build_parser().parse_args([
        "--manifest", str(args.manifest), "--checkpoint", str(args.checkpoint),
        "--model", args.model, "--device", args.device, "--history-mode", "off",
        "--history-prompt-policy", "legacy-image-only", "--skip-cache-validation", "--no-window",
    ])
    runtime = live.load_runtime(live_args)
    records = []
    for position, index in enumerate(indices):
        case = live.manifest_case(args.manifest.resolve(), index)
        result = live.run(live_args, case, runtime)
        records.append({"index": index, "sample_id": case.sample_id, "latency_ms": result["latency_ms"]})
        if (position + 1) % 10 == 0:
            print(json.dumps({"completed": position + 1, "total": len(indices)}), flush=True)
    summary = {
        "protocol": "one resident process; one untimed warmup; CUDA synchronization around every measured stage",
        "count": len(records), "indices": indices, "model_load_excluded": True,
        "stages_ms": {}, "records": records,
    }
    for stage in STAGES:
        values = [float(row["latency_ms"][stage]) for row in records]
        summary["stages_ms"][stage] = {
            "mean": statistics.fmean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p50": percentile(values, 0.50), "p95": percentile(values, 0.95),
            "min": min(values), "max": max(values),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
