#!/usr/bin/env python3
"""Aggregate JSON records written by live_visual_jev_demo.py."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


STAGES = (
    "qwen_visual",
    "qwen_text",
    "alignment_adapter",
    "jev_decision",
    "image_to_decision_e2e",
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.input_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no JSON records found in {args.input_dir}")

    rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    summary = {"count": len(rows), "stages_ms": {}}
    for stage in STAGES:
        values = [float(row["latency_ms"][stage]) for row in rows]
        summary["stages_ms"][stage] = {
            "mean": statistics.fmean(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "min": min(values),
            "max": max(values),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
