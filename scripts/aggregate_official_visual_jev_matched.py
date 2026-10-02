#!/usr/bin/env python3
"""Aggregate completed official Visual Jev matched runs without recomputation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


FIELDS = ("accuracy", "macro_f1", "nll", "brier", "ece")


def stats(values: list[float]) -> dict:
    return {"values": values, "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/results/official_visual_jev_matched"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    seed_summaries = {}
    for seed in args.seeds:
        path = args.root / f"seed-{seed}" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        seed_summaries[seed] = json.loads(path.read_text(encoding="utf-8"))
    names = list(seed_summaries[args.seeds[0]]["results"])
    aggregate = {"seeds": args.seeds, "datasets": {}}
    for name in names:
        aggregate["datasets"][name] = {
            field: stats([float(seed_summaries[seed]["results"][name][field]) for seed in args.seeds])
            for field in FIELDS
        }
        if name == "controlled":
            pair_rows = []
            for seed in args.seeds:
                detail = json.loads((args.root / f"seed-{seed}" / "controlled.json").read_text(encoding="utf-8"))
                pair_rows.append(detail["semantic_counterfactual"])
            aggregate["datasets"][name]["pair_both_accuracy"] = stats(
                [float(row["both_directions_accuracy"]) for row in pair_rows]
            )
            aggregate["datasets"][name]["pair_flip_rate"] = stats(
                [float(row["prediction_flip_rate"]) for row in pair_rows]
            )
    destination = args.root / "aggregate.json"
    destination.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
