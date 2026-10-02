#!/usr/bin/env python3
"""Create a compact CSV from completed Open-Jev-VL experiment runs."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("vl-results.csv"))
    args = parser.parse_args()
    fields = [
        "run", "seed", "tuning_mode", "lora_rank", "trainable_parameters",
        "split", "count", "accuracy", "nll", "brier", "multiclass_ece",
        "mean_latency_seconds", "peak_memory_gib", "reload_max_error",
    ]
    rows = []
    for summary_path in sorted(args.run_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        run = json.loads((summary_path.parent / "run.json").read_text())
        model = json.loads((summary_path.parent / "checkpoint/model.json").read_text())
        training = [json.loads(line) for line in
                    (summary_path.parent / "training.jsonl").read_text().splitlines() if line]
        peak = max((row["peak_memory_gib"] for row in training), default=None)
        for split in ("calibrated_test", "calibrated_ood"):
            metric = summary["metrics"][split]
            rows.append({
                "run": summary_path.parent.name,
                "seed": run["seed"],
                "tuning_mode": summary["tuning_mode"],
                "lora_rank": model["lora_rank"],
                "trainable_parameters": run["trainable_parameters"],
                "split": split,
                "count": metric["count"],
                "accuracy": metric["accuracy"],
                "nll": metric["nll"],
                "brier": metric["brier"],
                "multiclass_ece": metric["multiclass_ece"],
                "mean_latency_seconds": metric["mean_latency_seconds"],
                "peak_memory_gib": peak,
                "reload_max_error": summary["checkpoint_reload_max_error"],
            })
    if not rows:
        parser.error("no completed */summary.json runs found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
