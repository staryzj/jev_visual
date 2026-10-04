#!/usr/bin/env python3
"""Create auditable uncertainty summaries for full independent-dataset runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


OUTPUTS = ("coco", "aokvqa", "scienceqa", "iconqa")


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError(f"invalid binomial count: {successes}/{total}")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {
        "successes": successes,
        "total": total,
        "proportion": proportion,
        "lower_95": 0.0 if successes == 0 else max(0.0, center - half),
        "upper_95": 1.0 if successes == total else min(1.0, center + half),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("experiments/results/independent_datasets_full"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.result_root.resolve()
    rows = []
    missing = []
    for name in OUTPUTS:
        result_path = root / name / "result.json"
        if not result_path.is_file():
            missing.append(name)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = result["metrics"]
        total = int(metrics["count"])
        successes = round(float(metrics["accuracy"]) * total)
        interval = wilson(successes, total)
        row = {
            "dataset": result["dataset"],
            "test_n": total,
            "correct": successes,
            "accuracy": metrics["accuracy"],
            "accuracy_wilson_lower_95": interval["lower_95"],
            "accuracy_wilson_upper_95": interval["upper_95"],
            "macro_f1": metrics["macro_f1"],
            "nll": metrics["nll"],
            "brier": metrics["brier"],
            "ece": metrics["ece"],
        }
        pair = result["visual_dependency"].get("semantic_counterfactual", {})
        if pair.get("reported", True) and pair.get("pair_count"):
            pair_n = int(pair["pair_count"])
            both = round(float(pair["both_directions_accuracy"]) * pair_n)
            flip = round(float(pair["prediction_flip_rate"]) * pair_n)
            both_interval = wilson(both, pair_n)
            flip_interval = wilson(flip, pair_n)
            row.update(
                {
                    "semantic_pair_n": pair_n,
                    "pair_both": pair["both_directions_accuracy"],
                    "pair_both_wilson_lower_95": both_interval["lower_95"],
                    "pair_both_wilson_upper_95": both_interval["upper_95"],
                    "pair_flip": pair["prediction_flip_rate"],
                    "pair_flip_wilson_lower_95": flip_interval["lower_95"],
                    "pair_flip_wilson_upper_95": flip_interval["upper_95"],
                }
            )
        rows.append(row)

    if missing and not args.allow_incomplete:
        raise SystemExit("missing completed result(s): " + ", ".join(missing))
    payload = {
        "interval": "two-sided Wilson score, nominal 95%",
        "training_variance_included": False,
        "warning": "These intervals quantify held-out binomial sampling only; report multi-seed variation separately.",
        "missing": missing,
        "rows": rows,
    }
    (root / "statistics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if rows:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with (root / "statistics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
