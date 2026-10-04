"""Combine per-backbone Visual-JEV adapter results into JSON, CSV, and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") not in {"success", "completed"}:
        raise RuntimeError(f"incomplete result: {path}")
    return payload


def compact(payload: dict) -> dict:
    test = payload["test_uncalibrated"]
    calibrated = payload["test_calibrated"]
    extraction = payload["backbone_extraction"]
    visual = payload["visual_dependency"]
    latency = payload["latency"]
    return {
        "backbone": payload["backbone"],
        "test_examples": payload["split_counts"]["test"],
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "nll": test["nll"],
        "ece": test["ece"],
        "temperature": payload["temperature"],
        "calibrated_nll": calibrated["nll"],
        "calibrated_ece": calibrated["ece"],
        "wrong_image_js": visual["wrong_image"]["js_from_original"],
        "wrong_image_flip_rate": visual["wrong_image"]["flip_rate_from_original"],
        "blank_js": visual["blank"]["js_from_original"],
        "native_shapes": extraction["native_shapes"],
        "shared_token_count": extraction["token_count"],
        "hidden_dim": extraction["hidden_dim"],
        "extraction_mean_ms": extraction["mean_extraction_ms"],
        "extraction_p95_ms": extraction["p95_extraction_ms"],
        "adapter_plus_jev_mean_ms": latency["adapter_plus_jev_mean_ms"],
        "peak_gpu_memory_gib": extraction["peak_gpu_memory_gib"],
        "runtime_seconds": payload["runtime_seconds"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "claim_boundary": payload["claim_boundary"],
    }


def markdown(rows: list[dict]) -> str:
    lines = [
        "# Frozen-backbone adapter experiment summary",
        "",
        "All rows use 512/80/80/80 COCO hard-negative examples, a frozen visual backbone, "
        "cached Qwen candidate text features, a fixed Visual-JEV decision head, and a newly "
        "trained backbone-specific adapter. Results are controlled-subset evidence, not "
        "full-dataset or universal backbone-generality claims.",
        "",
        "| Backbone | Native tokens | Acc. | Macro-F1 | ECE | Cal. ECE | Extract mean (ms) | Adapter+JEV (ms) | Wrong-image JS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        shapes = ", ".join("x".join(map(str, shape)) for shape in row["native_shapes"])
        lines.append(
            f"| {row['backbone']} | {shapes} | {row['accuracy']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['ece']:.4f} | {row['calibrated_ece']:.4f} | "
            f"{row['extraction_mean_ms']:.2f} | {row['adapter_plus_jev_mean_ms']:.3f} | "
            f"{row['wrong_image_js']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: InternVL shows a material blank/noise response but little "
            "wrong-image sensitivity on this subset; SigLIP2 and LLaVA-OneVision also "
            "retain high accuracy under visual controls. Accuracy therefore cannot be "
            "treated as evidence of strong visual dependence by itself.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.glob("*/results.json"))
    if not paths:
        raise FileNotFoundError(f"no per-backbone results under {args.root}")
    rows = [compact(load_result(path)) for path in paths]
    preferred = {"siglip2": 0, "internvl": 1, "llava": 2}
    rows.sort(key=lambda row: preferred.get(row["backbone"], 99))
    (args.root / "summary.json").write_text(
        json.dumps({"status": "completed", "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "SUMMARY.md").write_text(markdown(rows), encoding="utf-8")
    print(markdown(rows))


if __name__ == "__main__":
    main()
