#!/usr/bin/env python3
"""Evaluate the official Visual Jev answer-SFT adapters on our fixed tests.

This script imports the official repository implementation, uses its exact
prompt/readout path, saves every raw logit, and computes metrics locally.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel


def macro_f1(predictions: list[int], labels: list[int]) -> float:
    classes = sorted(set(predictions) | set(labels))
    values = []
    for cls in classes:
        tp = sum(p == cls and y == cls for p, y in zip(predictions, labels))
        fp = sum(p == cls and y != cls for p, y in zip(predictions, labels))
        fn = sum(p != cls and y == cls for p, y in zip(predictions, labels))
        values.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return statistics.fmean(values)


def metrics(logits: list[list[float]], labels: list[int], bins: int = 10) -> dict[str, float | int]:
    rows = [torch.tensor(row, dtype=torch.float32) for row in logits]
    probs = [row.softmax(-1) for row in rows]
    predictions = [int(row.argmax()) for row in rows]
    confidences = np.asarray([float(row.max()) for row in probs])
    correct = np.asarray([float(p == y) for p, y in zip(predictions, labels)])
    nll = statistics.fmean(-math.log(max(float(row[y]), 1e-12)) for row, y in zip(probs, labels))
    brier = statistics.fmean(
        float(((row - F.one_hot(torch.tensor(y), row.numel()).float()) ** 2).sum())
        for row, y in zip(probs, labels)
    )
    ece = 0.0
    for low in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        high = low + 1.0 / bins
        mask = (confidences >= low) & (confidences < high if high < 1.0 else confidences <= high)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    return {
        "count": len(labels),
        "accuracy": float(correct.mean()),
        "macro_f1": macro_f1(predictions, labels),
        "nll": nll,
        "brier": brier,
        "ece": ece,
    }


def permute(candidates: list[str], label: int, index: int, seed: int) -> tuple[list[str], int, list[int]]:
    order = list(range(len(candidates)))
    random.Random(seed + index * 1009).shuffle(order)
    return [candidates[i] for i in order], order.index(label), order


@torch.no_grad()
def score(model, image_path: Path, question: str, candidates: list[str]) -> tuple[list[float], float]:
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    group = model.prepare_group(
        image,
        [{"qtype": "choice", "instruction": question, "candidates": candidates}],
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = model.run_independent(group)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return output["lm_option_logits"][0, : len(candidates)].float().cpu().tolist(), time.perf_counter() - started


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_manifest(model, path: Path, name: str, fixed_seed: int, output: Path) -> dict:
    rows = load_jsonl(path)
    raw = []
    logits, labels, timings = [], [], []
    for index, item in enumerate(rows):
        candidates = list(item["candidates"])
        label = int(item["label"])
        values, seconds = score(model, Path(item["image"]), item["question"], candidates)
        logits.append(values); labels.append(label); timings.append(seconds)
        raw.append({"uid": item.get("uid", f"{name}:{index}"), "label": label, "logits": values, "seconds": seconds})
        if (index + 1) % 50 == 0:
            print(json.dumps({"dataset": name, "completed": index + 1, "total": len(rows)}), flush=True)
    result = {"dataset": name, "manifest": str(path.resolve()), "metrics": metrics(logits, labels),
              "mean_model_seconds": statistics.fmean(timings), "records": raw}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def evaluate_controlled(model, repo: Path, fixed_seed: int, output: Path) -> dict:
    data_root = repo / "data/visual-jev-v2"
    records = load_jsonl(data_root / "splits/validation.jsonl")
    test_indices = [i for i in range(len(records)) if i % 2 == 1]
    raw, logits, labels, timings = [], [], [], []
    for position, index in enumerate(test_indices):
        item = records[index]
        original = [item["positive"], *item["hard_negatives"]]
        candidates, label, order = permute(original, 0, index, fixed_seed)
        values, seconds = score(model, data_root / item["image"], item["question"], candidates)
        logits.append(values); labels.append(label); timings.append(seconds)
        raw.append({"validation_index": index, "label": label, "candidate_order": order,
                    "logits": values, "seconds": seconds})
        if (position + 1) % 32 == 0:
            print(json.dumps({"dataset": "controlled_coco", "completed": position + 1, "total": len(test_indices)}), flush=True)

    pairs = load_jsonl(repo / "data/visual-jev-v3/pairs/validation.jsonl")
    allowed = set(test_indices)
    pair_rows = []
    for pair in pairs:
        source = int(pair["source_index"]); counterfactual = int(pair["counterfactual_index"])
        if source not in allowed or counterfactual not in allowed:
            continue
        candidates = list(pair["candidate_set"])
        source_logits, _ = score(model, data_root / records[source]["image"], "Which statement is supported by the image?", candidates)
        cf_logits, _ = score(model, data_root / records[counterfactual]["image"], "Which statement is supported by the image?", candidates)
        source_pred = int(np.argmax(source_logits)); cf_pred = int(np.argmax(cf_logits))
        pair_rows.append({"source_index": source, "counterfactual_index": counterfactual,
                          "source_target": 0, "counterfactual_target": 1,
                          "source_logits": source_logits, "counterfactual_logits": cf_logits,
                          "source_prediction": source_pred, "counterfactual_prediction": cf_pred})
    pair_n = len(pair_rows)
    pair_summary = {
        "pair_count": pair_n,
        "source_accuracy": statistics.fmean(float(r["source_prediction"] == 0) for r in pair_rows),
        "counterfactual_accuracy": statistics.fmean(float(r["counterfactual_prediction"] == 1) for r in pair_rows),
        "both_directions_accuracy": statistics.fmean(float(r["source_prediction"] == 0 and r["counterfactual_prediction"] == 1) for r in pair_rows),
        "prediction_flip_rate": statistics.fmean(float(r["source_prediction"] != r["counterfactual_prediction"]) for r in pair_rows),
        "records": pair_rows,
    }
    result = {"dataset": "controlled_coco", "test_indices": test_indices, "metrics": metrics(logits, labels),
              "mean_model_seconds": statistics.fmean(timings), "semantic_counterfactual": pair_summary,
              "records": raw}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-jev-root", type=Path, default=Path.cwd())
    parser.add_argument("--official-root", type=Path, default=Path("/home/jiezuo/projects/Visual-Jev-official"))
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default="guanxuyu/visual-jev-4b-answer-sft")
    parser.add_argument("--seed-index", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--fixed-eval-seed", type=int, default=20260928)
    parser.add_argument("--datasets", nargs="+", default=["controlled", "aokvqa", "scienceqa_image_only", "iconqa_choice", "coco_hard_negative"])
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/official_visual_jev_matched"))
    args = parser.parse_args()
    repo = args.open_jev_root.resolve(); official = args.official_root.resolve()
    sys.path.insert(0, str(official / "code"))
    from vdm.models.vdm_model import VDM

    output_root = args.output_root.resolve() / f"seed-{args.seed_index}"
    output_root.mkdir(parents=True, exist_ok=True)
    model = VDM(str((repo / args.model).resolve()), with_heads=False)
    kwargs = {} if args.seed_index == 0 else {"subfolder": f"seed{args.seed_index}"}
    model.backbone = PeftModel.from_pretrained(model.backbone, args.adapter, **kwargs).eval()
    model.keep_full_lm_logits = False

    results = {}
    for name in args.datasets:
        destination = output_root / f"{name}.json"
        if destination.exists():
            results[name] = json.loads(destination.read_text(encoding="utf-8"))
            continue
        if name == "controlled":
            results[name] = evaluate_controlled(model, repo, args.fixed_eval_seed, destination)
        else:
            path = repo / "data/independent_datasets/full" / name / "test.jsonl"
            results[name] = evaluate_manifest(model, path, name, args.fixed_eval_seed, destination)
    official_commit = subprocess.check_output(
        ["git", "-C", str(official), "rev-parse", "HEAD"], text=True
    ).strip()
    summary = {"official_repo": str(official), "official_commit": official_commit,
               "adapter": args.adapter, "seed_index": args.seed_index,
               "fixed_eval_seed": args.fixed_eval_seed,
               "results": {name: value["metrics"] for name, value in results.items()}}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
