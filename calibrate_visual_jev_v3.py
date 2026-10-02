"""Fit independent temperature scalers and evaluate them on held-out V3 test data."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Callable

import torch

from eval_visual_jev_v3 import classification_metrics, collect_original
from jev.benchmark_v1 import read_jsonl
from jev.calibration import TemperatureScaler
from scripts.run_visual_jev_v2_experiment import file_sha256, open_cached_dataset
from train_visual_jev_v2 import load_records
from train_visual_jev_v3 import load_config, open_shards
from visual_jev_v2 import VisualJEVV2, VisualJEVV2Config
from visual_jev_v3 import VisualJEVV3
from visual_jev_v3_pipeline import MeanPoolJEV, ThreeStageVisualJEV


def _models(config: dict[str, Any], device: str) -> dict[str, tuple[Any, str]]:
    paper_root = Path(config["paper_root"])
    torch.manual_seed(config["seed"])
    random_model = VisualJEVV2(VisualJEVV2Config(2560, 2560, 128, 8)).to(device).eval()
    mean_payload = torch.load(paper_root / "checkpoints" / "meanpool_jev.pt", map_location=device, weights_only=False)
    meanpool = MeanPoolJEV(mean_payload["vision_dim"], mean_payload["text_dim"], mean_payload["hidden_dim"]).to(device)
    meanpool.load_state_dict(mean_payload["state_dict"])
    v2, _ = VisualJEVV2.from_checkpoint(config["v2_checkpoint"], map_location=device)
    v3, _ = VisualJEVV3.from_checkpoint("checkpoints/visual-jev-v3-coco-2k-img5.pt", map_location=device)
    result: dict[str, tuple[Any, str]] = {
        "random_untrained_head": (random_model, "aligned"),
        "meanpool_mlp_jev": (meanpool.eval(), "aligned"),
        "v2_candidate_aware": (v2.to(device).eval(), "aligned"),
        "v3_simple_postmerger": (v3.to(device).eval(), "aligned"),
    }
    for name, filename in (
        ("v3_without_stage_a", "v3_without_stage_a.pt"),
        ("v3_without_stage_c", "v3_without_stage_c.pt"),
        ("v3_full", "v3_full.pt"),
    ):
        model, _ = ThreeStageVisualJEV.from_checkpoint(paper_root / "checkpoints" / filename, map_location=device)
        result[name] = (model.to(device).eval(), "pre_merger")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/visual_jev_v3_paper.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/calibration.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    device = config["device"]
    data_root = Path(config["data_root"])
    paper_root = Path(config["paper_root"])
    validation_path = data_root / "splits" / "validation.jsonl"
    records = load_records(validation_path, data_root)
    source_sha = file_sha256(validation_path)
    aligned = open_cached_dataset(data_root / "features" / "validation.pt", source_sha256=source_sha, item_key="features")
    pre_merger = open_shards(paper_root / "features" / "validation-shards", len(records))
    if aligned is None or pre_merger is None:
        raise RuntimeError("validation feature caches are incomplete")

    calibration_manifest = Path(config["calibration"]["manifest"])
    test_manifest = calibration_manifest.with_name("test.jsonl")
    calibration_indices = [int(item.metadata["source_index"]) for item in read_jsonl(calibration_manifest)]
    test_indices = [int(item.metadata["source_index"]) for item in read_jsonl(test_manifest)]
    if set(calibration_indices) & set(test_indices):
        raise RuntimeError("calibration and test partitions overlap")
    reports = {}
    ece_bins = int(config["ece_bins"])
    for name, (model, feature_kind) in _models(config, device).items():
        def score(index: int, text_override: torch.Tensor | None, model=model, feature_kind=feature_kind):
            text = aligned[index]["text_features"] if text_override is None else text_override
            if feature_kind == "pre_merger":
                return model(pre_merger[index]["pre_tokens"], text).scores
            output = model(aligned[index]["visual_tokens"], text)
            return output if torch.is_tensor(output) else output.scores

        calibration_logits, calibration_labels = collect_original(score, calibration_indices, aligned, config["seed"])
        test_logits, test_labels = collect_original(score, test_indices, aligned, config["seed"])
        scaler = TemperatureScaler()
        fit = scaler.fit(calibration_logits, calibration_labels)
        with torch.no_grad():
            scaled_calibration = scaler(calibration_logits).float()
            scaled_test = scaler(test_logits).float()
        reports[name] = {
            "temperature_fit": scaler.state_payload(fit),
            "calibration_before": classification_metrics(calibration_logits, calibration_labels, ece_bins),
            "calibration_after": classification_metrics(scaled_calibration, calibration_labels, ece_bins),
            "test_before": classification_metrics(test_logits, test_labels, ece_bins),
            "test_after": classification_metrics(scaled_test, test_labels, ece_bins),
        }

    payload = {
        "method": "single scalar temperature scaling",
        "optimizer": "LBFGS",
        "seed": config["seed"],
        "candidate_order": "deterministically permuted per sample",
        "calibration_partition": {
            "source": str(calibration_manifest),
            "rule": "benchmark_v1 manifest membership",
            "count": len(calibration_indices),
            "indices": calibration_indices,
        },
        "test_partition": {
            "source": str(test_manifest),
            "rule": "benchmark_v1 manifest membership",
            "count": len(test_indices),
            "indices": test_indices,
        },
        "overlap_count": 0,
        "models": reports,
        "limitation": "The legacy Stage-B/C checkpoints used the even validation partition for model selection; temperature is independent of the held-out odd test partition but not of legacy checkpoint selection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = ["model", "temperature", "calibration_nll_before", "calibration_nll_after", "test_nll_before", "test_nll_after", "test_brier_before", "test_brier_after", "test_ece_before", "test_ece_after", "test_accuracy"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, report in reports.items():
            writer.writerow({
                "model": name,
                "temperature": report["temperature_fit"]["temperature"],
                "calibration_nll_before": report["calibration_before"]["nll"],
                "calibration_nll_after": report["calibration_after"]["nll"],
                "test_nll_before": report["test_before"]["nll"],
                "test_nll_after": report["test_after"]["nll"],
                "test_brier_before": report["test_before"]["brier"],
                "test_brier_after": report["test_after"]["brier"],
                "test_ece_before": report["test_before"]["ece"],
                "test_ece_after": report["test_after"]["ece"],
                "test_accuracy": report["test_after"]["accuracy"],
            })
    print(json.dumps({"output": str(args.output), "csv": str(csv_path), "models": {name: {"temperature": report["temperature_fit"]["temperature"], "test_nll_before": report["test_before"]["nll"], "test_nll_after": report["test_after"]["nll"], "test_ece_before": report["test_before"]["ece"], "test_ece_after": report["test_after"]["ece"]} for name, report in reports.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
