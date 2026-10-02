"""Finalize audited negative-transfer results after the controlled runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from run_fix_negative_transfer import Context, compact_metrics, selection_score, sha256, write_json
from visual_jev_v3_pipeline import ThreeStageVisualJEV


ROOT = Path("experiments/results/benchmark_v1/fix_negative_transfer")
DOMAINS = ("aokvqa", "coco_hard_negative", "iconqa_choice", "scienceqa_image_only")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_row(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["test"]
    pair = payload["semantic_counterfactual"]
    return {
        "experiment": name,
        **compact_metrics(metrics),
        "pair_both": pair["both_directions_accuracy"],
        "pair_flip": pair["prediction_flip_rate"],
        "checkpoint": payload.get("checkpoint"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    context = Context("cuda:0")
    selected_path = ROOT / "stagec_ablation" / "E5_gated_stagec_from_failed.json"
    selected = load(selected_path)
    model, _ = ThreeStageVisualJEV.from_checkpoint(selected["checkpoint"], map_location="cuda:0")
    validation = context.evaluate(model, "validation")
    validation_pair = context.pair_metrics(model, "validation")
    base, worst, macro = selection_score(validation)
    validation_score = base + 0.5 * validation_pair["both_directions_accuracy"] + 0.25 * validation_pair["prediction_flip_rate"]

    best_dir = ROOT / "best_model_results"
    best_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = best_dir / "visual_jev_v3_best.pt"
    model.save_checkpoint(
        checkpoint,
        stage="best-negative-transfer-fix",
        step=2,
        metadata={
            "selected_from": "E5R_gated_stagec_from_failed",
            "selection_split": "validation",
            "selection_rule": "overall + 0.5*macro-domain + 0.5*worst-domain + 0.5*Pair-both + 0.25*Pair-flip",
            "test_not_used_to_fit_or_choose_epoch": True,
        },
    )
    baseline = load(Path("experiments/results/benchmark_v1/main_results.json"))["before"]
    baseline_pair = load(Path("experiments/results/benchmark_v1/visual_dependency.json"))["semantic_counterfactual"]["before"]
    calibration = load(ROOT / "calibration_after_fix" / "calibration_comparison.json")
    calibration_methods = calibration["methods"]
    best_payload = {
        "selected_experiment": "E5R_gated_stagec_from_failed",
        "selection_split": "validation",
        "validation_selection_score": validation_score,
        "validation": validation,
        "validation_semantic_counterfactual": validation_pair,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "test": selected["test"],
        "test_semantic_counterfactual": selected["semantic_counterfactual"],
        "test_visual_dependency": selected["visual_dependency_test"],
        "baseline_before_fix": baseline,
        "baseline_pair_before_fix": baseline_pair,
        "controlled_subset": True,
        "full_scale_pending": True,
    }
    write_json(best_dir / "best_model_results.json", best_payload)
    test = selected["test"]
    row = {
        "experiment": "E5R_gated_stagec_from_failed",
        **compact_metrics(test),
        "pair_both": selected["semantic_counterfactual"]["both_directions_accuracy"],
        "pair_flip": selected["semantic_counterfactual"]["prediction_flip_rate"],
        "blank_noise_kl_u": selected["visual_dependency_test"]["summary"]["invalid_mean_kl_u"],
        "invalid_confidence_drop": selected["visual_dependency_test"]["summary"]["invalid_confidence_drop"],
        "checkpoint": str(checkpoint.resolve()),
    }
    with (best_dir / "best_model_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)

    e0 = load(ROOT / "pipeline_regression" / "results.json")
    conflict = load(ROOT / "gradient_conflict" / "gradient_conflict.json")["summary"]
    before = baseline["overall"]
    after = selected["test"]["overall"]
    root_causes = {
        "pipeline_regression_bug": {
            "present": False,
            "evidence": "new schema replay exactly reproduced old COCO-only metrics",
            "accuracy": e0["metrics"]["accuracy"],
            "macro_f1": e0["metrics"]["macro_f1"],
        },
        "candidate_label_or_variable_k_bug": {
            "present": False,
            "evidence": "assertions cover shuffle/remap, K=2/3/5, padding mask, CE, Brier, ECE inputs, KL-U and entropy/log(K)",
        },
        "evaluation_distribution_mismatch": {
            "present": True,
            "evidence": "88.28% is COCO-only; 53.13% is an equal four-domain mixed test, so the 35.15-point headline drop was not like-for-like",
        },
        "dataset_imbalance": {
            "present": True,
            "severity": "secondary on controlled subset; important at full raw scale",
            "evidence": "raw-proportional sampling exposed COCO only 6.25% versus 25% balanced, but both proxy runs reached 56.25% overall",
        },
        "gradient_conflict": {
            "present": True,
            "conflict_fraction": conflict["conflict_fraction"],
            "mean_cosine": conflict["mean_cosine"],
            "evidence": "PCGrad did not win overall; GradNorm best preserved old semantic-pair behaviour, so conflict is real but not the sole cause",
        },
        "ungated_stage_c_assumption": {
            "present": True,
            "largest_safe_proxy_contribution_accuracy_points": 100 * (after["accuracy"] - before["accuracy"]),
            "evidence": "same failed checkpoint plus gated Stage C improved mixed accuracy, NLL/Brier/ECE and Pair-both while preserving COCO accuracy",
        },
    }
    write_json(ROOT / "root_cause_analysis.json", root_causes)

    run_files = {
        "E1": ROOT / "sampling_ablation" / "E1_proportional_plain.json",
        "E2": ROOT / "sampling_ablation" / "E2_balanced_plain.json",
        "S2": ROOT / "sampling_ablation" / "S2_doremi_groupdro_proxy.json",
        "E3": ROOT / "gradient_method_ablation" / "E3_balanced_pcgrad.json",
        "E4": ROOT / "gradient_method_ablation" / "E4_balanced_gradnorm.json",
        "E5_from_E3": ROOT / "stagec_ablation" / "E5_gated_stagec.json",
        "E6_from_E3": ROOT / "stagec_ablation" / "E6_curriculum_preference_anchor.json",
        "E5_from_E4": ROOT / "stagec_ablation" / "E5_gated_stagec_from_e4.json",
        "E6_from_E4": ROOT / "stagec_ablation" / "E6_curriculum_preference_anchor_from_e4.json",
        "E5R": ROOT / "stagec_ablation" / "E5_gated_stagec_from_failed.json",
        "E6R": ROOT / "stagec_ablation" / "E6_curriculum_preference_anchor_from_failed.json",
    }
    runs = {name: load(path) for name, path in run_files.items()}
    stage_names = ["E5_from_E3", "E6_from_E3", "E5_from_E4", "E6_from_E4", "E5R", "E6R"]
    write_csv(ROOT / "stagec_ablation" / "stagec_ablation.csv", [result_row(name, runs[name]) for name in stage_names])
    write_json(ROOT / "stagec_ablation" / "stagec_ablation.json", {
        "selected_on_validation": "E5R",
        "runs": {name: runs[name] for name in stage_names},
        "note": "E5/E6 requested branches plus recovery branches from the previously failed mixed checkpoint",
    })
    matrix_rows = [{
        "experiment": "E0_COCO_pipeline_regression",
        "evaluation_scope": "old COCO-only test",
        "overall_accuracy": e0["metrics"]["accuracy"],
        "overall_macro_f1": e0["metrics"]["macro_f1"],
        "pair_both": e0["semantic_counterfactual"]["both_directions_accuracy"],
        "pair_flip": e0["semantic_counterfactual"]["prediction_flip_rate"],
    }]
    for name, payload in runs.items():
        matrix_rows.append({"evaluation_scope": "mixed four-domain test", **result_row(name, payload)})
    write_csv(ROOT / "experiment_matrix.csv", matrix_rows)
    write_json(ROOT / "experiment_matrix.json", {
        "E0": e0, "runs": runs, "selected": best_payload,
        "selection_policy": "method/epoch selected on validation; test is reporting only",
    })
    config_path = ROOT / "full_config.json"
    config = load(config_path)
    config.update({
        "selected_experiment": "E5R_gated_stagec_from_failed",
        "selected_checkpoint": str(checkpoint.resolve()),
        "selection_split": "validation",
        "test_used_for_model_selection": False,
        "calibration_fit_split": "calibration only",
    })
    write_json(config_path, config)

    md = f"""# Visual-JEV V3 negative-transfer repair (controlled benchmark_v1)

## What actually failed

- The pipeline and candidate remapping did **not** fail: E0 replayed the old COCO test at {e0['metrics']['accuracy']:.4f} Accuracy, {e0['metrics']['macro_f1']:.4f} Macro-F1, Pair-both {e0['semantic_counterfactual']['both_directions_accuracy']:.4f}, Pair-flip {e0['semantic_counterfactual']['prediction_flip_rate']:.4f}.
- The apparent 88.28% -> 53.13% fall compared different distributions: old COCO-only versus a four-domain mixed test.
- Real interference remains: {conflict['conflict_fraction']:.2%} of measured cross-domain gradient pairs had cosine < 0.
- Raw-size sampling exposed COCO at only 6.25%; balanced sampling restored it to 25%, although sampling alone did not solve all domains.
- The original Stage C incorrectly treated every random wrong image as uniformly invalid. Gating that loss by visual-dependency type produced the largest safe proxy gain.

## Selected controlled-subset model

Selected on validation only: `E5R_gated_stagec_from_failed`.

| Metric | Before fix | After gated Stage C |
|---|---:|---:|
| Overall Accuracy | {before['accuracy']:.4f} | {after['accuracy']:.4f} |
| Overall Macro-F1 | {before['macro_f1']:.4f} | {after['macro_f1']:.4f} |
| NLL | {before['nll']:.4f} | {after['nll']:.4f} |
| Brier | {before['brier']:.4f} | {after['brier']:.4f} |
| ECE | {before['ece']:.4f} | {after['ece']:.4f} |
| COCO Accuracy | {baseline['coco_hard_negative']['accuracy']:.4f} | {test['coco_hard_negative']['accuracy']:.4f} |
| A-OKVQA Accuracy | {baseline['aokvqa']['accuracy']:.4f} | {test['aokvqa']['accuracy']:.4f} |
| IconQA Accuracy | {baseline['iconqa_choice']['accuracy']:.4f} | {test['iconqa_choice']['accuracy']:.4f} |
| ScienceQA Accuracy | {baseline['scienceqa_image_only']['accuracy']:.4f} | {test['scienceqa_image_only']['accuracy']:.4f} |
| Pair-both | {baseline_pair['both_directions_accuracy']:.4f} | {selected['semantic_counterfactual']['both_directions_accuracy']:.4f} |
| Pair-flip | {baseline_pair['prediction_flip_rate']:.4f} | {selected['semantic_counterfactual']['prediction_flip_rate']:.4f} |

IconQA changed by one example on the 16-example test slice, so worst-domain recovery is not established; the H100 full-scale run remains mandatory.

## Calibration after the decision-model fix

| Method | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
"""
    for method in ("uncalibrated", "global_ts", "multi_domain_ts", "visual_adaptive_ts", "dirichlet"):
        metrics = calibration_methods[method]["test"]["overall"]
        md += f"| {method} | {metrics['accuracy']:.4f} | {metrics['nll']:.4f} | {metrics['brier']:.4f} | {metrics['ece']:.4f} |\n"
    md += """

Multi-domain TS is the best calibrated compromise here (lowest NLL and nearly lowest ECE), but uncalibrated still has the lowest Brier. Dirichlet changes argmax and overfits; it is rejected.

## H100 follow-up

Run the selected gated objective on full per-domain training sets with at least three seeds, repeat the transfer matrix and gradient audit, expand calibration/test beyond 64 examples each, and add external POPEv2/Winoground/SugarCrepe evaluation without refitting calibration.
"""
    (ROOT / "summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

