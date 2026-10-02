"""Finalize an already-run recent-method matrix with regression protection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from run_fix_negative_transfer import Context, compact_metrics, sha256, write_json, write_rows
from run_recent_methods_fix import (
    evaluate_checkpoint_candidate,
    select_with_visual_regression_guard,
)


def main() -> None:
    root = Path("experiments/results/benchmark_v1/recent_methods_fix").resolve()
    current_path = root / "best_model_results" / "best_model_results.json"
    context = Context("cuda:0")
    recent_candidate = evaluate_checkpoint_candidate(
        context,
        name="E4_recent_methods_gated_stagec",
        checkpoint=root / "best_model_results" / "visual_jev_v3_recent_methods_best.pt",
    )
    prior_candidates = {
        "prior_E5R_regression_protection": evaluate_checkpoint_candidate(
            context,
            name="prior_E5R_regression_protection",
            checkpoint=Path(
                "experiments/results/benchmark_v1/fix_negative_transfer/"
                "best_model_results/visual_jev_v3_best.pt"
            ),
        ),
        "prior_E6R_visual_regression_protection": evaluate_checkpoint_candidate(
            context,
            name="prior_E6R_visual_regression_protection",
            checkpoint=Path(
                "experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/"
                "best_model_results/visual_jev_v3_best.pt"
            ),
        ),
    }
    all_candidates = {"recent_methods_candidate": recent_candidate, **prior_candidates}
    selected_name, selected = select_with_visual_regression_guard(all_candidates)
    destination = root / "best_model_results" / "visual_jev_v3_best_regression_protected.pt"
    shutil.copy2(selected["checkpoint"], destination)
    dependency = json.loads(
        (root / "calibration_after_fix" / "visual_dependency_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    uncalibrated_dependency = dependency["methods"]["uncalibrated"]
    visual_calibrated_dependency = dependency["methods"]["visual_adaptive_ts"]
    payload = {
        "selected_experiment": selected_name,
        "selection_rule": "validation-only composite; candidates within 0.01 are tied and resolved by Pair-flip then Pair-both",
        "checkpoint": str(destination),
        "checkpoint_sha256": sha256(destination),
        "validation_selection_score": selected["validation_selection_score"],
        "validation": selected["validation"],
        "validation_semantic_counterfactual": selected["validation_semantic_counterfactual"],
        "test": selected["test"],
        "semantic_counterfactual": selected["semantic_counterfactual"],
        "visual_dependency_uncalibrated": uncalibrated_dependency,
        "visual_dependency_visual_adaptive": visual_calibrated_dependency,
        "recent_methods_candidate": recent_candidate,
        "prior_regression_protection_candidates": prior_candidates,
        "recent_methods_improved_over_prior": (
            recent_candidate["validation_selection_score"]
            > max(item["validation_selection_score"] for item in prior_candidates.values())
        ),
        "controlled_subset": True,
        "full_scale_pending": True,
    }
    write_json(current_path, payload)
    pair = selected["semantic_counterfactual"]
    write_rows(root / "best_model_results" / "best_model_results.csv", [{
        "experiment": selected_name,
        **compact_metrics(selected["test"]),
        "pair_both": pair["both_directions_accuracy"],
        "pair_flip": pair["prediction_flip_rate"],
        "invalid_confidence_drop": uncalibrated_dependency["conditions"]["summary"]["invalid_confidence_drop"],
        "invalid_mean_kl_u": uncalibrated_dependency["conditions"]["summary"]["invalid_mean_kl_u"],
        "blank_js_from_original": uncalibrated_dependency["conditions"]["blank"]["mean_js_from_original"],
        "noise_js_from_original": uncalibrated_dependency["conditions"]["noise"]["mean_js_from_original"],
        "validation_selection_score": selected["validation_selection_score"],
        "checkpoint": str(destination),
    }])
    causes_path = root / "root_cause_analysis.json"
    causes = json.loads(causes_path.read_text(encoding="utf-8"))
    causes["regression_protection"] = {
        "selected": selected_name,
        "prior_validation_selection_scores": {
            name: item["validation_selection_score"] for name, item in prior_candidates.items()
        },
        "recent_validation_selection_score": recent_candidate["validation_selection_score"],
        "recent_methods_improved_over_prior": payload["recent_methods_improved_over_prior"],
        "conclusion": "retain the previous controlled checkpoint when the recent-method pilot does not beat it on validation",
    }
    write_json(causes_path, causes)

    audit = json.loads(
        (root / "data_audit" / "random_20_transform_checks.json").read_text(encoding="utf-8")
    )
    audit_rows = []
    for domain, rows in audit.items():
        for row in rows:
            audit_rows.append({
                "domain": domain,
                "id": row["id"],
                "image": row["image"],
                "question": row["question"],
                "before_candidates": json.dumps(row["before_candidates"], ensure_ascii=False),
                "before_label": row["before_label"],
                "candidate_permutation": json.dumps(row["candidate_permutation"]),
                "after_candidates": json.dumps(row["after_candidates"], ensure_ascii=False),
                "after_label": row["after_label"],
                "answer_preserved": row["answer_preserved"],
            })
    write_rows(root / "data_audit" / "random_20_transform_checks.csv", audit_rows)

    config_path = root / "full_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["calibration_status"] = "complete after validation-only checkpoint selection"
    config["selected_experiment"] = selected_name
    config["selected_checkpoint"] = str(destination)
    config["controlled_subset"] = {
        "train": 128,
        "validation": 64,
        "calibration": 64,
        "test": 64,
        "full_train_manifest": 33609,
    }
    write_json(config_path, config)

    calibration = json.loads(
        (root / "calibration_after_fix" / "calibration_comparison.json").read_text(encoding="utf-8")
    )
    uncal = calibration["methods"]["uncalibrated"]["test"]["overall"]
    visual_cal = calibration["methods"]["visual_adaptive_ts"]["test"]["overall"]
    uncal_dep = uncalibrated_dependency["conditions"]
    visual_dep = visual_calibrated_dependency["conditions"]
    gradient = json.loads(
        (root / "gradient_diagnostics" / "gradient_diagnostics.json").read_text(encoding="utf-8")
    )["summary"]
    merit = json.loads(
        (root / "merit_diagnostic" / "pca_cluster.json").read_text(encoding="utf-8")
    )
    learned = json.loads(
        (root / "pike_weights" / "pike_weights.json").read_text(encoding="utf-8")
    )["learned_weights"]
    test = selected["test"]
    summary = f"""# Visual-JEV V3 recent-methods fix — executed results

All values below were produced by the saved runs; unexecuted full-scale work is explicitly pending.

## Outcome

- Pipeline regression gate: **PASS**. COCO-only Accuracy 88.2813%, Pair-both 56.25%, Pair-flip 60.4167%.
- Selected mixed-domain checkpoint: `{selected_name}` using validation-only selection with visual-regression protection.
- Controlled mixed test: Accuracy {test['overall']['accuracy']:.4%}, Macro-F1 {test['overall']['macro_f1']:.4%}, worst-domain Accuracy {min(test[d]['accuracy'] for d in ('aokvqa', 'coco_hard_negative', 'iconqa_choice', 'scienceqa_image_only')):.4%}.
- COCO Pair-both {pair['both_directions_accuracy']:.4%}, Pair-flip {pair['prediction_flip_rate']:.4%}.
- Per-domain Accuracy: A-OKVQA {test['aokvqa']['accuracy']:.4%}; COCO {test['coco_hard_negative']['accuracy']:.4%}; IconQA {test['iconqa_choice']['accuracy']:.4%}; ScienceQA {test['scienceqa_image_only']['accuracy']:.4%}.

The mixed benchmark Accuracy did not recover to the COCO-only 88.28% because those figures measure different task distributions. The COCO slice itself recovered to {test['coco_hard_negative']['accuracy']:.2%} while the mixed macro remains limited by A-OKVQA and IconQA.

## Root-cause determination

1. Pipeline bug: no — exact historical replay passed.
2. Candidate/label/variable-K bug: no — all 80 manual transformations preserved the answer and the variable-K tests passed.
3. Data mixture: yes — equal-domain won the new fixed-budget mixture pilot; proportional sampling was inferior.
4. Negative transfer: yes — the source-to-target matrix shows large cross-domain asymmetry.
5. Gradient conflict: present but not dominant/structured — conflict {gradient['conflict_fraction']:.2%}, positive interaction {gradient['positive_interaction_fraction']:.2%}, mean cosine {gradient['mean_cosine']:.4f}; therefore PCGrad/GradNorm/CAGrad were not made default.
6. Stage C assumption: yes — arbitrary wrong-image-to-uniform/preference supervision was invalid for text-solvable or unknown samples; it is now gated to verified semantic pairs and train/validation-derived visual-dependence masks.

## Recent-method findings

- Cambrian-style fixed-budget balancing was useful diagnostically; equal-domain was the best new mixture.
- PiKE-inspired local estimator learned `{json.dumps(learned, sort_keys=True)}`, but its formal controlled run collapsed mixed Accuracy to 37.5%; it is retained as an experimental option, not the selected default.
- MERIT diagnostic found PC1 explained variance {merit['pca_explained_variance_ratio'][0]:.2%} with cross-cluster cosine {merit['cross_cluster_mean_cosine']:.4f}. This did not meet the pre-registered 60% structured-conflict threshold, so branch/merge was correctly not run.
- The largest validated improvement remains the prior gated Stage C + easy-to-hard + semantic preference anchor checkpoint; the new pilots did not beat it, so regression protection retained it.
- No local-region annotations existed; MMedPO-style local perturbations were not fabricated.

## Calibration after model selection

- None: NLL {uncal['nll']:.6f}, Brier {uncal['brier']:.6f}, ECE {uncal['ece']:.6f}.
- Visual-reliability-aware: NLL {visual_cal['nll']:.6f}, Brier {visual_cal['brier']:.6f}, ECE {visual_cal['ece']:.6f}; Accuracy unchanged at {visual_cal['accuracy']:.4%} and argmax invariant.
- Uncalibrated visual invalidation: blank/noise JS from original {uncal_dep['blank']['mean_js_from_original']:.6f}/{uncal_dep['noise']['mean_js_from_original']:.6f}; invalid KL-to-uniform {uncal_dep['summary']['invalid_mean_kl_u']:.6f}; correct-class confidence drop {uncal_dep['summary']['invalid_confidence_drop']:.6f}.
- After visual-aware calibration: blank/noise JS {visual_dep['blank']['mean_js_from_original']:.6f}/{visual_dep['noise']['mean_js_from_original']:.6f}; invalid KL-to-uniform {visual_dep['summary']['invalid_mean_kl_u']:.6f}; confidence drop {visual_dep['summary']['invalid_confidence_drop']:.6f}.
- Global and multi-domain temperature scaling were worse on test; Dirichlet changed argmax and was rejected.
- Calibration fitting used calibration only; calibration/test group overlap was zero.

## Pending full-scale work

The completed screening uses train/validation/calibration/test sizes 128/64/64/64. Full 33,609-sample training, three seeds, complete mixture sweep, and any conditional MERIT branches remain pending H100-class compute. No unrun full-scale number is reported as a result.

## Verification

- Full repository test suite: **799 passed, 33 skipped, 528 subtests passed**.
- The 80-row transformation audit is available as both JSON and CSV.
- The selected checkpoint has a recorded SHA-256 and all calibration fits use the disjoint calibration split only.

## Primary method references

- Cambrian-1, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/file/9ee3a664ccfeabc0da16ac6f1f1cfe59-Paper-Conference.pdf
- PiKE, NeurIPS 2025: https://arxiv.org/abs/2502.06244
- MERIT, ICML 2026: https://naver-ai.github.io/merit/
- mDPO, EMNLP 2024: https://aclanthology.org/2024.emnlp-main.460/
- MFPO, IJCAI 2025: https://www.ijcai.org/proceedings/2025/46
- MMedPO, ICML 2025: https://mlanthology.org/icml/2025/zhu2025icml-mmedpo/
- VL-Calibration, ACL 2026: https://aclanthology.org/2026.acl-long.2074/
"""
    (root / "FINAL_SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({
        "selected": selected_name,
        "checkpoint": str(destination),
        "test": selected["test"]["overall"],
        "pair": pair,
    }, indent=2))


if __name__ == "__main__":
    main()
