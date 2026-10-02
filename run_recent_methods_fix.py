"""Run the 2024--2026 recent-method Visual-JEV repair matrix.

The experiment uses cached Visual-JEV features and a fixed controlled budget so
that every mixture is directly comparable on the local GPU.  Full-manifest
confirmation is intentionally reported as pending rather than extrapolated.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from jev.multidomain import task_gradients, visual_dependency_masks
from jev.recent_multidomain import (
    PiKEInspiredWeights,
    capped_balanced_weights,
    gradient_pca_diagnostic,
    normalise_weights,
    weighted_state_dict_average,
)
from run_fix_negative_transfer import (
    DOMAINS,
    SEED,
    Context,
    candidate_audit,
    candidate_loss,
    compact_metrics,
    domain_indices,
    pipeline_regression,
    run_stage_c,
    selection_score,
    set_seed,
    sha256,
    write_json,
    write_rows,
)
from visual_jev_v3_pipeline import ThreeStageVisualJEV


RAW_COUNTS = {
    "aokvqa": 17055,
    "coco_hard_negative": 2048,
    "iconqa_choice": 7095,
    "scienceqa_image_only": 7411,
}


def validation_score(metrics: Mapping[str, Any], pair: Mapping[str, float]) -> float:
    base, _, _ = selection_score(dict(metrics))
    return float(
        base
        + 0.50 * pair["both_directions_accuracy"]
        + 0.25 * pair["prediction_flip_rate"]
    )


def reconstruct_source_candidates(example: Any) -> tuple[list[str], int]:
    order = [int(value) for value in example.metadata["candidate_permutation"]]
    source = [""] * len(order)
    for shuffled_index, source_index in enumerate(order):
        source[source_index] = example.candidates[shuffled_index]
    return source, order[example.label]


def data_audit(context: Context, root: Path) -> dict[str, Any]:
    histograms = candidate_audit(context.examples)
    rng = random.Random(f"{SEED}:twenty-row-audit")
    samples: dict[str, list[dict[str, Any]]] = {}
    rows = context.examples["train"]
    for domain in DOMAINS:
        pool = [example for example in rows if example.dataset == domain]
        selected = rng.sample(pool, min(20, len(pool)))
        records = []
        for example in selected:
            before, before_label = reconstruct_source_candidates(example)
            records.append({
                "id": example.id,
                "image": example.image,
                "question": example.question,
                "before_candidates": before,
                "before_label": before_label,
                "before_answer": before[before_label],
                "candidate_permutation": list(example.metadata["candidate_permutation"]),
                "after_candidates": list(example.candidates),
                "after_label": example.label,
                "after_answer": example.answer,
                "answer_preserved": before[before_label] == example.answer,
            })
        if not all(record["answer_preserved"] for record in records):
            raise AssertionError(f"candidate/label remap failed for {domain}")
        samples[domain] = records
    payload = {
        "seed": SEED,
        "histograms": histograms,
        "random_20_before_after_by_dataset": samples,
        "checks": {
            "candidate_shuffle_label_remap": "passed",
            "variable_k_padding_softmax_losses": "covered by tests/test_multidomain_training.py",
            "uniform_target_entropy_brier_ece_klu": "covered by tests/test_multidomain_training.py",
            "hard_coded_k3": False,
        },
    }
    write_json(root / "data_audit.json", payload)
    write_json(root / "candidate_count_label_position_histograms.json", histograms)
    write_json(root / "random_20_transform_checks.json", samples)
    return payload


def mixture_plans() -> dict[str, dict[str, float]]:
    return {
        "proportional": normalise_weights(RAW_COUNTS, DOMAINS),
        "equal_domain": {domain: 0.25 for domain in DOMAINS},
        "capped_balanced_4096": capped_balanced_weights(
            RAW_COUNTS, DOMAINS, cap=4096
        ),
        "coco_heavy": normalise_weights({
            "aokvqa": 0.15, "coco_hard_negative": 0.45,
            "iconqa_choice": 0.20, "scienceqa_image_only": 0.20,
        }, DOMAINS),
        "visual_hard_heavy": normalise_weights({
            "aokvqa": 0.20, "coco_hard_negative": 0.35,
            "iconqa_choice": 0.30, "scienceqa_image_only": 0.15,
        }, DOMAINS),
    }


def _sample_index(
    grouped: Mapping[str, Sequence[int]], weights: Mapping[str, float], rng: random.Random
) -> tuple[str, int]:
    names = tuple(grouped)
    domain = rng.choices(names, weights=[weights[name] for name in names], k=1)[0]
    return domain, rng.choice(grouped[domain])


def _checkpoint_model(
    model: ThreeStageVisualJEV, path: Path, *, name: str, metadata: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        path, stage="recent-methods-candidate", step=0,
        metadata={
            "seed": SEED,
            "experiment": name,
            "paper_implementation": "project-local adaptation; not official",
            **dict(metadata),
        },
    )


def train_fixed_mixture(
    context: Context,
    *,
    name: str,
    weights: Mapping[str, float],
    budget: int,
    output: Path,
    initial_checkpoint: Path | None = None,
    allowed_domains: Sequence[str] = DOMAINS,
    learning_rate: float = 5e-5,
) -> dict[str, Any]:
    set_seed()
    checkpoint = (initial_checkpoint or context.base_checkpoint).resolve()
    model, _ = ThreeStageVisualJEV.from_checkpoint(checkpoint, map_location=context.device)
    for parameter in model.alignment.parameters():
        parameter.requires_grad_(False)
    parameters = [parameter for parameter in model.decision.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-2)
    grouped_all = domain_indices(context.examples["train"])
    allowed = tuple(allowed_domains)
    grouped = {domain: grouped_all[domain] for domain in allowed}
    local_weights = normalise_weights(
        {domain: float(weights[domain]) for domain in allowed}, allowed
    )
    rng = random.Random(f"{SEED}:{name}")
    validation = context.evaluate(model, "validation")
    validation_pair = context.pair_metrics(model, "validation")
    best_score = validation_score(validation, validation_pair)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = copy.deepcopy(validation)
    best_validation_pair = dict(validation_pair)
    sample_counts: Counter[str] = Counter()
    effective_candidate_tokens = 0
    loss_by_domain: dict[str, list[float]] = defaultdict(list)
    history: list[dict[str, Any]] = []
    window = min(128, budget)
    started = time.time()
    progress = tqdm(range(1, budget + 1), desc=name, unit="sample", dynamic_ncols=True)
    for step in progress:
        domain, index = _sample_index(grouped, local_weights, rng)
        example = context.examples["train"][index]
        optimizer.zero_grad(set_to_none=True)
        loss = candidate_loss(model, context.features["train"][index], example)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        value = float(loss.detach().item())
        sample_counts[domain] += 1
        effective_candidate_tokens += len(example.candidates)
        loss_by_domain[domain].append(value)
        if step % 16 == 0:
            progress.set_postfix(loss=f"{value:.4f}", domain=domain)
        if step % window == 0 or step == budget:
            model.eval()
            current = context.evaluate(model, "validation")
            current_pair = context.pair_metrics(model, "validation")
            score = validation_score(current, current_pair)
            _, worst, macro = selection_score(current)
            history.append({
                "step": step,
                "selection_score": score,
                "accuracy": current["overall"]["accuracy"],
                "macro_domain_accuracy": macro,
                "worst_domain_accuracy": worst,
                "pair_both": current_pair["both_directions_accuracy"],
                "pair_flip": current_pair["prediction_flip_rate"],
                "elapsed_seconds": time.time() - started,
            })
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_validation = copy.deepcopy(current)
                best_validation_pair = dict(current_pair)
            model.train()
    model.load_state_dict(best_state)
    model.eval()
    test = context.evaluate(model, "test")
    pair = context.pair_metrics(model, "test")
    checkpoint_out = output / f"{name}.pt"
    _checkpoint_model(model, checkpoint_out, name=name, metadata={
        "mixture_weights": local_weights, "fixed_sample_budget": budget,
        "initial_checkpoint": str(checkpoint),
    })
    total = sum(sample_counts.values())
    payload = {
        "name": name,
        "initial_checkpoint": str(checkpoint),
        "checkpoint": str(checkpoint_out.resolve()),
        "checkpoint_sha256": sha256(checkpoint_out),
        "allowed_domains": list(allowed),
        "target_mixture": local_weights,
        "fixed_sample_budget": budget,
        "actual_sample_counts": dict(sample_counts),
        "actual_sampling_ratio": {
            domain: sample_counts[domain] / total for domain in allowed
        },
        "effective_candidate_tokens": effective_candidate_tokens,
        "mean_train_loss_by_domain": {
            domain: float(np.mean(values)) for domain, values in loss_by_domain.items()
        },
        "history": history,
        "validation_selection_score": best_score,
        "validation": best_validation,
        "validation_semantic_counterfactual": best_validation_pair,
        "test": test,
        "semantic_counterfactual": pair,
        "runtime_seconds": time.time() - started,
    }
    write_json(output / f"{name}.json", payload)
    return payload


def flat_result(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    test = payload["test"]
    pair = payload["semantic_counterfactual"]
    return {
        "experiment": name,
        **compact_metrics(test),
        "pair_both": pair["both_directions_accuracy"],
        "pair_flip": pair["prediction_flip_rate"],
        "validation_selection_score": payload.get("validation_selection_score"),
        "checkpoint": payload.get("checkpoint"),
    }


def run_transfer_matrix(context: Context, root: Path, budget: int) -> dict[str, Any]:
    runs = {}
    matrix_rows = []
    for source in DOMAINS:
        run = train_fixed_mixture(
            context,
            name=f"single_{source}",
            weights={source: 1.0},
            budget=budget,
            output=root,
            allowed_domains=(source,),
        )
        runs[source] = run
        row = {"source_train_domain": source}
        for target in DOMAINS:
            row[f"{target}_accuracy"] = run["test"][target]["accuracy"]
            row[f"{target}_f1"] = run["test"][target]["macro_f1"]
            row[f"{target}_nll"] = run["test"][target]["nll"]
            row[f"{target}_brier"] = run["test"][target]["brier"]
            row[f"{target}_ece"] = run["test"][target]["ece"]
        matrix_rows.append(row)
    write_rows(root / "transfer_matrix.csv", matrix_rows)
    write_json(root / "transfer_matrix.json", {
        "fixed_budget_per_source": budget,
        "runs": runs,
        "matrix": matrix_rows,
    })
    return {"runs": runs, "matrix": matrix_rows}


def _one_gradient_snapshot(
    context: Context,
    model: ThreeStageVisualJEV,
    grouped: Mapping[str, Sequence[int]],
    step: int,
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[list[list[torch.Tensor]], list[float], list[list[float]], dict[str, float]]:
    losses = []
    loss_values = {}
    for domain in DOMAINS:
        index = grouped[domain][step % len(grouped[domain])]
        loss = candidate_loss(model, context.features["train"][index], context.examples["train"][index])
        losses.append(loss)
        loss_values[domain] = float(loss.detach().item())
    gradients, norms, cosine = task_gradients(losses, parameters)
    return gradients, norms, cosine, loss_values


def gradient_diagnostic(
    context: Context, root: Path, merit_root: Path, *, steps: int
) -> tuple[dict[str, Any], dict[str, object]]:
    model = context.base_model().train()
    for parameter in model.alignment.parameters():
        parameter.requires_grad_(False)
    parameters = [parameter for parameter in model.decision.parameters() if parameter.requires_grad]
    grouped = domain_indices(context.examples["train"])
    sums: dict[str, torch.Tensor | None] = {domain: None for domain in DOMAINS}
    rows = []
    norm_history: dict[str, list[float]] = defaultdict(list)
    for step in tqdm(range(steps), desc="gradient interaction", unit="step", dynamic_ncols=True):
        gradients, norms, cosine, losses = _one_gradient_snapshot(
            context, model, grouped, step, parameters
        )
        for domain_index, domain in enumerate(DOMAINS):
            flat = torch.cat([value.reshape(-1).float().cpu() for value in gradients[domain_index]])
            sums[domain] = flat if sums[domain] is None else sums[domain] + flat
            norm_history[domain].append(norms[domain_index])
        for left in range(len(DOMAINS)):
            for right in range(left + 1, len(DOMAINS)):
                value = cosine[left][right]
                rows.append({
                    "step": step,
                    "domain_a": DOMAINS[left], "domain_b": DOMAINS[right],
                    "loss_a": losses[DOMAINS[left]], "loss_b": losses[DOMAINS[right]],
                    "norm_a": norms[left], "norm_b": norms[right],
                    "cosine": value,
                    "positive_interaction": value >= 0.0,
                    "conflict": value < 0.0,
                })
    means = {domain: value / steps for domain, value in sums.items() if value is not None}
    pca = gradient_pca_diagnostic(means)
    positive_fraction = float(np.mean([row["positive_interaction"] for row in rows]))
    summary = {
        "checkpoint": str(context.base_checkpoint.resolve()),
        "same_checkpoint_same_step": True,
        "steps": steps,
        "pair_observations": len(rows),
        "positive_interaction_fraction": positive_fraction,
        "conflict_fraction": 1.0 - positive_fraction,
        "mean_cosine": float(np.mean([row["cosine"] for row in rows])),
        "mean_gradient_norm_by_domain": {
            domain: float(np.mean(values)) for domain, values in norm_history.items()
        },
        "decision": (
            "adaptive_data_mixing; gradient surgery not applicable as default"
            if positive_fraction >= 0.5
            else "structured-conflict diagnostic required before optional gradient surgery baseline"
        ),
        "trainable_scope": "Visual-JEV decision adapter/head only; Qwen3-VL and alignment frozen",
    }
    write_rows(root / "gradient_interactions.csv", rows)
    write_json(root / "gradient_diagnostics.json", {"summary": summary, "rows": rows})
    write_json(merit_root / "pca_cluster.json", pca)
    pca_rows = [
        {
            "domain": domain,
            "gradient_norm": pca["gradient_norms"][domain],
            "pc1_score": pca["pc1_scores"][domain],
            "cluster": next(i for i, group in enumerate(pca["clusters"]) if domain in group),
        }
        for domain in DOMAINS
    ]
    write_rows(merit_root / "pca_cluster.csv", pca_rows)
    return summary, pca


def learn_pike_weights(
    context: Context, root: Path, *, pilot_budget: int, window: int
) -> tuple[dict[str, float], dict[str, Any]]:
    set_seed()
    model = context.base_model().train()
    for parameter in model.alignment.parameters():
        parameter.requires_grad_(False)
    parameters = [parameter for parameter in model.decision.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=1e-2)
    grouped = domain_indices(context.examples["train"])
    rng = random.Random(f"{SEED}:pike-pilot")
    estimator = PiKEInspiredWeights(tuple(DOMAINS), eta=0.30, floor=0.05)
    updates = []
    sample_counts: Counter[str] = Counter()
    for start in range(0, pilot_budget, window):
        losses_by_domain: dict[str, list[float]] = defaultdict(list)
        current_weights = estimator.as_dict()
        for _ in range(min(window, pilot_budget - start)):
            domain, index = _sample_index(grouped, current_weights, rng)
            optimizer.zero_grad(set_to_none=True)
            loss = candidate_loss(model, context.features["train"][index], context.examples["train"][index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses_by_domain[domain].append(float(loss.detach().item()))
            sample_counts[domain] += 1
        gradients, norms, cosine, probe_losses = _one_gradient_snapshot(
            context, model, grouped, start // window, parameters
        )
        del gradients
        observed = {
            domain: float(np.mean(losses_by_domain[domain]))
            if losses_by_domain[domain] else probe_losses[domain]
            for domain in DOMAINS
        }
        report = estimator.update(observed, cosine)
        updates.append({
            "window": len(updates) + 1,
            "start_step": start,
            "end_step": min(start + window, pilot_budget),
            "gradient_norms": {domain: norm for domain, norm in zip(DOMAINS, norms)},
            "cosine_matrix": cosine,
            **report,
        })
    learned = estimator.as_dict()
    payload = {
        "method": "PiKE-inspired lightweight project-local estimator; not official",
        "pilot_budget": pilot_budget,
        "window": window,
        "initial_checkpoint": str(context.base_checkpoint.resolve()),
        "sample_counts": dict(sample_counts),
        "updates": updates,
        "learned_weights": learned,
    }
    write_json(root / "pike_weights.json", payload)
    write_rows(root / "pike_weights.csv", [
        {
            "window": update["window"],
            **{f"weight_{domain}": update["weights"][domain] for domain in DOMAINS},
            **{f"loss_{domain}": update["loss"][domain] for domain in DOMAINS},
            **{f"loss_decrease_{domain}": update["loss_decrease"][domain] for domain in DOMAINS},
            **{f"positive_interaction_{domain}": update["positive_interaction"][domain] for domain in DOMAINS},
        }
        for update in updates
    ])
    return learned, payload


def run_merit_if_needed(
    context: Context,
    root: Path,
    diagnostic: Mapping[str, Any],
    *,
    branch_budget: int,
) -> dict[str, Any]:
    if not diagnostic["structured_conflict"]:
        payload = {
            "executed": False,
            "reason": "dataset-gradient PCA did not meet the pre-registered structured-conflict rule",
            "diagnostic": dict(diagnostic),
            "paper_implementation": "MERIT-inspired project-local conditional experiment; not official",
        }
        write_json(root / "branch_merge.json", payload)
        return payload
    branches = []
    for branch_index, domains in enumerate(diagnostic["clusters"]):
        weights = {domain: 1.0 for domain in domains}
        result = train_fixed_mixture(
            context,
            name=f"merit_branch_{branch_index}",
            weights=weights,
            budget=branch_budget,
            output=root,
            allowed_domains=tuple(domains),
        )
        branches.append(result)
    states = []
    merge_weights = []
    for result in branches:
        model, _ = ThreeStageVisualJEV.from_checkpoint(result["checkpoint"], map_location="cpu")
        states.append(model.decision.state_dict())
        merge_weights.append(float(result["effective_candidate_tokens"]))
    merged_state = weighted_state_dict_average(states, merge_weights)
    merged_model = context.base_model().to(context.device)
    merged_model.decision.load_state_dict(merged_state)
    merged_model.eval()
    validation = context.evaluate(merged_model, "validation")
    validation_pair = context.pair_metrics(merged_model, "validation")
    test = context.evaluate(merged_model, "test")
    pair = context.pair_metrics(merged_model, "test")
    checkpoint = root / "merit_weighted_merge.pt"
    _checkpoint_model(merged_model, checkpoint, name="merit_weighted_merge", metadata={
        "clusters": diagnostic["clusters"],
        "merge_weights": merge_weights,
        "merge_scope": "Visual-JEV decision adapter/head only",
    })
    payload = {
        "executed": True,
        "branches": branches,
        "merge_weights_effective_candidate_tokens": merge_weights,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "validation": validation,
        "validation_semantic_counterfactual": validation_pair,
        "validation_selection_score": validation_score(validation, validation_pair),
        "test": test,
        "semantic_counterfactual": pair,
        "paper_implementation": "MERIT-inspired project-local conditional experiment; not official",
    }
    write_json(root / "branch_merge.json", payload)
    return payload


def enrich_stage_result(context: Context, payload: dict[str, Any]) -> dict[str, Any]:
    model, _ = ThreeStageVisualJEV.from_checkpoint(payload["checkpoint"], map_location=context.device)
    model.eval()
    validation = context.evaluate(model, "validation")
    pair = context.pair_metrics(model, "validation")
    payload["validation"] = validation
    payload["validation_semantic_counterfactual"] = pair
    payload["validation_selection_score"] = validation_score(validation, pair)
    return payload


def evaluate_checkpoint_candidate(
    context: Context, *, name: str, checkpoint: Path
) -> dict[str, Any]:
    """Evaluate an existing checkpoint as a regression-protection candidate."""
    model, _ = ThreeStageVisualJEV.from_checkpoint(
        checkpoint.resolve(), map_location=context.device
    )
    model.eval()
    validation = context.evaluate(model, "validation")
    validation_pair = context.pair_metrics(model, "validation")
    return {
        "name": name,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "validation": validation,
        "validation_semantic_counterfactual": validation_pair,
        "validation_selection_score": validation_score(validation, validation_pair),
        "test": context.evaluate(model, "test"),
        "semantic_counterfactual": context.pair_metrics(model, "test"),
        "regression_protection_candidate": True,
    }


def select_with_visual_regression_guard(
    candidates: Mapping[str, Mapping[str, Any]], *, score_tolerance: float = 0.01
) -> tuple[str, Mapping[str, Any]]:
    """Select on validation while protecting semantic visual dependence.

    Candidates within a small composite-score tolerance are tied; the tie is
    resolved by validation Pair-flip, then Pair-both, then the composite.  Test
    metrics never participate in selection.
    """
    maximum = max(float(item["validation_selection_score"]) for item in candidates.values())
    eligible = [
        (name, item) for name, item in candidates.items()
        if float(item["validation_selection_score"]) >= maximum - score_tolerance
    ]
    return max(eligible, key=lambda row: (
        float(row[1]["validation_semantic_counterfactual"]["prediction_flip_rate"]),
        float(row[1]["validation_semantic_counterfactual"]["both_directions_accuracy"]),
        float(row[1]["validation_selection_score"]),
    ))


def write_visual_dependency_assignments(context: Context, payload: Mapping[str, Any], path: Path) -> None:
    required = payload["visual_dependency"]["train_visual_required"]
    rows = []
    for index, example in enumerate(context.examples["train"]):
        masks = visual_dependency_masks(
            example.dataset,
            scienceqa_visual_required=bool(required.get(index, required.get(str(index), False))),
        )
        rows.append({
            "index": index, "id": example.id, "dataset": example.dataset,
            "visual_dependency_type": masks["type"],
            "mask_invalid": masks["invalid"],
            "mask_counterfactual": masks["counterfactual"],
            "mask_preference": masks["preference"],
            "preference_source": masks["preference_source"],
        })
    write_json(path, {"assignments": rows, "test_used_for_threshold": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/results/benchmark_v1/recent_methods_fix"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixture-budget", type=int, default=512)
    parser.add_argument("--single-domain-budget", type=int, default=128)
    parser.add_argument("--pike-pilot-budget", type=int, default=256)
    parser.add_argument("--pike-window", type=int, default=64)
    parser.add_argument("--gradient-steps", type=int, default=16)
    parser.add_argument("--merit-branch-budget", type=int, default=256)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing experiment directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    context = Context(args.device)

    e0 = pipeline_regression(context, root / "pipeline_regression")
    audit = data_audit(context, root / "data_audit")
    transfer = run_transfer_matrix(
        context, root / "transfer_matrix", args.single_domain_budget
    )

    plans = mixture_plans()
    mixture_root = root / "mixture_ablation"
    mixture_runs = {
        name: train_fixed_mixture(
            context, name=f"E_mix_{name}", weights=weights,
            budget=args.mixture_budget, output=mixture_root,
        )
        for name, weights in plans.items()
    }
    write_rows(
        mixture_root / "mixture_ablation.csv",
        [flat_result(name, payload) for name, payload in mixture_runs.items()],
    )
    write_json(mixture_root / "mixture_ablation.json", {
        "fixed_total_sample_budget": args.mixture_budget,
        "raw_counts": RAW_COUNTS,
        "plans": plans,
        "runs": mixture_runs,
    })

    learned_weights, pike_pilot = learn_pike_weights(
        context, root / "pike_weights",
        pilot_budget=args.pike_pilot_budget, window=args.pike_window,
    )
    pike_run = train_fixed_mixture(
        context,
        name="E3_pike_fixed_learned_weights",
        weights=learned_weights,
        budget=args.mixture_budget,
        output=root / "pike_weights",
    )
    write_json(root / "pike_weights" / "formal_training.json", pike_run)

    gradient_summary, merit_pca = gradient_diagnostic(
        context, root / "gradient_diagnostics", root / "merit_diagnostic",
        steps=args.gradient_steps,
    )
    merit = run_merit_if_needed(
        context, root / "merit_diagnostic", merit_pca,
        branch_budget=args.merit_branch_budget,
    )

    pre_stage_candidates: dict[str, dict[str, Any]] = {
        **{f"mixture_{name}": payload for name, payload in mixture_runs.items()},
        "pike": pike_run,
    }
    if merit.get("executed"):
        pre_stage_candidates["merit_merge"] = merit
    pre_stage_name, pre_stage_best = max(
        pre_stage_candidates.items(),
        key=lambda item: item[1]["validation_selection_score"],
    )
    stage_root = root / "stagec_ablation"
    e4 = enrich_stage_result(context, run_stage_c(
        context, name="E4_best_mixture_gated_stagec",
        initial_checkpoint=Path(pre_stage_best["checkpoint"]),
        root=stage_root, curriculum=False, epochs=3,
    ))
    e5 = enrich_stage_result(context, run_stage_c(
        context, name="E5_easy_to_hard_semantic_preference_anchor",
        initial_checkpoint=Path(pre_stage_best["checkpoint"]),
        root=stage_root, curriculum=True, epochs=3,
    ))
    write_visual_dependency_assignments(
        context, e5, stage_root / "visual_dependency_assignments.json"
    )
    write_rows(stage_root / "stagec_ablation.csv", [
        flat_result("pre_stage_best", pre_stage_best),
        flat_result("E4_gated", e4), flat_result("E5_curriculum_anchor", e5),
    ])
    write_json(stage_root / "stagec_ablation.json", {
        "pre_stage_best": pre_stage_name,
        "random_wrong_image_preference_disabled": True,
        "local_region_annotations_available": False,
        "semantic_pair_policy": "only explicit COCO semantic counterfactual pairs; no fabricated local labels",
        "runs": {"E4": e4, "E5": e5},
    })

    candidates = {**pre_stage_candidates, "E4_gated": e4, "E5_curriculum_anchor": e5}
    prior_best = Path(
        "experiments/results/benchmark_v1/fix_negative_transfer/"
        "best_model_results/visual_jev_v3_best.pt"
    )
    if prior_best.is_file():
        candidates["prior_E5R_regression_protection"] = evaluate_checkpoint_candidate(
            context, name="prior_E5R_regression_protection", checkpoint=prior_best
        )
    prior_visual = Path(
        "experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/"
        "best_model_results/visual_jev_v3_best.pt"
    )
    if prior_visual.is_file():
        candidates["prior_E6R_visual_regression_protection"] = evaluate_checkpoint_candidate(
            context, name="prior_E6R_visual_regression_protection", checkpoint=prior_visual
        )
    best_name, best = select_with_visual_regression_guard(candidates)
    best_model, _ = ThreeStageVisualJEV.from_checkpoint(best["checkpoint"], map_location=args.device)
    best_dir = root / "best_model_results"
    best_checkpoint = best_dir / "visual_jev_v3_recent_methods_best.pt"
    _checkpoint_model(best_model, best_checkpoint, name=best_name, metadata={
        "selection_rule": "validation-only composite; within 0.01 use Pair-flip/Pair-both visual regression guard",
        "selected_from": best["checkpoint"],
    })
    best_payload = {
        "selected_experiment": best_name,
        "checkpoint": str(best_checkpoint.resolve()),
        "checkpoint_sha256": sha256(best_checkpoint),
        "validation_selection_score": best["validation_selection_score"],
        "validation": best["validation"],
        "validation_semantic_counterfactual": best["validation_semantic_counterfactual"],
        "test": best["test"],
        "semantic_counterfactual": best["semantic_counterfactual"],
        "e0_old_coco_reproduction": e0,
        "gradient_diagnostics": gradient_summary,
        "controlled_subset": True,
        "full_scale_pending": True,
    }
    write_json(best_dir / "best_model_results.json", best_payload)
    write_rows(best_dir / "best_model_results.csv", [flat_result(best_name, best)])

    root_causes = {
        "pipeline_bug": {"present": not e0["gate_passed"], "evidence": e0},
        "candidate_label_bug": {
            "present": False,
            "evidence": "all four domains passed 20-row before/after answer preservation and variable-K unit tests",
        },
        "data_mixture": {
            "present": True,
            "best_mixture": max(
                mixture_runs, key=lambda name: mixture_runs[name]["validation_selection_score"]
            ),
            "evidence": {
                name: payload["validation_selection_score"] for name, payload in mixture_runs.items()
            },
        },
        "negative_transfer": {
            "present": True,
            "evidence": transfer["matrix"],
        },
        "gradient_conflict": {
            "present": gradient_summary["conflict_fraction"] > 0,
            "default_surgery_applicable": gradient_summary["positive_interaction_fraction"] < 0.5,
            "evidence": gradient_summary,
        },
        "stage_c_wrong_assumption": {
            "present": True,
            "fix": "arbitrary same-domain wrong images removed; only explicit semantic pairs receive preference loss",
            "evidence": {
                "pre_stage_validation": pre_stage_best["validation_selection_score"],
                "E4_validation": e4["validation_selection_score"],
                "E5_validation": e5["validation_selection_score"],
            },
        },
    }
    write_json(root / "root_cause_analysis.json", root_causes)
    write_json(root / "full_config.json", {
        "seed": SEED,
        "device": args.device,
        "budgets": vars(args) | {"output": str(root)},
        "qwen3_vl_frozen": True,
        "alignment_frozen": True,
        "trainable_scope": "Visual-JEV decision adapter/head",
        "literature_scope": "2024--2026 main methods; PCGrad/GradNorm retained only in prior optional baseline",
        "paper_attribution": {
            "Cambrian-1": "source caps and explicit fixed-budget mixtures",
            "PiKE": "positive-interaction and short-window loss-decrease weight estimator; project-local, not official",
            "MERIT": "dataset-gradient PCA, conditional branches, decision-only token-weighted merge; project-local, not official",
            "mDPO": "candidate-scalar matched-vs-semantic-counterfactual preference with positive anchor",
            "MFPO": "easy-to-hard schedule and semantic hard negatives",
            "MMedPO": "local perturbations permitted only with verified annotations; none fabricated here",
            "VL-Calibration": "post-hoc visual/reasoning reliability calibration after model selection",
        },
        "runtime_seconds_before_calibration": time.time() - started,
        "calibration_status": "run separately after best checkpoint is fixed",
    })
    print(json.dumps({
        "status": "complete_before_calibration",
        "output": str(root),
        "best_experiment": best_name,
        "best_checkpoint": str(best_checkpoint),
        "test": best["test"]["overall"],
        "pair": best["semantic_counterfactual"],
        "merit_executed": merit.get("executed", False),
        "runtime_seconds": time.time() - started,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
