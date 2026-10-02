"""Paper-grade evaluation for Visual-JEV V3 checkpoints and baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.run_visual_jev_v2_experiment import file_sha256, open_cached_dataset
from scripts.run_visual_jev_v3_experiment import load_pairs
from train_visual_jev_v2 import load_records
from visual_jev_v2 import VisualJEVV2, VisualJEVV2Config
from visual_jev_v3 import VisualJEVV3
from visual_jev_v3_pipeline import AlignmentAdapterConfig, MeanPoolJEV, ThreeStageVisualJEV
from train_visual_jev_v3 import load_config, open_shards, subset_pairs


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, ece_bins: int) -> dict[str, float]:
    probabilities = logits.softmax(-1)
    predictions = probabilities.argmax(-1)
    accuracy = float((predictions == labels).float().mean().item())
    class_count = logits.shape[1]
    f1_values = []
    for class_index in range(class_count):
        predicted = predictions == class_index
        actual = labels == class_index
        tp = (predicted & actual).sum().item()
        fp = (predicted & ~actual).sum().item()
        fn = (~predicted & actual).sum().item()
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    nll = float(F.cross_entropy(logits, labels).item())
    targets = F.one_hot(labels, num_classes=class_count).float()
    brier = float(((probabilities - targets) ** 2).sum(-1).mean().item())
    confidence, _ = probabilities.max(-1)
    correct = (predictions == labels).float()
    ece = 0.0
    boundaries = torch.linspace(0, 1, ece_bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.float().mean() * (confidence[mask].mean() - correct[mask].mean()).abs())
    positive_probability = probabilities[torch.arange(len(labels)), labels]
    other = logits.masked_fill(F.one_hot(labels, class_count).bool(), -torch.inf).max(-1).values
    margin = logits[torch.arange(len(labels)), labels] - other
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    uniform_kl = math.log(class_count) - entropy
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "mean_correct_probability": float(positive_probability.mean().item()),
        "mean_margin": float(margin.mean().item()),
        "mean_entropy": float(entropy.mean().item()),
        "normalized_entropy": float((entropy / math.log(class_count)).mean().item()),
        "mean_uniform_kl": float(uniform_kl.mean().item()),
    }


def permutation_for(index: int, count: int, seed: int) -> torch.Tensor:
    order = list(range(count))
    random.Random(seed + index * 1009).shuffle(order)
    return torch.tensor(order, dtype=torch.long)


@torch.no_grad()
def collect_original(
    score: Callable[[int, torch.Tensor | None], torch.Tensor],
    indices: Sequence[int],
    text_data: Sequence[Any],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = []
    labels = []
    for index in indices:
        count = text_data[index]["text_features"].shape[0]
        order = permutation_for(index, count, seed)
        original_scores = score(index, None).detach().float().cpu()
        logits.append(original_scores[order])
        labels.append(int((order == 0).nonzero(as_tuple=False)[0].item()))
    return torch.stack(logits), torch.tensor(labels)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
        + (q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
    )


@torch.no_grad()
def visual_diagnostics(
    score_condition: Callable[[int, str], torch.Tensor],
    indices: Sequence[int],
) -> dict[str, Any]:
    conditions = ("original", "blank", "noise", "wrong_image", "image_swap")
    collected = {name: [] for name in conditions}
    for index in indices:
        for name in conditions:
            collected[name].append(score_condition(index, name).detach().float().cpu())
    logits = {name: torch.stack(values) for name, values in collected.items()}
    original_probabilities = logits["original"].softmax(-1)
    original_predictions = logits["original"].argmax(-1)
    output: dict[str, Any] = {}
    for name in conditions:
        probabilities = logits[name].softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        margin = logits[name][:, 0] - logits[name][:, 1:].max(-1).values
        kl = (
            original_probabilities
            * (original_probabilities.clamp_min(1e-12).log() - probabilities.clamp_min(1e-12).log())
        ).sum(-1)
        js = js_divergence(original_probabilities, probabilities)
        output[name] = {
            "accuracy": float((logits[name].argmax(-1) == 0).float().mean().item()),
            "mean_correct_probability": float(probabilities[:, 0].mean().item()),
            "mean_margin": float(margin.mean().item()),
            "mean_entropy": float(entropy.mean().item()),
            "normalized_entropy": float((entropy / math.log(logits[name].shape[1])).mean().item()),
            "mean_uniform_kl": float((math.log(logits[name].shape[1]) - entropy).mean().item()),
            "prediction_flip_rate_vs_original": float((logits[name].argmax(-1) != original_predictions).float().mean().item()),
            "mean_kl_from_original": float(kl.mean().item()),
            "mean_js_from_original": float(js.mean().item()),
            "mean_logit_l2_from_original": float((logits[name] - logits["original"]).norm(dim=-1).mean().item()),
        }
    output["visual_sensitivity"] = {
        "mean_js_invalid": float(np.mean([output[name]["mean_js_from_original"] for name in ("blank", "noise")])),
        "mean_js_semantic": output["wrong_image"]["mean_js_from_original"],
        "mean_js_swap": output["image_swap"]["mean_js_from_original"],
        "invalid_confidence_drop": output["original"]["mean_correct_probability"] - float(np.mean([output["blank"]["mean_correct_probability"], output["noise"]["mean_correct_probability"]])),
    }
    return output


@torch.no_grad()
def semantic_pair_metrics(
    score_pair: Callable[[int, int, torch.Tensor], torch.Tensor],
    pairs: dict[int, dict[str, Any]],
    pair_text: Sequence[Any],
) -> dict[str, float]:
    source_correct = partner_correct = both = flips = 0
    source_margins = []
    partner_margins = []
    for source, pair in pairs.items():
        partner = int(pair["counterfactual_index"])
        target = int(pair["counterfactual_target_index"])
        text = pair_text[int(pair["_cache_index"])] ["text_features"]
        source_scores = score_pair(source, source, text).detach().float().cpu()
        partner_scores = score_pair(partner, source, text).detach().float().cpu()
        a = int(source_scores.argmax().item())
        b = int(partner_scores.argmax().item())
        source_ok = a == 0
        partner_ok = b == target
        source_correct += int(source_ok)
        partner_correct += int(partner_ok)
        both += int(source_ok and partner_ok)
        flips += int(a != b)
        source_margins.append(float(source_scores[0] - source_scores[1:].max()))
        partner_margins.append(float(partner_scores[target] - torch.cat((partner_scores[:target], partner_scores[target + 1 :])).max()))
    count = max(1, len(pairs))
    return {
        "pair_count": len(pairs),
        "source_accuracy": source_correct / count,
        "counterfactual_accuracy": partner_correct / count,
        "both_directions_accuracy": both / count,
        "prediction_flip_rate": flips / count,
        "source_target_margin": float(np.mean(source_margins)) if source_margins else 0.0,
        "counterfactual_target_margin": float(np.mean(partner_margins)) if partner_margins else 0.0,
    }


def evaluate_native(
    config: dict[str, Any], records: Sequence[Any], indices: Sequence[int], output_path: Path
) -> dict[str, Any]:
    if output_path.is_file():
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            len(cached.get("samples", [])) == len(indices)
            and cached.get("candidate_order") == "deterministically permuted"
        ):
            return cached
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(config["model"])
    model = AutoModelForImageTextToText.from_pretrained(
        config["model"], dtype=torch.bfloat16
    ).to(config["device"]).eval()
    samples = []
    predictions: list[int] = []
    targets: list[int] = []
    started = time.time()
    for position, index in enumerate(indices):
        record = records[index]
        candidates = [record.positive, *record.negatives]
        order = permutation_for(index, len(candidates), config["seed"]).tolist()
        candidates = [candidates[item] for item in order]
        target = order.index(0)
        labels = [chr(ord("A") + i) for i in range(len(candidates))]
        prompt = "Which option is supported by the image? Reply with only the option letter.\n" + "\n".join(
            f"{label}. {candidate}" for label, candidate in zip(labels, candidates)
        )
        image = Image.open(record.image).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            [messages], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        inputs = {key: value.to(config["device"]) if torch.is_tensor(value) else value for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        answer = processor.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()
        match = re.search(r"\b([A-Z])\b", answer.upper())
        prediction = labels.index(match.group(1)) if match and match.group(1) in labels else -1
        predictions.append(prediction)
        targets.append(target)
        samples.append({"index": index, "answer": answer, "prediction": prediction, "target": target, "permutation": order})
        if (position + 1) % 16 == 0:
            print(json.dumps({"native_completed": position + 1, "total": len(indices)}), flush=True)
    correct = sum(int(prediction == target) for prediction, target in zip(predictions, targets))
    f1_values = []
    for class_index in range(3):
        tp = sum(p == class_index and t == class_index for p, t in zip(predictions, targets))
        fp = sum(p == class_index and t != class_index for p, t in zip(predictions, targets))
        fn = sum(p != class_index and t == class_index for p, t in zip(predictions, targets))
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    result = {
        "name": "Qwen3-VL native generative answer",
        "sample_count": len(indices),
        "accuracy": correct / max(1, len(indices)),
        "macro_f1": float(np.mean(f1_values)),
        "candidate_order": "deterministically permuted",
        "seconds": time.time() - started,
        "samples": samples,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/visual_jev_v3_paper.json"))
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    device = config["device"]
    data_root = Path(config["data_root"])
    pair_root = Path(config["pair_root"])
    paper_root = Path(config["paper_root"])
    results_root = Path(config.get("results_root", "experiments/results")).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    validation_path = data_root / "splits" / "validation.jsonl"
    validation_records = load_records(validation_path, data_root)
    validation_sha = file_sha256(validation_path)
    validation_features = open_cached_dataset(data_root / "features" / "validation.pt", source_sha256=validation_sha, item_key="features")
    validation_controls_post = open_cached_dataset(data_root / "features" / "validation-controls.pt", source_sha256=validation_sha, item_key="controls")
    validation_alignment = open_shards(paper_root / "features" / "validation-shards", len(validation_records))
    validation_controls_pre = open_shards(paper_root / "features" / "validation_controls-shards", len(validation_records))
    pair_path = pair_root / "pairs" / "validation.jsonl"
    pairs = load_pairs(pair_path, len(validation_records))
    pair_text = open_cached_dataset(pair_root / "features" / "validation-pair-text.pt", source_sha256=file_sha256(pair_path), item_key="pair_text")
    if any(x is None for x in (validation_features, validation_controls_post, validation_alignment, validation_controls_pre, pair_text)):
        raise RuntimeError("evaluation cache is incomplete")
    modulus = config["split"]["validation_modulus"]
    test_indices = [i for i in range(len(validation_records)) if i % modulus == config["split"]["test_remainder"]]
    test_pairs = subset_pairs(pairs, set(test_indices))
    next_test = {index: test_indices[(position + 1) % len(test_indices)] for position, index in enumerate(test_indices)}
    pair_partner = {source: int(pair["counterfactual_index"]) for source, pair in test_pairs.items()}

    reports: dict[str, Any] = {}
    torch.manual_seed(config["seed"])
    random_model = VisualJEVV2(VisualJEVV2Config(2560, 2560, 128, 8)).to(device).eval()
    mean_payload = torch.load(paper_root / "checkpoints" / "meanpool_jev.pt", map_location=device, weights_only=False)
    meanpool = MeanPoolJEV(mean_payload["vision_dim"], mean_payload["text_dim"], mean_payload["hidden_dim"]).to(device)
    meanpool.load_state_dict(mean_payload["state_dict"]); meanpool.eval()
    v2, _ = VisualJEVV2.from_checkpoint(config["v2_checkpoint"], map_location=device); v2 = v2.to(device).eval()
    v3_simple, _ = VisualJEVV3.from_checkpoint("checkpoints/visual-jev-v3-coco-2k-img5.pt", map_location=device); v3_simple = v3_simple.to(device).eval()
    composites = {}
    for key, filename in (
        ("v3_without_stage_a", "v3_without_stage_a.pt"),
        ("v3_without_stage_c", "v3_without_stage_c.pt"),
        ("v3_full", "v3_full.pt"),
    ):
        model, payload = ThreeStageVisualJEV.from_checkpoint(paper_root / "checkpoints" / filename, map_location=device)
        composites[key] = (model.to(device).eval(), payload)
    stage_a_payload = torch.load(
        paper_root / "checkpoints" / "stage_a_alignment.pt",
        map_location=device,
        weights_only=False,
    )
    torch.manual_seed(config["seed"] + 17)
    stage_a_only = ThreeStageVisualJEV(
        AlignmentAdapterConfig(**stage_a_payload["config"]),
        composites["v3_without_stage_c"][0].decision.config,
    ).to(device)
    stage_a_only.alignment.load_state_dict(stage_a_payload["state_dict"])
    composites["v3_stage_a_only"] = (
        stage_a_only.eval(),
        {"stage": "A_only_random_decision_head"},
    )

    teacher_models = {
        "random_untrained_head": lambda model=random_model: model,
        "meanpool_mlp_jev": lambda model=meanpool: model,
        "v2_candidate_aware": lambda model=v2: model,
        "v3_simple_postmerger": lambda model=v3_simple: model,
    }
    for name, get_model in teacher_models.items():
        model = get_model()
        def score(index: int, text_override: torch.Tensor | None, model=model):
            item = validation_features[index]
            text = item["text_features"] if text_override is None else text_override
            output = model(item["visual_tokens"], text)
            return output if torch.is_tensor(output) else output.scores
        logits, labels = collect_original(score, test_indices, validation_features, config["seed"])
        reports[name] = {"classification": classification_metrics(logits, labels, config["ece_bins"])}

    for name, (model, payload) in composites.items():
        def score(index: int, text_override: torch.Tensor | None, model=model):
            text = validation_features[index]["text_features"] if text_override is None else text_override
            return model(validation_alignment[index]["pre_tokens"], text).scores
        logits, labels = collect_original(score, test_indices, validation_features, config["seed"])

        def score_condition(index: int, condition: str, model=model):
            text = validation_features[index]["text_features"]
            if condition == "original":
                visual = validation_alignment[index]["pre_tokens"]
            elif condition == "blank":
                visual = validation_controls_pre[index]["blank_pre"]
            elif condition == "noise":
                visual = validation_controls_pre[index]["noise_pre"]
            elif condition == "image_swap":
                visual = validation_alignment[next_test[index]]["pre_tokens"]
            elif condition == "wrong_image":
                visual = validation_alignment[pair_partner.get(index, next_test[index])]["pre_tokens"]
            else:
                raise ValueError(condition)
            return model(visual, text).scores

        def score_pair(image_index: int, source_index: int, text: torch.Tensor, model=model):
            del source_index
            return model(validation_alignment[image_index]["pre_tokens"], text).scores

        reports[name] = {
            "checkpoint_stage": payload.get("stage"),
            "classification": classification_metrics(logits, labels, config["ece_bins"]),
            "visual_dependency": visual_diagnostics(score_condition, test_indices),
            "semantic_counterfactual": semantic_pair_metrics(score_pair, test_pairs, pair_text),
        }

    # Teacher-token diagnostics for the existing V2 and simple V3 baselines.
    for name, model in (("v2_candidate_aware", v2), ("v3_simple_postmerger", v3_simple)):
        def teacher_condition(index: int, condition: str, model=model):
            text = validation_features[index]["text_features"]
            if condition == "original": visual = validation_features[index]["visual_tokens"]
            elif condition == "blank": visual = validation_controls_post[index]["blank"]
            elif condition == "noise": visual = validation_controls_post[index]["noise"]
            elif condition == "image_swap": visual = validation_features[next_test[index]]["visual_tokens"]
            elif condition == "wrong_image": visual = validation_features[pair_partner.get(index, next_test[index])]["visual_tokens"]
            else: raise ValueError(condition)
            return model(visual, text).scores
        def teacher_pair(image_index: int, source_index: int, text: torch.Tensor, model=model):
            del source_index
            return model(validation_features[image_index]["visual_tokens"], text).scores
        reports[name]["visual_dependency"] = visual_diagnostics(teacher_condition, test_indices)
        reports[name]["semantic_counterfactual"] = semantic_pair_metrics(teacher_pair, test_pairs, pair_text)

    native = None
    if args.native:
        max_samples = min(config["native_baseline_max_samples"], len(test_indices))
        native = evaluate_native(config, validation_records, test_indices[:max_samples], results_root / "qwen3_vl_native_answers.json")

    summary = {
        "test_records": len(test_indices),
        "semantic_pair_count": len(test_pairs),
        "candidate_order": "deterministically permuted per sample",
        "models": reports,
        "qwen3_vl_native": native,
    }
    (results_root / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = results_root / "metrics.csv"
    fields = ["model", "accuracy", "macro_f1", "nll", "brier", "ece", "counterfactual_both_accuracy", "counterfactual_flip_rate", "blank_uniform_kl", "noise_uniform_kl", "visual_sensitivity_js"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for name, report in reports.items():
            c = report["classification"]; d = report.get("visual_dependency", {}); pair = report.get("semantic_counterfactual", {})
            writer.writerow({
                "model": name, "accuracy": c["accuracy"], "macro_f1": c["macro_f1"], "nll": c["nll"], "brier": c["brier"], "ece": c["ece"],
                "counterfactual_both_accuracy": pair.get("both_directions_accuracy"), "counterfactual_flip_rate": pair.get("prediction_flip_rate"),
                "blank_uniform_kl": d.get("blank", {}).get("mean_uniform_kl"), "noise_uniform_kl": d.get("noise", {}).get("mean_uniform_kl"),
                "visual_sensitivity_js": d.get("visual_sensitivity", {}).get("mean_js_semantic"),
            })
        if native is not None:
            writer.writerow({"model": "qwen3_vl_native", "accuracy": native["accuracy"], "macro_f1": native.get("macro_f1")})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
