"""Reproducible text-path and modality ablations for Open-JEV/Visual-JEV.

The script deliberately separates the released Open-JEV text decision model
from the later Visual-JEV V3 fusion head.  It never downloads a model: both
backbones and both checkpoints must already exist locally.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jev.api import candidate_prompts, compile_request  # noqa: E402
from jev.conversation_history import (  # noqa: E402
    HistoryContext,
    VisualConversationHistory,
    format_candidate_with_history,
)
from jev.model import DecisionModel  # noqa: E402
from jev.serving import load_predictor  # noqa: E402
from visual_jev_v2 import Qwen3VLFeatureExtractor  # noqa: E402
from visual_jev_v3_pipeline import ThreeStageVisualJEV  # noqa: E402


DEFAULT_OFFICIAL_CHECKPOINT = Path("checkpoints/open-jev-9b/package/checkpoint")
DEFAULT_OFFICIAL_BASE = Path("models/Qwen3.5-9B")
DEFAULT_VISUAL_CHECKPOINT = Path(
    "experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/"
    "best_model_results/visual_jev_v3_best.pt"
)
DEFAULT_VISUAL_BASE = Path("models/Qwen3-VL-4B-Instruct")
DEFAULT_IMAGE = Path("test.jpg")


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    category: str
    question: str
    candidates: tuple[str, ...]
    expected_index: int
    history: tuple[tuple[str, str], ...] = ()


CASES = (
    ProbeCase(
        "common_knowledge_capital",
        "common_knowledge",
        "法国的首都是哪座城市？",
        ("巴黎", "伦敦", "东京"),
        0,
    ),
    ProbeCase(
        "synonym_fast",
        "synonym",
        "哪个词与“迅速”的意思最接近？",
        ("快速", "缓慢", "犹豫"),
        0,
    ),
    ProbeCase(
        "antonym_increase",
        "antonym",
        "哪个词是“增加”的反义词？",
        ("减少", "扩大", "提升"),
        0,
    ),
    ProbeCase(
        "history_name_recall",
        "history_recall",
        "我刚才说狗叫什么？",
        ("小白", "小黑", "旺财"),
        0,
        (("我养了一只叫小白的柴犬。", "小白"),),
    ),
    ProbeCase(
        "previous_question_recall",
        "history_recall",
        "上一个问题是什么？",
        ("问狗的品种", "问狗的年龄", "问狗的身高"),
        0,
        (("这是什么品种？", "柴犬"),),
    ),
    ProbeCase(
        "multi_turn_recall",
        "multi_turn",
        "第二轮里我问了什么？",
        ("问狗的名字", "问狗的颜色", "问狗的年龄"),
        1,
        (
            ("这只狗叫什么名字？", "小白"),
            ("这只狗是什么颜色？", "黄色和白色"),
            ("这只狗大约几岁？", "三岁"),
        ),
    ),
)


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _distribution(
    scores: torch.Tensor,
    candidates: tuple[str, ...],
    expected_index: int,
    temperature: float,
) -> dict[str, Any]:
    values = scores.detach().float().cpu()
    probabilities = torch.softmax(values / temperature, dim=-1)
    order = torch.argsort(values, descending=True)
    top = int(order[0])
    runner_up = int(order[1])
    expected_other = torch.cat(
        (values[:expected_index], values[expected_index + 1 :])
    ).max()
    return {
        "candidates": [
            {
                "text": candidate,
                "raw_score": float(values[index]),
                "softmax": float(probabilities[index]),
            }
            for index, candidate in enumerate(candidates)
        ],
        "prediction_index": top,
        "prediction": candidates[top],
        "expected_index": expected_index,
        "expected": candidates[expected_index],
        "correct": top == expected_index,
        "raw_top1_margin": float(values[top] - values[runner_up]),
        "softmax_top1_margin": float(probabilities[top] - probabilities[runner_up]),
        "expected_raw_margin": float(values[expected_index] - expected_other),
        "score_std": float(values.std(unbiased=False)),
    }


def _case_variants(case: ProbeCase):
    if case.history:
        yield False
        yield True
    else:
        yield False


def _history_text(case: ProbeCase, enabled: bool) -> str:
    if not enabled:
        return "No conversation history is available."
    lines = ["Conversation history:"]
    for index, (question, answer) in enumerate(case.history, 1):
        lines.extend(
            (f"[Turn {index}]", f"Question: {question}", f"Selected answer: {answer}")
        )
    return "\n".join(lines)


def _aggregate(rows: list[dict[str, Any]], result_key: str = "result") -> dict[str, Any]:
    evaluated = [row[result_key] for row in rows]
    return {
        "n": len(evaluated),
        "accuracy": sum(item["correct"] for item in evaluated) / len(evaluated),
        "mean_raw_top1_margin": sum(item["raw_top1_margin"] for item in evaluated)
        / len(evaluated),
        "mean_score_std": sum(item["score_std"] for item in evaluated) / len(evaluated),
    }


def _load_official_model(
    checkpoint: Path, base_model: Path, device: str
) -> tuple[DecisionModel, float, dict[str, Any]]:
    config = json.loads((checkpoint / "model.json").read_text(encoding="utf-8"))
    model = DecisionModel(
        str(base_model),
        revision=None,
        device=device,
        lora_rank=0,
        max_length=int(config["max_length"]),
    )
    if int(config["lora_rank"]):
        from peft import PeftModel

        model.backbone = PeftModel.from_pretrained(model.backbone, checkpoint / "adapter")
        model.lora_rank = int(config["lora_rank"])
    model.head.load_state_dict(
        torch.load(checkpoint / "head.pt", map_location=device, weights_only=True)
    )
    model._validate_loaded_parameters()
    model.eval()
    temperature = json.loads(
        (checkpoint / "temperature.json").read_text(encoding="utf-8")
    )["temperature"]
    return model, float(temperature), config


@torch.inference_mode()
def run_official(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve(args.official_checkpoint)
    base_model = _resolve(args.official_base)
    started = time.perf_counter()
    model, temperature, config = _load_official_model(
        checkpoint, base_model, args.device
    )
    load_seconds = time.perf_counter() - started
    device = torch.device(args.device)
    rows = []
    prompt_examples = {}
    for case in CASES:
        for history_enabled in _case_variants(case):
            request = {
                "state": _history_text(case, history_enabled),
                "questions": {
                    case.case_id: {
                        "type": "choice",
                        "instructions": case.question,
                        "criteria": {candidate: None for candidate in case.candidates},
                    }
                },
            }
            record = compile_request(request["state"], request["questions"])[0]
            _synchronize(device)
            score_started = time.perf_counter()
            scores = model([record])[0]
            _synchronize(device)
            elapsed_ms = (time.perf_counter() - score_started) * 1000.0
            variant = "history_on" if history_enabled else "history_off"
            rows.append(
                {
                    "case": asdict(case),
                    "variant": variant,
                    "history_enabled": history_enabled,
                    "result": _distribution(
                        scores, case.candidates, case.expected_index, temperature
                    ),
                    "latency_ms": elapsed_ms,
                    "input_tokens": model.last_input_tokens,
                }
            )
            prompt_examples[f"{case.case_id}:{variant}"] = candidate_prompts(record)
    history_pairs = {}
    for case in CASES:
        if not case.history:
            continue
        off = next(
            row for row in rows if row["case"]["case_id"] == case.case_id and not row["history_enabled"]
        )
        on = next(
            row for row in rows if row["case"]["case_id"] == case.case_id and row["history_enabled"]
        )
        history_pairs[case.case_id] = {
            "prediction_off": off["result"]["prediction"],
            "prediction_on": on["result"]["prediction"],
            "expected_margin_off": off["result"]["expected_raw_margin"],
            "expected_margin_on": on["result"]["expected_raw_margin"],
            "expected_margin_gain": (
                on["result"]["expected_raw_margin"]
                - off["result"]["expected_raw_margin"]
            ),
        }
    return {
        "backend": "official_open_jev_text",
        "architecture": (
            "state + question + one candidate -> Qwen3.5 final non-padding hidden "
            "state -> trained scalar decision head"
        ),
        "checkpoint": str(checkpoint),
        "base_model": str(base_model),
        "checkpoint_config": config,
        "temperature": temperature,
        "device": args.device,
        "model_load_seconds": load_seconds,
        "rows": rows,
        "summary": _aggregate(rows),
        "history_on_off": history_pairs,
        "rendered_candidate_prompts": prompt_examples,
    }


def _make_history_context(
    case: ProbeCase,
    enabled: bool,
    image: Path,
    tokenizer: Any,
) -> HistoryContext:
    history = VisualConversationHistory(mode="recent" if enabled else "off")
    history.bind_image(str(image))
    for question, answer in case.history:
        history.add(
            question=question,
            candidates=(answer, "other"),
            answer=answer,
            answer_index=0,
            confidence=1.0,
        )
    return history.context(case.question, tokenizer=tokenizer)


def _text_only_probe(decision: torch.nn.Module, text_features: torch.Tensor) -> torch.Tensor:
    """Bypass visual attention to probe the trained text-side subnetwork.

    This is an architectural diagnostic, not a separately trained inference
    mode: the released Visual-JEV checkpoint was optimized through multimodal
    fusion, so the returned scores must not be presented as a calibrated model.
    """
    parameter = next(decision.parameters())
    text = text_features.to(parameter.device, parameter.dtype)
    projected = decision.text_projector(text)
    fused = decision.fusion_norm(projected)
    fused = decision.output_norm(fused + decision.fusion_mlp(fused))
    return decision.scalar_head(fused).squeeze(-1)


def _mean_off_diagonal_cosine(features: torch.Tensor) -> float:
    normalized = F.normalize(features.detach().float(), dim=-1)
    similarities = normalized @ normalized.T
    count = similarities.shape[0]
    mask = ~torch.eye(count, dtype=torch.bool, device=similarities.device)
    return float(similarities[mask].mean().cpu())


@torch.inference_mode()
def run_visual(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve(args.visual_checkpoint)
    base_model = _resolve(args.visual_base)
    image_path = _resolve(args.image)
    device = torch.device(args.device)
    started = time.perf_counter()
    predictor = load_predictor(
        model_id=str(base_model),
        device=args.device,
        max_length=512,
        batch_size=8,
        vision=True,
        image_root=image_path.parent,
    )
    extractor = Qwen3VLFeatureExtractor(predictor.scorer.model)
    model, payload = ThreeStageVisualJEV.from_checkpoint(checkpoint, map_location="cpu")
    model = model.to(device).eval()
    _synchronize(device)
    load_seconds = time.perf_counter() - started

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.load()
    visual_started = time.perf_counter()
    pre_tokens, post_tokens = extractor.encode_image_stages(image)
    aligned = model.alignment(pre_tokens)
    _synchronize(device)
    visual_encode_ms = (time.perf_counter() - visual_started) * 1000.0

    rows = []
    for case in CASES:
        for history_enabled in _case_variants(case):
            context = _make_history_context(
                case,
                history_enabled,
                image_path,
                extractor.processor.tokenizer,
            )
            prompts = [
                format_candidate_with_history(
                    case.question,
                    candidate,
                    context,
                    prompt_policy=args.history_prompt_policy,
                )
                for candidate in case.candidates
            ]
            text_started = time.perf_counter()
            text_features = extractor.encode_candidates(prompts)
            _synchronize(device)
            text_encode_ms = (time.perf_counter() - text_started) * 1000.0

            mode_scores = {
                "visual_on": model.decision(aligned, text_features).scores,
                "visual_off_zero_aligned": model.decision(
                    torch.zeros_like(aligned), text_features
                ).scores,
                "text_only_probe": _text_only_probe(model.decision, text_features),
            }
            mode_results = {
                mode: _distribution(scores, case.candidates, case.expected_index, 1.0)
                for mode, scores in mode_scores.items()
            }
            on_prob = torch.tensor(
                [item["softmax"] for item in mode_results["visual_on"]["candidates"]]
            )
            off_prob = torch.tensor(
                [
                    item["softmax"]
                    for item in mode_results["visual_off_zero_aligned"]["candidates"]
                ]
            )
            rows.append(
                {
                    "case": asdict(case),
                    "variant": "history_on" if history_enabled else "history_off",
                    "history_enabled": history_enabled,
                    "history_context": {
                        "used": context.used,
                        "reason": context.reason,
                        "selected_turn_ids": list(context.selected_turn_ids),
                        "token_count": context.token_count,
                    },
                    "prompt_policy": args.history_prompt_policy,
                    "prompts": prompts,
                    "text_feature_shape": list(text_features.shape),
                    "mean_off_diagonal_text_cosine": _mean_off_diagonal_cosine(text_features),
                    "text_encode_ms": text_encode_ms,
                    "modes": mode_results,
                    "visual_on_off_probability_l1": float((on_prob - off_prob).abs().sum()),
                    "visual_on_off_top1_flip": (
                        mode_results["visual_on"]["prediction_index"]
                        != mode_results["visual_off_zero_aligned"]["prediction_index"]
                    ),
                }
            )

    per_mode = {}
    for mode in ("visual_on", "visual_off_zero_aligned", "text_only_probe"):
        flattened = [{"result": row["modes"][mode]} for row in rows]
        per_mode[mode] = _aggregate(flattened)
    history_pairs = {}
    for case in CASES:
        if not case.history:
            continue
        off = next(
            row for row in rows if row["case"]["case_id"] == case.case_id and not row["history_enabled"]
        )
        on = next(
            row for row in rows if row["case"]["case_id"] == case.case_id and row["history_enabled"]
        )
        history_pairs[case.case_id] = {}
        for mode in per_mode:
            history_pairs[case.case_id][mode] = {
                "prediction_off": off["modes"][mode]["prediction"],
                "prediction_on": on["modes"][mode]["prediction"],
                "expected_margin_off": off["modes"][mode]["expected_raw_margin"],
                "expected_margin_on": on["modes"][mode]["expected_raw_margin"],
                "expected_margin_gain": (
                    on["modes"][mode]["expected_raw_margin"]
                    - off["modes"][mode]["expected_raw_margin"]
                ),
            }
    return {
        "backend": "visual_jev_v3",
        "architecture": (
            "Qwen3-VL candidate final hidden state + mandatory visual cross-attention "
            "+ gated fusion + Visual-JEV scalar head"
        ),
        "checkpoint": str(checkpoint),
        "base_model": str(base_model),
        "image": str(image_path),
        "checkpoint_stage": payload.get("stage"),
        "alignment_config": payload.get("alignment_config"),
        "decision_config": payload.get("decision_config"),
        "device": args.device,
        "model_load_seconds": load_seconds,
        "visual_encode_ms": visual_encode_ms,
        "pre_merger_shape": list(pre_tokens.shape),
        "post_merger_shape": list(post_tokens.shape),
        "rows": rows,
        "summary_by_mode": per_mode,
        "history_on_off": history_pairs,
        "text_only_probe_warning": (
            "The text-only path bypasses visual attention and was not separately trained or calibrated."
        ),
    }


def run_gate() -> dict[str, Any]:
    image = "/tmp/jev-text-gate-diagnostic.jpg"
    rows = []
    for mode in ("auto", "legacy-auto", "recent", "off"):
        history = VisualConversationHistory(mode=mode)
        history.bind_image(image)
        history.add(
            question="这是什么品种？",
            candidates=("柴犬", "金毛"),
            answer="柴犬",
            answer_index=0,
            confidence=0.9,
        )
        context = history.context("上一个问题是什么？")
        rows.append(
            {
                "mode": mode,
                "used": context.used,
                "reason": context.reason,
                "selected_turn_ids": list(context.selected_turn_ids),
                "rendered_context": context.text,
            }
        )
    return {
        "backend": "history_gate_only",
        "question": "上一个问题是什么？",
        "rows": rows,
        "fixed": next(row for row in rows if row["mode"] == "auto")["used"],
        "legacy_reproduced": not next(
            row for row in rows if row["mode"] == "legacy-auto"
        )["used"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("gate", "official", "visual"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL_CHECKPOINT)
    parser.add_argument("--official-base", type=Path, default=DEFAULT_OFFICIAL_BASE)
    parser.add_argument("--visual-checkpoint", type=Path, default=DEFAULT_VISUAL_CHECKPOINT)
    parser.add_argument("--visual-base", type=Path, default=DEFAULT_VISUAL_BASE)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--history-prompt-policy",
        choices=("adaptive", "legacy-image-only"),
        default="legacy-image-only",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.backend == "gate":
        report = run_gate()
    elif args.backend == "official":
        report = run_official(args)
    else:
        report = run_visual(args)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
