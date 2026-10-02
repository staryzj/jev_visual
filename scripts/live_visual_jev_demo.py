"""Continuous, cache-free image -> Visual-JEV decision demo.

The decision path is always live:

    image -> Qwen3-VL grouped pre-merger tokens -> alignment adapter
          -> VisualJEVV3 -> candidate scores/probabilities

For a manifest sample, the corresponding cache shard is loaded only after the
decision to verify that live ``pre_tokens`` match the training cache.  Cached
``pre_tokens`` and ``text_features`` are never used as model inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# Direct ``python scripts/live_visual_jev_demo.py`` execution must see the
# repository's top-level training-compatible feature modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jev.conversation_history import (
    VisualConversationHistory,
    format_candidate_with_history,
)
from jev.benchmark_v1 import BenchmarkExample, read_jsonl
from jev.serving import load_predictor
from visual_jev_v2 import Qwen3VLFeatureExtractor
from visual_jev_v3_pipeline import ThreeStageVisualJEV


DEFAULT_MANIFEST = Path("data/benchmark_v1_full/manifests/test.jsonl")
DEFAULT_CACHE_DIR = Path("experiments/benchmark_v1_full/features/test-shards")
DEFAULT_CHECKPOINT = Path(
    "experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/"
    "best_model_results/visual_jev_v3_best.pt"
)
DEFAULT_MODEL = "models/Qwen3-VL-4B-Instruct"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


@dataclass(frozen=True)
class DemoCase:
    image: Path
    question: str
    candidates: tuple[str, ...]
    sample_index: int | None = None
    sample_id: str | None = None
    dataset: str | None = None
    label: int | None = None


@dataclass
class DemoRuntime:
    extractor: Qwen3VLFeatureExtractor
    model: ThreeStageVisualJEV
    checkpoint_payload: dict[str, Any]
    checkpoint: Path
    history: VisualConversationHistory
    device: torch.device
    load_ms: float
    warmed_up: bool = False
    popup_process: subprocess.Popen | None = None
    popup_payload: Path | None = None


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def input_path(value: str) -> Path:
    """Resolve Linux/Windows input and infer a uniquely matching image suffix."""
    value = value.strip().strip('"')
    match = re.fullmatch(r"([A-Za-z]):[\\/](.*)", value)
    if match:
        drive, tail = match.groups()
        value = f"/mnt/{drive.lower()}/{tail.replace(chr(92), '/')}"
    path = Path(value).expanduser().resolve()
    if path.is_file() or path.suffix:
        return path
    matches = [path.with_suffix(suffix) for suffix in IMAGE_SUFFIXES]
    matches = [candidate for candidate in matches if candidate.is_file()]
    if len(matches) == 1:
        print(f"Resolved image path: {value} -> {matches[0]}")
        return matches[0]
    if len(matches) > 1:
        choices = ", ".join(candidate.name for candidate in matches)
        raise ValueError(
            f"multiple images match {value!r}: {choices}; include the extension"
        )
    return path


def prompt_image_path() -> Path:
    """Prompt until an existing image is selected, before asking other fields."""
    while True:
        raw = input("Image path: ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        image = input_path(raw)
        if image.is_file():
            return image
        suffix_hint = ", ".join(IMAGE_SUFFIXES)
        print(
            f"Image not found: {image}\n"
            f"Try the exact path, or omit only one of these extensions: {suffix_hint}"
        )


def manifest_case(manifest: Path, index: int) -> DemoCase:
    rows = list(read_jsonl(manifest))
    if not rows:
        raise ValueError(f"manifest is empty: {manifest}")
    if index < 0 or index >= len(rows):
        raise IndexError(f"sample index {index} is outside [0, {len(rows) - 1}]")
    row: BenchmarkExample = rows[index]
    image = Path(row.image).expanduser()
    if not image.is_absolute():
        image = manifest.parent.parent / image
    return DemoCase(
        image=image.resolve(),
        question=row.question,
        candidates=tuple(row.candidates),
        sample_index=index,
        sample_id=row.id,
        dataset=row.dataset,
        label=row.label,
    )


def interactive_case(
    manifest: Path,
    default_index: int,
    history: VisualConversationHistory,
) -> DemoCase | None:
    while True:
        print("\nChoose the next input:")
        print("  1) benchmark manifest sample")
        print("  2) arbitrary image + question + candidates")
        if history.active_image and Path(history.active_image).is_file():
            print("  d) continue the dialogue on the current image")
        print("  n) choose a new arbitrary image")
        print("  h) show conversation history")
        print("  c) clear conversation history")
        print("  q) end this live session")
        mode = input("Mode [1]: ").strip().lower() or "1"
        if mode in {"n", "next", "下一个", "继续"}:
            mode = "2"
        continue_dialogue = mode in {"d", "dialog", "dialogue", "对话", "追问"}
        if mode in {"q", "quit", "exit"}:
            return None
        if mode in {"h", "history", "历史"}:
            print(history.describe())
            continue
        if mode in {"c", "clear", "清空"}:
            history.clear()
            print("Conversation history cleared; loaded models were kept.")
            continue
        if mode == "1":
            raw = input(f"Sample index [{default_index}]: ").strip()
            return manifest_case(manifest, int(raw) if raw else default_index)
        if mode != "2" and not continue_dialogue:
            print("Mode must be 1, 2/n, d, h, c, or q.")
            continue

        if continue_dialogue:
            if not history.active_image or not Path(history.active_image).is_file():
                print("There is no current image yet. Choose 2/n for the first turn.")
                continue
            image = Path(history.active_image)
            print(f"Continuing with current image: {image}")
        else:
            image = prompt_image_path()
        question = input("Question: ").strip()
        candidates: list[str] = []
        print("Enter one candidate per line; submit an empty line when finished.")
        while True:
            candidate = input(f"Candidate {len(candidates) + 1}: ").strip()
            if not candidate:
                break
            candidates.append(candidate)
        return DemoCase(image=image, question=question, candidates=tuple(candidates))


def validate_case(case: DemoCase) -> None:
    if not case.image.is_file():
        raise FileNotFoundError(case.image)
    if not case.question.strip():
        raise ValueError("question must be non-empty")
    if len(case.candidates) < 2 or any(not value.strip() for value in case.candidates):
        raise ValueError("provide at least two non-empty candidates")
    if len(set(case.candidates)) != len(case.candidates):
        raise ValueError("candidates must be distinct")


def parse_case(args: argparse.Namespace, parser: argparse.ArgumentParser) -> DemoCase:
    manifest = args.manifest.expanduser().resolve()
    direct = args.image is not None or args.question is not None or args.candidate
    if direct:
        if args.image is None or args.question is None or not args.candidate:
            parser.error("custom mode requires --image, --question, and repeated --candidate")
        case = DemoCase(
            image=input_path(str(args.image)),
            question=args.question.strip(),
            candidates=tuple(value.strip() for value in args.candidate),
        )
    else:
        case = manifest_case(manifest, args.index)
    validate_case(case)
    return case


def timed(device: torch.device, function):
    sync(device)
    started = time.perf_counter()
    value = function()
    sync(device)
    return value, (time.perf_counter() - started) * 1000.0


def cache_validation(
    cache_dir: Path,
    index: int,
    live_pre: torch.Tensor,
    live_text: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    shard = cache_dir / f"{index:06d}.pt"
    if not shard.is_file():
        raise FileNotFoundError(f"cache validation shard is missing: {shard}")
    cached = torch.load(shard, map_location="cpu", weights_only=True)
    cached_pre = cached["pre_tokens"].float()
    current_pre = live_pre.detach().float().cpu()
    shape_match = tuple(current_pre.shape) == tuple(cached_pre.shape)
    if shape_match:
        difference = (current_pre - cached_pre).abs()
        max_abs = float(difference.max().item())
        mean_abs = float(difference.mean().item())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                current_pre.double().flatten(), cached_pre.double().flatten(), dim=0
            ).item()
        )
        cosine = min(1.0, max(-1.0, cosine))
        values_match = bool(torch.allclose(current_pre, cached_pre, atol=atol, rtol=rtol))
    else:
        max_abs = mean_abs = cosine = math.nan
        values_match = False

    text_report: dict[str, Any] | None = None
    if "text_features" in cached:
        cached_text = cached["text_features"].float()
        current_text = live_text.detach().float().cpu()
        text_shape_match = tuple(current_text.shape) == tuple(cached_text.shape)
        text_report = {
            "live_shape": list(current_text.shape),
            "cached_shape": list(cached_text.shape),
            "shape_match": text_shape_match,
            "max_abs_error": (
                float((current_text - cached_text).abs().max().item())
                if text_shape_match else math.nan
            ),
        }

    report = {
        "passed": shape_match and values_match,
        "shard": str(shard.resolve()),
        "cache_used_for_decision": False,
        "live_shape": list(current_pre.shape),
        "cached_shape": list(cached_pre.shape),
        "shape_match": shape_match,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "cosine_similarity": cosine,
        "atol": atol,
        "rtol": rtol,
        "live_text_vs_cache_diagnostic_only": text_report,
    }
    if not report["passed"]:
        raise RuntimeError(
            "live pre-merger tokens do not match the training cache: "
            + json.dumps(report, ensure_ascii=False)
        )
    return report


def print_result(result: dict[str, Any]) -> None:
    line = "=" * 88
    print(f"\n{line}")
    print("LIVE IMAGE -> VISUAL-JEV DECISION")
    print(line)
    if result.get("sample_index") is not None:
        print(f"Sample       : {result['sample_index']}  ({result.get('sample_id')})")
    print(f"Image        : {result['image']}")
    print(f"Question     : {result['question']}")
    history = result.get("history", {})
    if history.get("used"):
        turn_ids = ", ".join(str(value) for value in history["selected_turn_ids"])
        print(
            f"History      : turns {turn_ids} injected "
            f"({history['history_prompt_tokens']} tokens)"
        )
    elif history:
        print(f"History      : not injected ({history.get('reason', 'n/a')})")
    print(f"Live pre     : {tuple(result['features']['pre_merger_shape'])}")
    print(f"Live text    : {tuple(result['features']['text_shape'])}")
    print("\nLatency (CUDA synchronized, warm; model loading excluded)")
    latency = result["latency_ms"]
    print(f"  Qwen visual         {latency['qwen_visual']:10.3f} ms")
    print(f"  Qwen candidate text {latency['qwen_text']:10.3f} ms")
    print(f"  Alignment Adapter   {latency['alignment_adapter']:10.3f} ms")
    print(f"  VisualJEVV3         {latency['jev_decision']:10.3f} ms")
    print(f"  Full image->decision{latency['image_to_decision_e2e']:10.3f} ms")
    print("\nCandidate scores")
    print(f"  {'#':>2}  {'candidate':<36} {'raw score':>12} {'softmax':>11}")
    for item in result["candidates"]:
        mark = " <- TOP-1" if item["index"] == result["prediction_index"] else ""
        print(
            f"  {item['index'] + 1:>2}  {item['text'][:36]:<36} "
            f"{item['raw_score']:>12.6f} {item['probability'] * 100:>10.2f}%{mark}"
        )
    print(f"\nPrediction   : {result['prediction']}")
    if result.get("ground_truth") is not None:
        print(f"Ground truth : {result['ground_truth']}")
        print(f"Correct      : {'YES' if result['correct'] else 'NO'}")
    validation = result.get("cache_validation")
    if validation:
        print(
            "Cache check  : PASS (diagnostic only; not used for decision) | "
            f"shape={tuple(validation['live_shape'])} | "
            f"max_abs={validation['max_abs_error']:.6g} | "
            f"cos={validation['cosine_similarity']:.9f}"
        )
    print(line, flush=True)


def show_window(case: DemoCase, result: dict[str, Any]) -> None:
    """Show a dependency-free Tk result window suitable for a live demo."""
    try:
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk

        root = tk.Tk()
        root.title("Visual-JEV: Live Image to Decision")
        root.geometry("1420x820")
        root.minsize(1080, 680)
        root.lift()
        root.attributes("-topmost", True)
        root.after(600, lambda: root.attributes("-topmost", False))

        outer = ttk.Frame(root, padding=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=4)
        outer.columnconfigure(1, weight=6)
        outer.rowconfigure(0, weight=1)

        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        image = Image.open(case.image).convert("RGB")
        image.thumbnail((540, 570), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        image_label = ttk.Label(left, image=photo)
        image_label.image = photo
        image_label.pack(anchor="center", pady=(0, 12))
        ttk.Label(
            left,
            text=str(case.image),
            wraplength=520,
            foreground="#555555",
        ).pack(anchor="w")

        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        ttk.Label(right, text="LIVE IMAGE → DECISION", font=("Arial", 22, "bold")).pack(anchor="w")
        ttk.Label(
            right,
            text=case.question,
            font=("Arial", 16),
            wraplength=780,
        ).pack(anchor="w", pady=(12, 18))

        columns = ("candidate", "score", "probability", "result")
        table = ttk.Treeview(right, columns=columns, show="headings", height=max(4, len(case.candidates)))
        table.heading("candidate", text="Candidate")
        table.heading("score", text="Raw score")
        table.heading("probability", text="Softmax probability")
        table.heading("result", text="")
        table.column("candidate", width=330, anchor="w")
        table.column("score", width=120, anchor="e")
        table.column("probability", width=165, anchor="e")
        table.column("result", width=105, anchor="center")
        for item in result["candidates"]:
            top = item["index"] == result["prediction_index"]
            table.insert(
                "",
                "end",
                values=(
                    item["text"],
                    f"{item['raw_score']:.6f}",
                    f"{item['probability'] * 100:.2f}%",
                    "TOP-1" if top else "",
                ),
                tags=("top",) if top else (),
            )
        table.tag_configure("top", background="#dff5e1", foreground="#075e20")
        table.pack(fill="x")

        ttk.Label(
            right,
            text=f"Prediction: {result['prediction']}",
            font=("Arial", 18, "bold"),
            foreground="#075e20",
        ).pack(anchor="w", pady=(20, 8))
        if result.get("ground_truth") is not None:
            ttk.Label(
                right,
                text=(
                    f"Ground truth: {result['ground_truth']}   ·   "
                    f"{'CORRECT' if result['correct'] else 'INCORRECT'}"
                ),
                font=("Arial", 13),
            ).pack(anchor="w")

        latency = result["latency_ms"]
        timing_text = (
            f"Qwen visual  {latency['qwen_visual']:.3f} ms     "
            f"Qwen text  {latency['qwen_text']:.3f} ms\n"
            f"Adapter  {latency['alignment_adapter']:.3f} ms     "
            f"JEV  {latency['jev_decision']:.3f} ms\n"
            f"FULL IMAGE → DECISION  {latency['image_to_decision_e2e']:.3f} ms"
        )
        ttk.Separator(right).pack(fill="x", pady=18)
        ttk.Label(right, text=timing_text, font=("Consolas", 13)).pack(anchor="w")
        validation = result.get("cache_validation")
        if validation:
            ttk.Label(
                right,
                text=(
                    "✓ Live pre-merger tokens match training cache "
                    f"{tuple(validation['live_shape'])}; cache was not used for inference"
                ),
                foreground="#075e20",
            ).pack(anchor="w", pady=(18, 0))

        ttk.Label(
            right,
            text="查看完成后，点击下面按钮返回终端并输入下一题。",
            foreground="#555555",
        ).pack(anchor="e", pady=(22, 4))
        ttk.Button(
            right,
            text="继续下一个 / Next input",
            command=root.destroy,
        ).pack(anchor="e")
        root.mainloop()
    except Exception as error:
        print(f"[window unavailable] {type(error).__name__}: {error}", flush=True)


def load_runtime(args: argparse.Namespace) -> DemoRuntime:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    print("\nLoading Qwen3-VL vision/language model and Visual-JEV checkpoint...", flush=True)
    load_started = time.perf_counter()
    predictor = load_predictor(
        model_id=args.model,
        device=args.device,
        max_length=512,
        batch_size=8,
        vision=True,
        image_root=REPO_ROOT,
    )
    extractor = Qwen3VLFeatureExtractor(predictor.scorer.model)
    model, checkpoint_payload = ThreeStageVisualJEV.from_checkpoint(
        checkpoint, map_location="cpu"
    )
    model = model.to(device).eval()
    history = VisualConversationHistory(
        mode=args.history_mode,
        max_stored_turns=args.history_max_stored_turns,
        max_prompt_turns=args.history_max_prompt_turns,
        max_prompt_tokens=args.history_max_prompt_tokens,
        min_relevance=args.history_min_relevance,
    )
    if args.history_file and args.history_file.expanduser().is_file():
        history.load(args.history_file)
        print(
            f"Loaded {len(history.turns)} conversation turn(s) from "
            f"{args.history_file.expanduser().resolve()}.",
            flush=True,
        )
    sync(device)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    print(
        f"Models ready in {load_ms / 1000.0:.2f} s. "
        "They will stay resident for the whole session.",
        flush=True,
    )
    return DemoRuntime(
        extractor=extractor,
        model=model,
        checkpoint_payload=checkpoint_payload,
        checkpoint=checkpoint,
        history=history,
        device=device,
        load_ms=load_ms,
    )


@torch.inference_mode()
def run(args: argparse.Namespace, case: DemoCase, runtime: DemoRuntime) -> dict[str, Any]:
    device = runtime.device
    extractor = runtime.extractor
    model = runtime.model

    history_reset = runtime.history.bind_image(str(case.image))
    if history_reset:
        print("Image changed: previous same-image conversation history was cleared.")
    history_context = runtime.history.context(
        case.question,
        tokenizer=extractor.processor.tokenizer,
    )
    prompts = [
        format_candidate_with_history(
            case.question,
            candidate,
            history_context,
            prompt_policy=args.history_prompt_policy,
        )
        for candidate in case.candidates
    ]
    if history_context.used:
        print(
            "History injected into Qwen text features: turn(s) "
            + ", ".join(map(str, history_context.selected_turn_ids))
            + f" ({history_context.token_count} history tokens)."
        )
    elif runtime.history.turns:
        print(f"History not injected for this question: {history_context.reason}.")
    if not args.no_warmup and not runtime.warmed_up:
        print("Running one untimed live warmup...", flush=True)
        with Image.open(case.image) as source:
            warm_image = source.convert("RGB")
            warm_image.load()
        warm_pre, _ = extractor.encode_image_stages(warm_image)
        warm_text = extractor.encode_candidates(prompts)
        warm_aligned = model.alignment(warm_pre)
        model.decision(warm_aligned, warm_text)
        sync(device)
        del warm_pre, warm_text, warm_aligned, warm_image
        runtime.warmed_up = True

    sync(device)
    e2e_started = time.perf_counter()
    with Image.open(case.image) as source:
        image = source.convert("RGB")
        image.load()

    (pre_tokens, post_tokens), visual_ms = timed(
        device, lambda: extractor.encode_image_stages(image)
    )
    text_features, text_ms = timed(
        device, lambda: extractor.encode_candidates(prompts)
    )

    if pre_tokens.shape[-1] != model.alignment.config.pre_merger_dim:
        raise ValueError(
            f"live pre-merger width {pre_tokens.shape[-1]} != checkpoint width "
            f"{model.alignment.config.pre_merger_dim}"
        )
    if text_features.shape[-1] != model.decision.config.text_dim:
        raise ValueError(
            f"live text width {text_features.shape[-1]} != checkpoint width "
            f"{model.decision.config.text_dim}"
        )

    aligned, adapter_ms = timed(device, lambda: model.alignment(pre_tokens))
    output, jev_ms = timed(device, lambda: model.decision(aligned, text_features))
    scores_gpu = output.scores.detach().float()
    probabilities_gpu = torch.softmax(scores_gpu / args.temperature, dim=-1)
    scores = scores_gpu.cpu()
    probabilities = probabilities_gpu.cpu()
    sync(device)
    e2e_ms = (time.perf_counter() - e2e_started) * 1000.0

    prediction = int(probabilities.argmax().item())
    validation = None
    if case.sample_index is not None and not args.skip_cache_validation:
        validation = cache_validation(
            args.cache_dir.expanduser().resolve(),
            case.sample_index,
            pre_tokens,
            text_features,
            atol=args.validation_atol,
            rtol=args.validation_rtol,
        )

    result: dict[str, Any] = {
        "sample_index": case.sample_index,
        "sample_id": case.sample_id,
        "dataset": case.dataset,
        "image": str(case.image),
        "question": case.question,
        "candidates": [
            {
                "index": index,
                "text": candidate,
                "raw_score": float(scores[index].item()),
                "probability": float(probabilities[index].item()),
            }
            for index, candidate in enumerate(case.candidates)
        ],
        "prediction_index": prediction,
        "prediction": case.candidates[prediction],
        "ground_truth_index": case.label,
        "ground_truth": case.candidates[case.label] if case.label is not None else None,
        "correct": prediction == case.label if case.label is not None else None,
        "temperature": args.temperature,
        "history": {
            "mode": history_context.mode,
            "prompt_policy": args.history_prompt_policy,
            "used": history_context.used,
            "reason": history_context.reason,
            "selected_turn_ids": list(history_context.selected_turn_ids),
            "history_prompt_tokens": history_context.token_count,
            "stored_turns_before": history_context.stored_turn_count,
            "reset_on_image_change": history_reset,
        },
        "features": {
            "source": "Qwen3-VL visual.last_hidden_state before visual.merger",
            "grouping": "mean over each native spatial_merge_unit",
            "pre_merger_shape": list(pre_tokens.shape),
            "post_merger_shape": list(post_tokens.shape),
            "text_shape": list(text_features.shape),
            "text_source": "Qwen language_model final non-padding hidden state",
            "cached_features_used_for_decision": False,
        },
        "latency_ms": {
            "cold_model_load": runtime.load_ms,
            "qwen_visual": visual_ms,
            "qwen_text": text_ms,
            "alignment_adapter": adapter_ms,
            "jev_decision": jev_ms,
            "image_to_decision_e2e": e2e_ms,
        },
        "checkpoint": str(runtime.checkpoint),
        "checkpoint_stage": runtime.checkpoint_payload.get("stage"),
        "model": str(Path(args.model).expanduser().resolve()) if Path(args.model).expanduser().exists() else args.model,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "cache_validation": validation,
    }
    runtime.history.add(
        question=case.question,
        candidates=case.candidates,
        answer=case.candidates[prediction],
        answer_index=prediction,
        confidence=float(probabilities[prediction].item()),
    )
    result["history"]["stored_turns_after"] = len(runtime.history.turns)
    if args.history_file:
        result["history"]["saved_to"] = str(runtime.history.save(args.history_file))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/live_visual_jev_demo.py 0\n"
            "  python scripts/live_visual_jev_demo.py --interactive\n"
            "  python scripts/live_visual_jev_demo.py --image photo.jpg "
            "--question 'What is shown?' --candidate cat --candidate dog\n"
        ),
    )
    parser.add_argument("index", type=int, nargs="?", default=0, help="manifest sample index")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="load models once, then continuously prompt until q is entered",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image", help="arbitrary input image path")
    parser.add_argument("--question", help="question for arbitrary-image mode")
    parser.add_argument("--candidate", action="append", help="candidate answer; repeat at least twice")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--history-mode",
        choices=("off", "recent", "auto", "legacy-auto"),
        default="auto",
        help=(
            "off, always use a recent window, selectively retrieve relevant turns, "
            "or reproduce the original lexical-only auto gate"
        ),
    )
    parser.add_argument(
        "--history-prompt-policy",
        choices=("adaptive", "legacy-image-only"),
        default="legacy-image-only",
        help=(
            "legacy-image-only preserves the checkpoint's training prompt; "
            "adaptive is an experimental dialogue-aware instruction"
        ),
    )
    parser.add_argument("--history-max-stored-turns", type=int, default=128)
    parser.add_argument("--history-max-prompt-turns", type=int, default=8)
    parser.add_argument("--history-max-prompt-tokens", type=int, default=384)
    parser.add_argument("--history-min-relevance", type=float, default=0.12)
    parser.add_argument(
        "--history-file",
        type=Path,
        help="optional JSON file used to resume and persist the current image dialogue",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--validation-atol", type=float, default=5e-3)
    parser.add_argument("--validation-rtol", type=float, default=5e-3)
    parser.add_argument(
        "--skip-cache-validation",
        action="store_true",
        help="skip manifest cache parity check (cache is never used for inference)",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-window", action="store_true", help="terminal output only")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    return parser


def print_case(case: DemoCase) -> None:
    print("\nInput accepted")
    print(f"  image      : {case.image}")
    print(f"  question   : {case.question}")
    for index, candidate in enumerate(case.candidates, 1):
        print(f"  candidate {index}: {candidate}")


def save_result(path: Path, result: dict[str, Any], sequence: int | None = None) -> Path:
    output = path.expanduser().resolve()
    if sequence is not None:
        output = output.with_name(f"{output.stem}_{sequence:03d}{output.suffix or '.json'}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"JSON result saved to {output}")
    return output


def stop_popup(runtime: DemoRuntime) -> None:
    process = runtime.popup_process
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
    if runtime.popup_payload is not None:
        runtime.popup_payload.unlink(missing_ok=True)
    runtime.popup_process = None
    runtime.popup_payload = None


def launch_result_window(
    case: DemoCase,
    result: dict[str, Any],
    runtime: DemoRuntime,
) -> None:
    stop_popup(runtime)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="visual-jev-result-",
        delete=False,
    ) as handle:
        json.dump(
            {
                "case": {
                    "image": str(case.image),
                    "question": case.question,
                    "candidates": list(case.candidates),
                },
                "result": result,
            },
            handle,
            ensure_ascii=False,
        )
        payload_path = Path(handle.name)
    popup_script = REPO_ROOT / "scripts" / "live_visual_jev_popup.py"
    runtime.popup_payload = payload_path
    runtime.popup_process = subprocess.Popen(
        [sys.executable, str(popup_script), str(payload_path)],
        start_new_session=True,
    )


def present(
    args: argparse.Namespace,
    case: DemoCase,
    result: dict[str, Any],
    runtime: DemoRuntime,
) -> None:
    print_result(result)
    if not args.no_window:
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            print(
                "Result window opened without blocking the terminal. "
                "Enter n for the next custom image.",
                flush=True,
            )
            launch_result_window(case, result, runtime)
        else:
            print("[window unavailable] DISPLAY/WAYLAND_DISPLAY is not set", flush=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    if args.validation_atol < 0 or args.validation_rtol < 0:
        parser.error("validation tolerances must be non-negative")
    if min(
        args.history_max_stored_turns,
        args.history_max_prompt_turns,
        args.history_max_prompt_tokens,
    ) < 1:
        parser.error("history limits must be positive")
    if not 0.0 <= args.history_min_relevance <= 1.0:
        parser.error("--history-min-relevance must be in [0, 1]")
    direct = args.image is not None or args.question is not None or args.candidate
    if args.interactive and direct:
        parser.error("--interactive cannot be combined with direct custom input flags")

    # In continuous mode the expensive models are deliberately loaded before
    # asking for the first image/question, then retained until the user quits.
    if args.interactive:
        runtime = load_runtime(args)
        manifest = args.manifest.expanduser().resolve()
        completed = 0
        print("\nContinuous live session is ready. Enter q at the menu to stop.")
        while True:
            try:
                case = interactive_case(manifest, args.index, runtime.history)
                if case is None:
                    break
                validate_case(case)
                print_case(case)
                result = run(args, case, runtime)
                completed += 1
                if args.output:
                    save_result(args.output, result, completed)
                present(args, case, result, runtime)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            except Exception as error:
                print(
                    f"\n[request failed; models remain loaded] "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
        stop_popup(runtime)
        if args.history_file:
            saved = runtime.history.save(args.history_file)
            print(f"Conversation history saved to {saved}.")
        print(f"Live session ended after {completed} decision(s).")
        return

    case = parse_case(args, parser)
    print_case(case)
    runtime = load_runtime(args)
    result = run(args, case, runtime)
    if args.output:
        save_result(args.output, result)
    present(args, case, result, runtime)


if __name__ == "__main__":
    main()
