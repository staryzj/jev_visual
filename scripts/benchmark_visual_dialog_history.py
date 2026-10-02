"""Benchmark Visual-JEV dialogue-memory retrieval and token-budget behavior."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

from jev.benchmark_v1 import read_jsonl
from jev.conversation_history import VisualConversationHistory


CONTROLLED_TURNS = (
    ("Which animal is visible?", "zebra"),
    ("What color is the vehicle?", "blue"),
    ("How many people are standing?", "three"),
    ("What is the person holding?", "umbrella"),
    ("Where is the bicycle parked?", "beside the wall"),
    ("What food is on the plate?", "pizza"),
    ("Which sport is being played?", "tennis"),
    ("What object is closest to the camera?", "red suitcase"),
)


def add_turn(history: VisualConversationHistory, question: str, answer: str) -> None:
    history.add(
        question=question,
        candidates=(answer, "none of the above"),
        answer=answer,
        answer_index=0,
        confidence=0.9,
    )


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/benchmark_v1_full/manifests/test.jsonl"),
    )
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--manifest-probes", type=int, default=500)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--max-prompt-turns", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/visual_dialog_history/retrieval_benchmark.json"),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model.expanduser().resolve(), local_files_only=True
    )
    rng = random.Random(20260929)
    timings = []
    relevance_hits = coreference_hits = unrelated_skips = 0
    budget_violations = turn_limit_violations = 0
    selected_counts = []

    for repetition in range(args.repetitions):
        turns = list(CONTROLLED_TURNS)
        rng.shuffle(turns)
        history = VisualConversationHistory(
            mode="auto",
            max_prompt_turns=args.max_prompt_turns,
            max_prompt_tokens=args.max_prompt_tokens,
            min_relevance=0.12,
        )
        history.bind_image(f"/tmp/controlled-dialogue-{repetition}.png")
        ids = {}
        for question, answer in turns:
            turn = history.add(
                question=question,
                candidates=(answer, "none of the above"),
                answer=answer,
                answer_index=0,
                confidence=0.9,
            )
            ids[question] = turn.turn_id

        probes = (
            ("Which animal is visible?", ids["Which animal is visible?"], "relevance"),
            ("What is the person holding?", ids["What is the person holding?"], "relevance"),
            ("What color is it?", len(turns), "coreference"),
            ("Tell me more about that object.", len(turns), "coreference"),
            ("Is the weather cold?", None, "unrelated"),
        )
        for query, expected, kind in probes:
            start = time.perf_counter_ns()
            context = history.context(query, tokenizer=tokenizer)
            timings.append((time.perf_counter_ns() - start) / 1000.0)
            selected_counts.append(len(context.selected_turn_ids))
            budget_violations += context.token_count > args.max_prompt_tokens
            turn_limit_violations += len(context.selected_turn_ids) > args.max_prompt_turns
            if kind == "relevance":
                relevance_hits += expected in context.selected_turn_ids
            elif kind == "coreference":
                coreference_hits += expected in context.selected_turn_ids
            else:
                unrelated_skips += not context.selected_turn_ids

    rows = list(read_jsonl(args.manifest.expanduser().resolve()))[: args.manifest_probes]
    manifest_hits = 0
    manifest_timings = []
    manifest_budget_violations = 0
    for index, row in enumerate(rows):
        history = VisualConversationHistory(
            mode="auto",
            max_prompt_turns=args.max_prompt_turns,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        history.bind_image(row.image)
        distractors = rows[max(0, index - 7) : index]
        for distractor in distractors:
            add_turn(
                history,
                distractor.question,
                distractor.candidates[distractor.label],
            )
        target = history.add(
            question=row.question,
            candidates=tuple(row.candidates),
            answer=row.candidates[row.label],
            answer_index=row.label,
            confidence=1.0,
        )
        start = time.perf_counter_ns()
        context = history.context("Regarding it, " + row.question, tokenizer=tokenizer)
        manifest_timings.append((time.perf_counter_ns() - start) / 1000.0)
        manifest_hits += target.turn_id in context.selected_turn_ids
        manifest_budget_violations += context.token_count > args.max_prompt_tokens

    persistence = VisualConversationHistory(mode="auto")
    persistence.bind_image("/tmp/persistence.png")
    add_turn(persistence, "What animal is visible?", "dog")
    temporary = args.output.with_suffix(".roundtrip.json")
    persistence.save(temporary)
    restored = VisualConversationHistory(mode="auto")
    restored.load(temporary)
    roundtrip_exact = restored.to_dict() == persistence.to_dict()
    temporary.unlink(missing_ok=True)

    result = {
        "benchmark": "visual-dialog-history-retrieval-v1",
        "seed": 20260929,
        "mode": "auto",
        "tokenizer": str(args.model.expanduser().resolve()),
        "limits": {
            "max_prompt_turns": args.max_prompt_turns,
            "max_prompt_tokens": args.max_prompt_tokens,
        },
        "controlled": {
            "histories": args.repetitions,
            "probes": args.repetitions * 5,
            "relevance_target_recall": relevance_hits / (args.repetitions * 2),
            "coreference_latest_turn_recall": coreference_hits / (args.repetitions * 2),
            "unrelated_skip_rate": unrelated_skips / args.repetitions,
            "mean_selected_turns": statistics.mean(selected_counts),
            "token_budget_violations": budget_violations,
            "turn_limit_violations": turn_limit_violations,
            "mean_retrieval_us": statistics.mean(timings),
            "p50_retrieval_us": percentile(timings, 50),
            "p95_retrieval_us": percentile(timings, 95),
        },
        "manifest_stress": {
            "source": str(args.manifest.expanduser().resolve()),
            "probes": len(rows),
            "target_turn_recall": manifest_hits / max(1, len(rows)),
            "token_budget_violations": manifest_budget_violations,
            "mean_retrieval_us": statistics.mean(manifest_timings),
            "p95_retrieval_us": percentile(manifest_timings, 95),
            "boundary": (
                "manifest-backed questions with synthetic follow-up phrasing; "
                "this measures retrieval mechanics, not visual-dialog answer accuracy"
            ),
        },
        "persistence_roundtrip_exact": roundtrip_exact,
        "passed": all(
            (
                relevance_hits == args.repetitions * 2,
                coreference_hits == args.repetitions * 2,
                unrelated_skips == args.repetitions,
                budget_violations == 0,
                turn_limit_violations == 0,
                manifest_hits == len(rows),
                manifest_budget_violations == 0,
                roundtrip_exact,
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
