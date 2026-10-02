"""Build a deterministic, leakage-safe controlled benchmark_v1 manifest.

The controlled profile is intentionally small enough to run end-to-end on a
single 16 GB GPU. It uses labelled A-OKVQA, image-only ScienceQA, IconQA
``choose_txt`` and the existing COCO hard-negative corpus. POPE, POPEv2 and
MMMU are audited but reserved for external evaluation. SugarCrepe and
Winoground are only marked available when their actual data files exist.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from PIL import Image

from jev.benchmark_v1 import BENCHMARK_FORMAT, DEFAULT_SEED, BenchmarkExample, write_predefined_manifests

CAPS = {"train": 32, "validation": 16, "calibration": 16, "test": 16}
EVAL_SPLITS = ("validation", "calibration", "test")


def _digest(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return bytes(value["bytes"])
        if value.get("path"):
            return Path(value["path"]).read_bytes()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise ValueError("parquet image has neither bytes nor a readable path")


def _split_for(key: str, seed: int) -> str:
    value = int(_digest(f"{seed}\0{key}")[:16], 16) % len(EVAL_SPLITS)
    return EVAL_SPLITS[value]


def _candidate_shuffle(
    candidates: list[str], label: int, example_id: str, seed: int
) -> tuple[tuple[str, ...], int, list[int]]:
    order = list(range(len(candidates)))
    random.Random(f"{seed}:{example_id}").shuffle(order)
    return tuple(candidates[index] for index in order), order.index(label), order


class Selector:
    """Keep the deterministic lowest-ranked examples for every dataset/split."""

    def __init__(self, caps: dict[str, int], seed: int):
        self.caps = caps
        self.seed = seed
        self.heaps: dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)

    def add(self, dataset: str, split: str, source_id: str, payload: dict[str, Any]) -> None:
        cap = self.caps[split]
        rank = int(_digest(f"{self.seed}\0{dataset}\0{split}\0{source_id}")[:16], 16)
        item = (-rank, source_id, payload)
        heap = self.heaps[(dataset, split)]
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def selected(self) -> Iterable[tuple[str, str, dict[str, Any]]]:
        for (dataset, split), heap in sorted(self.heaps.items()):
            for _, _, payload in sorted(heap, reverse=True):
                yield dataset, split, payload


def _iter_parquet(path: Path, columns: list[str]) -> Iterable[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        yield from batch.to_pylist()


def _scan_aokvqa(root: Path, selector: Selector, audit: dict[str, Any]) -> None:
    files = sorted((root / "aokvqa" / "data").glob("*.parquet"))
    for path in files:
        source_split = "train" if path.name.startswith("train-") else "validation" if path.name.startswith("validation-") else "test"
        for row in _iter_parquet(path, ["image", "question_id", "question", "choices", "correct_choice_idx"]):
            audit["rows"]["aokvqa"] += 1
            if row["correct_choice_idx"] is None:
                audit["excluded"]["aokvqa_unlabelled"] += 1
                continue
            raw = _image_bytes(row["image"])
            image_sha = _digest(raw)
            split = "train" if source_split == "train" else _split_for(image_sha, selector.seed)
            selector.add("aokvqa", split, str(row["question_id"]), {
                "source_id": str(row["question_id"]), "source_split": source_split,
                "question": str(row["question"]), "candidates": [str(x) for x in row["choices"]],
                "label": int(row["correct_choice_idx"]), "image_bytes": raw, "image_sha256": image_sha,
            })


def _scan_scienceqa(root: Path, selector: Selector, audit: dict[str, Any]) -> None:
    for path in sorted((root / "scienceqa" / "data").glob("*.parquet")):
        source_split = "train" if path.name.startswith("train-") else "validation" if path.name.startswith("validation-") else "test"
        for index, row in enumerate(_iter_parquet(path, ["image", "question", "choices", "answer", "task"])):
            audit["rows"]["scienceqa"] += 1
            if row["image"] is None:
                audit["excluded"]["scienceqa_text_only"] += 1
                continue
            raw = _image_bytes(row["image"])
            image_sha = _digest(raw)
            source_id = f"{source_split}:{index}:{_digest(str(row['question']))[:12]}"
            split = "train" if source_split == "train" else _split_for(image_sha, selector.seed)
            selector.add("scienceqa_image_only", split, source_id, {
                "source_id": source_id, "source_split": source_split,
                "question": str(row["question"]), "candidates": [str(x) for x in row["choices"]],
                "label": int(row["answer"]), "image_bytes": raw, "image_sha256": image_sha,
            })


def _scan_iconqa(root: Path, selector: Selector, audit: dict[str, Any]) -> None:
    for path in sorted((root / "iconqa" / "data").glob("*.parquet")):
        source_split = "validation" if path.name.startswith("val-") else "test"
        columns = ["query_image", "question_id", "question", "choices", "answer", "ques_type"]
        for index, row in enumerate(_iter_parquet(path, columns)):
            audit["rows"]["iconqa"] += 1
            if row["ques_type"] != "choose_txt":
                audit["excluded"][f"iconqa_{row['ques_type']}"] += 1
                continue
            candidates = [item.strip() for item in str(row["choices"]).split(",") if item.strip()]
            try:
                label = candidates.index(str(row["answer"]).strip())
            except ValueError:
                audit["excluded"]["iconqa_answer_not_in_choices"] += 1
                continue
            raw = _image_bytes(row["query_image"])
            image_sha = _digest(raw)
            source_id = f"{source_split}:{row['question_id']}:{index}"
            split = "train" if source_split == "validation" else _split_for(image_sha, selector.seed)
            selector.add("iconqa_choice", split, source_id, {
                "source_id": source_id, "source_split": source_split,
                "question": str(row["question"]), "candidates": candidates, "label": label,
                "image_bytes": raw, "image_sha256": image_sha,
            })


def _scan_coco(project: Path, selector: Selector, audit: dict[str, Any]) -> None:
    data_root = project / "data" / "visual-jev-v2"
    for source_split in ("train", "validation"):
        path = data_root / "splits" / f"{source_split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                audit["rows"]["coco_hard_negative"] += 1
                image_path = (data_root / row["image"]).resolve()
                image_sha = _digest(image_path.read_bytes())
                image_id = row.get("metadata", {}).get("image_id", index)
                source_id = f"{source_split}:{image_id}:{index}"
                split = "train" if source_split == "train" else _split_for(image_sha, selector.seed)
                selector.add("coco_hard_negative", split, source_id, {
                    "source_id": source_id, "source_split": source_split,
                    "question": str(row.get("question") or "Which caption is supported by the image?"),
                    "candidates": [str(row["positive"]), *[str(x) for x in row["hard_negatives"]]],
                    "label": 0, "image_path": str(image_path), "image_sha256": image_sha,
                })


def _materialise_image(payload: dict[str, Any], image_root: Path) -> str:
    if payload.get("image_path"):
        return payload["image_path"]
    raw = payload.pop("image_bytes")
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        suffix = ".png" if image.format == "PNG" else ".jpg"
    path = image_root / f"{payload['image_sha256']}{suffix}"
    if not path.is_file():
        path.write_bytes(raw)
    return str(path.resolve())


def _external_audit(raw_root: Path) -> dict[str, Any]:
    def parquet_rows(paths: Iterable[Path]) -> tuple[int, int]:
        files = list(paths)
        return len(files), sum(pq.ParquetFile(path).metadata.num_rows for path in files)

    sugar_files = list((raw_root / "sugarcrepe").rglob("*.json")) if (raw_root / "sugarcrepe").exists() else []
    winoground = parquet_rows((raw_root / "winoground").rglob("*.parquet"))
    pope = parquet_rows((raw_root / "pope" / "Full").glob("*.parquet"))
    popev2 = parquet_rows((raw_root / "popev2").glob("*.parquet"))
    mmmu = parquet_rows((raw_root / "mmmu").rglob("*.parquet"))
    return {
        "sugarcrepe": {"role": "external_test", "available": bool(sugar_files), "files": len(sugar_files)},
        "winoground": {"role": "external_test", "available": winoground[0] > 0, "files": winoground[0], "rows": winoground[1], "note": "README/lock files only" if winoground[0] == 0 else "ready"},
        "pope": {"role": "external_test", "available": pope[0] > 0, "files": pope[0], "rows": pope[1]},
        "popev2": {"role": "external_test", "available": popev2[0] > 0, "files": popev2[0], "rows": popev2[1]},
        "mmmu": {"role": "external_test", "available": mmmu[0] > 0, "files": mmmu[0], "rows": mmmu[1]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/visual_jev_raw"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmark_v1/manifests"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-per-dataset", type=int, default=CAPS["train"])
    parser.add_argument("--eval-per-dataset", type=int, default=CAPS["validation"])
    parser.add_argument(
        "--full",
        action="store_true",
        help="Keep every eligible labelled record; overrides the per-dataset caps.",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    raw_root = args.raw_root.resolve()
    output = args.output.resolve()
    if args.full:
        caps = {name: 2**31 - 1 for name in ("train", *EVAL_SPLITS)}
    else:
        caps = {"train": args.train_per_dataset, **{name: args.eval_per_dataset for name in EVAL_SPLITS}}
    selector = Selector(caps, args.seed)
    audit: dict[str, Any] = {"rows": Counter(), "excluded": Counter()}
    _scan_aokvqa(raw_root, selector, audit)
    _scan_scienceqa(raw_root, selector, audit)
    _scan_iconqa(raw_root, selector, audit)
    _scan_coco(project, selector, audit)

    image_root = output.parent / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    partitions: dict[str, list[BenchmarkExample]] = defaultdict(list)
    selected_counts: Counter[str] = Counter()
    selected_rows = list(selector.selected())
    group_splits: dict[str, str] = {}
    for _dataset, proposed_split, payload in selected_rows:
        image_sha = str(payload["image_sha256"])
        current = group_splits.get(image_sha)
        if proposed_split == "train" or current is None:
            group_splits[image_sha] = proposed_split
    for dataset, proposed_split, payload in selected_rows:
        split = group_splits[str(payload["image_sha256"])]
        if split != proposed_split:
            audit["excluded"][f"{dataset}_group_split_reassigned_{proposed_split}_to_{split}"] += 1
        example_id = f"{dataset}:{payload['source_id']}"
        question = str(payload.get("question", "")).strip()
        raw_candidates = [str(value).strip() for value in payload.get("candidates", [])]
        source_label = int(payload.get("label", -1))
        valid_source_indices = [index for index, value in enumerate(raw_candidates) if value]
        if not question:
            audit["excluded"][f"{dataset}_empty_question"] += 1
            continue
        if source_label not in valid_source_indices or len(valid_source_indices) < 2:
            audit["excluded"][f"{dataset}_invalid_candidates"] += 1
            continue
        clean_candidates = [raw_candidates[index] for index in valid_source_indices]
        clean_label = valid_source_indices.index(source_label)
        candidates, label, order = _candidate_shuffle(
            clean_candidates, clean_label, example_id, args.seed
        )
        image_path = _materialise_image(payload, image_root)
        partitions[split].append(BenchmarkExample(
            id=example_id, dataset=dataset, image=image_path,
            question=question, candidates=candidates, label=label,
            group_id=f"image:{payload['image_sha256']}",
            metadata={
                "source_split": payload["source_split"], "source_id": payload["source_id"],
                "image_sha256": payload["image_sha256"], "candidate_permutation": order,
                "candidate_source_indices": valid_source_indices,
                "selection_profile": "full_eligible" if args.full else "controlled_equal_per_dataset",
            },
        ))
        selected_counts[f"{dataset}/{split}"] += 1

    manifest = write_predefined_manifests(
        partitions, output, seed=args.seed,
        split_method="source train -> train; labelled held-out source -> sha256(image) validation/calibration/test; deterministic capped sampling",
        provenance={
            "schema": BENCHMARK_FORMAT, "profile": "full_eligible" if args.full else "controlled_subset",
            "caps_per_dataset": caps, "candidate_order": "deterministic shuffle with stored permutation",
            "training_datasets": ["aokvqa", "scienceqa_image_only", "iconqa_choice", "coco_hard_negative"],
            "external_test_only": _external_audit(raw_root),
            "full_scale_pending": not args.full,
        },
    )
    audit_payload = {
        "seed": args.seed, "raw_rows": dict(sorted(audit["rows"].items())),
        "excluded": dict(sorted(audit["excluded"].items())),
        "selected": dict(sorted(selected_counts.items())),
        "external": _external_audit(raw_root), "manifest": manifest,
    }
    (output / "data_audit.json").write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
