"""Stable data and scoring interfaces for the Visual-JEV benchmark_v1.

The benchmark deliberately stores paths rather than image bytes.  Dataset-
specific preparation code is responsible for materialising an image locally;
all training, calibration and evaluation code then consumes the same schema.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

BENCHMARK_FORMAT = "open-jev-benchmark-v1"
DEFAULT_SEED = 20260928
SPLIT_NAMES = ("train", "validation", "calibration", "test")


@dataclass(frozen=True)
class BenchmarkExample:
    """One closed-ended multimodal question in the benchmark_v1 schema."""

    id: str
    dataset: str
    image: str
    question: str
    candidates: tuple[str, ...]
    label: int
    group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.dataset or not self.image or not self.question:
            raise ValueError("id, dataset, image and question must be non-empty")
        if len(self.candidates) < 2 or any(not str(item).strip() for item in self.candidates):
            raise ValueError("candidates must contain at least two non-empty strings")
        if not 0 <= self.label < len(self.candidates):
            raise ValueError("label is outside the candidate range")

    @property
    def answer(self) -> str:
        return self.candidates[self.label]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = list(self.candidates)
        payload["group_id"] = self.group_id or self.id
        payload["schema"] = BENCHMARK_FORMAT
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkExample":
        schema = payload.get("schema", BENCHMARK_FORMAT)
        if schema != BENCHMARK_FORMAT:
            raise ValueError(f"unsupported benchmark schema: {schema}")
        return cls(
            id=str(payload["id"]),
            dataset=str(payload["dataset"]),
            image=str(payload["image"]),
            question=str(payload["question"]),
            candidates=tuple(str(item) for item in payload["candidates"]),
            label=int(payload["label"]),
            group_id=str(payload.get("group_id") or payload["id"]),
            metadata=dict(payload.get("metadata", {})),
        )


@runtime_checkable
class DatasetAdapter(Protocol):
    """Contract implemented by dataset-specific benchmark converters."""

    name: str

    def examples(self) -> Iterable[BenchmarkExample]: ...


@runtime_checkable
class CandidateScorer(Protocol):
    """Common model interface; one scalar logit is returned per candidate."""

    name: str

    def score(self, example: BenchmarkExample) -> Sequence[float]: ...


def _row_id(dataset: str, source_id: Any) -> str:
    return f"{dataset}:{source_id}"


def aokvqa_example(row: Mapping[str, Any], image: str) -> BenchmarkExample:
    return BenchmarkExample(
        id=_row_id("aokvqa", row["question_id"]),
        dataset="aokvqa",
        image=image,
        question=str(row["question"]),
        candidates=tuple(row["choices"]),
        label=int(row["correct_choice_idx"]),
        group_id=str(row.get("image_id") or row["question_id"]),
        metadata={"source_split": row.get("source_split")},
    )


def scienceqa_example(row: Mapping[str, Any], image: str) -> BenchmarkExample:
    source_id = row.get("id") or row.get("question_id")
    if source_id is None:
        source_id = hashlib.sha256(
            (str(row["question"]) + "\0" + image).encode("utf-8")
        ).hexdigest()[:20]
    return BenchmarkExample(
        id=_row_id("scienceqa", source_id),
        dataset="scienceqa_image_only",
        image=image,
        question=str(row["question"]),
        candidates=tuple(row["choices"]),
        label=int(row["answer"]),
        group_id=str(row.get("image_id") or source_id),
        metadata={"source_split": row.get("source_split")},
    )


def iconqa_choice_example(row: Mapping[str, Any], image: str) -> BenchmarkExample:
    raw_choices = row["choices"]
    if isinstance(raw_choices, str):
        raw_choices = [item.strip() for item in raw_choices.split(",")]
    label = row.get("label")
    if label is None:
        answer = str(row["answer"])
        digits = "".join(character for character in answer if character.isdigit())
        if not digits:
            raise ValueError("IconQA row has neither label nor parseable answer")
        label = int(digits)
    source_id = row.get("question_id") or row.get("id")
    return BenchmarkExample(
        id=_row_id("iconqa", source_id),
        dataset="iconqa_choice",
        image=image,
        question=str(row["question"]),
        candidates=tuple(raw_choices),
        label=int(label),
        group_id=str(source_id),
        metadata={"source_split": row.get("source_split"), "ques_type": row.get("ques_type")},
    )


def coco_hard_negative_example(row: Mapping[str, Any], image: str) -> BenchmarkExample:
    negatives = row.get("negatives", ())
    candidates = tuple([str(row["positive"]), *(str(item) for item in negatives)])
    source_id = row.get("id") or row.get("image_id")
    return BenchmarkExample(
        id=_row_id("coco_hard_negative", source_id),
        dataset="coco_hard_negative",
        image=image,
        question=str(row.get("question") or "Which caption is supported by the image?"),
        candidates=candidates,
        label=int(row.get("label", 0)),
        group_id=str(row.get("image_id") or source_id),
        metadata={"source_split": row.get("source_split")},
    )


def read_jsonl(path: str | Path) -> Iterator[BenchmarkExample]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield BenchmarkExample.from_dict(json.loads(line))
                except Exception as error:
                    raise ValueError(f"invalid benchmark row {path}:{line_number}") from error


def write_jsonl(path: str | Path, examples: Iterable[BenchmarkExample]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    datasets: Counter[str] = Counter()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            encoded = (json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
            count += 1
            datasets[example.dataset] += 1
    return {"path": path.name, "count": count, "sha256": digest.hexdigest(), "datasets": dict(sorted(datasets.items()))}


def _split_for(group_id: str, seed: int, ratios: Mapping[str, float]) -> str:
    value = int.from_bytes(
        hashlib.sha256(f"{seed}\0{group_id}".encode("utf-8")).digest()[:8], "big"
    ) / 2**64
    cumulative = 0.0
    for name in SPLIT_NAMES:
        cumulative += ratios[name]
        if value < cumulative:
            return name
    return SPLIT_NAMES[-1]


def build_manifests(
    examples: Iterable[BenchmarkExample],
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    ratios: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Create deterministic, group-safe train/validation/calibration/test files."""
    ratios = dict(ratios or {"train": 0.70, "validation": 0.10, "calibration": 0.10, "test": 0.10})
    if set(ratios) != set(SPLIT_NAMES) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"ratios must contain {SPLIT_NAMES} and sum to one")
    if any(value < 0 for value in ratios.values()):
        raise ValueError("split ratios must be non-negative")
    partitions: dict[str, list[BenchmarkExample]] = defaultdict(list)
    seen_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    for example in examples:
        if example.id in seen_ids:
            raise ValueError(f"duplicate example id: {example.id}")
        seen_ids.add(example.id)
        group_id = example.group_id or example.id
        split = _split_for(group_id, seed, ratios)
        prior = group_splits.setdefault(group_id, split)
        if prior != split:
            raise AssertionError("group leakage detected")
        partitions[split].append(example)
    output_dir = Path(output_dir)
    split_metadata = {
        name: write_jsonl(output_dir / f"{name}.jsonl", sorted(partitions[name], key=lambda item: item.id))
        for name in SPLIT_NAMES
    }
    manifest = {
        "format": BENCHMARK_FORMAT,
        "seed": seed,
        "split_method": "sha256(seed, group_id)",
        "ratios": ratios,
        "total_count": len(seen_ids),
        "splits": split_metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def write_predefined_manifests(
    partitions: Mapping[str, Iterable[BenchmarkExample]],
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    split_method: str = "source-defined",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write explicit partitions after checking IDs and groups cannot leak."""
    materialised = {name: list(partitions.get(name, ())) for name in SPLIT_NAMES}
    seen_ids: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
    for split, examples in materialised.items():
        for example in examples:
            if example.id in seen_ids:
                raise ValueError(f"example {example.id} appears in {seen_ids[example.id]} and {split}")
            seen_ids[example.id] = split
            group_id = example.group_id or example.id
            prior = seen_groups.setdefault(group_id, split)
            if prior != split:
                raise ValueError(f"group {group_id} appears in {prior} and {split}")
    output_dir = Path(output_dir)
    split_metadata = {
        name: write_jsonl(output_dir / f"{name}.jsonl", sorted(materialised[name], key=lambda item: item.id))
        for name in SPLIT_NAMES
    }
    manifest = {
        "format": BENCHMARK_FORMAT,
        "seed": seed,
        "split_method": split_method,
        "total_count": len(seen_ids),
        "splits": split_metadata,
        "provenance": dict(provenance or {}),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
