"""Build deterministic counterfactual image pairs from Visual-JEV COCO splits.

For a source record with candidates ``T1`` (positive) and ``T2`` (an object
substitution), the script finds a different image whose annotated source
object is the object introduced in T2.  The resulting pair supervises
``I1:T1>T2`` and ``I2:T2>T1`` while keeping the candidate set fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata", {})
            if not isinstance(metadata.get("image_id"), int):
                raise TypeError(f"{path}:{line_number}: missing integer image_id")
            if not isinstance(metadata.get("source_object"), str):
                raise TypeError(f"{path}:{line_number}: missing source_object")
            rows.append(row)
    return rows


def contains_phrase(text: str, phrase: str) -> bool:
    return (
        re.search(
            r"(?<![\w])" + re.escape(phrase) + r"(?![\w])",
            text,
            re.IGNORECASE,
        )
        is not None
    )


def identify_target_object(
    negative: str, *, source_object: str, vocabulary: list[str]
) -> str | None:
    matches = [
        value
        for value in vocabulary
        if value != source_object and contains_phrase(negative, value)
    ]
    if not matches:
        return None
    return max(matches, key=lambda value: (len(value), value))


def build_pairs(
    rows: list[dict[str, Any]], *, split: str, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_object: dict[str, list[int]] = defaultdict(list)
    image_ids = []
    for index, row in enumerate(rows):
        metadata = row["metadata"]
        by_object[metadata["source_object"]].append(index)
        image_ids.append(metadata["image_id"])
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{split}: duplicate image_id violates isolation")

    vocabulary = sorted(by_object, key=lambda value: (-len(value), value))
    rng = random.Random(seed)
    pairs = []
    missing_target_object = 0
    missing_partner = 0
    for source_index, row in enumerate(rows):
        metadata = row["metadata"]
        source_id = metadata["image_id"]
        candidates = [row["positive"], *row.get("hard_negatives", [])]
        selected = None
        for candidate_index, negative in enumerate(candidates[1:], start=1):
            target_object = identify_target_object(
                negative,
                source_object=metadata["source_object"],
                vocabulary=vocabulary,
            )
            if target_object is None:
                missing_target_object += 1
                continue
            partners = [
                index
                for index in by_object[target_object]
                if rows[index]["metadata"]["image_id"] != source_id
            ]
            if not partners:
                missing_partner += 1
                continue
            partner_index = partners[rng.randrange(len(partners))]
            selected = (candidate_index, target_object, partner_index)
            break
        if selected is None:
            continue
        candidate_index, target_object, partner_index = selected
        partner = rows[partner_index]
        partner_id = partner["metadata"]["image_id"]
        if source_id == partner_id:
            raise AssertionError("counterfactual pair reused the same image_id")
        pairs.append(
            {
                "schema_version": 1,
                "split": split,
                "pair_type": "object_flip",
                "source_index": source_index,
                "counterfactual_index": partner_index,
                "source_image_id": source_id,
                "counterfactual_image_id": partner_id,
                "source_target_index": 0,
                "counterfactual_target_index": 1,
                "source_object": metadata["source_object"],
                "counterfactual_object": target_object,
                "candidate_set": [
                    f"The image contains a {metadata['source_object']}.",
                    f"The image contains a {target_object}.",
                ],
                "source_record_candidate_index": candidate_index,
                "source_record_candidates": candidates,
            }
        )
    stats = {
        "records": len(rows),
        "pairs": len(pairs),
        "unpaired_records": len(rows) - len(pairs),
        "missing_target_object_attempts": missing_target_object,
        "missing_partner_attempts": missing_partner,
    }
    return pairs, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/visual-jev-v2"))
    parser.add_argument("--output-root", type=Path, default=Path("data/visual-jev-v3"))
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    pair_dir = output_root / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)

    split_rows = {
        split: load_jsonl(source_root / "splits" / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    train_ids = {row["metadata"]["image_id"] for row in split_rows["train"]}
    validation_ids = {
        row["metadata"]["image_id"] for row in split_rows["validation"]
    }
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(
            f"train/validation image_id leakage: {sorted(overlap)[:5]}"
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_root": str(source_root),
        "seed": args.seed,
        "pair_policy": (
            "fixed candidate set; target object matched to a distinct COCO image; "
            "train and validation image_id sets remain disjoint"
        ),
        "splits": {},
    }
    for offset, split in enumerate(("train", "validation")):
        source_path = source_root / "splits" / f"{split}.jsonl"
        pairs, stats = build_pairs(
            split_rows[split], split=split, seed=args.seed + offset
        )
        output_path = pair_dir / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            **stats,
            "source_sha256": file_sha256(source_path),
            "pairs_sha256": file_sha256(output_path),
            "unique_source_image_ids": len(
                {pair["source_image_id"] for pair in pairs}
            ),
            "unique_counterfactual_image_ids": len(
                {pair["counterfactual_image_id"] for pair in pairs}
            ),
        }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"wrote counterfactual metadata: {manifest_path}")


if __name__ == "__main__":
    main()
