"""Build a reproducible COCO hard-negative subset for Visual-JEV V2.

The script reads official COCO 2017 caption and instance annotations, keeps
images whose caption names an annotated object, produces two absent-category
substitutions from the same COCO super-category, downloads only the selected
train images, and splits strictly by image id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ALIASES = {
    "airplane": ("airplane", "plane", "aircraft"),
    "bicycle": ("bicycle", "bike"),
    "couch": ("couch", "sofa"),
    "dining table": ("dining table", "table"),
    "motorcycle": ("motorcycle", "motorbike"),
    "potted plant": ("potted plant", "plant"),
    "tv": ("tv", "television"),
}

IRREGULAR_PLURALS = {
    "mouse": "mice",
    "person": "people",
    "sheep": "sheep",
}


def plural_forms(name: str) -> tuple[str, ...]:
    forms = [name]
    if name in IRREGULAR_PLURALS:
        forms.append(IRREGULAR_PLURALS[name])
    if not name.endswith("s"):
        forms.append(name + "s")
    if name.endswith("y"):
        forms.append(name[:-1] + "ies")
    return tuple(forms)


def aliases_for(name: str) -> tuple[str, ...]:
    return ALIASES.get(name, plural_forms(name))


def replace_phrase(text: str, old: str, new: str) -> str | None:
    for alias in sorted(aliases_for(old), key=len, reverse=True):
        match = re.search(rf"\b{re.escape(alias)}\b", text, flags=re.IGNORECASE)
        if match is None:
            continue
        replacement = new
        prefix = text[max(0, match.start() - 8) : match.start()]
        has_singular_marker = re.search(
            r"\b(a|an|one)\s+$", prefix, flags=re.IGNORECASE
        )
        looks_plural = has_singular_marker is None and (
            alias in IRREGULAR_PLURALS.values() or alias.endswith("s")
        )
        if looks_plural:
            replacement = IRREGULAR_PLURALS.get(replacement, replacement + "s")
        if match.group(0)[0].isupper():
            replacement = replacement.capitalize()
        result = text[: match.start()] + replacement + text[match.end() :]
        result = re.sub(r"\ba ([aeiou])", r"an \1", result, flags=re.IGNORECASE)
        result = re.sub(r"\ban ([^aeiou\W])", r"a \1", result, flags=re.IGNORECASE)
        return result
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(item: tuple[str, Path], *, attempts: int = 5) -> None:
    url, path = item
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Open-JEV/0.1"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
            return
        except (OSError, TimeoutError):
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 8) + random.random())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_records(
    captions_path: Path,
    instances_path: Path,
    *,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    captions_data = load_json(captions_path)
    instances_data = load_json(instances_path)
    images = {int(item["id"]): item for item in captions_data["images"]}
    captions: dict[int, list[str]] = defaultdict(list)
    for annotation in captions_data["annotations"]:
        captions[int(annotation["image_id"])].append(annotation["caption"].strip())

    categories = {int(item["id"]): item for item in instances_data["categories"]}
    by_supercategory: dict[str, list[str]] = defaultdict(list)
    for category in categories.values():
        by_supercategory[category["supercategory"]].append(category["name"])
    image_categories: dict[int, set[str]] = defaultdict(set)
    for annotation in instances_data["annotations"]:
        image_categories[int(annotation["image_id"])].add(
            categories[int(annotation["category_id"])]["name"]
        )

    candidates: list[dict[str, Any]] = []
    for image_id, positive_options in captions.items():
        present = image_categories.get(image_id, set())
        for positive in positive_options:
            for source_name in sorted(present, key=len, reverse=True):
                if replace_phrase(positive, source_name, source_name) is None:
                    continue
                supercategory = next(
                    item["supercategory"]
                    for item in categories.values()
                    if item["name"] == source_name
                )
                replacements = [
                    name
                    for name in by_supercategory[supercategory]
                    if name not in present and name != source_name
                ]
                random.Random(seed + image_id).shuffle(replacements)
                negatives = []
                for replacement in replacements:
                    negative = replace_phrase(positive, source_name, replacement)
                    if negative is not None and negative != positive:
                        negatives.append(negative)
                    if len(negatives) == 2:
                        break
                if len(negatives) < 2:
                    continue
                candidates.append(
                    {
                        "image_id": image_id,
                        "positive": positive,
                        "hard_negatives": negatives,
                        "source_object": source_name,
                        "supercategory": supercategory,
                    }
                )
                break
            else:
                continue
            break

    rng = random.Random(seed)
    rng.shuffle(candidates)
    unique: list[dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate["image_id"] in seen:
            continue
        seen.add(candidate["image_id"])
        unique.append(candidate)
        if len(unique) == count:
            break
    if len(unique) < count:
        raise ValueError(f"requested {count} records but only built {len(unique)}")
    return unique, images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/visual-jev-v2"))
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    annotations = root / "coco" / "annotations"
    records, images = build_records(
        annotations / "captions_train2017.json",
        annotations / "instances_train2017.json",
        count=args.train_size + args.validation_size,
        seed=args.seed,
    )
    image_dir = root / "coco" / "train2017"
    downloads = []
    for record in records:
        image = images[record["image_id"]]
        record["file_name"] = image["file_name"]
        url = image.get("coco_url") or (
            "http://images.cocodataset.org/train2017/" + image["file_name"]
        )
        downloads.append((url, image_dir / image["file_name"]))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(download_one, downloads))

    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_records = {
        "train": records[: args.train_size],
        "validation": records[args.train_size :],
    }
    manifest_records = []
    for split, items in split_records.items():
        output_path = split_dir / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for item in items:
                image_path = image_dir / item["file_name"]
                output = {
                    "image": str(image_path.relative_to(root)),
                    "question": "Which description is supported by the image?",
                    "positive": item["positive"],
                    "hard_negatives": item["hard_negatives"],
                    "metadata": {
                        "source": "MS COCO 2017 train",
                        "image_id": item["image_id"],
                        "image_sha256": sha256(image_path),
                        "negative_type": "same-supercategory object substitution",
                        "source_object": item["source_object"],
                        "supercategory": item["supercategory"],
                    },
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                manifest_records.append((split, output))

    manifest = {
        "schema_version": 1,
        "source": "MS COCO 2017 train captions and instances",
        "seed": args.seed,
        "split_policy": "disjoint image_id after deterministic shuffle",
        "train_records": args.train_size,
        "validation_records": args.validation_size,
        "hard_negatives_per_record": 2,
        "unique_images": len(
            {item["metadata"]["image_id"] for _, item in manifest_records}
        ),
        "files": {
            f"splits/{split}.jsonl": sha256(split_dir / f"{split}.jsonl")
            for split in split_records
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
