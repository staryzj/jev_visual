"""Auditable image+text dataset validation and a tiny wiring-only fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import SPLITS, read_split_directory, validate_records


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_vl_records(records, image_root):
    """Validate the JEV schema plus immutable, split-isolated local images."""
    rows = list(records)
    summary = validate_records(rows, extra_input_fields=("images",))
    root = Path(image_root).expanduser().resolve()
    image_splits, image_count = {}, 0
    for index, row in enumerate(rows, 1):
        prefix = f"record {index} ({row.get('id', '?')}): "
        images = row.get("images")
        if not isinstance(images, list) or not 1 <= len(images) <= 8:
            raise ValueError(prefix + "images must contain 1 to 8 local paths")
        if not all(isinstance(value, str) and value for value in images):
            raise ValueError(prefix + "image paths must be nonempty strings")
        expected = row.get("metadata", {}).get("image_sha256")
        if not isinstance(expected, list) or len(expected) != len(images):
            raise ValueError(prefix + "metadata.image_sha256 must align with images")
        for value, claimed in zip(images, expected):
            candidate = Path(value).expanduser()
            path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(prefix + "image is missing or outside image_root: " + value)
            try:
                from PIL import Image
                with Image.open(path) as source:
                    source.verify()
            except Exception as error:
                raise ValueError(prefix + "image is not decodable: " + value) from error
            actual = _sha256(path)
            if claimed != actual:
                raise ValueError(prefix + "image SHA256 differs: " + value)
            previous = image_splits.setdefault(actual, row["split"])
            if previous != row["split"]:
                raise ValueError(prefix + "identical image bytes appear across splits")
            image_count += 1
    return {**summary, "images": image_count, "unique_image_sha256": len(image_splits)}


def build_smoke_dataset(output_dir, image_root):
    """Build five rows only for code-path testing, never for paper results."""
    output = Path(output_dir)
    if output.exists():
        raise ValueError("refusing to overwrite existing VL smoke data")
    root = Path(image_root).expanduser().resolve()
    fixtures = {
        "train": "site/media/customer-context.jpg",
        "calibration": "site/media/citation-control.jpg",
        "validation": "site/media/gameplay-overview.jpg",
        "test": "test.jpg",
        "ood": "site/media/gameplay-doom.jpg",
    }
    rows = []
    for index, split in enumerate(SPLITS):
        image = fixtures[split]
        digest = _sha256(root / image)
        rows.append({
            "id": f"vl-smoke:{split}",
            "group_id": f"vl-smoke-group:{split}",
            "split": split,
            "source": "open-jev-vl/wiring-smoke",
            "state": {"task": "Inspect the attached project screenshot.", "fixture": index},
            "images": [image],
            "question": "Which broad visual category best matches the image?",
            "kind": "choice",
            "options": ["software interface", "animal"],
            "target": [1.0, 0.0],
            "metadata": {
                "target_basis": "manual_smoke_fixture",
                "image_sha256": [digest],
                "provenance": {
                    "type": "import",
                    "input_sha256": digest,
                    "source_url": "local Open-Jev repository fixture",
                    "original_id": image,
                    "license": "MIT",
                    "split_policy": "fixed_unique_images_smoke_only",
                },
            },
        })
    summary = validate_vl_records(rows, root)
    output.mkdir(parents=True)
    files = {}
    for split in SPLITS:
        path = output / f"{split}.jsonl"
        path.write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows if row["split"] == split), encoding="utf-8")
        files[path.name] = _sha256(path)
    manifest = {
        "schema_version": 1,
        "kind": "open_jev_vl_wiring_smoke_only",
        "paper_evidence": False,
        "model_input_fields": ["state", "images", "question", "kind", "options"],
        "summary": summary,
        "files_sha256": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-smoke")
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--image-root", required=True, type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("--data", required=True, type=Path)
    validate.add_argument("--image-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = (build_smoke_dataset(args.output_dir, args.image_root)
                  if args.command == "build-smoke"
                  else validate_vl_records(read_split_directory(args.data), args.image_root))
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
