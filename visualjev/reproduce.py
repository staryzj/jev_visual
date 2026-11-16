"""Prepare the sealed paper population, run published weights, and check accuracy."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

from .assets import ROOT, content_hash, download_file

DATASETS = ["aokvqa", "scienceqa", "iconqa", "iconqa_choose_img", "ai2d"]
EXPECTED_COUNTS = dict(zip(DATASETS, [1145, 4241, 6316, 11535, 3088]))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pixel_hash(image):
    image = image.convert("RGB")
    digest = hashlib.sha256(str(image.size).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def safe_extract(archive, destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    def target(name):
        output = (destination / name).resolve()
        if not output.is_relative_to(destination):
            raise ValueError("Unsafe archive member: " + name)
        return output
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                output = target(member.filename)
                if member.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    raw = source.read(member)
                    if output.exists() and output.read_bytes() != raw:
                        raise ValueError("Existing extracted file differs: " + str(output))
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(raw)
    else:
        with tarfile.open(archive) as source:
            for member in source:
                output = target(member.name)
                if not member.isfile():
                    raise ValueError("Only regular files are expected in this source archive")
                raw = source.extractfile(member).read()
                if output.exists() and output.read_bytes() != raw:
                    raise ValueError("Existing extracted file differs: " + str(output))
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(raw)


def fetch(args):
    lock = json.loads((ROOT / "release_metadata/reproduction_assets.json").read_text())
    for entry in lock["files"]:
        if entry["role"] == "source_archive" and not args.include_source:
            continue
        asset = dict(key="downloads", url_prefix=lock["url_prefix"])
        record = dict(path=entry["name"], size=entry["bytes"], sha256=entry["sha256"])
        download_file(asset, record, args.destination, lambda *values: None)
        archive = args.destination / "downloads" / entry["name"]
        folder = args.destination / ("paper" if entry["role"] == "benchmark" else "original_experiments")
        safe_extract(archive, folder)
        print(f"Verified and extracted {entry['name']} to {folder}", flush=True)


def _source_dataset(name, metadata, cache, source_cache):
    from datasets import load_dataset, load_from_disk
    if source_cache and (source_cache / name / "dataset").is_dir():
        return load_from_disk(str(source_cache / name / "dataset"))
    from huggingface_hub import hf_hub_download
    files = []
    for record in metadata["download"]["files"]:
        path = Path(hf_hub_download(metadata["repo"], record["path"], repo_type="dataset",
                                   revision=metadata["download"]["revision"], cache_dir=cache / "hf"))
        if content_hash(path) != record["sha256"]:
            raise ValueError("Dataset shard differs from the paper download: " + str(path))
        files.append(str(path))
    return load_dataset("parquet", data_files=files, split="train", cache_dir=str(cache / "datasets"))


def prepare(args):
    from PIL import Image
    rows = read_rows(args.bundle / "benchmark.jsonl")
    if Counter(x["dataset"] for x in rows) != Counter(EXPECTED_COUNTS):
        raise ValueError("The paper manifest does not contain the five complete populations")
    if len({x["uid"] for x in rows}) != 26325:
        raise ValueError("Duplicate or missing paper UIDs")
    metadata = json.loads((args.bundle / "dataset_sources.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / ".downloads"
    datasets = {}
    official_zip = None
    if not args.prepared_images:
        for name in DATASETS:
            if name != "iconqa_choose_img":
                datasets[name] = _source_dataset(name, metadata[name], cache, args.source_cache)
        if not args.iconqa_root:
            info = metadata["iconqa_choose_img"]["download"]
            filename = "iconqa_data.zip"
            download_file(dict(key="iconqa", url_prefix=info["url"].removesuffix(filename)),
                          dict(path=filename, size=info["size"], sha256=info["sha256"]),
                          cache, lambda *values: None)
            official_zip = zipfile.ZipFile(cache / "iconqa" / filename)
    prepared = []
    try:
        for index, row in enumerate(rows):
            result = dict(row)
            if row["image"]:
                target = args.output / row["image"]
                if args.prepared_images:
                    target = args.prepared_images / row["image"]
                    with Image.open(target) as image:
                        if pixel_hash(image) != row["pixel_sha256"]:
                            raise ValueError("Image pixels differ for " + row["uid"])
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if row["dataset"] == "iconqa_choose_img":
                        from PIL import ImageDraw
                        from io import BytesIO
                        pid = row["uid"].split(":", 1)[1]
                        relative = Path("iconqa/test/choose_img") / pid
                        def icon_image(name):
                            if args.iconqa_root:
                                with Image.open(args.iconqa_root / relative / name) as image:
                                    return image.convert("RGB")
                            with Image.open(BytesIO(official_zip.read("iconqa_data/" +
                                                   (relative / name).as_posix()))) as image:
                                return image.convert("RGB")
                        first = icon_image("image.png")
                        options = [icon_image(f"choice_{i}.png") for i in range(len(row["candidates"]))]
                        cell_width = max(image.width for image in options) + 20
                        width = max(first.width, cell_width * len(options))
                        height = first.height + 20 + max(image.height for image in options) + 36
                        image = Image.new("RGB", (width, height), "white")
                        image.paste(first, ((width - first.width) // 2, 0))
                        draw = ImageDraw.Draw(image)
                        y = first.height + 20
                        for i, option in enumerate(options):
                            draw.text((i * cell_width + 10, y), f"Option {i}", fill="black")
                            image.paste(option, (i * cell_width + 10, y + 24))
                    else:
                        source_index = int(Path(row["image"]).stem)
                        original = datasets[row["dataset"]][source_index]
                        image = original.get("query_image", original.get("image")).convert("RGB")
                    if pixel_hash(image) != row["pixel_sha256"]:
                        raise ValueError("Regenerated pixels differ for " + row["uid"])
                    image.save(target, format="PNG")
                result["image"] = str(target.resolve())
            prepared.append(result)
            if (index + 1) % 1000 == 0:
                print(f"Verified {index + 1}/26325 questions", flush=True)
    finally:
        if official_zip:
            official_zip.close()
    for shard in range(8):
        data = [row for row in prepared if row["shard"] == shard]
        (args.output / f"shard{shard}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data), encoding="utf-8")
    (args.output / "questions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared), encoding="utf-8")
    print(json.dumps({"questions": len(prepared), "image_questions": sum(bool(x["image"]) for x in prepared),
                      "verified_pixels": True, "output": str(args.output)}))


def score(manifest, predictions, allow_subset=False):
    truth = {row["uid"]: row for row in manifest}
    seen = set()
    counts = {}
    for row in predictions:
        uid = row["uid"]
        if uid not in truth or uid in seen:
            raise ValueError("Unknown or duplicated prediction UID: " + uid)
        seen.add(uid)
        expected = truth[uid]
        if "candidates" in row and row["candidates"] != expected["candidates"]:
            raise ValueError("Candidate order changed for " + uid)
        if "label" in row and row["label"] != expected["label"]:
            raise ValueError("Prediction label changed for " + uid)
        if not row.get("supported", True):
            continue
        correct = row.get("prediction") == expected["label"]
        for key in ["overall", expected["dataset"]] + (
                ["scienceqa_image"] if expected["dataset"] == "scienceqa" and expected["image"] else []):
            entry = counts.setdefault(key, {"N": 0, "correct": 0, "execution_errors": 0})
            entry["N"] += 1
            entry["correct"] += int(correct)
            entry["execution_errors"] += int(row.get("execution_error", False))
    if not allow_subset and seen != set(truth):
        raise ValueError(f"Incomplete prediction coverage: {len(seen)}/{len(truth)}")
    for entry in counts.values():
        entry["accuracy_pct"] = 100 * entry["correct"] / entry["N"]
    return counts


def verify_reference(bundle):
    manifest = read_rows(bundle / "benchmark.jsonl")
    expected = json.loads((bundle / "expected_summary.json").read_text())
    result = {}
    for route, original in expected["routes"].items():
        measured = score(manifest, read_rows(bundle / "reference_predictions" / (route + ".jsonl")))
        for name, entry in measured.items():
            target = original["overall"] if name == "overall" else (
                original["scienceqa_image"] if name == "scienceqa_image" else original["tasks"][name])
            if abs(entry["accuracy_pct"] - target["accuracy_pct"]) > 1e-10:
                raise ValueError("Recorded accuracy does not reproduce: " + route + "/" + name)
        result[route] = measured
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("fetch")
    get.add_argument("--destination", type=Path, required=True)
    get.add_argument("--include-source", action="store_true")
    prep = sub.add_parser("prepare")
    prep.add_argument("--bundle", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--source-cache", type=Path, help="Existing dataset_name/dataset saved datasets")
    prep.add_argument("--prepared-images", type=Path, help="Existing dataset_name/images image folders")
    prep.add_argument("--iconqa-root", type=Path, help="Extracted official iconqa_data directory")
    check = sub.add_parser("verify-reference")
    check.add_argument("--bundle", type=Path, required=True)
    check.add_argument("--output", type=Path)
    evaluate = sub.add_parser("score")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, nargs="+", required=True)
    evaluate.add_argument("--allow-subset", action="store_true")
    evaluate.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch(args)
    elif args.command == "prepare":
        prepare(args)
    else:
        if args.command == "verify-reference":
            result = verify_reference(args.bundle)
        else:
            predictions = [row for path in args.predictions for row in read_rows(path)]
            result = score(read_rows(args.bundle / "benchmark.jsonl"), predictions, args.allow_subset)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
