"""Create one uploadable Open-Jev source bundle with audited artifacts."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import tarfile
from pathlib import Path


EXCLUDED_TOP_LEVEL = {
    ".git", ".venv", "models", "data", "experiments", "checkpoints",
    "reports", "node_modules", ".pytest_cache", "__pycache__",
}
EXCLUDED_PATTERNS = ("*.pyc", "*.pyo", "*.safetensors", "*.bin", "*.pt", "*.pth")
ARTIFACTS = (
    "experiments/results/benchmark_v1/fix_negative_transfer/summary.md",
    "experiments/results/benchmark_v1/fix_negative_transfer/root_cause_analysis.json",
    "experiments/results/benchmark_v1/fix_negative_transfer/experiment_matrix.csv",
    "experiments/results/benchmark_v1/fix_negative_transfer/full_config.json",
    "experiments/results/benchmark_v1/fix_negative_transfer/best_model_results/best_model_results.json",
    "experiments/results/benchmark_v1/fix_negative_transfer/best_model_results/visual_jev_v3_best.pt",
    "experiments/results/benchmark_v1/fix_negative_transfer/calibration_after_fix/calibration_comparison.json",
)
RUNTIME_RESOURCES = (
    "data/benchmark_v1/manifests",
    "data/visual-jev-v2/splits",
    "data/visual-jev-v2/features/validation-shards",
    "data/visual-jev-v3/pairs",
    "data/visual-jev-v3/features/train-pair-text-shards",
    "data/visual-jev-v3/features/validation-pair-text-shards",
    "experiments/benchmark_v1_controlled/features",
    "experiments/benchmark_v1_controlled/checkpoints",
    "experiments/visual_jev_v3_paper/features/train-shards",
    "experiments/visual_jev_v3_paper/features/validation-shards",
    "experiments/visual_jev_v3_paper/checkpoints/v3_full.pt",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def include_source(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if not relative.parts or relative.parts[0] in EXCLUDED_TOP_LEVEL:
        return False
    if any(part in {"__pycache__", ".pytest_cache"} for part in relative.parts):
        return False
    return not any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDED_PATTERNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--include-runtime-resources", action="store_true",
        help="include the minimal cached tensors/checkpoints needed to run the controlled matrix",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    compression = 1 if args.include_runtime_resources else 6
    with tarfile.open(output, "w:gz", compresslevel=compression) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if not include_source(path, root):
                continue
            relative = path.relative_to(root)
            archive.add(path, arcname=Path("Open-Jev-server") / relative, recursive=False)
            manifest.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": digest(path)})
        for relative_name in ARTIFACTS:
            path = root / relative_name
            if not path.is_file():
                continue
            artifact_relative = Path("artifacts") / Path(relative_name).relative_to(
                "experiments/results/benchmark_v1/fix_negative_transfer"
            )
            archive.add(path, arcname=Path("Open-Jev-server") / artifact_relative, recursive=False)
            manifest.append({"path": str(artifact_relative), "bytes": path.stat().st_size, "sha256": digest(path)})
        if args.include_runtime_resources:
            for relative_name in RUNTIME_RESOURCES:
                resource = root / relative_name
                if not resource.exists():
                    raise FileNotFoundError(resource)
                paths = [resource] if resource.is_file() else sorted(
                    path for path in resource.rglob("*") if path.is_file()
                )
                for path in paths:
                    relative = path.relative_to(root)
                    archive.add(path, arcname=Path("Open-Jev-server") / relative, recursive=False)
                    manifest.append({
                        "path": str(relative), "bytes": path.stat().st_size,
                        "sha256": None, "role": "runtime_resource",
                    })
        payload = json.dumps({
            "format": "open-jev-server-source-bundle-v1",
            "source_root": str(root),
            "files": manifest,
            "excluded_large_resources": sorted(EXCLUDED_TOP_LEVEL),
            "resource_instructions": "SERVER_RUN.md",
            "runtime_resources_included": args.include_runtime_resources,
        }, ensure_ascii=False, indent=2).encode("utf-8")
        info = tarfile.TarInfo("Open-Jev-server/BUNDLE_MANIFEST.json")
        info.size = len(payload)
        import io
        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps({
        "output": str(output), "bytes": output.stat().st_size,
        "sha256": digest(output), "files": len(manifest),
    }, indent=2))


if __name__ == "__main__":
    main()

