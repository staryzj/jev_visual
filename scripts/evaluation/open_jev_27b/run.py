"""Run one route and shard using local model artifacts and a runtime factory."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from .connector import from_checkpoint
from .measurement import measure


def load_factory(target):
    module, separator, name = target.partition(":")
    if not separator or not module or not name:
        raise ValueError("Runtime factory must use module:function syntax")
    return getattr(importlib.import_module(module), name)


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("llava", "internvl", "qwen", "siglip2"), required=True)
    parser.add_argument("--kind", choices=("external", "short", "rationale", "matching"), required=True)
    parser.add_argument("--items", type=Path, required=True, help="Shard JSON with row and cluster fields")
    parser.add_argument("--output", type=Path, required=True, help="New measurement JSONL file")
    parser.add_argument("--runtime", required=True, help="Local factory, module:function")
    parser.add_argument("--checkpoint", type=Path, help="Locked adapter checkpoint for external scoring")
    args = parser.parse_args()
    if (args.kind == "external") != (args.checkpoint is not None):
        parser.error("--checkpoint is required for external scoring only")
    if args.kind != "external" and (args.encoder == "siglip2") != (args.kind == "matching"):
        parser.error("The native SigLIP2 route uses matching")
    if args.output.exists():
        parser.error("--output must name a new file")
    if torch.cuda.device_count() != 1:
        parser.error("Expose exactly one CUDA device")
    torch.set_num_threads(3)
    torch.manual_seed(20260928)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    items = json.loads(args.items.read_text(encoding="utf-8"))
    if not items or len({item["row"]["uid"] for item in items}) != len(items):
        parser.error("The shard must contain unique UIDs")
    recipes = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8"))
    recipe = recipes["methods"][args.encoder]
    bridge = None
    if args.kind == "external":
        if file_sha256(args.checkpoint) != recipe["checkpoint_sha256"]:
            parser.error("Adapter checkpoint differs from the locked recipe")
        bridge = from_checkpoint(args.checkpoint, device="cuda:0")
    runtime = load_factory(args.runtime)(args.encoder, args.kind)
    model = runtime["model"]
    if args.kind == "external":
        if any(p.requires_grad for p in model.mm.parameters()):
            raise ValueError("The receiver must be frozen")
        if any(p.requires_grad for p in runtime["encoder"].model.parameters()):
            raise ValueError("The external encoder must be frozen")
    elif any(p.requires_grad for p in model.parameters()):
        raise ValueError("The native comparator must be frozen")

    def measure_row(item):
        return measure(item, kind=args.kind, model=model, bridge=bridge, tap=recipe["tap"],
                       processor=runtime.get("processor"), encoder=runtime.get("encoder"))

    failures = 0
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for item in items[:recipes["warmup_per_worker"]]:
            measure_row(item)
        with args.output.open("x", encoding="utf-8") as stream:
            for item in items:
                row = item["row"]
                fatal = None
                try:
                    value = measure_row(item)
                    value["execution_error"] = False
                except Exception as exc:
                    failures += 1
                    value = {
                        "uid": row["uid"], "dataset": row["dataset"], "cluster": item["cluster"],
                        "label": row["label"], "candidate_N": len(row["candidates"]),
                        "has_image": bool(row["images"]), "supported": True, "correct": False,
                        "prediction": None, "execution_error": True, "error_type": type(exc).__name__,
                    }
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    elif isinstance(exc, RuntimeError) and any(
                        message in str(exc) for message in ("illegal memory", "device-side assert")
                    ):
                        fatal = exc
                stream.write(json.dumps(value, allow_nan=False) + "\n")
                stream.flush()
                if fatal is not None:
                    raise fatal

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
