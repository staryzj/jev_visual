from __future__ import annotations

import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModel,
    InternVLForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
)

DEVICE = "cuda:0"
DTYPE = torch.bfloat16

MANIFEST = Path("data/benchmark_v1_full/manifests/test.jsonl")

BACKBONES = {
    "siglip2": Path("models/SigLIP2-Base-Patch16-224"),
    "internvl": Path("models/InternVL3_5-1B-HF"),
    "llava": Path("models/LLaVA-OneVision-0.5B"),
}

WARMUP = 5
RUNS = 30


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def move_inputs(inputs):
    result = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                result[key] = value.to(DEVICE, dtype=DTYPE)
            else:
                result[key] = value.to(DEVICE)
        else:
            result[key] = value
    return result


def normalize_tokens(output):
    """
    Convert one-image visual output to [tokens, hidden_dim].

    Handles:
      [B, T, D]
      [B, N, T, D]
      [N, T, D] from a one-image feature list
      [T, D]
    """

    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state

    # LLaVA commonly returns a list with one tensor per image.
    if isinstance(output, (tuple, list)):
        if len(output) == 0:
            raise RuntimeError("Empty visual feature output")

        tensors = []

        for x in output:
            if hasattr(x, "last_hidden_state"):
                x = x.last_hidden_state

            if torch.is_tensor(x):
                tensors.append(x)

        if not tensors:
            raise RuntimeError(
                f"No tensor found in output type {type(output)}"
            )

        # We benchmark batch size = 1.
        output = tensors[0]

        # For LLaVA this can be [num_patches, tokens, dim].
        if output.ndim == 3 and output.shape[0] > 1:
            return output.reshape(-1, output.shape[-1])

    if not torch.is_tensor(output):
        raise RuntimeError(
            f"Unsupported feature type: {type(output)}"
        )

    x = output

    if x.ndim == 4:
        # [B, N, T, D]
        if x.shape[0] != 1:
            raise RuntimeError(
                f"Expected batch size 1, got {tuple(x.shape)}"
            )
        x = x[0].reshape(-1, x.shape[-1])

    elif x.ndim == 3:
        # Normal VLM output: [B, T, D]
        if x.shape[0] == 1:
            x = x[0]
        else:
            # One-image multi-patch representation: [N, T, D]
            x = x.reshape(-1, x.shape[-1])

    elif x.ndim == 2:
        pass

    elif x.ndim == 1:
        x = x.unsqueeze(0)

    else:
        raise RuntimeError(
            f"Unsupported visual feature shape: {tuple(x.shape)}"
        )

    return x


class SigLIP2Extractor:
    def __init__(self, path):
        self.processor = AutoProcessor.from_pretrained(
            path,
            local_files_only=True,
        )

        # This checkpoint declares model_type="siglip".
        # Let AutoModel select the checkpoint-compatible architecture.
        self.model = AutoModel.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        ).to(DEVICE).eval()

        if not hasattr(self.model, "vision_model"):
            raise RuntimeError(
                f"{type(self.model)} has no vision_model"
            )

    @torch.inference_mode()
    def encode(self, image):
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )
        inputs = move_inputs(inputs)

        # Important:
        # We need token-level visual representations for our Adapter,
        # not only a pooled image vector.
        output = self.model.vision_model(
            pixel_values=inputs["pixel_values"],
            return_dict=True,
        )

        return normalize_tokens(
            output.last_hidden_state
        )


class InternVLExtractor:
    def __init__(self, path):
        self.processor = AutoProcessor.from_pretrained(
            path,
            local_files_only=True,
        )

        self.model = InternVLForConditionalGeneration.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        ).to(DEVICE).eval()

    @torch.inference_mode()
    def encode(self, image):
        processor = getattr(self.processor, "image_processor", self.processor)
        inputs = processor(images=image, return_tensors="pt")
        inputs = move_inputs(inputs)

        output = self.model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

        return normalize_tokens(output)


class LLaVAExtractor:
    def __init__(self, path):
        self.processor = AutoProcessor.from_pretrained(
            path,
            local_files_only=True,
        )

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        ).to(DEVICE).eval()

    @torch.inference_mode()
    def encode(self, image):
        processor = getattr(self.processor, "image_processor", self.processor)
        inputs = processor(images=image, return_tensors="pt")

        inputs = move_inputs(inputs)

        kwargs = {
            "pixel_values": inputs["pixel_values"],
        }

        if "image_sizes" in inputs:
            kwargs["image_sizes"] = inputs["image_sizes"]
        else:
            kwargs["image_sizes"] = torch.tensor(
                [[image.height, image.width]],
                device=DEVICE,
            )

        output = self.model.get_image_features(**kwargs)

        return normalize_tokens(output)


EXTRACTORS = {
    "siglip2": SigLIP2Extractor,
    "internvl": InternVLExtractor,
    "llava": LLaVAExtractor,
}


def clear_gpu():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


records = []

with MANIFEST.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

        if len(records) >= WARMUP + RUNS:
            break


results = {}

for name, path in BACKBONES.items():

    print()
    print("=" * 70)
    print(name)
    print(path)
    print("=" * 70)

    extractor = None
    try:
        clear_gpu()
        extractor = EXTRACTORS[name](path)

        with Image.open(records[0]["image"]) as source:
            first_image = source.convert("RGB")
            tokens = extractor.encode(first_image)

        print(
            "Visual tokens:",
            tuple(tokens.shape),
            tokens.dtype,
            tokens.device,
        )

        print(f"Warmup: {WARMUP}")
        for record in records[:WARMUP]:
            with Image.open(record["image"]) as source:
                image = source.convert("RGB")
                _ = extractor.encode(image)
        sync()

        times = []
        shapes = set()
        print(f"Benchmark: {RUNS}")
        for index, record in enumerate(
            records[WARMUP:WARMUP + RUNS],
            start=1,
        ):
            with Image.open(record["image"]) as source:
                image = source.convert("RGB")
                sync()
                t0 = time.perf_counter()
                tokens = extractor.encode(image)
                sync()
                t1 = time.perf_counter()
            ms = (t1 - t0) * 1000.0
            times.append(ms)
            shapes.add(tuple(tokens.shape))
            if index % 10 == 0:
                print(
                    f"{index:3d}/{RUNS}: "
                    f"{ms:.3f} ms  "
                    f"shape={tuple(tokens.shape)}"
                )

        arr = np.asarray(times, dtype=np.float64)
        result = {
            "status": "success",
            "backbone": name,
            "model_path": str(path),
            "token_shape": list(tokens.shape),
            "observed_token_shapes": [list(shape) for shape in sorted(shapes)],
            "hidden_dim": int(tokens.shape[-1]),
            "dtype": str(tokens.dtype),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "fps": float(1000.0 / arr.mean()),
        }
        results[name] = result
        print(json.dumps(result, indent=2))
    except Exception as error:
        traceback.print_exc()
        result = {
            "status": "failed",
            "backbone": name,
            "model_path": str(path),
            "error": f"{type(error).__name__}: {error}",
        }
        results[name] = result
        print(json.dumps(result, indent=2))
    finally:
        if extractor is not None:
            del extractor
        clear_gpu()


output = Path(
    "experiments/results/backbone_generality/preflight.json"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output.write_text(
    json.dumps(results, indent=2) + "\n",
    encoding="utf-8",
)

print()
print("Saved:", output)
