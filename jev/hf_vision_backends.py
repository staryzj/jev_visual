"""Frozen Hugging Face vision backbones behind one token-level interface."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    InternVLForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
)


BACKBONE_NAMES = ("siglip2", "internvl", "llava")


def _move_inputs(inputs: Any, device: torch.device, dtype: torch.dtype) -> dict:
    result = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            result[key] = value.to(
                device=device,
                dtype=dtype if value.is_floating_point() else value.dtype,
            )
        else:
            result[key] = value
    return result


def normalize_visual_tokens(output: Any) -> torch.Tensor:
    """Normalize a one-image feature output to ``[tokens, hidden_dim]``."""
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    elif hasattr(output, "image_embeds"):
        output = output.image_embeds
    if isinstance(output, (tuple, list)):
        tensors = []
        for item in output:
            if hasattr(item, "last_hidden_state"):
                item = item.last_hidden_state
            if torch.is_tensor(item):
                tensors.append(item)
        if not tensors:
            raise RuntimeError("visual feature output contains no tensor")
        output = tensors[0]
    if not torch.is_tensor(output):
        raise RuntimeError(f"unsupported visual feature output: {type(output)}")
    if output.ndim == 4:
        if output.shape[0] != 1:
            raise RuntimeError(f"expected one image, got shape {tuple(output.shape)}")
        output = output[0].reshape(-1, output.shape[-1])
    elif output.ndim == 3:
        output = output[0] if output.shape[0] == 1 else output.reshape(-1, output.shape[-1])
    elif output.ndim == 1:
        output = output.unsqueeze(0)
    elif output.ndim != 2:
        raise RuntimeError(f"unsupported visual feature shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise RuntimeError("visual features contain non-finite values")
    return output.contiguous()


def resample_visual_tokens(tokens: torch.Tensor, token_count: int) -> torch.Tensor:
    """Deterministically map a variable token sequence to a shared token count."""
    if tokens.ndim != 2 or token_count < 1:
        raise ValueError("expected [tokens, dim] and a positive target count")
    if tokens.shape[0] == token_count:
        return tokens
    pooled = F.adaptive_avg_pool1d(
        tokens.transpose(0, 1).unsqueeze(0).float(), token_count
    )
    return pooled.squeeze(0).transpose(0, 1).to(tokens.dtype).contiguous()


class HFVisionTokenExtractor:
    """Load one frozen backbone and expose native or resampled visual tokens."""

    def __init__(
        self,
        name: str,
        model_path: str | Path,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        if name not in BACKBONE_NAMES:
            raise ValueError(f"unknown backbone {name!r}; choose from {BACKBONE_NAMES}")
        self.name = name
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = torch.device(device)
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=name == "internvl",
        )
        common = dict(
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if name == "siglip2":
            self.model = AutoModel.from_pretrained(self.model_path, **common)
            if not hasattr(self.model, "vision_model"):
                raise RuntimeError("SigLIP checkpoint has no vision_model")
        elif name == "internvl":
            self.model = InternVLForConditionalGeneration.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                **common,
            )
        else:
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_path,
                **common,
            )
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, image: Image.Image, token_count: int | None = None) -> torch.Tensor:
        if self.name == "siglip2":
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = _move_inputs(inputs, self.device, self.dtype)
            output = self.model.vision_model(
                pixel_values=inputs["pixel_values"], return_dict=True
            ).last_hidden_state
        else:
            image_processor = getattr(self.processor, "image_processor", self.processor)
            inputs = image_processor(images=image, return_tensors="pt")
            inputs = _move_inputs(inputs, self.device, self.dtype)
            kwargs = {"pixel_values": inputs["pixel_values"]}
            if self.name == "llava":
                kwargs["image_sizes"] = inputs.get(
                    "image_sizes",
                    torch.tensor(
                        [[image.height, image.width]],
                        device=self.device,
                        dtype=torch.long,
                    ),
                )
            output = self.model.get_image_features(**kwargs)
        tokens = normalize_visual_tokens(output)
        return (
            resample_visual_tokens(tokens, token_count)
            if token_count is not None
            else tokens
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def close(self) -> None:
        self.model.to("cpu")
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

