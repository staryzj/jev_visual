"""Frozen, swappable vision encoders for backend-agnostic Visual-JEV training.

The contract deliberately exposes only a sequence of visual tokens.  Training
code must not depend on a backend's processor, embedding width, grid size, or
internal module names after construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch import nn


@dataclass(frozen=True)
class VisionBackendSpec:
    name: str
    family: str
    feature_dim: int
    token_layout: str
    weight_id: str
    weight_url: str


BACKEND_NAMES = ("torchvision_vit_b16", "torchvision_convnext_tiny")


class FrozenVisionBackend:
    """Wrap a frozen encoder behind a common ``[batch, tokens, dim]`` API."""

    def __init__(self, name: str, device: str | torch.device):
        if name not in BACKEND_NAMES:
            raise ValueError(f"unknown vision backend {name!r}; choose from {BACKEND_NAMES}")
        from torchvision.models import (
            ConvNeXt_Tiny_Weights,
            ViT_B_16_Weights,
            convnext_tiny,
            vit_b_16,
        )

        self.name = name
        self.device = torch.device(device)
        if name == "torchvision_vit_b16":
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.model = vit_b_16(weights=weights)
            self.spec = VisionBackendSpec(
                name=name,
                family="vision_transformer_patch_tokens",
                feature_dim=768,
                token_layout="14x14_patch_grid_without_cls",
                weight_id="ViT_B_16_Weights.IMAGENET1K_V1",
                weight_url=weights.url,
            )
        else:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            self.model = convnext_tiny(weights=weights)
            self.spec = VisionBackendSpec(
                name=name,
                family="convolutional_feature_grid",
                feature_dim=768,
                token_layout="7x7_final_feature_grid",
                weight_id="ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
                weight_url=weights.url,
            )
        self.transform = weights.transforms(antialias=True)
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def metadata(self) -> dict[str, object]:
        import torchvision

        return {
            **asdict(self.spec),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "frozen": True,
            "transform": repr(self.transform),
        }

    @torch.inference_mode()
    def encode(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("images must not be empty")
        batch = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(
            self.device
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast:
            if self.name == "torchvision_vit_b16":
                tokens = self.model._process_input(batch)
                cls = self.model.class_token.expand(tokens.shape[0], -1, -1)
                tokens = self.model.encoder(torch.cat((cls, tokens), dim=1))[:, 1:]
            else:
                feature_map = self.model.features(batch)
                tokens = feature_map.flatten(2).transpose(1, 2)
        if tokens.ndim != 3 or tokens.shape[-1] != self.spec.feature_dim:
            raise RuntimeError(
                f"{self.name} violated the token contract: {tuple(tokens.shape)}"
            )
        if not torch.isfinite(tokens).all():
            raise RuntimeError(f"{self.name} produced non-finite features")
        return tokens.float().cpu()


class TextOnlyCandidateScorer(nn.Module):
    """Shared candidate scorer used as the visual-information control."""

    def __init__(self, text_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        if text_features.ndim != 3:
            raise ValueError("text_features must have shape [batch, candidates, dim]")
        batch, candidates, width = text_features.shape
        return self.network(text_features.reshape(batch * candidates, width)).reshape(
            batch, candidates
        )


def batched_candidate_scores(
    model: nn.Module,
    visual_tokens: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """Score fixed-K batches without making the model backend-aware."""
    if visual_tokens.ndim != 3 or text_features.ndim != 3:
        raise ValueError("expected visual [B,N,D] and text [B,K,E]")
    if visual_tokens.shape[0] != text_features.shape[0]:
        raise ValueError("visual and text batch sizes differ")
    batch, candidates, text_dim = text_features.shape
    token_count, vision_dim = visual_tokens.shape[1:]
    expanded = visual_tokens[:, None].expand(-1, candidates, -1, -1)
    output = model(
        expanded.reshape(batch * candidates, token_count, vision_dim),
        text_features.reshape(batch * candidates, text_dim),
    )
    return output.scores.reshape(batch, candidates)


def cached_weight_sha256(weight_url: str) -> str | None:
    """Return the exact downloaded torchvision weight hash when available."""
    import hashlib
    from urllib.parse import urlparse

    filename = Path(urlparse(weight_url).path).name
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
