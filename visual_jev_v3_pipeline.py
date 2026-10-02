"""Three-stage Visual-JEV V3 components used by the paper experiments.

This module deliberately leaves :mod:`visual_jev_v2`, :mod:`visual_jev_v3`,
``jev/vl_smoke.py`` and ``vision_test.py`` unchanged.  The Stage-A adapter
learns the frozen Qwen3-VL pre-merger -> post-merger boundary, while the
existing candidate-aware V3 module remains the Stage-B/C decision model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from visual_jev_v3 import VisualJEVV3, VisualJEVV3Config


@dataclass(frozen=True)
class AlignmentAdapterConfig:
    pre_merger_dim: int = 1024
    teacher_dim: int = 2560
    hidden_dim: int = 1024
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.pre_merger_dim, self.teacher_dim, self.hidden_dim) <= 0:
            raise ValueError("alignment dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class PreMergerAlignmentAdapter(nn.Module):
    """Map grouped pre-merger tokens into Qwen's aligned visual space."""

    def __init__(self, config: AlignmentAdapterConfig):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.pre_merger_dim)
        self.projector = nn.Sequential(
            nn.Linear(config.pre_merger_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.teacher_dim),
        )
        self.output_scale = nn.Parameter(torch.ones(config.teacher_dim))
        self.output_bias = nn.Parameter(torch.zeros(config.teacher_dim))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, pre_merger_tokens: torch.Tensor) -> torch.Tensor:
        if pre_merger_tokens.ndim != 2:
            raise ValueError("pre_merger_tokens must have shape [tokens, dim]")
        if pre_merger_tokens.shape[-1] != self.config.pre_merger_dim:
            raise ValueError(
                f"expected pre_merger_dim={self.config.pre_merger_dim}, "
                f"got {pre_merger_tokens.shape[-1]}"
            )
        parameter = next(self.parameters())
        inputs = pre_merger_tokens.to(
            device=parameter.device, dtype=parameter.dtype
        )
        projected = self.projector(self.input_norm(inputs))
        return projected * self.output_scale + self.output_bias


@dataclass
class AlignmentLossOutput:
    total: torch.Tensor
    cosine: torch.Tensor
    mse: torch.Tensor
    norm: torch.Tensor


def alignment_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    cosine_weight: float = 1.0,
    mse_weight: float = 1.0,
    norm_weight: float = 0.1,
) -> AlignmentLossOutput:
    """Token-level cosine/MSE alignment with an explicit feature-norm term."""
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("student and teacher must share [tokens, dim] shape")
    student_f = student.float()
    teacher_f = teacher.to(student.device).float()
    cosine = (1.0 - F.cosine_similarity(student_f, teacher_f, dim=-1)).mean()
    # Scale-normalized MSE keeps its magnitude comparable across checkpoints.
    teacher_variance = teacher_f.square().mean().detach().clamp_min(1e-6)
    mse = F.mse_loss(student_f, teacher_f) / teacher_variance
    student_norm = student_f.norm(dim=-1)
    teacher_norm = teacher_f.norm(dim=-1)
    norm = F.smooth_l1_loss(
        student_norm / teacher_norm.mean().detach().clamp_min(1e-6),
        teacher_norm / teacher_norm.mean().detach().clamp_min(1e-6),
    )
    total = cosine_weight * cosine + mse_weight * mse + norm_weight * norm
    return AlignmentLossOutput(total=total, cosine=cosine, mse=mse, norm=norm)


class ThreeStageVisualJEV(nn.Module):
    """Stage-A alignment adapter followed by the existing V3 decision model."""

    checkpoint_format_version = 1

    def __init__(
        self,
        alignment_config: AlignmentAdapterConfig,
        decision_config: VisualJEVV3Config,
    ):
        super().__init__()
        if alignment_config.teacher_dim != decision_config.vision_dim:
            raise ValueError("alignment teacher_dim must equal decision vision_dim")
        self.alignment = PreMergerAlignmentAdapter(alignment_config)
        self.decision = VisualJEVV3(decision_config)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        pre_merger_tokens: torch.Tensor,
        text_features: torch.Tensor,
    ):
        return self.decision(self.alignment(pre_merger_tokens), text_features)

    def score_aligned(
        self,
        aligned_visual_tokens: torch.Tensor,
        text_features: torch.Tensor,
    ):
        return self.decision(aligned_visual_tokens, text_features)

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        stage: str,
        step: int,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": self.checkpoint_format_version,
                "model_type": "visual-jev-v3-three-stage",
                "alignment_config": asdict(self.alignment.config),
                "decision_config": asdict(self.decision.config),
                "state_dict": self.state_dict(),
                "stage": stage,
                "step": step,
                "metadata": metadata or {},
            },
            path,
        )
        return path

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["ThreeStageVisualJEV", dict[str, Any]]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("model_type") != "visual-jev-v3-three-stage":
            raise ValueError("checkpoint is not a three-stage Visual-JEV V3")
        if payload.get("format_version") != cls.checkpoint_format_version:
            raise ValueError("unsupported three-stage checkpoint format")
        model = cls(
            AlignmentAdapterConfig(**payload["alignment_config"]),
            VisualJEVV3Config(**payload["decision_config"]),
        )
        model.load_state_dict(payload["state_dict"])
        return model, payload


class MeanPoolJEV(nn.Module):
    """Mean-pool + MLP + shared scalar-head paper baseline."""

    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.visual = nn.Sequential(
            nn.LayerNorm(vision_dim), nn.Linear(vision_dim, hidden_dim), nn.GELU()
        )
        self.text = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.scalar_head = nn.Linear(hidden_dim, 1)

    def forward(self, visual_tokens: torch.Tensor, text_features: torch.Tensor):
        if visual_tokens.ndim != 2 or text_features.ndim != 2:
            raise ValueError("expected [tokens, dim] visual and [candidates, dim] text")
        parameter = next(self.parameters())
        visual_tokens = visual_tokens.to(parameter.device, parameter.dtype)
        text_features = text_features.to(parameter.device, parameter.dtype)
        v = self.visual(visual_tokens.mean(0, keepdim=True))
        t = self.text(text_features)
        v = v.expand(t.shape[0], -1)
        fused = torch.cat((v, t, v * t, (v - t).abs()), dim=-1)
        return self.scalar_head(self.fusion(fused)).squeeze(-1)


def grouped_pre_merger_tokens(
    pre_merger_tokens: torch.Tensor, post_token_count: int
) -> torch.Tensor:
    """Average consecutive native spatial groups to the merger token count."""
    if pre_merger_tokens.ndim != 2:
        raise ValueError("pre-merger tokens must have shape [tokens, dim]")
    if post_token_count <= 0 or pre_merger_tokens.shape[0] % post_token_count:
        raise ValueError("pre/post token counts are not integer aligned")
    group = pre_merger_tokens.shape[0] // post_token_count
    return pre_merger_tokens.reshape(post_token_count, group, -1).mean(dim=1)
