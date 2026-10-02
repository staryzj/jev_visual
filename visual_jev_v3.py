"""Visual-JEV V3: candidate-aware scoring with explicit modality dependence.

V3 intentionally keeps the V2 adapter parameterization so an existing V2
checkpoint can initialize it.  The behavioural change is supplied by losses
that make a decisive distribution valid only when both modalities agree:
counterfactual image/candidate flips, invalid-image uniformity, text-null
uniformity, and cross-image ranking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from visual_jev_v2 import VisualJEVV2, VisualJEVV2Config, VisualJEVV2Output


@dataclass(frozen=True)
class VisualJEVV3Config(VisualJEVV2Config):
    """V3 uses the V2 candidate-aware adaptive adapter unchanged."""


VisualJEVV3Output = VisualJEVV2Output


class VisualJEVV3(VisualJEVV2):
    """V2-compatible candidate-aware adapter trained with V3 objectives."""

    checkpoint_format_version = 1

    def __init__(self, config: VisualJEVV3Config):
        super().__init__(config)
        self.config = config

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format_version": self.checkpoint_format_version,
            "model_type": "visual-jev-v3",
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "step": step,
            "metadata": metadata or {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, path)
        return path

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        allow_v2: bool = True,
    ) -> tuple[VisualJEVV3, dict[str, Any]]:
        """Load V3, or migrate a V2 checkpoint without altering the source."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model_type = payload.get("model_type")
        accepted = {"visual-jev-v3"}
        if allow_v2:
            accepted.add("visual-jev-v2")
        if model_type not in accepted:
            raise ValueError(f"unsupported checkpoint model_type: {model_type!r}")
        if payload.get("format_version") != cls.checkpoint_format_version:
            raise ValueError(
                f"unsupported checkpoint format {payload.get('format_version')}"
            )
        model = cls(VisualJEVV3Config(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=strict)
        return model, payload


def _as_score_batch(scores: torch.Tensor, name: str) -> torch.Tensor:
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError(f"{name} must have shape [batch, candidates>=2]")
    if not torch.isfinite(scores).all():
        raise ValueError(f"{name} contains non-finite values")
    return scores


def uniformity_loss(
    scores: torch.Tensor,
    *,
    divergence: Literal["kl", "js"] = "kl",
) -> torch.Tensor:
    """Penalize departure of a candidate softmax from the uniform prior.

    ``kl`` computes KL(p || U), which is ``log(K) - H(p)``.  ``js`` is the
    symmetric, bounded Jensen-Shannon divergence between p and U.
    """
    scores = _as_score_batch(scores, "scores")
    log_p = F.log_softmax(scores.float(), dim=-1)
    p = log_p.exp()
    candidate_count = scores.shape[-1]
    log_u = -torch.log(
        torch.tensor(float(candidate_count), device=scores.device)
    )
    if divergence == "kl":
        return (p * (log_p - log_u)).sum(dim=-1).mean()
    if divergence == "js":
        u = torch.full_like(p, 1.0 / candidate_count)
        m = 0.5 * (p + u)
        kl_pm = (p * (log_p - m.log())).sum(dim=-1)
        kl_um = (u * (log_u - m.log())).sum(dim=-1)
        return (0.5 * (kl_pm + kl_um)).mean()
    raise ValueError("divergence must be 'kl' or 'js'")


def _target_tensor(
    target: int | torch.Tensor, scores: torch.Tensor, name: str
) -> torch.Tensor:
    if isinstance(target, int):
        targets = torch.full(
            (scores.shape[0],), target, device=scores.device, dtype=torch.long
        )
    else:
        targets = target.to(device=scores.device, dtype=torch.long)
        if targets.ndim == 0:
            targets = targets.expand(scores.shape[0])
    if targets.shape != (scores.shape[0],):
        raise ValueError(f"{name} must contain one index per batch item")
    if ((targets < 0) | (targets >= scores.shape[1])).any():
        raise ValueError(f"{name} contains an out-of-range candidate index")
    return targets


def counterfactual_visual_dependency_loss(
    image1_scores: torch.Tensor,
    image2_scores: torch.Tensor,
    *,
    image1_target: int | torch.Tensor = 0,
    image2_target: int | torch.Tensor = 1,
    loss_type: Literal["cross_entropy", "ranking"] = "cross_entropy",
    margin: float = 0.2,
) -> torch.Tensor:
    """Require the winner for one fixed candidate set to flip across images."""
    image1_scores = _as_score_batch(image1_scores, "image1_scores")
    image2_scores = _as_score_batch(image2_scores, "image2_scores")
    if image1_scores.shape != image2_scores.shape:
        raise ValueError("counterfactual score tensors must have identical shapes")
    target1 = _target_tensor(image1_target, image1_scores, "image1_target")
    target2 = _target_tensor(image2_target, image2_scores, "image2_target")
    if torch.equal(target1, target2):
        raise ValueError("counterfactual targets must flip between the two images")
    if loss_type == "cross_entropy":
        return 0.5 * (
            F.cross_entropy(image1_scores, target1)
            + F.cross_entropy(image2_scores, target2)
        )
    if loss_type == "ranking":
        losses = []
        for scores, targets in ((image1_scores, target1), (image2_scores, target2)):
            positive = scores.gather(1, targets[:, None])
            mask = F.one_hot(targets, scores.shape[1]).bool()
            hardest_other = scores.masked_fill(mask, float("-inf")).max(1).values
            losses.append(F.relu(margin - positive.squeeze(1) + hardest_other).mean())
        return 0.5 * (losses[0] + losses[1])
    raise ValueError("loss_type must be 'cross_entropy' or 'ranking'")


def cross_image_ranking_loss(
    matched_scores: torch.Tensor,
    mismatched_scores: torch.Tensor,
    *,
    positive_index: int = 0,
    margin: float = 0.2,
) -> torch.Tensor:
    """Rank a candidate higher with its matched image than with a wrong image."""
    matched_scores = _as_score_batch(matched_scores, "matched_scores")
    mismatched_scores = _as_score_batch(mismatched_scores, "mismatched_scores")
    if matched_scores.shape != mismatched_scores.shape:
        raise ValueError("matched and mismatched scores must have identical shapes")
    if not 0 <= positive_index < matched_scores.shape[1]:
        raise ValueError("positive_index is out of range")
    return F.relu(
        margin
        - matched_scores[:, positive_index]
        + mismatched_scores[:, positive_index]
    ).mean()


@dataclass(frozen=True)
class VisualJEVV3LossWeights:
    counterfactual: float = 1.0
    invalid_image: float = 0.5
    text_null: float = 0.5
    cross_image_rank: float = 0.5

    def __post_init__(self) -> None:
        if min(
            self.counterfactual,
            self.invalid_image,
            self.text_null,
            self.cross_image_rank,
        ) < 0:
            raise ValueError("loss weights must be non-negative")


@dataclass
class VisualJEVV3LossOutput:
    total: torch.Tensor
    candidate: torch.Tensor
    counterfactual: torch.Tensor
    invalid_image: torch.Tensor
    text_null: torch.Tensor
    cross_image_rank: torch.Tensor


def combine_v3_losses(
    *,
    candidate: torch.Tensor,
    counterfactual: torch.Tensor,
    invalid_image: torch.Tensor,
    text_null: torch.Tensor,
    cross_image_rank: torch.Tensor,
    weights: VisualJEVV3LossWeights,
) -> VisualJEVV3LossOutput:
    total = (
        candidate
        + weights.counterfactual * counterfactual
        + weights.invalid_image * invalid_image
        + weights.text_null * text_null
        + weights.cross_image_rank * cross_image_rank
    )
    return VisualJEVV3LossOutput(
        total=total,
        candidate=candidate,
        counterfactual=counterfactual,
        invalid_image=invalid_image,
        text_null=text_null,
        cross_image_rank=cross_image_rank,
    )


def text_null_features(
    text_features: torch.Tensor,
    *,
    mode: Literal["zero", "mean"] = "zero",
) -> torch.Tensor:
    """Remove candidate identity while preserving the candidate tensor shape."""
    if text_features.ndim != 2 or text_features.shape[0] < 2:
        raise ValueError("text_features must have shape [candidates>=2, dim]")
    if mode == "zero":
        return torch.zeros_like(text_features)
    if mode == "mean":
        return text_features.mean(dim=0, keepdim=True).expand_as(text_features)
    raise ValueError("mode must be 'zero' or 'mean'")


def mean_uniformity_loss(
    score_sets: Sequence[torch.Tensor], *, divergence: Literal["kl", "js"] = "kl"
) -> torch.Tensor:
    if not score_sets:
        raise ValueError("score_sets must not be empty")
    return torch.stack(
        [uniformity_loss(scores, divergence=divergence) for scores in score_sets]
    ).mean()
