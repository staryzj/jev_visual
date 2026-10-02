"""Utilities for variable-K and multi-domain Visual-JEV training.

The optimisation routines in this module are project-local reimplementations
inspired by PCGrad, GradNorm, and DoReMi/Group-DRO.  They are deliberately not
presented as the authors' official implementations.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


def permute_candidates(
    candidates: Sequence[str], label: int, order: Sequence[int]
) -> tuple[tuple[str, ...], int]:
    """Apply a candidate permutation and remap the gold label."""
    order = tuple(int(index) for index in order)
    if sorted(order) != list(range(len(candidates))):
        raise ValueError("order must be a permutation of every candidate index")
    if not 0 <= label < len(candidates):
        raise ValueError("label is outside the candidate range")
    return tuple(candidates[index] for index in order), order.index(label)


def pad_variable_candidates(
    rows: Sequence[torch.Tensor], labels: Sequence[int], *, pad_value: float = -torch.inf
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable candidate logits and return logits, validity mask, labels."""
    if not rows or len(rows) != len(labels):
        raise ValueError("rows and labels must be non-empty and have equal length")
    device = rows[0].device
    dtype = rows[0].dtype
    width = max(int(row.numel()) for row in rows)
    padded = torch.full((len(rows), width), pad_value, device=device, dtype=dtype)
    valid = torch.zeros((len(rows), width), device=device, dtype=torch.bool)
    target = torch.as_tensor(labels, device=device, dtype=torch.long)
    for index, (row, label) in enumerate(zip(rows, labels)):
        if row.ndim != 1 or row.device != device:
            raise ValueError("each row must be a one-dimensional tensor on one device")
        if not 0 <= int(label) < row.numel():
            raise ValueError("label is outside a row's valid candidates")
        padded[index, : row.numel()] = row
        valid[index, : row.numel()] = True
    return padded, valid, target


def masked_log_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Log-softmax over valid candidates only; padding receives -inf."""
    if logits.shape != valid.shape or valid.dtype is not torch.bool:
        raise ValueError("logits and boolean valid mask must have the same shape")
    if logits.ndim != 2 or not bool(valid.any(dim=-1).all()):
        raise ValueError("every row must contain at least one valid candidate")
    return F.log_softmax(logits.masked_fill(~valid, -torch.inf), dim=-1)


def masked_cross_entropy(
    logits: torch.Tensor, valid: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    log_probability = masked_log_softmax(logits, valid)
    if bool((~valid.gather(1, labels[:, None])).any()):
        raise ValueError("a label points to padding")
    return -log_probability.gather(1, labels[:, None]).mean()


def uniform_targets(valid: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Per-row uniform target over valid candidates (never over padding)."""
    counts = valid.sum(dim=-1, keepdim=True).clamp_min(1)
    return valid.to(dtype=dtype) / counts


def masked_probabilities(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return masked_log_softmax(logits, valid).exp().masked_fill(~valid, 0.0)


def normalized_entropy(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    probability = masked_probabilities(logits, valid)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    counts = valid.sum(dim=-1).to(logits.dtype)
    denominator = counts.log().clamp_min(torch.finfo(logits.dtype).eps)
    return torch.where(counts > 1, entropy / denominator, torch.zeros_like(entropy))


def masked_brier(
    logits: torch.Tensor, valid: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    probability = masked_probabilities(logits, valid)
    target = torch.zeros_like(probability).scatter_(1, labels[:, None], 1.0)
    return ((probability - target).square() * valid).sum(dim=-1).mean()


def masked_kl_to_uniform(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    probability = masked_probabilities(logits, valid)
    target = uniform_targets(valid, dtype=logits.dtype)
    return (
        probability
        * (probability.clamp_min(1e-12).log() - target.clamp_min(1e-12).log())
        * valid
    ).sum(dim=-1).mean()


def _trainable(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return [parameter for parameter in parameters if parameter.requires_grad]


def task_gradients(
    losses: Sequence[torch.Tensor], parameters: Iterable[torch.nn.Parameter]
) -> tuple[list[list[torch.Tensor]], list[float], list[list[float]]]:
    """Return per-task gradients, norms and pairwise cosine similarities."""
    parameters = _trainable(parameters)
    task_grads: list[list[torch.Tensor]] = []
    flat: list[torch.Tensor] = []
    for loss in losses:
        raw = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        grads = [torch.zeros_like(p) if g is None else g.detach().clone() for p, g in zip(parameters, raw)]
        task_grads.append(grads)
        flat.append(torch.cat([gradient.reshape(-1).float() for gradient in grads]))
    norms = [float(vector.norm().item()) for vector in flat]
    cosine: list[list[float]] = []
    for left in flat:
        row = []
        for right in flat:
            denominator = left.norm() * right.norm()
            row.append(float((left.dot(right) / denominator.clamp_min(1e-12)).item()))
        cosine.append(row)
    return task_grads, norms, cosine


def _assign_average_gradient(
    parameters: Sequence[torch.nn.Parameter], task_grads: Sequence[Sequence[torch.Tensor]]
) -> None:
    for parameter_index, parameter in enumerate(parameters):
        parameter.grad = torch.stack(
            [grads[parameter_index].to(parameter.dtype) for grads in task_grads]
        ).mean(dim=0)


def pcgrad_backward(
    losses: Sequence[torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
    *,
    rng: random.Random,
) -> dict[str, object]:
    """Project conflicting task gradients, following the PCGrad update rule."""
    parameters = _trainable(parameters)
    task_grads, norms, cosine = task_gradients(losses, parameters)
    projected = [[gradient.clone() for gradient in gradients] for gradients in task_grads]
    task_order = list(range(len(projected)))
    for task_index in range(len(projected)):
        other_order = [index for index in task_order if index != task_index]
        rng.shuffle(other_order)
        for other_index in other_order:
            dot = sum((left * right).sum() for left, right in zip(projected[task_index], task_grads[other_index]))
            if float(dot.item()) < 0.0:
                denominator = sum(right.square().sum() for right in task_grads[other_index]).clamp_min(1e-12)
                coefficient = dot / denominator
                projected[task_index] = [
                    left - coefficient * right
                    for left, right in zip(projected[task_index], task_grads[other_index])
                ]
    _assign_average_gradient(parameters, projected)
    return {"norms": norms, "cosine": cosine}


def gradnorm_backward(
    losses: Sequence[torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
    *,
    initial_losses: Sequence[float],
    alpha: float = 1.5,
) -> dict[str, object]:
    """Project-local GradNorm-inspired gradient magnitude balancing.

    This uses GradNorm's relative inverse training-rate target but applies the
    resulting scale directly to task gradients instead of learning separate
    scalar loss-weight parameters.
    """
    parameters = _trainable(parameters)
    task_grads, norms, cosine = task_gradients(losses, parameters)
    current = torch.tensor([float(loss.detach().item()) for loss in losses])
    initial = torch.tensor(initial_losses).clamp_min(1e-12)
    rate = current / initial
    inverse_rate = rate / rate.mean().clamp_min(1e-12)
    norm_tensor = torch.tensor(norms).clamp_min(1e-12)
    target = norm_tensor.mean() * inverse_rate.pow(alpha)
    scales = (target / norm_tensor).clamp(0.1, 10.0)
    scaled = [
        [gradient * float(scales[index].item()) for gradient in gradients]
        for index, gradients in enumerate(task_grads)
    ]
    _assign_average_gradient(parameters, scaled)
    return {"norms": norms, "cosine": cosine, "scales": scales.tolist()}


@dataclass
class AdaptiveDomainWeights:
    """Exponentiated Group-DRO/DoReMi-inspired proxy-domain weights."""

    domains: tuple[str, ...]
    eta: float = 0.5
    floor: float = 0.05

    def __post_init__(self) -> None:
        self._weights = torch.ones(len(self.domains), dtype=torch.float64) / len(self.domains)

    def update(self, excess_losses: Mapping[str, float]) -> dict[str, float]:
        excess = torch.tensor([float(excess_losses[name]) for name in self.domains], dtype=torch.float64)
        excess = excess - excess.mean()
        self._weights *= torch.exp(self.eta * excess)
        self._weights = self._weights.clamp_min(self.floor)
        self._weights /= self._weights.sum()
        return self.as_dict()

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.domains, self._weights)}


def visual_dependency_masks(
    dataset: str,
    *,
    scienceqa_visual_required: bool = False,
    semantic_counterfactual_available: bool = False,
    local_region_annotation_available: bool = False,
) -> dict[str, object]:
    """Return Stage-C gates without treating arbitrary wrong images as negatives.

    ``preference`` is enabled only when a sample has an explicit semantic pair
    or a verified local-region perturbation.  This follows the data discipline
    suggested by MFPO/MMedPO while remaining a project-local adaptation: no
    synthetic local annotation is inferred from dataset membership.
    """
    verified_preference = bool(
        semantic_counterfactual_available or local_region_annotation_available
    )
    if dataset == "coco_hard_negative":
        kind = "strong_visual_counterfactual"
        return {
            "type": kind,
            "invalid": True,
            "counterfactual": True,
            "preference": verified_preference,
            "preference_source": (
                "semantic_pair" if semantic_counterfactual_available
                else "local_region" if local_region_annotation_available
                else "none"
            ),
        }
    if dataset in {"aokvqa", "iconqa_choice"}:
        kind = "visual_required"
        return {
            "type": kind,
            "invalid": True,
            "counterfactual": False,
            "preference": verified_preference,
            "preference_source": (
                "semantic_pair" if semantic_counterfactual_available
                else "local_region" if local_region_annotation_available
                else "none"
            ),
        }
    if dataset == "scienceqa_image_only" and scienceqa_visual_required:
        kind = "visual_required"
        return {
            "type": kind,
            "invalid": True,
            "counterfactual": False,
            "preference": verified_preference,
            "preference_source": (
                "semantic_pair" if semantic_counterfactual_available
                else "local_region" if local_region_annotation_available
                else "none"
            ),
        }
    if dataset == "scienceqa_image_only":
        kind = "mixed_or_text_solvable"
        return {
            "type": kind, "invalid": False, "counterfactual": False,
            "preference": False, "preference_source": "none",
        }
    return {
        "type": "unknown", "invalid": False, "counterfactual": False,
        "preference": False, "preference_source": "none",
    }


def preference_anchor_loss(
    matched_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
    *,
    label: int,
    margin: float = 0.2,
    anchor: float = 0.0,
) -> torch.Tensor:
    """mDPO-inspired image preference plus a positive-score anchor."""
    positive = matched_scores[label]
    rejected = rejected_scores[label]
    preference = F.relu(margin - positive + rejected)
    positive_anchor = F.softplus(anchor - positive)
    return preference + positive_anchor

