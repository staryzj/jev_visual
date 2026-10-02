"""Recent-method helpers for Visual-JEV multi-domain experiments.

These routines are small, auditable project-local adaptations inspired by
Cambrian-1 (data-source balance), PiKE (positive-interaction task weighting),
and MERIT (dataset-gradient PCA, branching and merge).  They are not official
implementations of those papers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch


def normalise_weights(
    weights: Mapping[str, float],
    domains: Sequence[str],
    *,
    floor: float = 0.0,
) -> dict[str, float]:
    """Validate and normalise a domain distribution with an optional floor."""
    values = torch.tensor([float(weights[name]) for name in domains], dtype=torch.float64)
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("domain weights must be finite and non-negative")
    if float(values.sum()) <= 0:
        raise ValueError("at least one domain weight must be positive")
    values /= values.sum()
    if floor:
        if floor * len(domains) >= 1.0:
            raise ValueError("floor is too large for the number of domains")
        values = values.clamp_min(float(floor))
        values /= values.sum()
    return {name: float(value) for name, value in zip(domains, values)}


def capped_balanced_weights(
    counts: Mapping[str, int], domains: Sequence[str], *, cap: int
) -> dict[str, float]:
    """Cambrian-style source cap before conversion to a sampling ratio."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    return normalise_weights(
        {name: min(int(counts[name]), cap) for name in domains}, domains
    )


def _standardise(values: torch.Tensor) -> torch.Tensor:
    centred = values - values.mean()
    scale = centred.std(unbiased=False)
    return centred / scale.clamp_min(1e-12)


@dataclass
class PiKEInspiredWeights:
    """Lightweight positive-interaction domain weight estimator.

    The update favours domains that remain difficult, improve slowly over a
    short window, and have non-negative interactions with the other domains.
    This deliberately uses only observable losses and gradient cosines rather
    than claiming to reproduce the authors' complete optimizer.
    """

    domains: tuple[str, ...]
    eta: float = 0.30
    floor: float = 0.05
    loss_ema: float = 0.6
    _weights: torch.Tensor = field(init=False, repr=False)
    _losses: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.domains:
            raise ValueError("domains must be non-empty")
        self._weights = torch.ones(len(self.domains), dtype=torch.float64)
        self._weights /= self._weights.sum()

    def update(
        self,
        losses: Mapping[str, float],
        cosine: Sequence[Sequence[float]],
    ) -> dict[str, object]:
        current = torch.tensor(
            [float(losses[name]) for name in self.domains], dtype=torch.float64
        )
        matrix = torch.as_tensor(cosine, dtype=torch.float64)
        expected = (len(self.domains), len(self.domains))
        if tuple(matrix.shape) != expected:
            raise ValueError(f"cosine matrix must have shape {expected}")
        if not bool(torch.isfinite(current).all()) or not bool(torch.isfinite(matrix).all()):
            raise ValueError("losses and cosine values must be finite")

        previous = current if self._losses is None else self._losses
        decrease = previous - current
        self._losses = (
            current if self._losses is None
            else self.loss_ema * self._losses + (1.0 - self.loss_ema) * current
        )
        off_diagonal = ~torch.eye(len(self.domains), dtype=torch.bool)
        positive = matrix.clamp_min(0.0)
        positive_share = positive.masked_select(off_diagonal).reshape(
            len(self.domains), len(self.domains) - 1
        ).mean(dim=1)
        difficulty = _standardise(current)
        stagnation = _standardise(-decrease)
        compatibility = _standardise(positive_share)
        utility = 0.50 * difficulty + 0.35 * stagnation + 0.15 * compatibility
        self._weights *= torch.exp(self.eta * utility).clamp(0.5, 2.0)
        self._weights = self._weights.clamp_min(self.floor)
        self._weights /= self._weights.sum()
        return {
            "weights": self.as_dict(),
            "loss": {name: float(value) for name, value in zip(self.domains, current)},
            "loss_decrease": {
                name: float(value) for name, value in zip(self.domains, decrease)
            },
            "positive_interaction": {
                name: float(value) for name, value in zip(self.domains, positive_share)
            },
            "utility": {name: float(value) for name, value in zip(self.domains, utility)},
        }

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(value) for name, value in zip(self.domains, self._weights)
        }


def gradient_pca_diagnostic(
    mean_gradients: Mapping[str, torch.Tensor],
    *,
    explained_threshold: float = 0.60,
    cross_cosine_threshold: float = -0.05,
) -> dict[str, object]:
    """Run PCA on dataset-level mean gradients and form a two-way split."""
    domains = tuple(mean_gradients)
    if len(domains) < 2:
        raise ValueError("at least two domain gradients are required")
    matrix = torch.stack(
        [mean_gradients[name].detach().double().flatten().cpu() for name in domains]
    )
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("mean gradients must be finite")
    norms = matrix.norm(dim=1)
    cosine = matrix @ matrix.T / (norms[:, None] * norms[None, :]).clamp_min(1e-12)
    centred = matrix - matrix.mean(dim=0, keepdim=True)
    u, singular, _ = torch.linalg.svd(centred, full_matrices=False)
    variance = singular.square()
    explained = variance / variance.sum().clamp_min(1e-12)
    scores = u[:, 0] * singular[0] if singular.numel() else torch.zeros(len(domains))
    median = scores.median()
    first = [name for name, value in zip(domains, scores) if float(value) >= float(median)]
    second = [name for name in domains if name not in first]
    if not first or not second:
        order = sorted(range(len(domains)), key=lambda index: float(scores[index]))
        midpoint = max(1, len(order) // 2)
        second = [domains[index] for index in order[:midpoint]]
        first = [domains[index] for index in order[midpoint:]]
    index = {name: position for position, name in enumerate(domains)}
    cross = [
        float(cosine[index[left], index[right]]) for left in first for right in second
    ]
    pair_values = [
        float(cosine[left, right])
        for left in range(len(domains)) for right in range(left + 1, len(domains))
    ]
    top_explained = float(explained[0]) if explained.numel() else 0.0
    cross_mean = float(sum(cross) / len(cross))
    structured = bool(
        top_explained >= explained_threshold and cross_mean <= cross_cosine_threshold
    )
    return {
        "domains": list(domains),
        "cosine_matrix": cosine.tolist(),
        "gradient_norms": {name: float(value) for name, value in zip(domains, norms)},
        "pca_explained_variance_ratio": [float(value) for value in explained],
        "pc1_scores": {name: float(value) for name, value in zip(domains, scores)},
        "clusters": [first, second],
        "cross_cluster_mean_cosine": cross_mean,
        "pair_conflict_fraction": sum(value < 0.0 for value in pair_values) / len(pair_values),
        "structured_conflict": structured,
        "decision_rule": {
            "top_pca_explained_at_least": explained_threshold,
            "cross_cluster_mean_cosine_at_most": cross_cosine_threshold,
        },
    }


def weighted_state_dict_average(
    states: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[float]
) -> dict[str, torch.Tensor]:
    """Average merge-ready trainable state dictionaries."""
    if not states or len(states) != len(weights):
        raise ValueError("states and weights must be non-empty and aligned")
    normalised = torch.as_tensor(weights, dtype=torch.float64)
    if bool((normalised < 0).any()) or float(normalised.sum()) <= 0:
        raise ValueError("merge weights must be non-negative with a positive sum")
    normalised /= normalised.sum()
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise ValueError("all state dictionaries must have identical ordered keys")
    merged: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [state[key] for state in states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"shape mismatch for {key}")
        if tensors[0].is_floating_point():
            accumulator = torch.zeros_like(tensors[0], dtype=torch.float64)
            for weight, tensor in zip(normalised, tensors):
                accumulator.add_(tensor.detach().cpu().double(), alpha=float(weight))
            merged[key] = accumulator.to(dtype=tensors[0].dtype)
        else:
            merged[key] = tensors[0].detach().cpu().clone()
    return merged
