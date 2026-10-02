"""Post-hoc probability calibration for variable-choice JEV logits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _validate_batches(
    logits: torch.Tensor | Sequence[torch.Tensor], labels: torch.Tensor | Sequence[int]
) -> tuple[list[torch.Tensor], torch.Tensor]:
    batches = [row for row in logits] if isinstance(logits, torch.Tensor) else list(logits)
    target = labels.detach().long().cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
    if not batches or len(batches) != len(target):
        raise ValueError("logits and labels must be non-empty and have equal length")
    normalised = []
    for index, (row, label) in enumerate(zip(batches, target.tolist())):
        row = torch.as_tensor(row, dtype=torch.float64).flatten()
        if row.numel() < 2 or not torch.isfinite(row).all():
            raise ValueError(f"invalid logits at row {index}")
        if not 0 <= label < row.numel():
            raise ValueError(f"label outside candidate range at row {index}")
        normalised.append(row)
    return normalised, target


def mean_nll(logits: Sequence[torch.Tensor], labels: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    losses = [
        -F.log_softmax(row / temperature, dim=0)[int(label)]
        for row, label in zip(logits, labels.tolist())
    ]
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    calibration_nll_before: float
    calibration_nll_after: float
    sample_count: int
    optimizer: str = "LBFGS"


class TemperatureScaler(nn.Module):
    """A single positive temperature fitted only on a calibration partition."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature), dtype=torch.float64))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.to(logits.device, logits.dtype)

    def fit(
        self,
        logits: torch.Tensor | Sequence[torch.Tensor],
        labels: torch.Tensor | Sequence[int],
        *,
        max_iter: int = 100,
    ) -> TemperatureFit:
        rows, target = _validate_batches(logits, labels)
        before = float(mean_nll(rows, target, torch.tensor(1.0, dtype=torch.float64)).item())
        optimizer = torch.optim.LBFGS(
            [self.log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = mean_nll(rows, target, self.temperature)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.log_temperature.clamp_(math.log(0.05), math.log(20.0))
            after = float(mean_nll(rows, target, self.temperature).item())
        return TemperatureFit(
            temperature=float(self.temperature.item()),
            calibration_nll_before=before,
            calibration_nll_after=after,
            sample_count=len(rows),
        )

    def apply_rows(self, logits: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(logits, torch.Tensor):
            return self(logits)
        return [self(row) for row in logits]

    def state_payload(self, fit: TemperatureFit) -> dict[str, float | int | str]:
        return asdict(fit)


@dataclass(frozen=True)
class AdaptiveTemperatureFit:
    calibration_nll_before: float
    calibration_nll_after: float
    sample_count: int
    feature_count: int
    regularization: float
    mean_temperature: float
    min_temperature: float
    max_temperature: float
    optimizer: str = "LBFGS"


class AdaptiveTemperatureScaler(nn.Module):
    """Predict one positive temperature per sample from label-free features.

    A linear log-temperature model is deliberately used instead of a larger
    MLP: benchmark_v1 has only 64 calibration examples, so this is the most
    conservative adaptive model that still lets entropy, visual sensitivity,
    gates, and attention modulate confidence sample by sample.
    """

    def __init__(self, feature_count: int, *, initial_temperature: float = 1.0):
        super().__init__()
        if feature_count <= 0 or initial_temperature <= 0:
            raise ValueError("feature_count and initial_temperature must be positive")
        self.linear = nn.Linear(feature_count, 1, dtype=torch.float64)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, math.log(initial_temperature))
        self.register_buffer("feature_mean", torch.zeros(feature_count, dtype=torch.float64))
        self.register_buffer("feature_std", torch.ones(feature_count, dtype=torch.float64))

    def temperatures(self, features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(features, dtype=torch.float64, device=self.feature_mean.device)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2 or features.shape[1] != self.feature_mean.numel():
            raise ValueError("features must have shape [samples, feature_count]")
        standardised = (features - self.feature_mean) / self.feature_std
        return self.linear(standardised).squeeze(-1).exp().clamp(0.05, 20.0)

    def fit(
        self,
        logits: torch.Tensor | Sequence[torch.Tensor],
        labels: torch.Tensor | Sequence[int],
        features: torch.Tensor,
        *,
        regularization: float = 0.1,
        max_iter: int = 200,
    ) -> AdaptiveTemperatureFit:
        rows, target = _validate_batches(logits, labels)
        matrix = torch.as_tensor(features, dtype=torch.float64)
        if matrix.shape != (len(rows), self.feature_mean.numel()):
            raise ValueError("one feature row is required for every logit row")
        if not torch.isfinite(matrix).all():
            raise ValueError("features contain non-finite values")
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        with torch.no_grad():
            self.feature_mean.copy_(matrix.mean(0))
            self.feature_std.copy_(matrix.std(0, unbiased=False).clamp_min(1e-6))
        before = float(mean_nll(rows, target, torch.tensor(1.0, dtype=torch.float64)).item())
        optimizer = torch.optim.LBFGS(
            self.parameters(), lr=0.2, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def objective() -> torch.Tensor:
            temperatures = self.temperatures(matrix)
            losses = [
                -F.log_softmax(row / temperatures[index], dim=0)[int(label)]
                for index, (row, label) in enumerate(zip(rows, target.tolist()))
            ]
            penalty = regularization * self.linear.weight.square().mean()
            return torch.stack(losses).mean() + penalty

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            temperatures = self.temperatures(matrix)
            after = float(torch.stack([
                -F.log_softmax(row / temperatures[index], dim=0)[int(label)]
                for index, (row, label) in enumerate(zip(rows, target.tolist()))
            ]).mean().item())
        return AdaptiveTemperatureFit(
            calibration_nll_before=before,
            calibration_nll_after=after,
            sample_count=len(rows),
            feature_count=matrix.shape[1],
            regularization=regularization,
            mean_temperature=float(temperatures.mean().item()),
            min_temperature=float(temperatures.min().item()),
            max_temperature=float(temperatures.max().item()),
        )

    def apply_rows(
        self, logits: Sequence[torch.Tensor], features: torch.Tensor
    ) -> list[torch.Tensor]:
        temperatures = self.temperatures(features).detach().cpu()
        if len(logits) != len(temperatures):
            raise ValueError("one feature row is required for every logit row")
        return [row / temperatures[index].to(row.dtype) for index, row in enumerate(logits)]


@dataclass(frozen=True)
class DirichletFit:
    class_count: int
    calibration_nll_before: float
    calibration_nll_after: float
    sample_count: int
    regularization: float
    optimizer: str = "LBFGS"


class DirichletCalibrator(nn.Module):
    """Regularised Dirichlet calibration for one fixed candidate count."""

    def __init__(self, class_count: int):
        super().__init__()
        if class_count < 2:
            raise ValueError("class_count must be at least two")
        self.class_count = class_count
        self.weight = nn.Parameter(torch.eye(class_count, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(class_count, dtype=torch.float64))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.as_tensor(logits, dtype=torch.float64, device=self.weight.device)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.ndim != 2 or logits.shape[1] != self.class_count:
            raise ValueError("logits must have shape [samples, class_count]")
        log_probabilities = F.log_softmax(logits, dim=-1)
        return F.linear(log_probabilities, self.weight, self.bias)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor | Sequence[int],
        *,
        regularization: float = 0.1,
        max_iter: int = 200,
    ) -> DirichletFit:
        matrix = torch.as_tensor(logits, dtype=torch.float64)
        target = labels.detach().long().cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        if matrix.ndim != 2 or matrix.shape[1] != self.class_count or matrix.shape[0] != len(target):
            raise ValueError("fixed-width logits and labels must have equal sample counts")
        if not torch.isfinite(matrix).all() or ((target < 0) | (target >= self.class_count)).any():
            raise ValueError("invalid logits or labels")
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        before = float(F.cross_entropy(matrix, target).item())
        identity = torch.eye(self.class_count, dtype=torch.float64)
        optimizer = torch.optim.LBFGS(
            self.parameters(), lr=0.2, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            transformed = self(matrix)
            penalty = regularization * (
                (self.weight - identity).square().mean() + self.bias.square().mean()
            )
            loss = F.cross_entropy(transformed, target) + penalty
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            after = float(F.cross_entropy(self(matrix), target).item())
        return DirichletFit(
            class_count=self.class_count,
            calibration_nll_before=before,
            calibration_nll_after=after,
            sample_count=len(target),
            regularization=regularization,
        )

    def state_payload(self, fit: DirichletFit) -> dict[str, Any]:
        return {
            **asdict(fit),
            "weight": self.weight.detach().cpu().tolist(),
            "bias": self.bias.detach().cpu().tolist(),
        }
