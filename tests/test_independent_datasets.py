from __future__ import annotations

import torch

from run_independent_datasets import DATASETS, extended_metrics, passes_regression_floor


def test_independent_dataset_list_has_no_duplicate_or_mixed_alias() -> None:
    assert len(DATASETS) == len(set(DATASETS)) == 4
    assert "mixed" not in DATASETS


def test_extended_metrics_reports_normalized_entropy_and_variable_k() -> None:
    logits = [torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0, -1.0])]
    metrics = extended_metrics(logits, [0, 1])
    assert metrics["accuracy"] == 1.0
    assert 0.0 <= metrics["normalized_entropy"] <= 1.0
    assert metrics["count"] == 2


def test_positive_temperature_preserves_argmax() -> None:
    logits = [torch.tensor([-1.0, 3.0, 2.0]), torch.tensor([4.0, 0.0])]
    for temperature in (0.05, 0.5, 1.0, 20.0):
        assert [int(row.argmax()) for row in logits] == [
            int((row / temperature).argmax()) for row in logits
        ]


def test_regression_floor_accepts_improvement_and_rejects_large_drop() -> None:
    assert passes_regression_floor(0.92, target=0.88, tolerance=0.02)
    assert passes_regression_floor(0.86, target=0.88, tolerance=0.02)
    assert not passes_regression_floor(0.859, target=0.88, tolerance=0.02)
