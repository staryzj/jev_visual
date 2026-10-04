"""Candidate-classification metrics shared by the benchmark evaluator.

Extracted unchanged from the historical comparison runner so evaluation has
no dependency on that unrelated model or its PEFT loading path.
"""
import math
import statistics
import numpy as np
import torch
import torch.nn.functional as F


def macro_f1(predictions: list[int], labels: list[int]) -> float:
    classes = sorted(set(predictions) | set(labels))
    values = []
    for cls in classes:
        tp = sum(p == cls and y == cls for p, y in zip(predictions, labels))
        fp = sum(p == cls and y != cls for p, y in zip(predictions, labels))
        fn = sum(p != cls and y == cls for p, y in zip(predictions, labels))
        values.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return statistics.fmean(values)


def metrics(logits: list[list[float]], labels: list[int], bins: int = 10) -> dict[str, float | int]:
    rows = [torch.tensor(row, dtype=torch.float32) for row in logits]
    probs = [row.softmax(-1) for row in rows]
    predictions = [int(row.argmax()) for row in rows]
    confidences = np.asarray([float(row.max()) for row in probs])
    correct = np.asarray([float(p == y) for p, y in zip(predictions, labels)])
    nll = statistics.fmean(-math.log(max(float(row[y]), 1e-12)) for row, y in zip(probs, labels))
    brier = statistics.fmean(
        float(((row - F.one_hot(torch.tensor(y), row.numel()).float()) ** 2).sum())
        for row, y in zip(probs, labels)
    )
    ece = 0.0
    for low in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        high = low + 1.0 / bins
        mask = (confidences >= low) & (confidences < high if high < 1.0 else confidences <= high)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    return {
        'count': len(labels),
        'accuracy': float(correct.mean()),
        'macro_f1': macro_f1(predictions, labels),
        'nll': nll,
        'brier': brier,
        'ece': ece,
    }
