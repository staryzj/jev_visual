import torch

from jev.calibration import AdaptiveTemperatureScaler, DirichletCalibrator


def test_adaptive_temperature_is_positive_and_argmax_invariant():
    logits = [
        torch.tensor([5.0, 0.0]),
        torch.tensor([4.0, 0.0]),
        torch.tensor([3.0, 0.0]),
        torch.tensor([2.0, 0.0]),
    ]
    labels = [0, 0, 1, 1]
    features = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]])
    scaler = AdaptiveTemperatureScaler(2)
    fit = scaler.fit(logits, labels, features, regularization=1.0, max_iter=30)
    transformed = scaler.apply_rows(logits, features)
    assert fit.min_temperature > 0
    assert fit.max_temperature <= 20.0
    assert [int(row.argmax()) for row in transformed] == [int(row.argmax()) for row in logits]


def test_dirichlet_calibrator_accepts_fixed_width_logits():
    logits = torch.tensor(
        [[4.0, 0.0, -1.0], [0.0, 3.0, -1.0], [2.0, 0.0, 1.0], [0.0, 2.0, 1.0]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 1, 2, 2])
    calibrator = DirichletCalibrator(3)
    fit = calibrator.fit(logits, labels, regularization=1.0, max_iter=30)
    transformed = calibrator(logits)
    assert transformed.shape == logits.shape
    assert torch.isfinite(transformed).all()
    assert fit.sample_count == len(labels)
