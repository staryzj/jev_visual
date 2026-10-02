import torch

from run_vision_backend_pair_proof import pair_metrics


def test_pair_both_requires_opposite_image_conditioned_predictions():
    source = torch.tensor([[3.0, 0.0], [2.0, 0.0]])
    counterfactual = torch.tensor([[0.0, 3.0], [0.0, 2.0]])
    metrics = pair_metrics(source, counterfactual)
    assert metrics["both_directions_accuracy"] == 1.0
    assert metrics["prediction_flip_rate"] == 1.0


def test_text_identical_prediction_cannot_pass_pair_both():
    source = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    counterfactual = source.clone()
    metrics = pair_metrics(source, counterfactual)
    assert metrics["both_directions_accuracy"] == 0.0
    assert metrics["prediction_flip_rate"] == 0.0
