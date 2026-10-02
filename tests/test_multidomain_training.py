import math
import random

import pytest
import torch

from jev.multidomain import (
    gradnorm_backward,
    masked_brier,
    masked_cross_entropy,
    masked_kl_to_uniform,
    masked_probabilities,
    normalized_entropy,
    pad_variable_candidates,
    pcgrad_backward,
    permute_candidates,
    preference_anchor_loss,
    uniform_targets,
    visual_dependency_masks,
)
from jev.recent_multidomain import (
    PiKEInspiredWeights,
    capped_balanced_weights,
    gradient_pca_diagnostic,
    weighted_state_dict_average,
)


def test_candidate_shuffle_remaps_label_to_same_answer():
    candidates = ("correct", "hard-a", "hard-b", "hard-c")
    shuffled, label = permute_candidates(candidates, 0, (2, 0, 3, 1))
    assert shuffled[label] == "correct"
    assert label == 1
    with pytest.raises(ValueError, match="permutation"):
        permute_candidates(candidates, 0, (0, 0, 2, 3))


def test_variable_k_padding_never_enters_losses_or_probabilities():
    rows = [torch.tensor([2.0, -1.0]), torch.tensor([1.0, 4.0, 0.0, -2.0])]
    logits, valid, labels = pad_variable_candidates(rows, [0, 1])
    probability = masked_probabilities(logits, valid)
    assert torch.allclose(probability[0, :2], rows[0].softmax(-1))
    assert torch.equal(probability[0, 2:], torch.zeros(2))
    assert torch.allclose(probability.sum(-1), torch.ones(2))
    assert torch.allclose(masked_cross_entropy(logits, valid, labels), torch.stack([
        -rows[0].log_softmax(-1)[0], -rows[1].log_softmax(-1)[1]
    ]).mean())
    assert torch.isfinite(masked_brier(logits, valid, labels))
    assert torch.isfinite(masked_kl_to_uniform(logits, valid))


def test_uniform_target_and_normalized_entropy_use_each_rows_k():
    logits, valid, _ = pad_variable_candidates(
        [torch.zeros(2), torch.zeros(3), torch.zeros(5)], [0, 0, 0]
    )
    target = uniform_targets(valid)
    assert torch.allclose(target[0, :2], torch.full((2,), 0.5))
    assert torch.allclose(target[1, :3], torch.full((3,), 1 / 3))
    assert torch.allclose(target[2], torch.full((5,), 0.2))
    assert torch.all(target[~valid] == 0)
    assert torch.allclose(normalized_entropy(logits, valid), torch.ones(3), atol=1e-6)
    assert masked_kl_to_uniform(logits, valid).item() == pytest.approx(0.0, abs=1e-7)


def test_pcgrad_removes_a_two_task_conflict():
    value = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    left = value[0] + value[1]
    right = -value[0] + value[1]
    report = pcgrad_backward([left, right], [value], rng=random.Random(7))
    assert report["cosine"][0][1] == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(value.grad).all()


def test_gradnorm_and_preference_anchor_are_finite():
    value = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    losses = [(value.square()).sum(), ((value - 3).square()).sum()]
    report = gradnorm_backward(losses, [value], initial_losses=[5.0, 2.0])
    assert len(report["scales"]) == 2
    assert torch.isfinite(value.grad).all()
    anchor = preference_anchor_loss(torch.tensor([0.3, -0.1]), torch.tensor([-0.2, 0.1]), label=0)
    assert math.isfinite(anchor.item()) and anchor.item() > 0


def test_stage_c_gates_do_not_make_scienceqa_uniform_by_default():
    assert visual_dependency_masks("coco_hard_negative")["counterfactual"]
    assert visual_dependency_masks("aokvqa")["invalid"]
    assert not visual_dependency_masks("aokvqa")["preference"]
    assert visual_dependency_masks(
        "aokvqa", semantic_counterfactual_available=True
    )["preference"]
    science = visual_dependency_masks("scienceqa_image_only")
    assert science["type"] == "mixed_or_text_solvable"
    assert not science["invalid"]
    assert visual_dependency_masks(
        "scienceqa_image_only", scienceqa_visual_required=True
    )["invalid"]


def test_capped_balance_and_pike_weights_are_normalised():
    domains = ("a", "b", "c")
    capped = capped_balanced_weights({"a": 100, "b": 10, "c": 50}, domains, cap=20)
    assert capped == pytest.approx({"a": 0.4, "b": 0.2, "c": 0.4})
    estimator = PiKEInspiredWeights(domains, eta=0.2)
    report = estimator.update(
        {"a": 1.0, "b": 0.5, "c": 0.8},
        [[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]],
    )
    assert sum(report["weights"].values()) == pytest.approx(1.0)
    assert all(value > 0 for value in report["weights"].values())


def test_gradient_pca_and_weighted_merge():
    gradients = {
        "a": torch.tensor([1.0, 0.0]),
        "b": torch.tensor([0.8, 0.1]),
        "c": torch.tensor([-1.0, 0.0]),
        "d": torch.tensor([-0.8, -0.1]),
    }
    report = gradient_pca_diagnostic(gradients)
    assert report["structured_conflict"]
    assert set(report["clusters"][0]) | set(report["clusters"][1]) == set(gradients)
    merged = weighted_state_dict_average(
        [{"w": torch.tensor([1.0])}, {"w": torch.tensor([3.0])}], [1.0, 3.0]
    )
    assert merged["w"].item() == pytest.approx(2.5)

