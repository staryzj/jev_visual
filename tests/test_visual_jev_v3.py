import pytest
import torch

from scripts.prepare_visual_jev_v3_pairs import build_pairs
from visual_jev_v2 import VisualJEVV2, VisualJEVV2Config, candidate_pair_loss
from visual_jev_v3 import (
    VisualJEVV3,
    VisualJEVV3Config,
    VisualJEVV3LossWeights,
    combine_v3_losses,
    counterfactual_visual_dependency_loss,
    cross_image_ranking_loss,
    text_null_features,
    uniformity_loss,
)


def make_model() -> VisualJEVV3:
    torch.manual_seed(17)
    return VisualJEVV3(
        VisualJEVV3Config(
            vision_dim=8,
            text_dim=10,
            adapter_dim=12,
            num_heads=3,
            dropout=0.0,
        )
    )


def test_v3_forward_backward_all_objectives():
    model = make_model()
    visual1 = torch.randn(1, 6, 8)
    visual2 = torch.randn(1, 5, 8)
    blank = torch.zeros(1, 4, 8)
    noise = torch.randn(1, 7, 8)
    text = torch.randn(3, 10)

    scores1 = model(visual1, text).scores
    scores2 = model(visual2, text).scores
    blank_scores = model(blank, text).scores
    noise_scores = model(noise, text).scores
    null_scores = model(visual1, text_null_features(text, mode="mean")).scores
    losses = combine_v3_losses(
        candidate=candidate_pair_loss(scores1[:1], scores1[1:].unsqueeze(0)),
        counterfactual=counterfactual_visual_dependency_loss(
            scores1, scores2, image2_target=1
        ),
        invalid_image=0.5
        * (uniformity_loss(blank_scores) + uniformity_loss(noise_scores)),
        text_null=uniformity_loss(null_scores),
        cross_image_rank=cross_image_ranking_loss(scores1, scores2),
        weights=VisualJEVV3LossWeights(),
    )
    losses.total.backward()

    assert scores1.shape == (3,)
    assert torch.isfinite(losses.total)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_uniformity_loss_is_zero_for_uniform_and_positive_for_peaked_logits():
    uniform = torch.zeros(4, 3, requires_grad=True)
    peaked = torch.tensor([[8.0, 0.0, -1.0]], requires_grad=True)
    uniform_kl = uniformity_loss(uniform)
    peaked_kl = uniformity_loss(peaked)
    js = uniformity_loss(peaked, divergence="js")
    (uniform_kl + peaked_kl + js).backward()

    assert uniform_kl.item() == pytest.approx(0.0, abs=1e-7)
    assert peaked_kl.item() > 0.5
    assert 0.0 < js.item() < torch.log(torch.tensor(2.0)).item()
    assert peaked.grad is not None


def test_text_null_removes_candidate_identity_and_forces_uniform_distribution():
    model = make_model().eval()
    visual = torch.randn(1, 6, 8)
    text = torch.randn(3, 10)
    null_text = text_null_features(text, mode="mean")
    scores = model(visual, null_text).scores

    assert torch.equal(null_text[0], null_text[1])
    assert torch.allclose(scores, scores[0].expand_as(scores), atol=1e-6)
    assert uniformity_loss(scores).item() == pytest.approx(0.0, abs=1e-7)


def test_counterfactual_loss_rewards_the_required_prediction_flip():
    correct_i1 = torch.tensor([4.0, 0.0, -1.0])
    correct_i2 = torch.tensor([0.0, 4.0, -1.0])
    wrong_i2 = torch.tensor([4.0, 0.0, -1.0])

    good = counterfactual_visual_dependency_loss(
        correct_i1, correct_i2, image2_target=1
    )
    bad = counterfactual_visual_dependency_loss(
        correct_i1, wrong_i2, image2_target=1
    )

    assert good.item() < bad.item()
    with pytest.raises(ValueError, match="must flip"):
        counterfactual_visual_dependency_loss(
            correct_i1, correct_i2, image1_target=0, image2_target=0
        )


def test_v3_checkpoint_round_trip(tmp_path):
    model = make_model().eval()
    visual = torch.randn(1, 4, 8)
    text = torch.randn(3, 10)
    expected = model(visual, text).scores.detach()
    checkpoint = tmp_path / "visual-jev-v3-test.pt"

    model.save_checkpoint(checkpoint, step=12, metadata={"version": 3})
    restored, payload = VisualJEVV3.from_checkpoint(checkpoint)
    actual = restored.eval()(visual, text).scores.detach()

    assert payload["model_type"] == "visual-jev-v3"
    assert payload["step"] == 12
    assert torch.equal(expected, actual)


def test_v3_loads_v2_checkpoint_without_modifying_it(tmp_path):
    torch.manual_seed(23)
    v2 = VisualJEVV2(
        VisualJEVV2Config(
            vision_dim=8,
            text_dim=10,
            adapter_dim=12,
            num_heads=3,
            dropout=0.0,
        )
    ).eval()
    checkpoint = tmp_path / "visual-jev-v2.pt"
    v2.save_checkpoint(checkpoint, step=7)
    before = checkpoint.read_bytes()
    visual = torch.randn(1, 4, 8)
    text = torch.randn(3, 10)

    v3, payload = VisualJEVV3.from_checkpoint(checkpoint)

    assert payload["model_type"] == "visual-jev-v2"
    assert torch.equal(v2(visual, text).scores, v3.eval()(visual, text).scores)
    assert checkpoint.read_bytes() == before


def test_counterfactual_pair_builder_uses_distinct_images_and_flipped_object():
    rows = [
        {
            "positive": "a bird on a branch",
            "hard_negatives": ["a cat on a branch"],
            "metadata": {"image_id": 10, "source_object": "bird"},
        },
        {
            "positive": "a cat by a chair",
            "hard_negatives": ["a bird by a chair"],
            "metadata": {"image_id": 20, "source_object": "cat"},
        },
    ]

    pairs, stats = build_pairs(rows, split="train", seed=1)

    assert stats["pairs"] == 2
    assert all(pair["source_image_id"] != pair["counterfactual_image_id"] for pair in pairs)
    assert {pair["counterfactual_target_index"] for pair in pairs} == {1}


def test_counterfactual_pair_builder_rejects_duplicate_image_ids():
    rows = [
        {
            "positive": "a bird",
            "hard_negatives": ["a cat"],
            "metadata": {"image_id": 10, "source_object": "bird"},
        },
        {
            "positive": "a cat",
            "hard_negatives": ["a bird"],
            "metadata": {"image_id": 10, "source_object": "cat"},
        },
    ]
    with pytest.raises(ValueError, match="duplicate image_id"):
        build_pairs(rows, split="train", seed=1)
