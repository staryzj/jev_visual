import torch
from torch import nn

from visual_jev import VisualJEV, VisualJev
from visual_jep import VisualJEV as LegacyVisualJEV


def test_visual_jev_scores_one_logit_per_candidate():
    model = VisualJEV(vision_dim=4, hidden_dim=6)
    visual_tokens = torch.randn(3, 5, 4)
    text_feature = torch.randn(3, 6)

    logits = model(visual_tokens, text_feature)
    probabilities = torch.softmax(logits, dim=0)

    assert logits.shape == (3,)
    assert probabilities.shape == (3,)
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0))


def test_visual_jev_can_reuse_existing_decision_head():
    head = nn.Linear(6, 1, dtype=torch.float32)
    model = VisualJEV(vision_dim=4, hidden_dim=6, decision_head=head)

    assert model.decision_head is head
    assert VisualJev is VisualJEV
    assert LegacyVisualJEV is VisualJEV


def test_visual_jev_rejects_mismatched_candidate_batches():
    model = VisualJEV(vision_dim=4, hidden_dim=6)

    try:
        model(torch.randn(2, 5, 4), torch.randn(3, 6))
    except ValueError as error:
        assert "equal shapes" in str(error)
    else:
        raise AssertionError("mismatched candidate batches must fail")
