import pytest
import torch

from jev.vision_backends import (
    TextOnlyCandidateScorer,
    batched_candidate_scores,
)
from run_vision_backend_compatibility import cache_is_valid, sha256, write_json
from visual_jev_v3 import VisualJEVV3, VisualJEVV3Config


@pytest.mark.parametrize("vision_dim,token_count", [(64, 49), (96, 196)])
def test_same_decision_code_accepts_different_backend_shapes(vision_dim, token_count):
    model = VisualJEVV3(
        VisualJEVV3Config(
            vision_dim=vision_dim,
            text_dim=32,
            adapter_dim=32,
            num_heads=4,
        )
    )
    visual = torch.randn(3, token_count, vision_dim)
    text = torch.randn(3, 5, 32)
    scores = batched_candidate_scores(model, visual, text)
    assert scores.shape == (3, 5)
    assert torch.isfinite(scores).all()


def test_text_only_control_is_candidate_count_agnostic():
    model = TextOnlyCandidateScorer(text_dim=24, hidden_dim=16)
    assert model(torch.randn(2, 3, 24)).shape == (2, 3)
    assert model(torch.randn(2, 7, 24)).shape == (2, 7)


def test_backend_batch_contract_rejects_mismatched_batch():
    model = VisualJEVV3(
        VisualJEVV3Config(vision_dim=16, text_dim=12, adapter_dim=16, num_heads=4)
    )
    with pytest.raises(ValueError, match="batch sizes differ"):
        batched_candidate_scores(model, torch.randn(2, 4, 16), torch.randn(3, 2, 12))


def test_feature_cache_is_bound_to_exact_weight_identity(tmp_path):
    tensor_path = tmp_path / "train.pt"
    torch.save({"visual_tokens": torch.randn(2, 4, 8)}, tensor_path)
    write_json(
        tensor_path.with_suffix(".json"),
        {
            "backend": "encoder-a",
            "manifest_sha256": "manifest-sha",
            "count": 2,
            "tensor_sha256": sha256(tensor_path),
            "weight_id": "weights-v1",
            "weight_sha256": "weight-sha-v1",
        },
    )
    assert cache_is_valid(
        tensor_path,
        "manifest-sha",
        "encoder-a",
        2,
        weight_id="weights-v1",
        weight_sha256="weight-sha-v1",
    )
    assert not cache_is_valid(
        tensor_path,
        "manifest-sha",
        "encoder-a",
        2,
        weight_id="weights-v2",
        weight_sha256="weight-sha-v1",
    )
