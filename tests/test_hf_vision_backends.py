from types import SimpleNamespace

import pytest
import torch

from jev.hf_vision_backends import normalize_visual_tokens, resample_visual_tokens


def test_normalize_visual_tokens_removes_singleton_batch() -> None:
    tokens = torch.randn(1, 17, 32)
    result = normalize_visual_tokens(SimpleNamespace(last_hidden_state=tokens))
    assert result.shape == (17, 32)
    assert torch.equal(result, tokens[0])


def test_resample_visual_tokens_has_fixed_shape_and_is_finite() -> None:
    tokens = torch.randn(1025, 64, dtype=torch.bfloat16)
    result = resample_visual_tokens(tokens, 196)
    assert result.shape == (196, 64)
    assert result.dtype == torch.bfloat16
    assert torch.isfinite(result).all()


def test_resample_visual_tokens_validates_inputs() -> None:
    with pytest.raises(ValueError):
        resample_visual_tokens(torch.randn(2, 3, 4), 196)
    with pytest.raises(ValueError):
        resample_visual_tokens(torch.randn(2, 3), 0)
