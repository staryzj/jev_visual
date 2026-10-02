from types import SimpleNamespace

import torch
from torch import nn

from jev.conversation_history import VisualConversationHistory
from scripts.live_visual_jev_demo import cache_validation, input_path, interactive_case
from visual_jev_v2 import Qwen3VLFeatureExtractor


class _Processor:
    def apply_chat_template(self, *args, **kwargs):
        return {
            "pixel_values": torch.zeros(8, 1),
            "image_grid_thw": torch.tensor([[1, 2, 4]]),
        }


class _Visual(nn.Module):
    spatial_merge_unit = 4

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, pixel_values, *, grid_thw, return_dict):
        native = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        post = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        return SimpleNamespace(last_hidden_state=native, pooler_output=post)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Visual()
        self.language_model = nn.Linear(2, 2)


def test_qwen_extractor_returns_grouped_pre_merger_tokens():
    decision_model = SimpleNamespace(processor=_Processor(), backbone=_Backbone())
    extractor = Qwen3VLFeatureExtractor(decision_model)

    grouped, post = extractor.encode_image_stages(object())

    native = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    assert torch.equal(grouped, native.reshape(2, 4, 3).mean(dim=1))
    assert grouped.shape == (2, 3)
    assert post.shape == (2, 5)


def test_cache_validation_is_diagnostic_and_exact(tmp_path):
    pre = torch.randn(3, 4, dtype=torch.bfloat16)
    text = torch.randn(2, 5, dtype=torch.bfloat16)
    torch.save(
        {"pre_tokens": pre.clone(), "text_features": text.clone()},
        tmp_path / "000007.pt",
    )

    report = cache_validation(
        tmp_path, 7, pre, text, atol=0.0, rtol=0.0
    )

    assert report["passed"] is True
    assert report["cache_used_for_decision"] is False
    assert report["max_abs_error"] == 0.0
    assert report["live_shape"] == [3, 4]


def test_input_path_infers_a_unique_image_extension(tmp_path, monkeypatch):
    image = tmp_path / "222.png"
    image.write_bytes(b"placeholder")
    monkeypatch.chdir(tmp_path)

    assert input_path("222") == image


def test_input_path_requires_extension_when_stem_is_ambiguous(tmp_path, monkeypatch):
    (tmp_path / "222.png").write_bytes(b"placeholder")
    (tmp_path / "222.jpg").write_bytes(b"placeholder")
    monkeypatch.chdir(tmp_path)

    try:
        input_path("222")
    except ValueError as error:
        assert "multiple images match" in str(error)
    else:
        raise AssertionError("ambiguous image stem should require an extension")


def test_dialogue_shortcut_reuses_current_image(tmp_path, monkeypatch):
    image = tmp_path / "dog.png"
    image.write_bytes(b"placeholder")
    history = VisualConversationHistory(mode="auto")
    history.bind_image(image)
    answers = iter(["d", "它是动物吗？", "是", "不是", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    case = interactive_case(tmp_path / "unused.jsonl", 0, history)

    assert case is not None
    assert case.image == image
    assert case.question == "它是动物吗？"
    assert case.candidates == ("是", "不是")
