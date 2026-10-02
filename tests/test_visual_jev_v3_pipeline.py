from pathlib import Path

import torch

from visual_jev_v3 import VisualJEVV3Config
from visual_jev_v3_pipeline import (
    AlignmentAdapterConfig,
    MeanPoolJEV,
    ThreeStageVisualJEV,
    alignment_loss,
    grouped_pre_merger_tokens,
)
from jev.benchmark_v1 import BenchmarkExample, build_manifests, read_jsonl, write_predefined_manifests
from jev.calibration import TemperatureScaler


def test_grouped_pre_merger_and_alignment_backward():
    pre = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    grouped = grouped_pre_merger_tokens(pre, 2)
    assert grouped.shape == (2, 4)
    config = AlignmentAdapterConfig(4, 6, 8)
    model = ThreeStageVisualJEV(config, VisualJEVV3Config(6, 5, 6, 2))
    teacher = torch.randn(2, 6)
    output = model.alignment(grouped)
    loss = alignment_loss(output, teacher)
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert all(parameter.grad is not None for parameter in model.alignment.parameters())


def test_three_stage_checkpoint_roundtrip(tmp_path: Path):
    torch.manual_seed(7)
    model = ThreeStageVisualJEV(
        AlignmentAdapterConfig(4, 6, 8), VisualJEVV3Config(6, 5, 6, 2)
    )
    pre = torch.randn(3, 4)
    text = torch.randn(3, 5)
    expected = model(pre, text).scores.detach()
    path = model.save_checkpoint(tmp_path / "model.pt", stage="C", step=9)
    restored, payload = ThreeStageVisualJEV.from_checkpoint(path)
    actual = restored(pre, text).scores.detach()
    assert payload["stage"] == "C"
    assert payload["step"] == 9
    assert torch.allclose(expected, actual)


def test_meanpool_shared_scalar_head_backward():
    model = MeanPoolJEV(6, 5, 4)
    scores = model(torch.randn(7, 6), torch.randn(3, 5))
    assert scores.shape == (3,)
    scores.sum().backward()
    assert model.scalar_head.weight.grad is not None


def test_benchmark_v1_manifest_is_deterministic_and_group_safe(tmp_path: Path):
    examples = [
        BenchmarkExample(
            id=f"toy:{index}", dataset="toy", image=f"{index}.jpg",
            question="Which?", candidates=("a", "b"), label=index % 2,
            group_id=f"group-{index // 2}",
        )
        for index in range(20)
    ]
    first = build_manifests(examples, tmp_path / "first")
    second = build_manifests(reversed(examples), tmp_path / "second")
    assert first["seed"] == 20260928
    assert [first["splits"][name]["sha256"] for name in ("train", "validation", "calibration", "test")] == [
        second["splits"][name]["sha256"] for name in ("train", "validation", "calibration", "test")
    ]
    location = {}
    for split in ("train", "validation", "calibration", "test"):
        for example in read_jsonl(tmp_path / "first" / f"{split}.jsonl"):
            prior = location.setdefault(example.group_id, split)
            assert prior == split


def test_temperature_scaling_uses_calibration_without_changing_predictions():
    logits = torch.tensor([[8.0, 0.0], [7.0, 0.0], [6.0, 0.0], [5.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    scaler = TemperatureScaler()
    fit = scaler.fit(logits, labels)
    scaled = scaler(logits.float())
    assert fit.temperature > 1.0
    assert fit.calibration_nll_after < fit.calibration_nll_before
    assert torch.equal(logits.argmax(-1), scaled.argmax(-1))


def test_temperature_scaling_supports_variable_candidate_counts():
    logits = [torch.tensor([3.0, 1.0]), torch.tensor([0.0, 4.0, 1.0])]
    scaler = TemperatureScaler()
    fit = scaler.fit(logits, [0, 1])
    scaled = scaler.apply_rows(logits)
    assert fit.sample_count == 2
    assert [row.shape for row in scaled] == [torch.Size([2]), torch.Size([3])]


def test_predefined_manifests_reject_group_leakage(tmp_path: Path):
    left = BenchmarkExample("left", "toy", "a.jpg", "q", ("a", "b"), 0, "same")
    right = BenchmarkExample("right", "toy", "a.jpg", "q", ("a", "b"), 1, "same")
    try:
        write_predefined_manifests({"train": [left], "test": [right]}, tmp_path)
    except ValueError as error:
        assert "group" in str(error)
    else:
        raise AssertionError("group leakage was not rejected")
