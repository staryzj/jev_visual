import json

import torch

from scripts.run_visual_jev_v2_experiment import (
    DiskFeatureDataset,
    open_cached_dataset,
)
from train_visual_jev_v2 import load_records
from visual_jev_v2 import (
    VisualJEVV2,
    VisualJEVV2Config,
    candidate_pair_loss,
    pad_visual_tokens,
)


def make_model() -> VisualJEVV2:
    torch.manual_seed(7)
    return VisualJEVV2(
        VisualJEVV2Config(
            vision_dim=8,
            text_dim=10,
            adapter_dim=12,
            num_heads=3,
            dropout=0.0,
        )
    )


def test_v2_scores_candidates_with_normalized_attention():
    model = make_model()
    visual_tokens = torch.randn(1, 6, 8)
    text_features = torch.randn(3, 10)

    output = model(visual_tokens, text_features)

    assert output.scores.shape == (3,)
    assert output.attention.shape == (3, 6)
    assert output.gates.shape == (3, 12)
    assert model.scalar_head.out_features == 1
    assert torch.allclose(output.attention.sum(-1), torch.ones(3), atol=1e-6)
    assert torch.all((0.0 <= output.gates) & (output.gates <= 1.0))


def test_v2_attention_respects_padded_visual_mask():
    model = make_model()
    tokens, mask = pad_visual_tokens((torch.randn(2, 8), torch.randn(5, 8)))
    text_features = torch.randn(2, 10)

    output = model(tokens, text_features, mask)

    assert output.attention.shape == (2, 5)
    assert torch.equal(output.attention[0, 2:], torch.zeros(3))
    assert torch.allclose(output.attention.sum(-1), torch.ones(2), atol=1e-6)


def test_v2_backward_reaches_every_trainable_component():
    model = make_model()
    output = model(torch.randn(1, 7, 8), torch.randn(3, 10))
    loss = candidate_pair_loss(output.scores[:1], output.scores[1:].unsqueeze(0))

    loss.backward()

    prefixes = (
        "visual_projector",
        "text_projector",
        "cross_attention",
        "gate",
        "fusion_norm",
        "fusion_mlp",
        "output_norm",
        "scalar_head",
    )
    named_parameters = dict(model.named_parameters())
    for prefix in prefixes:
        gradients = [
            parameter.grad
            for name, parameter in named_parameters.items()
            if name.startswith(prefix)
        ]
        assert gradients
        assert any(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        ), prefix


def test_v2_checkpoint_round_trip_preserves_scores(tmp_path):
    model = make_model().eval()
    visual_tokens = torch.randn(1, 4, 8)
    text_features = torch.randn(2, 10)
    expected = model(visual_tokens, text_features).scores.detach()
    checkpoint = tmp_path / "adapter.pt"

    model.save_checkpoint(checkpoint, step=9, metadata={"test": True})
    restored, payload = VisualJEVV2.from_checkpoint(checkpoint)
    actual = restored.eval()(visual_tokens, text_features).scores.detach()

    assert payload["step"] == 9
    assert payload["metadata"] == {"test": True}
    assert torch.equal(expected, actual)


def test_candidate_pair_loss_supports_cross_entropy_and_hard_negative_ranking():
    positive = torch.tensor([2.0, 1.5], requires_grad=True)
    negatives = torch.tensor([[0.0, -1.0], [0.5, 0.25]], requires_grad=True)

    cross_entropy = candidate_pair_loss(positive, negatives)
    ranking = candidate_pair_loss(positive, negatives, loss_type="ranking", margin=0.2)
    (cross_entropy + ranking).backward()

    assert cross_entropy.item() > 0.0
    assert ranking.item() == 0.0
    assert positive.grad is not None
    assert negatives.grad is not None


def test_training_records_accept_negative_and_hard_negatives(tmp_path):
    (tmp_path / "one.jpg").write_bytes(b"placeholder")
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "image": "one.jpg",
                        "positive": "a chart",
                        "negative": "a dog",
                    }
                ),
                json.dumps(
                    {
                        "image": "one.jpg",
                        "positive": "a chart",
                        "hard_negatives": ["a table", "a map"],
                    }
                ),
            )
        ),
        encoding="utf-8",
    )

    records = load_records(data_path)

    assert records[0].negatives == ("a dog",)
    assert records[1].negatives == ("a table", "a map")


def test_legacy_feature_cache_converts_to_disk_shards(tmp_path):
    cache_path = tmp_path / "train.pt"
    features = [
        {
            "visual_tokens": torch.randn(3 + index, 8),
            "text_features": torch.randn(2, 10),
        }
        for index in range(3)
    ]
    torch.save(
        {"source_sha256": "test-source", "features": features},
        cache_path,
    )

    dataset = open_cached_dataset(
        cache_path,
        source_sha256="test-source",
        item_key="features",
    )

    assert isinstance(dataset, DiskFeatureDataset)
    assert len(dataset) == 3
    assert dataset[2]["visual_tokens"].shape == (5, 8)
    assert (tmp_path / "train-shards" / "manifest.json").is_file()
