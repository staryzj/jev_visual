from __future__ import annotations

import torch

from run_benchmark_v1_experiment import _cache_valid, _write_feature_manifest


def test_feature_cache_rejects_zero_byte_interrupted_shard(tmp_path) -> None:
    cache = tmp_path / "train-shards"
    cache.mkdir()
    _write_feature_manifest(cache, "source-hash", 2)
    torch.save(
        {
            "pre_tokens": torch.zeros(1, 2),
            "teacher_tokens": torch.zeros(1, 2),
            "text_features": torch.zeros(2, 2),
        },
        cache / "000000.pt",
    )
    (cache / "000001.pt").touch()

    assert not _cache_valid(cache, "source-hash", 2)

