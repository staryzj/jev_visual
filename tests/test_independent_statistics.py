from scripts.summarize_independent_statistics import wilson


def test_wilson_interval_contains_observed_proportion() -> None:
    interval = wilson(76, 83)
    assert interval["successes"] == 76
    assert interval["total"] == 83
    assert interval["lower_95"] < 76 / 83 < interval["upper_95"]


def test_wilson_interval_handles_boundaries() -> None:
    assert wilson(0, 10)["lower_95"] == 0.0
    assert wilson(10, 10)["upper_95"] == 1.0

