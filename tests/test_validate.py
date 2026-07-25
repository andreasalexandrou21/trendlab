"""Validation-layer checks.

These tests are about arithmetic, not markets, so they use synthetic returns with known
properties. The point is that the statistics behave correctly when handed data whose true
answer is known in advance, especially data with NO edge.
"""

import numpy as np
import pandas as pd
import pytest

from trendlab.validate import (
    deflated_sharpe_ratio,
    effective_breadth,
    expected_max_sharpe,
    log_trial,
    probabilistic_sharpe_ratio,
    read_trials,
    sharpe_ratio,
    spa_test,
    walk_forward,
)


def _series(mu, sigma, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(rng.normal(mu, sigma, n), index=idx)


def test_sharpe_annualisation_is_explicit():
    r = _series(0.0005, 0.01)
    assert sharpe_ratio(r, annualise=True) == pytest.approx(
        sharpe_ratio(r) * np.sqrt(365), rel=1e-9
    )


def test_psr_is_half_for_a_zero_sharpe_strategy():
    """No edge means a coin flip on whether the true Sharpe is above zero."""
    r = _series(0.0, 0.01, n=5000, seed=3)
    assert probabilistic_sharpe_ratio(r, benchmark_sr=sharpe_ratio(r)) == pytest.approx(
        0.5, abs=0.02
    )


def test_psr_rises_with_sample_length():
    """The same Sharpe observed over a longer sample is more believable."""
    short = _series(0.0005, 0.01, n=200, seed=4)
    long = _series(0.0005, 0.01, n=4000, seed=4)
    assert probabilistic_sharpe_ratio(long) > probabilistic_sharpe_ratio(short)


def test_psr_punishes_negative_skew():
    """Two streams, same mean and volatility, one with a fat left tail. The skewed one
    should command less confidence: that is the whole reason PSR exists rather than a t-test."""
    rng = np.random.default_rng(9)
    n = 3000
    idx = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")

    symmetric = pd.Series(rng.normal(0, 1, n), index=idx)
    skewed = pd.Series(-rng.gumbel(0, 1, n), index=idx)  # left-tailed

    symmetric = (symmetric - symmetric.mean()) / symmetric.std() * 0.01 + 0.0004
    skewed = (skewed - skewed.mean()) / skewed.std() * 0.01 + 0.0004

    assert sharpe_ratio(symmetric) == pytest.approx(sharpe_ratio(skewed), rel=1e-6)
    assert probabilistic_sharpe_ratio(skewed) < probabilistic_sharpe_ratio(symmetric)


def test_expected_max_sharpe_grows_with_trials():
    """Search harder over noise and the best result gets better, by construction."""
    dispersion = 0.03
    values = [expected_max_sharpe(n, dispersion) for n in (2, 10, 100, 10_000)]
    assert values == sorted(values)
    assert expected_max_sharpe(1, dispersion) == 0.0


def test_expected_max_sharpe_scales_with_dispersion():
    assert expected_max_sharpe(100, 0.06) == pytest.approx(2 * expected_max_sharpe(100, 0.03))


def test_deflated_sharpe_falls_as_the_search_widens():
    """The headline result: an identical return stream is less impressive if you tried
    more things to find it. This is the correction that most published backtests skip."""
    r = _series(0.0006, 0.01, n=3000, seed=11)
    rng = np.random.default_rng(2)

    few = rng.normal(0, 0.02, 5)
    many = rng.normal(0, 0.02, 5000)

    dsr_few, sr0_few = deflated_sharpe_ratio(r, few)
    dsr_many, sr0_many = deflated_sharpe_ratio(r, many)

    assert sr0_many > sr0_few
    assert dsr_many < dsr_few


def test_deflated_sharpe_needs_more_than_one_trial():
    r = _series(0.0005, 0.01)
    with pytest.raises(ValueError, match="at least 2 trials"):
        deflated_sharpe_ratio(r, [0.05])


def test_walk_forward_never_scores_the_training_window():
    """Out-of-sample slices must be disjoint from every training window that preceded them."""
    candidates = {
        "a": _series(0.0004, 0.01, n=2500, seed=1),
        "b": _series(0.0001, 0.01, n=2500, seed=2),
    }
    wf = walk_forward(candidates, train_years=3.0, test_years=1.0)

    assert len(wf.folds) >= 2
    for fold in wf.folds:
        assert fold.test_start > fold.train_end
    starts = [f.test_start for f in wf.folds]
    assert starts == sorted(starts)
    assert len(wf.oos_returns) == len(set(wf.oos_returns.index))


def test_walk_forward_picks_the_best_in_sample_candidate():
    """A candidate that is clearly better in training must be the one selected."""
    n = 2500
    good = _series(0.002, 0.01, n=n, seed=5)
    bad = _series(-0.002, 0.01, n=n, seed=6)
    wf = walk_forward({"good": good, "bad": bad}, train_years=3.0, test_years=1.0)

    assert all(f.chosen == "good" for f in wf.folds)


def test_walk_forward_shows_shrinkage_on_pure_noise():
    """Twenty no-edge candidates. Selecting the in-sample winner produces a flattering
    training Sharpe and roughly nothing out of sample. That gap is the whole lesson."""
    candidates = {f"noise_{i}": _series(0.0, 0.01, n=3000, seed=100 + i) for i in range(20)}
    wf = walk_forward(candidates, train_years=3.0, test_years=1.0)

    is_mean = np.mean([f.train_sharpe for f in wf.folds])
    oos = sharpe_ratio(wf.oos_returns)
    assert is_mean > 0
    assert oos < is_mean


def test_walk_forward_rejects_too_little_history():
    with pytest.raises(ValueError, match="not enough history"):
        walk_forward({"a": _series(0.0, 0.01, n=100)}, train_years=3.0, test_years=1.0)


def test_spa_does_not_flag_noise_as_skill():
    """Ten no-edge candidates against a no-edge benchmark. The consistent p-value should
    not reject: any apparent winner is explained by the search itself."""
    benchmark = _series(0.0, 0.01, n=1500, seed=50)
    candidates = {f"c{i}": _series(0.0, 0.01, n=1500, seed=60 + i) for i in range(10)}

    p = spa_test(benchmark, candidates, reps=300)
    assert p["pvalue_consistent"] > 0.05


def test_spa_detects_a_genuinely_better_model():
    benchmark = _series(0.0, 0.01, n=1500, seed=70)
    candidates = {
        "real_edge": _series(0.002, 0.01, n=1500, seed=71),
        "noise": _series(0.0, 0.01, n=1500, seed=72),
    }
    p = spa_test(benchmark, candidates, reps=300)
    assert p["pvalue_consistent"] < 0.05


def test_effective_breadth_counts_independent_bets():
    idx = pd.date_range("2020-01-01", periods=600, freq="D", tz="UTC")
    rng = np.random.default_rng(21)

    independent = pd.DataFrame(rng.normal(0, 0.01, (600, 4)), index=idx)
    assert effective_breadth(independent) == pytest.approx(4.0, rel=0.25)

    one_factor = pd.Series(rng.normal(0, 0.01, 600), index=idx)
    identical = pd.DataFrame({c: one_factor for c in range(4)})
    assert effective_breadth(identical) == pytest.approx(1.0, abs=1e-6)


def test_trial_log_appends_and_reads_back(tmp_path):
    path = tmp_path / "trials.jsonl"
    log_trial("ewmac_16_64", {"rules": [(16, 64)]}, {"sharpe": 0.3}, path=path)
    log_trial("ewmac_32_128", {"rules": [(32, 128)]}, {"sharpe": 0.1}, path=path)

    trials = read_trials(path)
    assert len(trials) == 2
    assert trials[0]["name"] == "ewmac_16_64"
    assert trials[1]["stats"]["sharpe"] == 0.1


def test_trial_log_keeps_duplicates(tmp_path):
    """Testing the same idea twice is two chances to get lucky, and the deflated Sharpe
    must be told about both. Deduping the log would quietly understate the search."""
    path = tmp_path / "trials.jsonl"
    for _ in range(3):
        log_trial("same", {"x": 1}, {"sharpe": 0.2}, path=path)
    assert len(read_trials(path)) == 3


def test_read_trials_on_missing_file_is_empty(tmp_path):
    assert read_trials(tmp_path / "nope.jsonl") == []
