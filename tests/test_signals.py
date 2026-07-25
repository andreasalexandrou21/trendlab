"""Signal and sizing checks.

The theme: every estimated quantity here (volatility, forecast scalar, diversification
multiplier, universe rank) must be computable from the past alone. A signal that peeks is
the same bug as a backtester that peeks, just moved one file over.
"""

import numpy as np
import pandas as pd
import pytest

from trendlab.data import point_in_time_universe
from trendlab.portfolio import annualised_volatility, diversification_multiplier, target_weights
from trendlab.signals import (
    FORECAST_CAP,
    combined_forecast,
    ewmac_forecast,
    raw_ewmac,
)


def _trending(n=800, slope=0.002, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    rets = rng.normal(slope, 0.02, size=n)
    return pd.DataFrame({"A": 100 * np.exp(np.cumsum(rets))}, index=idx)


def test_fast_span_must_be_shorter_than_slow():
    prices = _trending()
    with pytest.raises(ValueError, match="fast span"):
        raw_ewmac(prices, 64, 16)


def test_forecast_signs_a_monotone_trend_correctly():
    """Deterministic ramps, so the answer is not a coin flip.

    Note what this test is NOT: an assertion about the last bar of a noisy uptrend. A random
    series with positive drift routinely ends mid-pullback, where a negative EWMAC is the
    correct reading, not a bug. Testing direction on noise tests the seed.
    """
    idx = pd.date_range("2020-01-01", periods=800, freq="D", tz="UTC")
    up = pd.DataFrame({"A": np.linspace(100, 400, 800)}, index=idx)
    down = pd.DataFrame({"A": np.linspace(400, 100, 800)}, index=idx)

    assert ewmac_forecast(up, 16, 64).iloc[-1, 0] > 0
    assert ewmac_forecast(down, 16, 64).iloc[-1, 0] < 0


def test_forecast_leans_long_on_average_through_an_uptrend():
    """Over a noisy but persistently rising series, the average forecast must be positive."""
    up = _trending(n=1200, slope=0.004)
    mean_forecast = ewmac_forecast(up, 16, 64).mean().iloc[0]
    assert mean_forecast > 1.0


def test_forecast_respects_the_cap():
    prices = _trending(slope=0.02, seed=3)  # violent, sustained trend
    forecast = combined_forecast(prices)
    assert forecast.abs().max().max() <= FORECAST_CAP + 1e-9


def test_forecast_is_scale_invariant():
    """A price series and the same series denominated 1000x differently must forecast alike.

    This is what dividing by price-unit volatility buys, and it is why one set of rules can
    span a EUR 60,000 BTC and a EUR 0.16 ADA.
    """
    prices = _trending()
    scaled = prices * 1000.0

    a = combined_forecast(prices).dropna()
    b = combined_forecast(scaled).dropna()
    assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-9)


def test_signals_are_causal():
    """Appending future bars must not change any forecast already computed.

    The strongest available test for look-ahead in a signal: recompute on a longer series
    and require the overlapping prefix to be identical. An expanding scalar passes; a
    full-sample normalisation would fail here.
    """
    prices = _trending(n=1400)
    prefix = prices.iloc[:1000]

    full = combined_forecast(prices).iloc[:1000]
    partial = combined_forecast(prefix)

    mask = full.notna() & partial.notna()
    assert mask.to_numpy().sum() > 100
    assert np.allclose(full[mask].dropna().to_numpy(), partial[mask].dropna().to_numpy())


def test_volatility_estimate_is_causal():
    prices = _trending(n=1400)
    full = annualised_volatility(prices).iloc[:1000]
    partial = annualised_volatility(prices.iloc[:1000])
    mask = full.notna() & partial.notna()
    assert np.allclose(full[mask].dropna().to_numpy(), partial[mask].dropna().to_numpy())


def test_idm_is_one_for_perfectly_correlated_assets():
    """Identical assets offer no diversification, so no scaling up is allowed."""
    idx = pd.date_range("2020-01-01", periods=600, freq="D", tz="UTC")
    rng = np.random.default_rng(11)
    series = pd.Series(rng.normal(0, 0.02, 600), index=idx)
    returns = pd.DataFrame({"A": series, "B": series, "C": series})
    active = pd.DataFrame(True, index=idx, columns=["A", "B", "C"])

    idm = diversification_multiplier(returns, active)
    assert idm.iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_idm_rises_for_uncorrelated_assets_and_respects_the_cap():
    idx = pd.date_range("2020-01-01", periods=600, freq="D", tz="UTC")
    rng = np.random.default_rng(12)
    returns = pd.DataFrame(rng.normal(0, 0.02, (600, 3)), index=idx, columns=["A", "B", "C"])
    active = pd.DataFrame(True, index=idx, columns=["A", "B", "C"])

    idm = diversification_multiplier(returns, active)
    assert 1.4 < idm.iloc[-1] <= 2.5  # sqrt(3) = 1.73 for truly independent


def test_target_weights_respect_max_gross():
    prices = _trending(n=700)
    prices["B"] = prices["A"] * 1.01
    forecast = combined_forecast(prices)

    weights = target_weights(prices, forecast, target_vol=0.15, max_gross=1.5)
    assert weights.abs().sum(axis=1).max() <= 1.5 + 1e-9


def test_low_volatility_instrument_gets_a_bigger_weight():
    """Volatility targeting means a calm asset earns more capital for the same conviction."""
    idx = pd.date_range("2020-01-01", periods=700, freq="D", tz="UTC")
    rng = np.random.default_rng(5)
    calm = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.005, 700)))
    wild = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.050, 700)))
    prices = pd.DataFrame({"CALM": calm, "WILD": wild}, index=idx)

    forecast = pd.DataFrame(10.0, index=idx, columns=["CALM", "WILD"])  # equal conviction
    weights = target_weights(prices, forecast, max_gross=99.0, use_idm=False)

    assert weights["CALM"].iloc[-1] > weights["WILD"].iloc[-1] * 3


def test_universe_selection_uses_only_the_past():
    """A coin that becomes liquid late must not be selected early."""
    idx = pd.date_range("2020-01-01", periods=800, freq="D", tz="UTC")
    dv = pd.DataFrame(
        {"OLD": 1e9, "LATE": 1.0},
        index=idx,
    )
    dv.loc[idx[600:], "LATE"] = 1e12  # becomes the most liquid thing on the venue, late

    mask = point_in_time_universe(dv, n=1, lookback=90, min_history=250)

    assert mask.loc[idx[500], "OLD"]
    assert not mask.loc[idx[500], "LATE"]
    assert mask.loc[idx[-1], "LATE"]


def test_universe_requires_minimum_history():
    idx = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC")
    dv = pd.DataFrame({"A": 1e9}, index=idx)
    mask = point_in_time_universe(dv, n=5, lookback=90, min_history=250)

    assert not mask.iloc[:249].to_numpy().any()
    assert mask.iloc[-1].to_numpy().all()
