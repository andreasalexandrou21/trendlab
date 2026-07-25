"""The load-bearing suite.

Every one of these tests exists because the corresponding bug is silent: a backtest with
look-ahead bias does not crash, it just prints a number you like.
"""

import numpy as np
import pandas as pd

from trendlab import FRICTIONLESS, run


def test_zero_signal_is_exactly_flat(random_walk):
    """No position, no cost, no change. If this drifts, the accounting is wrong."""
    weights = pd.DataFrame(0.0, index=random_walk.index, columns=random_walk.columns)
    result = run(random_walk, weights, costs=FRICTIONLESS, initial_capital=1000.0)

    assert np.allclose(result.equity.to_numpy(), 1000.0)
    assert result.trade_costs.sum() == 0.0
    assert result.financing_costs.sum() == 0.0


def test_cheating_beats_not_cheating(random_walk):
    """The guard has to be load-bearing.

    The signal is `sign of THIS bar's return`, which is only knowable after the bar closes.
    With the honest lag of 1 it is stale information and earns nothing. With lag=0 the
    engine would be applying it to the very bar it was derived from, i.e. perfect foresight,
    and equity should explode.

    If these two come out similar, the shift in `backtest.run` is not doing its job.
    """
    rets = random_walk.pct_change().fillna(0.0)
    peek = np.sign(rets) / random_walk.shape[1]

    honest = run(random_walk, peek, costs=FRICTIONLESS, execution_lag=1)
    cheating = run(random_walk, peek, costs=FRICTIONLESS, execution_lag=0)

    assert cheating.equity.iloc[-1] > honest.equity.iloc[-1] * 100
    # Perfect foresight cannot lose on any bar.
    assert (cheating.returns.iloc[1:] >= -1e-12).all()


def test_honest_run_has_no_edge_on_a_random_walk(random_walk):
    """Sanity floor: a stale sign signal on a random walk should not print a real Sharpe."""
    rets = random_walk.pct_change().fillna(0.0)
    stale = np.sign(rets) / random_walk.shape[1]
    result = run(random_walk, stale, costs=FRICTIONLESS, execution_lag=1)

    assert abs(result.stats()["sharpe"]) < 1.0


def test_weights_are_shifted_not_the_caller_s_job(random_walk):
    """`weights` on the result is what was actually held, and it lags the target by one bar."""
    target = pd.DataFrame(0.0, index=random_walk.index, columns=random_walk.columns)
    target.iloc[10] = 0.5

    result = run(random_walk, target, costs=FRICTIONLESS)

    assert result.weights.iloc[10].sum() == 0.0
    assert np.isclose(result.weights.iloc[11].sum(), 0.5 * random_walk.shape[1])


def test_missing_price_forces_zero_weight():
    """An instrument that was not trading cannot be held, whatever the signal claimed."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    prices = pd.DataFrame({"A": np.linspace(100, 110, 10), "B": np.nan}, index=idx)
    prices.loc[idx[5:], "B"] = np.linspace(50, 55, 5)

    target = pd.DataFrame(1.0, index=idx, columns=["A", "B"])
    result = run(prices, target, costs=FRICTIONLESS)

    assert (result.weights.loc[idx[:5], "B"] == 0.0).all()
    assert (result.weights.loc[idx[6:], "B"] == 1.0).all()


def test_full_long_reproduces_buy_and_hold_frictionless(single_asset):
    """A constant 100% long, with no costs, must equal the price ratio exactly.

    Offset by the execution lag: you are not in the market for the first bar.
    """
    target = pd.DataFrame(1.0, index=single_asset.index, columns=["A"])
    result = run(single_asset, target, costs=FRICTIONLESS, initial_capital=1000.0)

    expected = 1000.0 * single_asset["A"].iloc[-1] / single_asset["A"].iloc[0]
    assert np.isclose(result.equity.iloc[-1], expected, rtol=1e-9)
