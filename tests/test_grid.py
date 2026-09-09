"""Grid trading checks.

The point of these tests is not that the code runs. It is to make the strategy's risk
profile a matter of record: inventory that only grows against you, no exit, and a win rate
that stays beautiful while the account bleeds.
"""

import numpy as np
import pandas as pd
import pytest

from trendlab import FRICTIONLESS, CostModel, run
from trendlab.grid import GridConfig, grid_weights, inventory_stats


def _index(n):
    return pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")


def _ranging(n=1500, amplitude=0.15, period=60, seed=0):
    """Oscillates around a flat mean. The market grid bots are sold for."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    px = 100 * (1 + amplitude * np.sin(2 * np.pi * t / period)) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"A": px}, index=_index(n))


def _downtrend(n=1500, total_drop=0.6, seed=1):
    """A grinding decline. The market that ends grid bots."""
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, np.log(1 - total_drop), n)
    px = 100 * np.exp(drift + np.cumsum(rng.normal(0, 0.004, n)))
    return pd.DataFrame({"A": px}, index=_index(n))


def test_config_validation():
    for kwargs in ({"levels": 1}, {"spacing_pct": 0}, {"position_per_level": -1}):
        with pytest.raises(ValueError):
            GridConfig(**kwargs)


def test_grid_maxes_out_early_then_rides_the_rest_of_the_decline():
    """The defining behaviour, stated precisely.

    Not simply "exposure rises as price falls". The ladder only spans levels * spacing, so
    a 20-rung grid at 1% spacing is fully deployed after a 20% fall. In a 60% decline it is
    holding its maximum position for the remaining 40%, with nothing left to average into
    and no rule that closes anything.

    Correlation is the wrong measure for this and reads weak (-0.23) precisely BECAUSE the
    exposure saturates: it is a clipped step function, not a linear response.
    """
    prices = _downtrend(total_drop=0.6)
    cfg = GridConfig(levels=20, spacing_pct=0.01, position_per_level=0.05)
    w = grid_weights(prices, "A", cfg)

    exposure = w["A"]
    full = cfg.levels * cfg.position_per_level

    maxed_from = exposure[exposure >= full - 1e-9].index[0]
    price_at_max = prices.loc[maxed_from, "A"]
    start_price = prices["A"].iloc[:250].mean()

    # Fully deployed after roughly the ladder's own span, far from the bottom.
    assert price_at_max / start_price > 0.70
    # And still fully deployed at the end, having ridden the whole rest of the fall down.
    assert exposure.iloc[-1] == pytest.approx(full, abs=1e-9)
    assert prices["A"].iloc[-1] / price_at_max < 0.75


def test_grid_never_exits_on_the_way_down():
    """There is no stop. Falling through every rung leaves you holding the whole ladder.

    This is the risk in one assertion: the strategy's only response to being wrong is to
    increase the position.
    """
    prices = _downtrend(total_drop=0.7)
    cfg = GridConfig(levels=20, spacing_pct=0.01, position_per_level=0.05)
    w = grid_weights(prices, "A", cfg)

    tail = w["A"].iloc[-200:]
    assert (tail > 0).all()
    assert tail.iloc[-1] == pytest.approx(cfg.levels * cfg.position_per_level, abs=1e-9)


def test_grid_profits_in_a_range_and_loses_in_a_trend():
    """Both halves matter. It genuinely works in the market it is sold for."""
    ranging = run(_ranging(), grid_weights(_ranging(), "A"), costs=FRICTIONLESS)
    falling = run(_downtrend(), grid_weights(_downtrend(), "A"), costs=FRICTIONLESS)

    assert ranging.equity.iloc[-1] > ranging.initial_capital
    assert falling.equity.iloc[-1] < falling.initial_capital


def test_grid_drawdown_is_far_worse_in_a_trend_than_a_range():
    ranging = run(_ranging(), grid_weights(_ranging(), "A"), costs=CostModel())
    falling = run(_downtrend(), grid_weights(_downtrend(), "A"), costs=CostModel())

    assert falling.stats()["max_drawdown"] < ranging.stats()["max_drawdown"] - 0.05


def test_high_win_rate_coexists_with_losing_money():
    """The number every grid vendor quotes, and why it is meaningless.

    Almost every bar the position is held through is a small gain, because the strategy
    only ever adds on weakness into a falling market. The losses sit in the open inventory,
    so the proportion of up-days can look excellent while the account is deeply down.
    """
    prices = _downtrend(total_drop=0.6)
    result = run(prices, grid_weights(prices, "A"), costs=FRICTIONLESS)

    invested = result.returns[result.weights["A"].shift(1).fillna(0) != 0]
    win_rate = float((invested > 0).mean())

    assert win_rate > 0.40
    assert result.equity.iloc[-1] < result.initial_capital


def test_inventory_stats_expose_the_hidden_risk():
    prices = _downtrend()
    w = grid_weights(prices, "A", GridConfig(levels=20, spacing_pct=0.01))
    stats = inventory_stats(w, "A")

    assert stats["max_exposure"] > 0.5
    assert stats["pct_bars_invested"] > 0.5
    assert stats["max_consecutive_invested"] > 200  # underwater for years, not days


def test_grid_is_causal():
    """The anchor is a trailing mean, so future prices cannot move a past rung."""
    prices = _ranging(n=2000)
    full = grid_weights(prices, "A").iloc[:1200]
    partial = grid_weights(prices.iloc[:1200], "A")

    assert np.allclose(full.to_numpy(), partial.to_numpy(), atol=1e-12)
