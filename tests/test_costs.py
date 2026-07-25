"""Cost model checks, hand-computed.

The financing tests matter more than the commission ones. Commission drag is visible in a
backtest; financing drag on a flat market is invisible right up until the account is empty.
"""

import numpy as np
import pandas as pd
import pytest

from trendlab import CostModel, run
from trendlab.costs import DAYS_PER_YEAR


def test_trade_cost_is_hand_computable():
    model = CostModel(commission_bps=40.0, slippage_bps=5.0, multiplier=2.0)
    # (40 + 5) bps doubled = 90 bps = 0.9% of traded notional
    assert model.round_trip_bps == 90.0
    assert np.isclose(model.trade_cost(10_000.0), 90.0)


def test_multiplier_scales_trade_cost_linearly():
    single = CostModel(commission_bps=40.0, slippage_bps=5.0, multiplier=1.0)
    double = CostModel(commission_bps=40.0, slippage_bps=5.0, multiplier=2.0)
    assert np.isclose(double.trade_cost(10_000.0), 2 * single.trade_cost(10_000.0))


def test_trade_cost_ignores_sign():
    model = CostModel()
    assert model.trade_cost(-5_000.0) == model.trade_cost(5_000.0)


def test_unlevered_book_pays_no_financing():
    model = CostModel()
    assert model.financing_cost(gross_notional=1_000.0, equity=1_000.0, days=1) == 0.0
    assert model.financing_cost(gross_notional=400.0, equity=1_000.0, days=1) == 0.0


def test_financing_is_charged_on_borrowed_notional():
    """EUR 1,000 equity at 3x gross borrows EUR 2,000. One year at 5% is EUR 100."""
    model = CostModel(benchmark_rate=0.02, financing_spread=0.03)
    assert model.financing_rate == pytest.approx(0.05)

    one_year = model.financing_cost(gross_notional=3_000.0, equity=1_000.0, days=DAYS_PER_YEAR)
    assert one_year == pytest.approx(100.0)

    one_day = model.financing_cost(gross_notional=3_000.0, equity=1_000.0, days=1)
    assert one_day == pytest.approx(2_000.0 * 0.05 / DAYS_PER_YEAR)


def test_engine_charges_the_hand_computed_financing_on_bar_one(flat_prices):
    """This is the number that answers the leverage question.

    EUR 1,000 of equity held at 3x gross on a flat market: borrowed notional is EUR 2,000,
    financed at 5%, so bar one costs 2000 * 0.05 / 365.
    """
    model = CostModel(
        commission_bps=0.0, slippage_bps=0.0, benchmark_rate=0.02, financing_spread=0.03
    )
    target = pd.DataFrame({"A": 3.0, "B": 0.0}, index=flat_prices.index)

    result = run(flat_prices, target, costs=model, initial_capital=1_000.0)

    assert result.financing_costs.iloc[1] == pytest.approx(2_000.0 * 0.05 / DAYS_PER_YEAR)


def test_leverage_bleeds_a_flat_market_dry(flat_prices):
    """Prices never move. The unlevered book is untouched, the levered one bleeds.

    Nothing about the strategy changed. This is the carry cost alone.
    """
    model = CostModel(
        commission_bps=0.0, slippage_bps=0.0, benchmark_rate=0.02, financing_spread=0.03
    )

    unlevered = run(
        flat_prices,
        pd.DataFrame({"A": 1.0, "B": 0.0}, index=flat_prices.index),
        costs=model,
        initial_capital=1_000.0,
    )
    levered = run(
        flat_prices,
        pd.DataFrame({"A": 20.0, "B": 0.0}, index=flat_prices.index),
        costs=model,
        initial_capital=1_000.0,
    )

    assert unlevered.equity.iloc[-1] == pytest.approx(1_000.0)
    assert levered.equity.iloc[-1] < 1_000.0
    assert levered.financing_costs.sum() > 20 * unlevered.financing_costs.sum() + 1.0


def test_first_trade_cost_is_exact(flat_prices):
    """Entering a 100% long from flat trades exactly one unit of equity."""
    model = CostModel(
        commission_bps=40.0,
        slippage_bps=5.0,
        multiplier=2.0,
        benchmark_rate=0.0,
        financing_spread=0.0,
    )
    target = pd.DataFrame({"A": 1.0, "B": 0.0}, index=flat_prices.index)

    result = run(flat_prices, target, costs=model, initial_capital=1_000.0)

    assert result.trade_costs.iloc[0] == 0.0
    assert result.trade_costs.iloc[1] == pytest.approx(1_000.0 * 90.0 / 1e4)
