"""Engine mechanics: accounting, ruin, gross caps, controls."""

import numpy as np
import pandas as pd
import pytest

from trendlab import FRICTIONLESS, CostModel, buy_and_hold, run


def test_index_and_column_mismatch_is_rejected(random_walk):
    bad = pd.DataFrame(0.0, index=random_walk.index[:-1], columns=random_walk.columns)
    with pytest.raises(ValueError, match="index"):
        run(random_walk, bad)

    wrong_cols = pd.DataFrame(0.0, index=random_walk.index, columns=["X", "Y", "Z"])
    with pytest.raises(ValueError, match="columns"):
        run(random_walk, wrong_cols)


def test_negative_execution_lag_is_rejected(random_walk):
    weights = pd.DataFrame(0.0, index=random_walk.index, columns=random_walk.columns)
    with pytest.raises(ValueError, match="execution_lag"):
        run(random_walk, weights, execution_lag=-1)


def test_position_drift_avoids_phantom_turnover(single_asset):
    """Holding a constant weight through a moving market should trade once, not daily.

    If the engine forgot that positions drift with price, it would bill a rebalance on
    every bar and the cost drag would be pure fiction.
    """
    target = pd.DataFrame(1.0, index=single_asset.index, columns=["A"])
    result = run(single_asset, target, costs=CostModel(), initial_capital=1_000.0)

    entry = result.trade_costs.iloc[1]
    subsequent = result.trade_costs.iloc[2:].sum()
    assert entry > 0
    assert subsequent < entry * 0.05


def test_leverage_can_ruin_the_account():
    """A 60% fall at 5x gross wipes the account out and the engine says so."""
    idx = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    prices = pd.DataFrame({"A": 100.0}, index=idx)
    prices.iloc[10:] = 40.0  # a single 60% gap down

    target = pd.DataFrame(5.0, index=idx, columns=["A"])
    result = run(prices, target, costs=FRICTIONLESS, initial_capital=1_000.0)

    assert result.ruined
    assert result.equity.iloc[-1] == 0.0
    assert result.stats()["max_drawdown"] == pytest.approx(-1.0)


def test_ruin_is_absorbing():
    """Once wiped out, a later rally must not resurrect the account."""
    idx = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC")
    prices = pd.DataFrame({"A": 100.0}, index=idx)
    prices.iloc[10:20] = 20.0
    prices.iloc[20:] = 500.0

    target = pd.DataFrame(5.0, index=idx, columns=["A"])
    result = run(prices, target, costs=FRICTIONLESS, initial_capital=1_000.0)

    assert result.ruined
    assert (result.equity.iloc[20:] == 0.0).all()


def test_max_gross_caps_exposure(random_walk):
    target = pd.DataFrame(2.0, index=random_walk.index, columns=random_walk.columns)  # 6x gross
    result = run(random_walk, target, costs=FRICTIONLESS, max_gross=1.0)

    gross = result.weights.abs().sum(axis=1)
    assert gross.max() <= 1.0 + 1e-12
    assert np.isclose(gross.iloc[-1], 1.0)


def test_max_gross_does_not_inflate_small_positions(random_walk):
    """The cap scales down, never up. A 0.3x book stays 0.3x."""
    target = pd.DataFrame(0.1, index=random_walk.index, columns=random_walk.columns)
    result = run(random_walk, target, costs=FRICTIONLESS, max_gross=1.0)

    assert np.isclose(result.weights.abs().sum(axis=1).iloc[-1], 0.3)


def test_short_positions_make_money_when_price_falls():
    idx = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
    prices = pd.DataFrame({"A": np.linspace(100, 50, 50)}, index=idx)
    target = pd.DataFrame(-1.0, index=idx, columns=["A"])

    result = run(prices, target, costs=FRICTIONLESS, initial_capital=1_000.0)
    assert result.equity.iloc[-1] > 1_000.0


def test_buy_and_hold_starts_equal_weight_then_drifts(random_walk):
    result = buy_and_hold(random_walk, costs=FRICTIONLESS, initial_capital=1_000.0)

    assert np.allclose(result.weights.iloc[1].to_numpy(), 1 / 3, atol=1e-12)
    # Volatile assets must not stay pinned at 1/N; that would mean daily rebalancing.
    assert not np.allclose(result.weights.iloc[-1].to_numpy(), 1 / 3, atol=1e-3)
    assert np.isclose(result.weights.iloc[-1].sum(), 1.0)


def test_buy_and_hold_trades_exactly_once(random_walk):
    """The control must buy on the entry bar and then never trade again.

    Regression test: an earlier version passed a constant 1/N target, which the engine
    correctly rebalanced back to 1/N every single bar. That is a daily-rebalanced
    portfolio wearing a buy-and-hold label, and on 80%-vol assets it invented several
    percent a year of cost drag that a real buy-and-hold investor never pays.
    """
    free = buy_and_hold(random_walk, costs=FRICTIONLESS, initial_capital=1_000.0)
    assert free.traded_notional.iloc[1] > 0
    assert free.traded_notional.iloc[2:].sum() == pytest.approx(0.0, abs=1e-9)

    # With costs the residual is not exactly zero: a fee reduces equity without reducing
    # position value, so the held fraction creeps above the normalised target and the engine
    # trims it. That is real behaviour, not an accounting error, and it must stay negligible.
    priced = buy_and_hold(random_walk, costs=CostModel(), initial_capital=1_000.0)
    assert priced.traded_notional.iloc[2:].sum() < 0.02 * priced.traded_notional.iloc[1]
    assert priced.stats()["ann_cost_drag"] < 0.01


def test_buy_and_hold_waits_for_all_symbols_to_list():
    """A late listing must not be bought retroactively at its debut price."""
    idx = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    prices = pd.DataFrame({"A": np.linspace(100, 120, 20), "B": np.nan}, index=idx)
    prices.loc[idx[10:], "B"] = np.linspace(50, 60, 10)

    result = buy_and_hold(prices, costs=FRICTIONLESS)

    assert (result.weights.iloc[:10].to_numpy() == 0.0).all()
    assert np.allclose(result.weights.iloc[11].to_numpy(), 0.5, atol=1e-12)


def test_turnover_measures_real_trades_not_target_changes(random_walk):
    """Regression test for a metric that lied.

    A constant 1/N target on drifting assets rebalances every bar, so real turnover is
    large. Reading turnover off `weights.diff()` reports ~0 because the *target* never
    moves. That understates cost by an order of magnitude on exactly the strategies
    where cost decides the answer.
    """
    constant = pd.DataFrame(1 / 3, index=random_walk.index, columns=random_walk.columns)
    rebalanced = run(random_walk, constant, costs=FRICTIONLESS)

    assert rebalanced.weights.diff().abs().sum(axis=1).sum() == pytest.approx(1.0, abs=1e-9)
    assert rebalanced.stats()["ann_turnover"] > 1.0

    held = buy_and_hold(random_walk, costs=FRICTIONLESS)
    assert rebalanced.stats()["ann_turnover"] > 20 * held.stats()["ann_turnover"]


def test_stats_are_self_consistent(random_walk):
    target = pd.DataFrame(0.3, index=random_walk.index, columns=random_walk.columns)
    result = run(random_walk, target, costs=CostModel(), initial_capital=1_000.0)
    stats = result.stats()

    assert stats["final_equity"] == pytest.approx(result.equity.iloc[-1])
    assert stats["max_drawdown"] <= 0.0
    assert stats["ann_vol"] >= 0.0
    assert stats["total_trade_costs"] >= 0.0
    assert stats["ruined"] == 0.0


def test_drawdown_stop_flattens_and_stays_flat():
    """The kill switch halts at the threshold and does not resume on its own.

    The live rule needs a human to restart it, so a backtest that auto-resumes would be
    measuring a strategy nobody is running.
    """
    idx = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")
    prices = pd.DataFrame({"A": 100.0}, index=idx)
    prices.iloc[20:40] = 60.0  # -40%
    prices.iloc[40:] = 300.0  # full recovery and then some

    target = pd.DataFrame(1.0, index=idx, columns=["A"])
    stopped = run(prices, target, costs=FRICTIONLESS, drawdown_stop=0.25)
    free = run(prices, target, costs=FRICTIONLESS)

    assert stopped.meta["stopped"]
    assert stopped.meta["stop_date"] is not None
    assert (stopped.weights.iloc[45:].to_numpy() == 0.0).all()
    assert stopped.stats()["max_drawdown"] > -0.45
    # It cost real money here: flat through the rally is the price of the insurance.
    assert stopped.equity.iloc[-1] < free.equity.iloc[-1]


def test_drawdown_stop_does_not_fire_below_threshold():
    idx = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
    prices = pd.DataFrame({"A": 100.0}, index=idx)
    prices.iloc[20:] = 90.0  # -10%, inside a 25% budget

    target = pd.DataFrame(1.0, index=idx, columns=["A"])
    result = run(prices, target, costs=FRICTIONLESS, drawdown_stop=0.25)

    assert not result.meta["stopped"]
    assert result.weights.iloc[-1, 0] == pytest.approx(1.0)


def test_drawdown_stop_validates_its_range(random_walk):
    weights = pd.DataFrame(0.0, index=random_walk.index, columns=random_walk.columns)
    for bad in (0.0, 1.0, -0.1, 5.0):
        with pytest.raises(ValueError, match="drawdown_stop"):
            run(random_walk, weights, drawdown_stop=bad)


def test_bars_per_year_is_inferred_not_assumed():
    """~261 for weekdays-only, ~365 for 24/7, and ~252 once real holidays are removed.

    Hardcoding 365 inflates an exchange-traded strategy's annualised Sharpe by
    sqrt(365/252) = 1.20 for no reason other than a constant.
    """
    from trendlab.backtest import bars_per_year

    weekdays = pd.date_range("2015-01-01", periods=2520, freq="B", tz="UTC")
    daily = pd.date_range("2015-01-01", periods=3650, freq="D", tz="UTC")

    # freq="B" is every weekday with no market holidays, so 261 is the correct answer here.
    assert bars_per_year(weekdays) == pytest.approx(261, rel=0.02)
    assert bars_per_year(daily) == pytest.approx(365, rel=0.02)

    # A real exchange calendar drops ~9 holidays a year and lands near 252.
    holidays = weekdays[~weekdays.isin(weekdays[::29])]
    assert 248 < bars_per_year(holidays) < 256


def test_sharpe_uses_the_inferred_bar_rate():
    """Same return stream on a business-day calendar must not be annualised as if 24/7."""
    rng = np.random.default_rng(31)
    n = 2000
    rets = rng.normal(0.0004, 0.01, n)

    def equity_from(idx):
        prices = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rets))}, index=idx)
        target = pd.DataFrame(1.0, index=idx, columns=["A"])
        return run(prices, target, costs=FRICTIONLESS)

    b = equity_from(pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC"))
    d = equity_from(pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC"))

    ratio = d.stats()["sharpe"] / b.stats()["sharpe"]
    expected = np.sqrt(d.periods_per_year / b.periods_per_year)
    assert ratio == pytest.approx(expected, rel=0.001)
    assert ratio > 1.15  # the constant would have been worth this much free Sharpe


def test_financing_accrues_over_calendar_days_not_bars():
    """A Friday-to-Monday gap is three days of interest, not one.

    Charging one day per bar understates the carry on a levered equity book by ~40%,
    which is exactly the size of the effect that decides whether leverage is viable.
    """
    model = CostModel(
        commission_bps=0.0, slippage_bps=0.0, benchmark_rate=0.02, financing_spread=0.03
    )
    # Thu, Fri, Mon, Tue: the third bar carries a 3-day gap.
    idx = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"], tz="UTC")
    prices = pd.DataFrame({"A": 100.0}, index=idx)
    target = pd.DataFrame(3.0, index=idx, columns=["A"])

    result = run(prices, target, costs=model, initial_capital=1_000.0)

    weekday = result.financing_costs.iloc[1]
    weekend = result.financing_costs.iloc[2]
    assert weekend == pytest.approx(3 * weekday, rel=0.01)


def test_trade_buffer_cuts_turnover_without_abandoning_the_signal(random_walk):
    """A no-trade zone should remove most rebalancing churn while still tracking the signal.

    Without it, a slow signal still turns over enormously because the book chases exact
    volatility targets every single bar. That turnover is an implementation artefact, not
    a property of the strategy, and it is pure cost.
    """
    rng = np.random.default_rng(77)
    slow = pd.DataFrame(
        rng.normal(0, 1, random_walk.shape).cumsum(axis=0) * 0.001 + 0.2,
        index=random_walk.index,
        columns=random_walk.columns,
    )

    naive = run(random_walk, slow, costs=CostModel(), trade_buffer=0.0)
    buffered = run(random_walk, slow, costs=CostModel(), trade_buffer=0.10)

    assert buffered.stats()["ann_turnover"] < 0.5 * naive.stats()["ann_turnover"]
    assert buffered.stats()["ann_cost_drag"] < naive.stats()["ann_cost_drag"]
    # Still following the same signal: positions must stay correlated with the naive book.
    corr = buffered.weights.stack().corr(naive.weights.stack())
    assert corr > 0.9


def test_trade_buffer_still_honours_a_flat_target(random_walk):
    """Going to zero is not optional: the buffer must never keep you in a closed position."""
    target = pd.DataFrame(0.5, index=random_walk.index, columns=random_walk.columns)
    target.iloc[500:] = 0.0

    result = run(random_walk, target, costs=FRICTIONLESS, trade_buffer=0.25)
    assert (result.weights.iloc[502:].to_numpy() == 0.0).all()
