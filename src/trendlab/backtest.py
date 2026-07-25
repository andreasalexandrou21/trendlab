"""Daily backtest engine.

The single most important property of this module: **the caller cannot introduce look-ahead
bias by accident.** `target_weights.loc[t]` means "the weight I decided on, using information
up to and including bar t's close". The engine applies it to bar `t + execution_lag`. The
shift happens here, once, and never in strategy code.

`execution_lag=0` is deliberately reachable so that `tests/test_lookahead.py` can prove the
guard is load-bearing: if cheating does not measurably beat not-cheating, the shift is broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel

PERIODS_PER_YEAR = 365  # daily bars, 24/7 market


@dataclass
class BacktestResult:
    equity: pd.Series
    weights: pd.DataFrame
    trade_costs: pd.Series
    financing_costs: pd.Series
    traded_notional: pd.Series
    initial_capital: float
    ruined: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)

    @property
    def drawdown(self) -> pd.Series:
        peak = self.equity.cummax()
        return self.equity / peak - 1.0

    @property
    def turnover(self) -> pd.Series:
        """Fraction of equity actually traded each bar.

        Measured from the trades the engine really executed, NOT from the change in target
        weights. Those two differ whenever positions drift with price: a constant 1/N target
        on volatile assets looks like zero turnover but rebalances every single bar. Reading
        turnover off the target weights understates real trading, and therefore real costs,
        by an order of magnitude on exactly the strategies where it matters most.
        """
        return (self.traded_notional / self.equity.shift(1)).fillna(0.0)

    def stats(self) -> dict[str, float]:
        r = self.returns
        years = len(r) / PERIODS_PER_YEAR
        final = float(self.equity.iloc[-1])
        cagr = (
            (final / self.initial_capital) ** (1 / years) - 1 if years > 0 and final > 0 else -1.0
        )
        vol = float(r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
        sharpe = (
            float(r.mean() / r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if r.std(ddof=1) else 0.0
        )
        return {
            "final_equity": final,
            "cagr": cagr,
            "ann_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": float(self.drawdown.min()),
            "ann_turnover": float(self.turnover.sum() / years) if years > 0 else 0.0,
            "total_trade_costs": float(self.trade_costs.sum()),
            "total_financing_costs": float(self.financing_costs.sum()),
            # Costs as a share of the equity that was actually at risk when they were paid.
            # Dividing cumulative costs by *initial* capital is meaningless once the book has
            # compounded: it reports 563% for a strategy whose real drag was a few percent a year.
            "ann_cost_drag": float(
                ((self.trade_costs + self.financing_costs) / self.equity.shift(1)).fillna(0.0).sum()
                / years
            )
            if years > 0
            else 0.0,
            "ruined": float(self.ruined),
        }

    def __str__(self) -> str:
        s = self.stats()
        return (
            f"equity {self.initial_capital:,.0f} -> {s['final_equity']:,.0f} | "
            f"CAGR {s['cagr']:+.1%} | vol {s['ann_vol']:.1%} | Sharpe {s['sharpe']:.2f} | "
            f"maxDD {s['max_drawdown']:.1%} | turnover {s['ann_turnover']:.1f}x/yr | "
            f"cost drag {s['ann_cost_drag']:.1%}/yr" + (" | RUINED" if self.ruined else "")
        )


def run(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    costs: CostModel | None = None,
    initial_capital: float = 1_000.0,
    execution_lag: int = 1,
    max_gross: float | None = None,
) -> BacktestResult:
    """Run a daily backtest.

    Args:
        prices: close prices, index = UTC timestamps, columns = symbols. NaN means the
            instrument was not trading; its weight is forced to zero on those bars.
        target_weights: desired fraction of equity per instrument, decided on bar t's close.
            Signed (negative = short). May sum to more than 1.0 (leverage), which is what
            makes `costs.financing_cost` bite.
        costs: cost model. Defaults to the pessimistic `CostModel()`.
        initial_capital: starting equity.
        execution_lag: bars between decision and fill. 1 is the honest default. 0 is
            cheating and exists only so tests can prove the shift matters.
        max_gross: optional cap on sum of absolute weights, applied before execution.

    Returns:
        BacktestResult with the equity path, the weights actually held, and the two cost
        streams kept separate.
    """
    costs = costs or CostModel()

    if not prices.index.equals(target_weights.index):
        raise ValueError("prices and target_weights must share an index")
    if list(prices.columns) != list(target_weights.columns):
        raise ValueError("prices and target_weights must share columns")
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0")

    rets = prices.pct_change()

    # The shift. This is the load-bearing line of the whole project.
    held = target_weights.shift(execution_lag)

    # An instrument with no price cannot be held, whatever the signal said.
    held = held.where(prices.notna(), 0.0).fillna(0.0)

    if max_gross is not None:
        gross = held.abs().sum(axis=1)
        scale = (max_gross / gross).clip(upper=1.0).replace([np.inf, -np.inf], 1.0).fillna(1.0)
        held = held.mul(scale, axis=0)

    weights_arr = held.to_numpy(dtype=float)
    returns_arr = rets.to_numpy(dtype=float)
    returns_arr = np.nan_to_num(returns_arr, nan=0.0, posinf=0.0, neginf=0.0)

    n_bars, n_assets = weights_arr.shape
    equity = np.empty(n_bars, dtype=float)
    trade_cost_arr = np.zeros(n_bars, dtype=float)
    fin_cost_arr = np.zeros(n_bars, dtype=float)
    traded_arr = np.zeros(n_bars, dtype=float)
    equity[0] = initial_capital

    drift = np.zeros(n_assets, dtype=float)
    ruined = False

    for t in range(1, n_bars):
        prev_equity = equity[t - 1]
        if ruined:
            equity[t] = 0.0
            continue

        target = weights_arr[t]

        # Rebalance from wherever price drift left us to the new target.
        traded_notional = float(np.abs(target - drift).sum()) * prev_equity
        tc = float(costs.trade_cost(traded_notional))

        # Financing accrues on borrowed notional for the bar we are about to hold.
        gross_notional = float(np.abs(target).sum()) * prev_equity
        fc = float(costs.financing_cost(gross_notional, prev_equity, days=1.0))

        pnl = float(target @ returns_arr[t]) * prev_equity
        new_equity = prev_equity + pnl - tc - fc

        trade_cost_arr[t] = tc
        fin_cost_arr[t] = fc
        traded_arr[t] = traded_notional

        if new_equity <= 0.0:
            equity[t] = 0.0
            ruined = True
            drift = np.zeros(n_assets, dtype=float)
            continue

        equity[t] = new_equity
        # Positions drift with price, so next bar only pays to trade the difference.
        drift = target * (1.0 + returns_arr[t]) * prev_equity / new_equity

    idx = prices.index
    return BacktestResult(
        equity=pd.Series(equity, index=idx, name="equity"),
        weights=held,
        trade_costs=pd.Series(trade_cost_arr, index=idx, name="trade_costs"),
        financing_costs=pd.Series(fin_cost_arr, index=idx, name="financing_costs"),
        traded_notional=pd.Series(traded_arr, index=idx, name="traded_notional"),
        initial_capital=initial_capital,
        ruined=ruined,
        meta={"execution_lag": execution_lag, "max_gross": max_gross},
    )


def buy_and_hold(
    prices: pd.DataFrame,
    symbols: list[str] | None = None,
    costs: CostModel | None = None,
    initial_capital: float = 1_000.0,
) -> BacktestResult:
    """True buy-and-hold control: buy once, never trade again.

    The target weights returned to the engine are the *drifted* weights, so the engine
    trades on the entry bar and nothing thereafter. A constant 1/N target would instead be
    a daily-rebalanced portfolio, which is a completely different (and far more expensive)
    strategy that people routinely mislabel as buy-and-hold.

    Starts from the first bar on which every requested symbol is trading, so a late listing
    cannot be retroactively bought at its debut price.

    Every strategy variant has to beat this net of doubled costs, or it is a negative
    result and gets written down as one.
    """
    cols = symbols or list(prices.columns)
    sub = prices[cols]
    live_from = sub.dropna(how="any").index
    if len(live_from) == 0:
        raise ValueError(f"no bar where all of {cols} are trading simultaneously")
    start = live_from[0]

    growth = sub.loc[start:] / sub.loc[start]
    drifted = growth.div(growth.sum(axis=1), axis=0)  # equal initial stake, then pure drift

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.loc[start:, cols] = drifted
    return run(prices, weights, costs=costs, initial_capital=initial_capital)
