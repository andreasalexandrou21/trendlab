"""Batch 4: was breadth the problem?

Batch 3 killed EWMAC on crypto and blamed effective breadth of 1.7 against the 30+ that
trend following needs. That is a hypothesis, not a conclusion, and it is testable: run the
identical pipeline on a universe with genuinely different return drivers and see whether
breadth rises and the result follows.

Same engine, same costs, same rules, same validation. Only the universe changes. If the
answer flips, the constraint was capital and market choice. If it does not, the approach is
wrong and no amount of capital fixes it.

Run: python run_batch4.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from trendlab import FRICTIONLESS, CostModel, buy_and_hold, run
from trendlab.backtest import bars_per_year
from trendlab.data import load_cross_asset
from trendlab.portfolio import diversification_multiplier, target_weights
from trendlab.report import comparison_table, tearsheet
from trendlab.signals import combined_forecast
from trendlab.validate import (
    deflated_sharpe_ratio,
    effective_breadth,
    log_trial,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    spa_test,
    walk_forward,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")

CAPITAL = 1_000.0
TARGET_VOL = 0.15
MAX_GROSS = 2.0

# Carver's standard no-trade zone. Pre-specified at one value, NOT swept, so this adds one
# trial per rule set rather than a parameter search the deflated Sharpe would have to pay for.
TRADE_BUFFER = 0.10

# ETFs are cheaper than crypto: tighter spreads, IBKR tiered commissions. Still doubled.
ETF_COSTS = CostModel(commission_bps=3.0, slippage_bps=2.0, multiplier=2.0)

RULE_SETS: dict[str, tuple[tuple[int, int], ...]] = {
    "ewmac 8/32": ((8, 32),),
    "ewmac 16/64": ((16, 64),),
    "ewmac 32/128": ((32, 128),),
    "ewmac 64/256": ((64, 256),),
    "ewmac 8/32+16/64": ((8, 32), (16, 64)),
    "ewmac 16/64+32/128": ((16, 64), (32, 128)),
    "ewmac 32/128+64/256": ((32, 128), (64, 256)),
    "ewmac 16/64+64/256": ((16, 64), (64, 256)),
    "ewmac 3-rule": ((16, 64), (32, 128), (64, 256)),
    "ewmac 4-rule": ((8, 32), (16, 64), (32, 128), (64, 256)),
}


def main() -> None:
    prices = load_cross_asset()
    returns = prices.pct_change()
    universe = prices.notna()

    print(
        f"instruments {prices.shape[1]}, bars {prices.shape[0]}, "
        f"{prices.index[0].date()} to {prices.index[-1].date()}"
    )

    breadth = effective_breadth(returns, lookback=2000)
    idm = diversification_multiplier(returns, universe)
    print(f"effective breadth {breadth:.1f} of {prices.shape[1]} nominal   (crypto was 1.7 of 10)")
    print(f"mean IDM {idm.mean():.2f}   (crypto was 1.21)")
    print()

    net_returns: dict[str, pd.Series] = {}
    gross_returns: dict[str, pd.Series] = {}
    rows = []

    for name, rules in RULE_SETS.items():
        forecast = combined_forecast(prices, rules=rules)
        weights = target_weights(
            prices, forecast, universe=universe, target_vol=TARGET_VOL, max_gross=MAX_GROSS
        )
        net = run(
            prices, weights, costs=ETF_COSTS, initial_capital=CAPITAL, trade_buffer=TRADE_BUFFER
        )
        gross = run(
            prices, weights, costs=FRICTIONLESS, initial_capital=CAPITAL, trade_buffer=TRADE_BUFFER
        )

        first = weights.abs().sum(axis=1).replace(0.0, np.nan).first_valid_index()
        net_returns[name] = net.returns.loc[first:]
        gross_returns[name] = gross.returns.loc[first:]

        log_trial(
            f"cross-asset {name}",
            {
                "rules": [list(r) for r in rules],
                "target_vol": TARGET_VOL,
                "universe": "etf15",
                "trade_buffer": TRADE_BUFFER,
            },
            net.stats(),
        )
        ppy = net.periods_per_year
        rows.append(
            (
                name,
                sharpe_ratio(gross_returns[name]) * np.sqrt(ppy),
                sharpe_ratio(net_returns[name]) * np.sqrt(ppy),
                net.stats()["ann_cost_drag"],
            )
        )

    print(f"{'rule set':<24}{'gross SR':>10}{'net SR':>9}{'cost/yr':>10}")
    print("-" * 53)
    for name, g, n, c in rows:
        print(f"{name:<24}{g:>10.2f}{n:>9.2f}{c:>10.1%}")
    print()

    best_name = max(net_returns, key=lambda k: sharpe_ratio(net_returns[k]))
    ppy = bars_per_year(prices.index)  # ~252 for an exchange calendar, not 365

    for label, streams in (("GROSS", gross_returns), ("NET", net_returns)):
        sharpes = {k: sharpe_ratio(v) for k, v in streams.items()}
        best = max(sharpes, key=lambda k: sharpes[k])
        dsr, sr0 = deflated_sharpe_ratio(streams[best], list(sharpes.values()))
        print(f"{label}: best = {best}")
        print(f"  observed Sharpe        {sharpes[best] * np.sqrt(ppy):>7.2f} annualised")
        print(f"  expected max from luck {sr0 * np.sqrt(ppy):>7.2f} annualised")
        print(f"  PSR  (vs zero)         {probabilistic_sharpe_ratio(streams[best]):>7.3f}")
        print(f"  DSR  (vs the search)   {dsr:>7.3f}   {'PASS' if dsr > 0.95 else 'FAIL (<0.95)'}")
        print()

    wf = walk_forward(net_returns, train_years=5.0, test_years=2.0)
    print(wf.summary())
    print()

    spy = buy_and_hold(prices, symbols=["SPY"], costs=ETF_COSTS, initial_capital=CAPITAL)
    cash = pd.Series(0.0, index=next(iter(net_returns.values())).index)
    for bench_name, bench in (("cash", cash), ("hold SPY", spy.returns)):
        p = spa_test(bench, net_returns, reps=2000)
        verdict = "beats it" if p["pvalue_consistent"] < 0.05 else "does NOT beat it"
        print(
            f"SPA vs {bench_name:<9} consistent p = {p['pvalue_consistent']:.3f}  -> best {verdict}"
        )
    print()

    best_weights = target_weights(
        prices,
        combined_forecast(prices, rules=RULE_SETS[best_name]),
        universe=universe,
        target_vol=TARGET_VOL,
        max_gross=MAX_GROSS,
    )
    results = {
        "hold SPY": spy,
        "60/40 hold": buy_and_hold(
            prices, symbols=["SPY", "IEF"], costs=ETF_COSTS, initial_capital=CAPITAL
        ),
        f"trend ({best_name})": run(
            prices,
            best_weights,
            costs=ETF_COSTS,
            initial_capital=CAPITAL,
            trade_buffer=TRADE_BUFFER,
        ),
        "trend, no buffer": run(prices, best_weights, costs=ETF_COSTS, initial_capital=CAPITAL),
        "trend + 25% stop": run(
            prices,
            best_weights,
            costs=ETF_COSTS,
            initial_capital=CAPITAL,
            drawdown_stop=0.25,
            trade_buffer=TRADE_BUFFER,
        ),
    }
    print(comparison_table(results))
    print()
    path = tearsheet(
        results,
        path="reports/cross_asset.png",
        title="trendlab Batch 4: cross-asset EWMAC",
    )
    print(f"tearsheet -> {path}")


if __name__ == "__main__":
    main()
