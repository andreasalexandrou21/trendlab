"""Batch 2 experiment: does EWMAC trend following beat buying and holding?

Run: python run_batch2.py

Deliberately a flat script, not a CLI. Every run is the same run, so there is no hidden
parameter surface to wander around in. When Batch 3 adds trial logging, this is what gets
logged.
"""

from __future__ import annotations

import logging

import pandas as pd

from trendlab import CostModel, buy_and_hold, run
from trendlab.data import CANDIDATES, dollar_volume, load_frames, point_in_time_universe
from trendlab.portfolio import diversification_multiplier, target_weights
from trendlab.report import comparison_table, tearsheet
from trendlab.signals import combined_forecast

logging.basicConfig(level=logging.WARNING, format="%(message)s")

CAPITAL = 1_000.0
TARGET_VOL = 0.15
MAX_GROSS = 2.0
DRAWDOWN_STOP = 0.25


def main() -> None:
    frames = load_frames(CANDIDATES, since="2017-01-01")
    prices = pd.DataFrame({s: f["close"] for s, f in frames.items()}).sort_index()
    dv = dollar_volume(frames).reindex_like(prices)

    universe = point_in_time_universe(dv, n=10, lookback=90, min_history=250)
    forecast = combined_forecast(prices)
    costs = CostModel()

    print(
        f"candidates {prices.shape[1]}, bars {prices.shape[0]}, "
        f"{prices.index[0].date()} to {prices.index[-1].date()}"
    )
    print(f"mean instruments held: {universe.sum(axis=1).replace(0, float('nan')).mean():.1f}")

    # Effective breadth. Carver wants 30+ genuinely diversified instruments; crypto is one
    # factor wearing sixty tickers, so the honest count is far below the nominal ten.
    active = prices.notna() & forecast.notna() & universe
    idm = diversification_multiplier(prices.pct_change(), active)
    print(
        f"mean IDM: {idm.mean():.2f}  ->  effective breadth ~{idm.mean() ** 2:.1f} "
        f"of {universe.sum(axis=1).max():.0f} nominal"
    )
    print()

    results: dict[str, object] = {}

    # Controls first, so the bar to clear is set before any strategy is looked at.
    results["hold BTC"] = buy_and_hold(
        prices, symbols=["BTC/USDT"], costs=costs, initial_capital=CAPITAL
    )

    weights = target_weights(
        prices, forecast, universe=universe, target_vol=TARGET_VOL, max_gross=MAX_GROSS
    )
    results["trend, PIT universe"] = run(prices, weights, costs=costs, initial_capital=CAPITAL)
    results["trend + 25% stop"] = run(
        prices, weights, costs=costs, initial_capital=CAPITAL, drawdown_stop=DRAWDOWN_STOP
    )

    # The bias measurement: identical rules, universe fixed to today's survivors.
    survivors = [
        s
        for s in [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "LTC/USDT",
            "BCH/USDT",
            "LINK/USDT",
            "ADA/USDT",
            "DOGE/USDT",
            "AVAX/USDT",
        ]
        if s in prices.columns
    ]
    biased_mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    biased_mask[survivors] = True
    biased_weights = target_weights(
        prices, forecast, universe=biased_mask, target_vol=TARGET_VOL, max_gross=MAX_GROSS
    )
    results["trend, survivor universe"] = run(
        prices, biased_weights, costs=costs, initial_capital=CAPITAL
    )

    # Leverage. Note this raises the VOLATILITY TARGET, not the gross cap. Raising the cap
    # does nothing here: at a 15% target against 60-100% instrument volatility the book runs
    # at roughly 0.2x gross, so a 2.0 cap is never binding and "2x gross" would have been a
    # mislabelled copy of the base run.
    for mult, label in ((2.0, "2x risk"), (4.0, "4x risk")):
        results[f"trend, {label}"] = run(
            prices,
            target_weights(
                prices,
                forecast,
                universe=universe,
                target_vol=TARGET_VOL * mult,
                max_gross=MAX_GROSS * mult,
            ),
            costs=costs,
            initial_capital=CAPITAL,
        )

    # Frictionless, to separate "no edge" from "edge eaten by costs".
    results["trend, zero costs"] = run(
        prices,
        weights,
        costs=CostModel(
            commission_bps=0.0,
            slippage_bps=0.0,
            multiplier=0.0,
            benchmark_rate=0.0,
            financing_spread=0.0,
        ),
        initial_capital=CAPITAL,
    )

    print(comparison_table(results))
    print()
    path = tearsheet(results, title="trendlab Batch 2: EWMAC vs buy-and-hold")
    print(f"tearsheet -> {path}")


if __name__ == "__main__":
    main()
