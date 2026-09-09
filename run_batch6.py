"""Batch 6: grid trading, the most widely sold retail bot strategy.

Measured rather than dismissed, on the same instrument as everything else.

The interesting question is not "does it make money" but "when does it stop making money and
how much does it cost when it does". So this runs the same grid through three regimes taken
from real history rather than from a simulator:

    2013-2019  a long grinding equity bull market with shallow pullbacks
    2008-2009  the financial crisis
    2022       the rate shock

Run: python run_batch6.py
"""

from __future__ import annotations

import logging

from trendlab import CostModel, buy_and_hold, run
from trendlab.data import load_cross_asset
from trendlab.grid import GridConfig, grid_weights, inventory_stats
from trendlab.report import comparison_table
from trendlab.validate import log_trial

logging.basicConfig(level=logging.WARNING, format="%(message)s")

CAPITAL = 1_000.0
COSTS = CostModel(commission_bps=3.0, slippage_bps=2.0, multiplier=2.0)
CONFIG = GridConfig(levels=20, spacing_pct=0.01, position_per_level=0.05)

REGIMES = {
    "full sample 2007-2026": ("2007-01-01", "2026-12-31"),
    "quiet bull 2013-2019": ("2013-01-01", "2019-12-31"),
    "crisis 2008-2009": ("2007-06-01", "2009-12-31"),
    "rate shock 2022": ("2021-06-01", "2023-06-30"),
}


def main() -> None:
    prices = load_cross_asset()
    symbol = "SPY"

    print(
        f"grid: {CONFIG.levels} levels, {CONFIG.spacing_pct:.0%} spacing, "
        f"{CONFIG.position_per_level:.0%} per level"
    )
    print(
        f"ladder spans {CONFIG.levels * CONFIG.spacing_pct:.0%} of price, "
        f"max exposure {CONFIG.levels * CONFIG.position_per_level:.0%} of equity\n"
    )

    header = f"{'regime':<24}{'grid CAGR':>11}{'grid maxDD':>12}{'hold CAGR':>11}{'hold maxDD':>12}"
    print(header)
    print("-" * len(header))

    for label, (start, end) in REGIMES.items():
        window = prices.loc[start:end]
        if len(window) < 400:
            continue

        weights = grid_weights(window, symbol, CONFIG)
        grid = run(window, weights, costs=COSTS, initial_capital=CAPITAL)
        hold = buy_and_hold(window, symbols=[symbol], costs=COSTS, initial_capital=CAPITAL)

        g, h = grid.stats(), hold.stats()
        print(
            f"{label:<24}{g['cagr']:>11.1%}{g['max_drawdown']:>12.1%}"
            f"{h['cagr']:>11.1%}{h['max_drawdown']:>12.1%}"
        )
        log_trial(
            f"grid {symbol} {label}",
            {"levels": CONFIG.levels, "spacing_pct": CONFIG.spacing_pct},
            g,
        )

    print()

    # --- the marketing number versus the real one -----------------------------------------
    weights = grid_weights(prices, symbol, CONFIG)
    grid = run(prices, weights, costs=COSTS, initial_capital=CAPITAL)
    held = grid.weights[symbol].shift(1).fillna(0.0)
    invested = grid.returns[held != 0]
    win_rate = float((invested > 0).mean())
    stats = inventory_stats(grid.weights, symbol)

    print("what a vendor would quote, and what it hides")
    print(f"  win rate on invested bars    {win_rate:>8.1%}")
    print(
        f"  final equity                 {grid.stats()['final_equity']:>8,.0f} from {CAPITAL:,.0f}"
    )
    print(f"  max drawdown                 {grid.stats()['max_drawdown']:>8.1%}")
    print(f"  max exposure                 {stats['max_exposure']:>8.0%} of equity")
    print(f"  longest unbroken time held   {stats['max_consecutive_invested']:>8.0f} bars")
    print(f"  share of bars holding stock  {stats['pct_bars_invested']:>8.1%}")
    print()

    results = {
        "grid SPY": grid,
        "hold SPY": buy_and_hold(prices, symbols=["SPY"], costs=COSTS, initial_capital=CAPITAL),
        "60/40": buy_and_hold(prices, symbols=["SPY", "IEF"], costs=COSTS, initial_capital=CAPITAL),
    }
    print(comparison_table(results))


if __name__ == "__main__":
    main()
