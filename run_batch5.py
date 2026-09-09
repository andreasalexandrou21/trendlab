"""Batch 5: pairs trading with a Kalman-filtered hedge ratio.

The canonical statistical-arbitrage strategy, run through the same instrument as everything
else: same engine, same doubled costs, same walk-forward, same deflated Sharpe.

Two versions are deliberately run side by side:

  ECONOMIC  - 27 pairs chosen because they have a reason to co-move (share classes, a metal
              and its miners, adjacent points on one curve, countries with the same export
              basket). 27 tests.
  MINED     - every pair in the universe screened for cointegration. ~1,100 tests, of which
              ~55 come back significant on noise alone at a 5% level.

Running both is the point. The gap between them is the size of the data-snooping effect,
measured rather than asserted, on a strategy where mining is the standard practice.

Run: python run_batch5.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from trendlab import CostModel, buy_and_hold, run
from trendlab.backtest import bars_per_year
from trendlab.data import load_pairs_universe
from trendlab.pairs import pair_weights, screen_pairs
from trendlab.report import comparison_table, tearsheet
from trendlab.validate import (
    deflated_sharpe_ratio,
    log_trial,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    spa_test,
    walk_forward,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")

CAPITAL = 1_000.0
ETF_COSTS = CostModel(commission_bps=3.0, slippage_bps=2.0, multiplier=2.0)

ENTRY_Z = 1.0  # Chan's canonical threshold
EXIT_Z = 0.0
BURN_IN = 250
MAX_PAIRS = 10
GROSS_PER_PAIR = 1.0  # scaled down later by the number of live pairs

# Pairs with an economic reason to co-move, written out before looking at any p-value.
ECONOMIC_PAIRS = [
    # near-identical exposures: the positive control. If these do not screen, we are broken.
    ("SPY", "IVV"),
    ("SPY", "VOO"),
    ("IVV", "VOO"),
    ("SPY", "VTI"),
    ("QQQ", "QQQM"),
    ("IWM", "IWO"),
    # a metal and its miners
    ("GLD", "IAU"),
    ("GLD", "SLV"),
    ("GDX", "GDXJ"),
    ("GLD", "GDX"),
    ("SLV", "SIL"),
    # energy complex
    ("XLE", "XOP"),
    ("USO", "BNO"),
    # sector vs its concentrated cousin
    ("XLF", "KRE"),
    ("XLK", "VGT"),
    ("XLV", "IBB"),
    ("XLP", "XLY"),
    # adjacent points on one curve
    ("SHY", "IEF"),
    ("IEF", "TLT"),
    ("AGG", "IEF"),
    ("TIP", "IEF"),
    ("LQD", "HYG"),
    ("HYG", "JNK"),
    # countries with shared drivers; EWA/EWC is Chan's textbook example
    ("EWA", "EWC"),
    ("EEM", "VWO"),
    ("EFA", "VEA"),
    ("EWU", "EWG"),
]


def build_portfolio(prices, pairs, label):
    """Equal-capital allocation across the selected pairs, summed into one weight matrix."""
    if not pairs:
        return None
    scale = GROSS_PER_PAIR / len(pairs)
    total = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for cand in pairs:
        total = total + pair_weights(
            prices,
            cand.y,
            cand.x,
            entry_z=ENTRY_Z,
            exit_z=EXIT_Z,
            burn_in=BURN_IN,
            gross_per_pair=scale,
        )
    logging.getLogger(__name__).info("%s: %d pairs", label, len(pairs))
    return total


def main() -> None:
    prices = load_pairs_universe()
    ann = np.sqrt(bars_per_year(prices.index))
    print(
        f"universe {prices.shape[1]} tickers, {prices.shape[0]} bars, "
        f"{prices.index[0].date()} to {prices.index[-1].date()}\n"
    )

    # --- screen both ways -----------------------------------------------------------------
    economic = screen_pairs(prices, candidates=ECONOMIC_PAIRS, max_pvalue=0.05)
    n_economic_tests = len(ECONOMIC_PAIRS)

    all_cols = list(prices.columns)
    all_combos = [(a, b) for i, a in enumerate(all_cols) for b in all_cols[i + 1 :]]
    mined = screen_pairs(prices, candidates=all_combos, max_pvalue=0.05)
    n_mined_tests = len(all_combos)

    print(f"ECONOMIC screen: {len(economic):>3} pairs pass, from {n_economic_tests} tests")
    print(
        f"MINED    screen: {len(mined):>3} pairs pass, from {n_mined_tests} tests "
        f"(~{int(0.05 * n_mined_tests)} expected from noise alone)\n"
    )

    print("top economic pairs:")
    for cand in economic[:MAX_PAIRS]:
        print(f"  {cand}")
    print()

    # --- every screened pair gets its own backtest, and that IS the trial ------------------
    #
    # The deflated Sharpe needs the dispersion of the results you searched over. Fabricating
    # that distribution from a normal draw, which is the tempting shortcut, makes the number
    # meaningless. So each candidate is actually traded on its own and its Sharpe recorded.
    def per_pair_sharpes(candidates) -> list[float]:
        out = []
        for cand in candidates:
            w = pair_weights(
                prices,
                cand.y,
                cand.x,
                entry_z=ENTRY_Z,
                exit_z=EXIT_Z,
                burn_in=BURN_IN,
                gross_per_pair=1.0,
            )
            r = run(prices, w, costs=ETF_COSTS, initial_capital=CAPITAL)
            sr = sharpe_ratio(r.returns.iloc[BURN_IN:])
            out.append(sr)
            log_trial(
                f"pair {cand.y}/{cand.x}",
                {
                    "entry_z": ENTRY_Z,
                    "exit_z": EXIT_Z,
                    "pvalue": cand.pvalue,
                    "half_life": cand.half_life,
                },
                r.stats(),
            )
        return out

    economic_sharpes = per_pair_sharpes(economic)
    mined_sharpes = per_pair_sharpes(mined)
    print(
        f"single-pair Sharpes, economic: median {np.median(economic_sharpes) * ann:+.2f}, "
        f"best {max(economic_sharpes) * ann:+.2f}"
    )
    print(
        f"single-pair Sharpes, mined:    median {np.median(mined_sharpes) * ann:+.2f}, "
        f"best {max(mined_sharpes) * ann:+.2f}\n"
    )

    # --- backtest both --------------------------------------------------------------------
    results = {}
    streams = {}
    trial_pool = {"pairs, economic": economic_sharpes, "pairs, mined": mined_sharpes}

    for label, selected, n_tests in (
        ("pairs, economic", economic[:MAX_PAIRS], n_economic_tests),
        ("pairs, mined", mined[:MAX_PAIRS], n_mined_tests),
    ):
        weights = build_portfolio(prices, selected, label)
        if weights is None:
            print(f"{label}: no pairs passed, skipping")
            continue
        result = run(prices, weights, costs=ETF_COSTS, initial_capital=CAPITAL)
        results[label] = result
        streams[label] = result.returns.iloc[BURN_IN:]
        log_trial(
            label,
            {
                "entry_z": ENTRY_Z,
                "exit_z": EXIT_Z,
                "n_pairs": len(selected),
                "n_tests": n_tests,
                "pairs": [f"{c.y}/{c.x}" for c in selected],
            },
            result.stats(),
        )

    results["hold SPY"] = buy_and_hold(
        prices, symbols=["SPY"], costs=ETF_COSTS, initial_capital=CAPITAL
    )
    results["60/40 SPY+IEF"] = buy_and_hold(
        prices, symbols=["SPY", "IEF"], costs=ETF_COSTS, initial_capital=CAPITAL
    )

    print(comparison_table(results))
    print()

    # --- validate -------------------------------------------------------------------------
    for label, stream in streams.items():
        n_tests = n_economic_tests if "economic" in label else n_mined_tests
        # Measured trial dispersion: the Sharpe every screened pair actually produced.
        trial_sharpes = trial_pool[label]
        print(f"{label}")
        print(f"  observed Sharpe        {sharpe_ratio(stream) * ann:>7.2f} annualised")
        print(f"  PSR  (vs zero)         {probabilistic_sharpe_ratio(stream):>7.3f}")
        if len(trial_sharpes) < 2:
            # The economic screen can legitimately return a single pair, and one trial has no
            # dispersion to estimate. That is not a failure to report: with one trial there was
            # no search, so there is nothing for the deflation to remove.
            print(
                f"  DSR  (vs the search)       n/a   "
                f"({len(trial_sharpes)} pair traded, {n_tests} screened; needs 2+ to deflate)"
            )
        else:
            dsr, sr0 = deflated_sharpe_ratio(stream, trial_sharpes)
            print(
                f"  expected max from luck {sr0 * ann:>7.2f}  "
                f"({len(trial_sharpes)} pairs traded, {n_tests} screened)"
            )
            print(
                f"  DSR  (vs the search)   {dsr:>7.3f}   "
                f"{'PASS' if dsr > 0.95 else 'FAIL (<0.95)'}"
            )
        print()

    if len(streams) >= 2:
        wf = walk_forward(streams, train_years=5.0, test_years=2.0)
        print(wf.summary())
        print()

    sixty = results["60/40 SPY+IEF"].returns
    for label, stream in streams.items():
        common = stream.index.intersection(sixty.index)
        p = spa_test(sixty.loc[common], {label: stream.loc[common]}, reps=2000)
        verdict = "beats it" if p["pvalue_consistent"] < 0.05 else "does NOT beat it"
        print(f"SPA {label:<18} vs 60/40:  p = {p['pvalue_consistent']:.3f}  -> {verdict}")

    path = tearsheet(
        results, path="reports/pairs.png", title="trendlab Batch 5: Kalman pairs trading"
    )
    print(f"\ntearsheet -> {path}")


if __name__ == "__main__":
    main()
