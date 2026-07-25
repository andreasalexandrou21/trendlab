# trendlab

A systematic trend-following research system for crypto majors. Daily frequency, honest costs,
and a backtester built before any strategy, so it has no incentive to flatter one.

**Status: Batch 1 of 4. There is no strategy yet, and no result to report.** What exists is the
measuring instrument and the evidence that it measures correctly.

## Why the engine comes first

A backtester written after you already have a strategy you like will quietly be built to
flatter it. So this repo builds the instrument first, proves it cannot lie in the four specific
ways that matter, and only then points it at a signal.

The four ways a backtest lies silently:

| Failure | Guard | Test |
|---|---|---|
| Look-ahead bias | The decision-to-fill shift lives inside `backtest.run`, never in strategy code | `test_cheating_beats_not_cheating` |
| Phantom or missing turnover | Turnover is measured from trades actually executed, not from target-weight changes | `test_turnover_measures_real_trades_not_target_changes` |
| Financing ignored | Charged on borrowed notional every bar, as a first-class term | `test_engine_charges_the_hand_computed_financing_on_bar_one` |
| Ruin papered over | Equity floors at zero and stays there | `test_ruin_is_absorbing` |

Each guard is verified by mutation: break the line, confirm the suite goes red.

```
shift removed            -> 3 tests fail
drift accounting removed -> 1 test fails
financing on equity      -> 3 tests fail
ruin not absorbing       -> 2 tests fail
```

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
pytest && ruff check .
```

## Use

```python
from trendlab import CostModel, buy_and_hold, run
from trendlab.data import load

prices = load(since="2017-01-01")  # cached to data/*.parquet
result = run(prices, target_weights, costs=CostModel())
print(result)
```

`target_weights.loc[t]` means "the weight I decided using information up to and including bar
t's close". The engine applies it to bar `t + 1`. Strategy code never shifts anything.

## What the engine says about leverage

Constant leverage on BTC/USDT, 2017-08 to 2026-07, with commissions and financing modelled
and then doubled. Bitcoin rose roughly fourteen-fold over this window.

| Exposure | EUR 1,000 becomes | Max drawdown | Turnover | Cost drag |
|---|---|---|---|---|
| 1x | **14,767** | -83% | 0.1x/yr | 0.1%/yr |
| 2x | **389** | -99.5% | 17.7x/yr | 21%/yr |
| 3x | **0** (ruined) | -100% | 19.3x/yr | 20%/yr |
| 5x | **0** (ruined) | -100% | 49.3x/yr | 45%/yr |

Three separate forces, none of them bad luck:

1. **Volatility drag.** Constant leverage on a 67%-vol asset compounds against you even when
   the asset finishes far higher.
2. **Financing on notional.** 2x on EUR 1,000 borrows EUR 1,000 and pays interest on it daily,
   whether or not the position makes money.
3. **Rebalancing cost.** Holding constant 2x through 67% volatility requires trading nearly 18
   times your equity per year. That alone is 21% a year in fees.

The asset went up fourteen times. Two times leverage lost 61% of the account. Three times lost
all of it.

## Known biases, unfixed

Stated here rather than discovered later.

1. **Survivorship bias in the universe.** `data.UNIVERSE` is today's ten majors, selected
   because they survived. The 2018 major list also held EOS, TRX, XLM, IOTA and NEO. Every
   number produced on this universe is an upper bound, not an estimate. Fix is a point-in-time
   universe from rolling dollar volume. Batch 2.
2. **Research venue is not the execution venue.** Binance has the deepest USDT history
   (2017-08) but withdrew from the EU on 1 July 2026 under MiCA, so it can never be where
   orders go. Measured depth: binance 2017-08, bybit 2021-07, kraken 721 bars only (its OHLC
   endpoint ignores `since`). The paper phase must re-validate on the execution venue's own bars.
3. **Short sample.** Nine years, one asset class, dominated by a single factor. Every
   instrument here is largely a BTC beta, so effective breadth is far below the nominal ten.
   Measuring it is a Batch 3 deliverable.
4. **No validation layer yet.** No walk-forward, no deflated Sharpe, no trial counting. Until
   Batch 3 exists, no number from this repo should be believed, including the ones above.

## Roadmap

- **Batch 1 (done):** data layer, cost model, engine, mutation-verified test suite.
- **Batch 2:** EWMAC forecast family, volatility targeting at 15% annualised, half-Kelly cap,
  kill switch at -25% from high-water mark, point-in-time universe.
- **Batch 3:** walk-forward, deflated Sharpe ratio, automatic trial logging, Hansen SPA,
  effective breadth. Everything from Batch 2 re-run through it.
- **Batch 4:** paper execution, but only if anything survives Batch 3.

Negative results get published here alongside positive ones. A strategy that fails to beat
buy-and-hold net of doubled costs is a finding, not a gap.

## Reading

Robert Carver, *Systematic Trading* and *Advanced Futures Trading Strategies*, for the
framework and for honest Sharpe expectations. Bailey and Lopez de Prado on the deflated Sharpe
ratio and the probability of backtest overfitting, for why most published backtests are noise.
Harvey, Liu and Zhu (RFS 2016) for why the significance hurdle is t > 3.

## Licence

MIT.
