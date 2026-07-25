# trendlab

A systematic trend-following research system for crypto majors. Daily frequency, honest costs,
and a backtester built before any strategy, so it has no incentive to flatter one.

**Status: Batch 2 of 4. The strategy is built and it loses money.** Details below. That is a
result, not a gap, and it is reported here in the same place a good result would have gone.

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

## Result: EWMAC trend following on crypto majors does not work after costs

`python run_batch2.py`. Sixty candidate instruments, top ten by trailing 90-day dollar volume
selected point-in-time, 2017-08 to 2026-07. EWMAC 16/64, 32/128, 64/256, equal weighted,
volatility targeted at 15% annualised. Commissions and financing modelled and then doubled.

| Strategy | Final EUR | CAGR | Vol | Sharpe | MaxDD | Turn/yr | Cost/yr |
|---|---|---|---|---|---|---|---|
| hold BTC (control) | **14,767** | +35.1% | 67.3% | 0.79 | -83.2% | 0.1 | 0.1% |
| trend, point-in-time universe | **821** | -2.2% | 13.6% | -0.09 | -41.5% | 6.8 | 6.2% |
| trend + 25% kill switch | 957 | -0.5% | 10.5% | 0.01 | -25.1% | 3.3 | 3.0% |
| trend, survivor universe | 866 | -1.6% | 14.7% | -0.04 | -46.6% | 6.1 | 5.5% |
| trend, 2x risk | 557 | -6.3% | 27.2% | -0.10 | -70.2% | 14.0 | 12.6% |
| trend, 4x risk | 133 | -20.2% | 54.5% | -0.14 | -95.1% | 29.4 | 27.1% |
| **trend, zero costs** | 1,425 | +4.0% | 13.6% | **0.36** | -23.5% | 6.9 | 0.0% |

Read the first and last rows together. **There is a faint real signal: gross Sharpe 0.36,
+4.0% a year before costs.** And costs are 6.2% a year. The edge is real and it is smaller
than the toll.

Note what was *not* done: nothing was fitted. The three span pairs were fixed in advance,
rule weights are equal, and the forecast scalar and diversification multiplier are estimated
on expanding windows. There is no parameter that was tuned to make this number look good,
which is exactly why the number is believable.

### Why: effective breadth is 1.5, not 10

```
mean IDM: 1.21  ->  effective breadth ~1.5 of 10 nominal
```

Trend following is a diversification strategy. It earns its Sharpe by taking many small,
weakly correlated bets. Carver's benchmark is 30 or more genuinely distinct instruments.

Crypto is one factor wearing sixty tickers. The measured diversification multiplier is 1.21,
implying roughly **1.5 independent bets**. Holding ten coins here buys about as much
diversification as holding one and a half. The strategy is structurally unable to earn the
thing trend following is paid for, and no amount of parameter tuning fixes that.

### Leverage makes it monotonically worse

2x risk turns -2.2% into -6.3%. 4x turns it into -20.2% and a 95% drawdown. Leverage
multiplies a negative expected return *and* scales turnover linearly, so the cost drag climbs
from 6.2% to 27.1% a year. There is no leverage setting that rescues a strategy whose gross
edge is below its costs. This is the same arithmetic as the BTC table below, arrived at from
the opposite direction.

### The kill switch worked, and that is not the same as helping

The 25% stop fired on 2023-03-05 and, by design, never restarted. It capped the drawdown at
-25.1% versus -41.5% and left the account slightly better off. It did its job. It also means
the strategy has been switched off since March 2023, which is a fact about the strategy, not
an endorsement of the stop.

### Survivorship bias, measured

Same rules, universe frozen to today's ten majors: -1.6% a year versus -2.2% point-in-time.
So the bias is worth about **0.6% a year** here. Smaller than expected, real, and in the
predicted direction. Worth knowing that it is a modest effect on this dataset rather than
assuming it either way.

### What this does not yet establish

No walk-forward, no deflated Sharpe, no trial count. Batch 3 adds those. The result above is
a single configuration, and a single configuration is a data point, not a conclusion.

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

1. **Survivorship bias: mostly fixed, measured at 0.6%/yr.** `data.point_in_time_universe`
   now ranks 60 candidates by trailing dollar volume, so selection uses only what was known
   at the time. Residual bias remains: coins delisted so long ago that the exchange serves no
   bars cannot be ranked at all. The candidate pool includes fallen majors that still trade
   (EOS, TRX, XLM, IOTA, NEO, ZEC, DASH, WAVES) to shrink that gap.
2. **Research venue is not the execution venue.** Binance has the deepest USDT history
   (2017-08) but withdrew from the EU on 1 July 2026 under MiCA, so it can never be where
   orders go. Measured depth: binance 2017-08, bybit 2021-07, kraken 721 bars only (its OHLC
   endpoint ignores `since`). The paper phase must re-validate on the execution venue's own bars.
3. **Short sample and near-zero breadth.** Nine years, one asset class. Effective breadth
   measured at ~1.5 against 10 nominal instruments. This is the finding that most likely
   sinks the whole approach, and no parameter choice addresses it.
4. **No validation layer yet.** No walk-forward, no deflated Sharpe, no trial counting. Until
   Batch 3 exists, no number from this repo should be believed, including the ones above.

## Roadmap

- **Batch 1 (done):** data layer, cost model, engine, mutation-verified test suite.
- **Batch 2 (done):** EWMAC forecasts, volatility targeting, kill switch, point-in-time
  universe, effective-breadth measurement. Result: negative after costs.
- **Batch 3:** walk-forward, deflated Sharpe ratio, automatic trial logging, Hansen SPA.
  The question it answers is whether the gross 0.36 Sharpe is real or noise, since a
  strategy that only works at zero cost is worth knowing about but not worth trading.
- **Batch 4:** paper execution, but only if anything survives Batch 3. On current evidence
  it will not, and stopping is the correct outcome rather than a failure.

Negative results get published here alongside positive ones. A strategy that fails to beat
buy-and-hold net of doubled costs is a finding, not a gap.

## Reading

Robert Carver, *Systematic Trading* and *Advanced Futures Trading Strategies*, for the
framework and for honest Sharpe expectations. Bailey and Lopez de Prado on the deflated Sharpe
ratio and the probability of backtest overfitting, for why most published backtests are noise.
Harvey, Liu and Zhu (RFS 2016) for why the significance hurdle is t > 3.

## Licence

MIT.
