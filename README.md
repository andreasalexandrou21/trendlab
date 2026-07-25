# trendlab

A systematic trend-following research system for crypto majors. Daily frequency, honest costs,
and a backtester built before any strategy, so it has no incentive to flatter one.

**Status: Batch 4. The trend signal is statistically real. It is still not worth trading.**

Batch 3 blamed the crypto failure on effective breadth of 1.7. Batch 4 tested that on a
15-instrument cross-asset universe, breadth doubled to 3.3, and the gross deflated Sharpe went
from 0.926 (fail) to **0.977 (pass)**. The hypothesis was right.

Net of costs it is Sharpe 0.33, against **0.85 for holding two ETFs and never touching them.**
See [the cross-asset result](#result-2-cross-asset-the-signal-is-real-and-still-loses-to-6040).

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

## Result 2, cross-asset: the signal is real, and still loses to 60/40

`python run_batch4.py`. 15 ETFs spanning US/international/EM equity, REITs, treasuries,
credit, gold, silver, commodities, oil and the dollar. 2007-01 to 2026-07, so it contains the
GFC, COVID and the 2022 rate shock. IBKR-style costs (3bp commission + 2bp slippage), doubled.

**Breadth roughly doubled, exactly as predicted:**

```
effective breadth  3.3 of 15 nominal   (crypto was 1.7 of 10)
mean IDM           1.89                (crypto was 1.21)
```

**And that flipped the gross result:**

| | Crypto (Batch 3) | Cross-asset (Batch 4) |
|---|---|---|
| Best gross Sharpe | 0.73 | 0.64 |
| **Gross DSR** | 0.926 FAIL | **0.977 PASS** |
| Best net Sharpe | 0.07 | 0.33 |
| Net DSR | 0.430 FAIL | 0.382 FAIL |
| Walk-forward OOS | -0.42 | -0.12 |

The gross deflated Sharpe passing is the first statistically defensible finding in this
repo. A slow cross-asset trend signal exists, it survives correction for having tested ten
rule sets, and it is consistent with the published literature.

### Turnover was an implementation bug, and fixing it was not enough

A 128-day signal was turning over 20.5x a year, because the book chased exact volatility
targets every single bar. Adding a no-trade buffer (`run(..., trade_buffer=...)`) fixes it:

| Buffer | Turnover/yr | Cost/yr | Gross SR | Net SR |
|---|---|---|---|---|
| 0.00 | 20.5 | 5.4% | 0.60 | 0.12 |
| 0.25 | 11.6 | 4.5% | 0.62 | 0.21 |
| 1.00 | 5.7 | 3.7% | 0.66 | 0.24 |
| **2.00** | **3.1** | **2.2%** | **0.67** | **0.33** |

Cost drag fell by 60% and net Sharpe nearly tripled. It still was not enough.

### The row that ends it

| Strategy | CAGR | Vol | Sharpe | MaxDD | Turnover | Cost/yr |
|---|---|---|---|---|---|---|
| trend, best config | 2.1% | 6.8% | 0.33 | -21.3% | 3.1x | 2.2% |
| trend, at scale (2% on idle cash) | 2.4% | 6.8% | 0.39 | -21.3% | 3.0x | 1.9% |
| hold SPY | 10.8% | 19.6% | 0.62 | -55.2% | 0.1x | 0.0% |
| **60/40 SPY + IEF** | **8.2%** | **9.8%** | **0.85** | **-22.1%** | **0.1x** | **0.0%** |

```
SPA, trend vs 60/40:  consistent p = 0.491   -> does not beat it
correlation of the two return streams: 0.41  -> not even a useful diversifier
```

Buying two ETFs and never touching them again produces a higher Sharpe, four times the
return, the same drawdown, zero turnover, zero cost, and zero code.

Two honest caveats, both of which make the strategy look better than it is, not worse:
- At buffer 2.0 realised volatility is 6.8% against a 15% target, so the book runs at half
  its intended risk. Scaling to 15% would roughly lift the return to ~5% and the drawdown to
  ~-45%. Still worse than 60/40 on both axes.
- Idle cash earns nothing in the base case, which is correct for a small account (IBKR pays
  no interest below USD 10,000) but understates the strategy at scale. The 2% row models it.

### Conclusion

Breadth was a genuine constraint and fixing it produced a genuine, statistically significant
gross signal. The signal is simply too small: ~6.7% a year gross at 11% volatility, against a
2-5% cost floor that no amount of buffering removes. What is left after costs does not beat a
passive two-fund portfolio and cannot be distinguished from it statistically.

**The instrument works. The strategy does not. Those are different sentences, and only the
first one was ever worth building.**

## Verdict on crypto: the signal is real gross, gone net, and unfindable in real time

`python run_batch3.py`. Ten EWMAC rule sets, fixed before looking at any result and written
out explicitly so the trial count cannot quietly grow. Every evaluation is appended to
[`trials.jsonl`](trials.jsonl), which is committed: a deflated Sharpe quoted without a
verifiable trial count is just a Sharpe with extra decimals.

| Rule set | Gross SR | Net SR | Cost/yr |
|---|---|---|---|
| ewmac 8/32 | **0.73** | **0.07** | 10.9% |
| ewmac 8/32+16/64 | 0.62 | 0.03 | 9.3% |
| ewmac 4-rule | 0.54 | 0.00 | 7.0% |
| ewmac 16/64 | 0.49 | -0.03 | 7.9% |
| ewmac 16/64+32/128 | 0.45 | -0.06 | 6.9% |
| ewmac 16/64+64/256 | 0.43 | -0.06 | 6.2% |
| ewmac 3-rule (batch 2) | 0.39 | -0.10 | 6.2% |
| ewmac 32/128 | 0.37 | -0.10 | 6.3% |
| ewmac 32/128+64/256 | 0.30 | -0.17 | 5.8% |
| ewmac 64/256 | 0.25 | -0.20 | 5.5% |

The columns tell the story on their own. **Gross Sharpe rises monotonically as the rules get
faster, and so does the cost, and the cost wins every time.** Faster trend signals genuinely
carry more information in crypto. None of it survives a 90bp round trip.

### Deflated Sharpe: the best of ten is not a discovery

| | Gross | Net |
|---|---|---|
| Best candidate | ewmac 8/32 | ewmac 8/32 |
| Observed Sharpe (annualised) | 0.73 | 0.07 |
| Expected max from luck alone (n=10) | 0.23 | 0.13 |
| PSR vs zero | 0.982 | 0.582 |
| **DSR vs the search** | **0.926** | **0.430** |

Gross, the result *nearly* clears the 0.95 bar. Read honestly: there is probably a weak real
trend effect in crypto, which is what the published literature says too, and this sample is
not strong enough to prove it alone.

Net, the observed Sharpe of 0.07 is **below** the 0.13 that searching ten no-edge variants
would have produced anyway. A DSR of 0.43 means the result is worse than the search predicts.
There is nothing here.

### Walk-forward: choosing actively made it worse

Expanding training window, one-year out-of-sample blocks, rule set chosen on training data
only and never refit on the test window.

| Train end | Test window | Chosen | IS SR | OOS SR |
|---|---|---|---|---|
| 2021-04-21 | 2021-04 to 2022-04 | ewmac 4-rule | 0.83 | **-1.07** |
| 2022-04-21 | 2022-04 to 2023-04 | ewmac 8/32 | 0.38 | -0.18 |
| 2023-04-21 | 2023-04 to 2024-04 | ewmac 8/32 | 0.26 | +0.61 |
| 2024-04-20 | 2024-04 to 2025-04 | ewmac 8/32 | 0.31 | **-1.12** |
| 2025-04-20 | 2025-04 to 2026-04 | ewmac 8/32 | 0.10 | -0.30 |

```
mean in-sample Sharpe (selected)    0.38
pooled out-of-sample Sharpe        -0.42
shrinkage (the cost of choosing)    0.80
```

Four of five folds negative. The in-sample winner was systematically the wrong pick, and
selection destroyed 0.80 of Sharpe. This is the number that answers "would I have found this
in real time", and the answer is no.

### Superior Predictive Ability

```
SPA vs cash      consistent p = 0.549  -> best candidate does NOT beat it
SPA vs hold BTC  consistent p = 0.976  -> best candidate does NOT beat it
```

Hansen's test, 2000 bootstrap replications, accounting for the fact that ten candidates were
searched. The best of them does not beat holding cash, let alone holding Bitcoin.

### Conclusion

Four independent methods agree, which is the only reason to believe any of them:

1. Net Sharpe is negative for eight of ten rule sets and ~zero for the other two.
2. Deflated Sharpe of the best net candidate is 0.43, below the coin-flip line.
3. Walk-forward out-of-sample Sharpe is -0.42 against +0.38 in sample.
4. SPA cannot distinguish the best candidate from cash.

Root cause is unchanged from Batch 2 and is structural, not fixable by tuning: **effective
breadth of 1.7 against 30+ needed.** Trend following is paid for diversification. Sixty crypto
tickers are one factor, so there is nothing to be paid for, and a 6-11% annual cost is charged
for finding that out.

**Batch 4 is cancelled.** Not writing the live execution layer is the correct outcome. The
deliverable of this project is the instrument, the method, and a negative result that is
actually trustworthy, which is worth more than an equity curve that is not.

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
- **Batch 3 (done):** walk-forward, deflated Sharpe, trial logging, Hansen SPA, effective
  breadth. Verdict: no tradeable edge. See above.
- **Batch 4 (done):** cross-asset universe, calendar-day financing, inferred bars-per-year,
  no-trade buffer, cash-interest credit. Gross signal confirmed real (DSR 0.977). Net result
  still loses to 60/40. Live execution not built, because there is nothing worth executing.

Negative results get published here alongside positive ones. A strategy that fails to beat
buy-and-hold net of doubled costs is a finding, not a gap.

## Reading

Robert Carver, *Systematic Trading* and *Advanced Futures Trading Strategies*, for the
framework and for honest Sharpe expectations. Bailey and Lopez de Prado on the deflated Sharpe
ratio and the probability of backtest overfitting, for why most published backtests are noise.
Harvey, Liu and Zhu (RFS 2016) for why the significance hurdle is t > 3.

## Licence

MIT.
