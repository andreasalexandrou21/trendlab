"""Validation: deciding whether a backtest number means anything.

Three questions, three tools.

1. *Would I have found this in real time?* Walk-forward. Choose the rule set on data available
   at the time, evaluate on data that came after, never refit on the test window.
2. *Is this the best of many tries dressed up as a discovery?* The deflated Sharpe ratio,
   which penalises the observed Sharpe by how many variants were tested, how dispersed their
   results were, how short the sample is, and how fat the return distribution's tails are.
3. *Does the best variant actually beat the benchmark, allowing for the search?* Hansen's
   Superior Predictive Ability test.

The trial log exists because the deflated Sharpe needs an honest count of attempts, and human
memory of "how many things did I try" is biased downward in exactly the direction that makes
results look better. Every evaluation appends to `trials.jsonl` automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PERIODS_PER_YEAR = 365
EULER_MASCHERONI = 0.5772156649015329
TRIALS_PATH = Path(__file__).resolve().parents[2] / "trials.jsonl"


# --------------------------------------------------------------------------- trial log


def log_trial(
    name: str,
    params: dict,
    result_stats: dict,
    path: Path = TRIALS_PATH,
) -> None:
    """Append one evaluation to the trial log. Never overwrites, never dedupes.

    Duplicates are the point: testing the same idea twice is two chances to get lucky, and
    the deflated Sharpe should be told about both.
    """
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "name": name,
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
        "stats": {k: float(v) for k, v in result_stats.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def read_trials(path: Path = TRIALS_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------- Sharpe statistics


def sharpe_ratio(returns: pd.Series, annualise: bool = False) -> float:
    """Per-observation Sharpe by default.

    The deflated Sharpe formulas below are defined on per-observation Sharpe together with
    the observation count. Feeding them an annualised number silently inflates significance
    by a factor of sqrt(365), which is the single easiest way to get this wrong.
    """
    sd = returns.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    sr = float(returns.mean() / sd)
    return sr * np.sqrt(PERIODS_PER_YEAR) if annualise else sr


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true Sharpe > benchmark), adjusting for skew, kurtosis and sample length.

    Bailey and Lopez de Prado. Both Sharpe figures are per-observation. Negative skew and
    fat tails, which is what a short-volatility-shaped return stream has, reduce confidence:
    the same Sharpe means less when the distribution can produce a large sudden loss.
    """
    returns = returns.dropna()
    n = len(returns)
    if n < 3:
        return float("nan")

    sr = sharpe_ratio(returns)
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))  # non-excess

    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    if denom <= 0:
        return float("nan")

    z = (sr - benchmark_sr) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_dispersion: float) -> float:
    """Sharpe you should expect from the best of `n_trials` strategies that have NO edge.

    This is the number people forget. Test enough zero-edge variants and one of them prints
    a good Sharpe with certainty. The expected maximum grows with the number of trials and
    with how much the trial results vary, so a wide search over noisy variants sets a high
    bar before anything counts as a discovery.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0

    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sharpe_dispersion * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b))


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_sharpes: list[float] | np.ndarray,
) -> tuple[float, float]:
    """Returns (DSR, expected max Sharpe under the null), both per-observation.

    DSR is the probability that the strategy's true Sharpe exceeds what the best of this
    many no-edge trials would have produced anyway. Below 0.95 the result does not clear the
    usual bar; below 0.5 the result is worse than the search alone would predict.
    """
    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    trial_sharpes = trial_sharpes[np.isfinite(trial_sharpes)]
    n_trials = len(trial_sharpes)
    if n_trials < 2:
        raise ValueError("need at least 2 trials to estimate Sharpe dispersion")

    dispersion = float(trial_sharpes.std(ddof=1))
    sr0 = expected_max_sharpe(n_trials, dispersion)
    return probabilistic_sharpe_ratio(returns, benchmark_sr=sr0), sr0


# ------------------------------------------------------------------------ walk-forward


@dataclass
class Fold:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen: str
    train_sharpe: float
    test_sharpe: float


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    folds: list[Fold]

    @property
    def oos_equity(self) -> pd.Series:
        return (1.0 + self.oos_returns).cumprod()

    def summary(self) -> str:
        lines = [
            f"{'train end':<12}{'test window':<26}{'chosen':<26}{'IS SR':>8}{'OOS SR':>9}",
            "-" * 81,
        ]
        for f in self.folds:
            window = f"{f.test_start.date()} to {f.test_end.date()}"
            lines.append(
                f"{str(f.train_end.date()):<12}{window:<26}{f.chosen:<26}"
                f"{f.train_sharpe * np.sqrt(PERIODS_PER_YEAR):>8.2f}"
                f"{f.test_sharpe * np.sqrt(PERIODS_PER_YEAR):>9.2f}"
            )
        is_mean = np.mean([f.train_sharpe for f in self.folds]) * np.sqrt(PERIODS_PER_YEAR)
        oos = sharpe_ratio(self.oos_returns, annualise=True)
        lines.append("-" * 81)
        lines.append(f"{'mean in-sample Sharpe (selected)':<64}{is_mean:>17.2f}")
        lines.append(f"{'pooled out-of-sample Sharpe':<64}{oos:>17.2f}")
        lines.append(f"{'shrinkage (the cost of choosing)':<64}{is_mean - oos:>17.2f}")
        return "\n".join(lines)


def walk_forward(
    candidate_returns: dict[str, pd.Series],
    train_years: float = 3.0,
    test_years: float = 1.0,
) -> WalkForwardResult:
    """Pick the best candidate on the training window, score it on the next window.

    Takes precomputed per-candidate return streams rather than recomputing strategies per
    fold. That is safe here only because every signal in this project is causal: a return on
    date d never depends on data after d, so slicing a full-sample run into folds gives the
    same numbers as running each fold separately, at a fraction of the cost.

    The selection is the thing being tested. Any single candidate's full-sample Sharpe is a
    number you could only have known at the end; this measures what picking would have done.
    """
    if not candidate_returns:
        raise ValueError("no candidates")

    frame = pd.DataFrame(candidate_returns).dropna(how="all")
    index = frame.index
    train_bars = int(train_years * PERIODS_PER_YEAR)
    test_bars = int(test_years * PERIODS_PER_YEAR)
    if len(index) < train_bars + test_bars:
        raise ValueError("not enough history for one fold")

    folds: list[Fold] = []
    pieces: list[pd.Series] = []

    start = train_bars
    while start + test_bars <= len(index):
        train = frame.iloc[:start]
        test = frame.iloc[start : start + test_bars]

        train_sharpes = {c: sharpe_ratio(train[c].dropna()) for c in frame.columns}
        chosen = max(train_sharpes, key=lambda c: train_sharpes[c])

        folds.append(
            Fold(
                train_end=index[start - 1],
                test_start=test.index[0],
                test_end=test.index[-1],
                chosen=chosen,
                train_sharpe=train_sharpes[chosen],
                test_sharpe=sharpe_ratio(test[chosen].dropna()),
            )
        )
        pieces.append(test[chosen])
        start += test_bars

    return WalkForwardResult(oos_returns=pd.concat(pieces), folds=folds)


# --------------------------------------------------------------- superior predictive ability


def spa_test(
    benchmark_returns: pd.Series,
    candidate_returns: dict[str, pd.Series],
    reps: int = 1000,
    block_size: int = 10,
    seed: int = 20260725,
) -> dict[str, float]:
    """Hansen's SPA. Null: no candidate genuinely beats the benchmark.

    arch's SPA is written in terms of losses, where smaller is better, so returns are
    negated on the way in. A consistent p-value above 0.05 means the best candidate's
    outperformance is within what data snooping alone would produce.
    """
    from arch.bootstrap import SPA

    frame = pd.DataFrame(candidate_returns)
    aligned = pd.concat([benchmark_returns.rename("__benchmark__"), frame], axis=1).dropna()
    if aligned.empty:
        raise ValueError("no overlapping observations between benchmark and candidates")

    losses_benchmark = -aligned["__benchmark__"].to_numpy()
    losses_models = -aligned.drop(columns="__benchmark__").to_numpy()

    spa = SPA(losses_benchmark, losses_models, reps=reps, block_size=block_size, seed=seed)
    spa.compute()
    return {
        "pvalue_lower": float(spa.pvalues["lower"]),
        "pvalue_consistent": float(spa.pvalues["consistent"]),
        "pvalue_upper": float(spa.pvalues["upper"]),
    }


# ----------------------------------------------------------------------- effective breadth


def effective_breadth(returns: pd.DataFrame, lookback: int = 250) -> float:
    """How many independent bets a book of correlated instruments really contains.

    1 / (w' C w) for equal weights: the squared diversification multiplier. Ten instruments
    that all track one factor give a number near one, and trend following is paid precisely
    for the breadth this measures.
    """
    corr = returns.tail(lookback).corr().to_numpy(dtype=float)
    corr = corr[~np.isnan(corr).all(axis=1)][:, ~np.isnan(corr).all(axis=0)]
    n = corr.shape[0]
    if n == 0:
        return 0.0
    corr = np.nan_to_num(corr, nan=0.0)
    w = np.full(n, 1.0 / n)
    var = float(w @ corr @ w)
    return 1.0 / var if var > 0 else float(n)
