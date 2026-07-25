"""EWMAC: exponentially weighted moving average crossover forecasts.

Carver's construction, chosen because it has the smallest overfitting surface of anything
with a published record. Two spans per rule, no thresholds, no optimisation, and the
economic story (under-reaction to news plus positive-feedback flows from stops and
risk-parity rebalancing) is stated in advance rather than reverse-engineered from a backtest.

Everything here is causal. Volatility, the forecast scalar and the diversification multiplier
are all estimated from data up to and including bar t, and `backtest.run` then applies the
result to bar t+1. Nothing in this module shifts anything: that is the engine's job, once.

A note on the forecast scalar. Carver publishes fixed scalars per rule (EWMAC16,64 -> 3.75,
and so on) estimated on futures. Hard-coding them here would import a constant fitted on a
different asset class, and estimating them on the full crypto sample would be in-sample
fitting. So the scalar is estimated on an *expanding* window: at every date it uses only
history available then. It converges to a stable value and costs nothing in look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Carver's target average absolute forecast. Purely a scaling convention: a forecast of 10
#: means "average conviction", 20 means "twice that", and the cap sits at 20.
TARGET_ABS_FORECAST = 10.0
FORECAST_CAP = 20.0

#: Span pairs, slow enough that costs stay a small share of returns at daily frequency.
#: Faster pairs (2,8 / 4,16) are omitted deliberately: they are where retail trend following
#: gets eaten alive by spread, and adding them would be adding trials for a known-bad reason.
DEFAULT_RULES: tuple[tuple[int, int], ...] = ((16, 64), (32, 128), (64, 256))

VOL_SPAN = 32
MIN_VOL_PERIODS = 32
# Burn-in for the expanding estimates. Kept modest because it stacks: the slowest rule
# needs `slow` bars before it produces anything, then the scalar needs this many more, then
# the diversification multiplier needs this many again. At 250 that was 750 bars, two whole
# years of a nine-year sample thrown away before the first trade. The scalar is a scaling
# convention rather than an edge parameter, and it is pooled across instruments, so 100
# dates across ten instruments is a thousand observations and plenty.
MIN_SCALAR_PERIODS = 100


def price_volatility(prices: pd.DataFrame, span: int = VOL_SPAN) -> pd.DataFrame:
    """Volatility in price units: sigma of returns times price.

    EWMAC's raw crossover is a price difference, so dividing by a price-unit volatility is
    what makes the forecast comparable across a EUR 60,000 BTC and a EUR 0.16 ADA.
    """
    returns = prices.pct_change()
    return_vol = returns.ewm(span=span, min_periods=MIN_VOL_PERIODS).std()
    return (return_vol * prices).replace(0.0, np.nan)


def raw_ewmac(prices: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Scale-free crossover: (fast EWMA minus slow EWMA) divided by price volatility."""
    if fast >= slow:
        raise ValueError(f"fast span must be shorter than slow span, got {fast} and {slow}")
    crossover = (
        prices.ewm(span=fast, min_periods=fast).mean()
        - prices.ewm(span=slow, min_periods=slow).mean()
    )
    return crossover / price_volatility(prices)


def _expanding_scalar(raw: pd.DataFrame) -> pd.Series:
    """Scalar that maps the average absolute raw forecast onto TARGET_ABS_FORECAST.

    Pooled across instruments (one scalar per rule, not per instrument, which would be a
    per-instrument fitted parameter) and expanding through time, so it is causal.
    """
    pooled = raw.abs().mean(axis=1)
    mean_abs = pooled.expanding(min_periods=MIN_SCALAR_PERIODS).mean()
    return (TARGET_ABS_FORECAST / mean_abs).replace([np.inf, -np.inf], np.nan)


def ewmac_forecast(prices: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """A single scaled, capped EWMAC forecast in Carver units."""
    raw = raw_ewmac(prices, fast, slow)
    scaled = raw.mul(_expanding_scalar(raw), axis=0)
    return scaled.clip(-FORECAST_CAP, FORECAST_CAP)


def combined_forecast(
    prices: pd.DataFrame,
    rules: tuple[tuple[int, int], ...] = DEFAULT_RULES,
) -> pd.DataFrame:
    """Equal-weighted average of the rule forecasts, rescaled and capped.

    Equal weights are deliberate. Optimising rule weights on the sample would be fitting
    three more parameters to noise, and Carver's own result is that equal weighting is
    within noise of optimised weights for correlated trend rules.

    Averaging correlated forecasts shrinks the average absolute value below the target, so
    a forecast diversification multiplier restores it. That multiplier is also estimated on
    an expanding window and capped, for the same causality reason as the scalar.
    """
    if not rules:
        raise ValueError("at least one rule is required")

    forecasts = [ewmac_forecast(prices, fast, slow) for fast, slow in rules]
    combined = sum(forecasts) / len(forecasts)

    pooled = combined.abs().mean(axis=1)
    mean_abs = pooled.expanding(min_periods=MIN_SCALAR_PERIODS).mean()
    fdm = (TARGET_ABS_FORECAST / mean_abs).clip(upper=2.5).replace([np.inf, -np.inf], np.nan)

    return combined.mul(fdm, axis=0).clip(-FORECAST_CAP, FORECAST_CAP)
