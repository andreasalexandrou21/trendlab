"""Position sizing: turn forecasts into weights at a controlled level of risk.

The sizing rule is volatility targeting, which is the single highest-value risk overlay in
systematic trading and the reason a diversified trend book survives assets that routinely
move 80% a year.

    w_i = (forecast_i / 10) * (target_vol / vol_i) * instrument_weight * IDM

Read left to right: conviction, scaled so that a fixed risk budget is spent regardless of how
volatile the instrument currently is, split across instruments, then multiplied back up
because a diversified book realises less volatility than the sum of its parts.

On Kelly. For a lognormal book the growth-optimal volatility target equals the Sharpe ratio,
so a strategy with a true Sharpe of 0.5 is full-Kelly at a 50% volatility target. The default
here is 15%, which is under half Kelly for any Sharpe this project could plausibly have, and
that is intentional: full Kelly assumes the edge is known exactly, and overestimating it
pushes past the peak into negative expected log growth. Half Kelly keeps roughly 75% of the
growth rate at half the volatility and cuts the chance of ever halving capital from 1/2 to 1/8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import TARGET_ABS_FORECAST

PERIODS_PER_YEAR = 365
VOL_SPAN = 32
MIN_VOL_PERIODS = 32
IDM_LOOKBACK = 250
IDM_CAP = 2.5

#: Annualised volatility target. See the module docstring for why this is not higher.
DEFAULT_TARGET_VOL = 0.15

#: Ceiling on gross exposure, as a multiple of equity. Binds when many instruments are at a
#: full forecast simultaneously, which in crypto means "everything is a BTC beta and they all
#: trend together". Without it, a diversified-looking book quietly becomes a levered directional
#: bet at exactly the moment correlations go to one.
DEFAULT_MAX_GROSS = 2.0


def annualised_volatility(prices: pd.DataFrame, span: int = VOL_SPAN) -> pd.DataFrame:
    """Causal EWM estimate of annualised return volatility."""
    returns = prices.pct_change()
    return returns.ewm(span=span, min_periods=MIN_VOL_PERIODS).std() * np.sqrt(PERIODS_PER_YEAR)


def diversification_multiplier(
    returns: pd.DataFrame,
    active: pd.DataFrame,
    lookback: int = IDM_LOOKBACK,
    cap: float = IDM_CAP,
) -> pd.Series:
    """How much to scale up because instruments are less than perfectly correlated.

    IDM = 1 / sqrt(w' C w) for equal weights w over the currently active instruments.
    Perfectly correlated instruments give 1.0 (no free diversification); uncorrelated ones
    give sqrt(N). Crypto sits close to the former, which is precisely the finding that
    matters for whether this strategy can work at all.

    Estimated on a trailing window and capped, because a correlation matrix estimated on
    250 daily observations across dozens of assets is noisy, and the noise is asymmetric:
    underestimating correlation inflates leverage right before it hurts.
    """
    n_active = active.sum(axis=1)
    idm = pd.Series(1.0, index=returns.index)

    corr = returns.rolling(lookback, min_periods=lookback).corr()
    if corr.empty:
        return idm

    for date in returns.index:
        n = int(n_active.loc[date])
        if n <= 1:
            continue
        try:
            block = corr.loc[date]
        except KeyError:
            continue
        cols = active.loc[date]
        cols = cols[cols].index
        block = block.loc[cols, cols].to_numpy(dtype=float)
        if np.isnan(block).any():
            continue
        w = np.full(n, 1.0 / n)
        portfolio_var = float(w @ block @ w)
        if portfolio_var > 0:
            idm.loc[date] = min(1.0 / np.sqrt(portfolio_var), cap)

    return idm.ffill().fillna(1.0)


def target_weights(
    prices: pd.DataFrame,
    forecast: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    target_vol: float = DEFAULT_TARGET_VOL,
    max_gross: float = DEFAULT_MAX_GROSS,
    use_idm: bool = True,
) -> pd.DataFrame:
    """Convert forecasts into target portfolio weights.

    Args:
        prices: close prices.
        forecast: Carver-scaled forecasts, same shape as `prices`.
        universe: optional boolean mask of tradeable instruments per bar. Use
            `data.point_in_time_universe` to avoid selecting on survivorship.
        target_vol: annualised volatility target for the whole book.
        max_gross: ceiling on the sum of absolute weights.
        use_idm: scale up for diversification. Turning this off is the conservative choice.

    Returns:
        Weights decided on bar t's close. `backtest.run` applies them to bar t+1.
    """
    active = prices.notna() & forecast.notna()
    if universe is not None:
        active = active & universe.reindex_like(active).fillna(False)

    vol = annualised_volatility(prices)
    active = active & vol.notna() & (vol > 0)

    n_active = active.sum(axis=1).replace(0, np.nan)
    instrument_weight = 1.0 / n_active

    conviction = (forecast / TARGET_ABS_FORECAST).where(active, 0.0)
    risk_scaling = (target_vol / vol).where(active, 0.0)
    weights = conviction * risk_scaling
    weights = weights.mul(instrument_weight, axis=0)

    if use_idm:
        idm = diversification_multiplier(prices.pct_change(), active)
        weights = weights.mul(idm, axis=0)

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    gross = weights.abs().sum(axis=1)
    scale = (max_gross / gross).clip(upper=1.0).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return weights.mul(scale, axis=0)
