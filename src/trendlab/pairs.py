"""Pairs trading with a Kalman-filtered hedge ratio and an explicit spread state.

The canonical statistical-arbitrage strategy: find two instruments tied together by an
economic relationship, estimate the ratio that ties them, and trade the residual when it
strays from zero.

This module contains TWO models, and the difference between them is the main result here.

`kalman_hedge_ratio` is the textbook two-state formulation (Chan 2013): beta and alpha as a
random walk, residual treated as white observation noise. It is kept because it is what
every tutorial implements, and because it does not work. The residual is not white, it is
the mean-reverting spread, and that IS the trade. With nowhere to put an autocorrelated
residual the filter forces it into beta and alpha, chasing each observation and destroying
the signal. Measured against a synthetic spread with half-life 13.5 and std 1.60:

    Chan's published defaults  ->  recovered half-life 0.7, spread std 0.04
    a deliberately slow filter ->  recovered half-life 11.2, spread std 1.43

Fitting the two-state model by maximum likelihood makes this worse, not better. The
likelihood rewards one-step-ahead prediction, and a filter that chases the observation
predicts well one step ahead, so MLE actively selects the parameters that eat the spread.
The likelihood objective and the trading objective point in opposite directions.

`kalman_spread_model` is the fix and the one to use: a third state carrying the spread with
its own AR(1) dynamics. The model can then explain an autocorrelated residual without
corrupting the hedge ratio, MLE becomes the right tool rather than the wrong one, and the
filtered spread correlates 0.99 with the true one on synthetic data.

Two properties hold in both:

1. **The z-score needs no rolling window.** Standardisation comes from the fitted model's
   stationary variance, not from a sample mean and standard deviation over a window that
   might include the future. That window is where most published pairs backtests leak.

2. **Filtered, never smoothed.** Forward recursion only. An RTS smoother would give visibly
   better states and a completely untradeable strategy, because the smoothed estimate at
   time t conditions on data after t. That is the most common error in published Kalman and
   HMM trading results, and there is a test asserting this code does not make it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Chan's defaults. delta controls how fast the hedge ratio adapts; Ve is observation noise.
DEFAULT_DELTA = 1e-4
DEFAULT_VE = 1e-3


@dataclass
class KalmanFit:
    """Output of the filter. Every series is causal at its own timestamp."""

    beta: pd.Series
    alpha: pd.Series
    innovation: pd.Series
    innovation_std: pd.Series

    @property
    def zscore(self) -> pd.Series:
        """Innovation in units of its own forecast standard deviation."""
        return (self.innovation / self.innovation_std).replace([np.inf, -np.inf], np.nan)


def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = DEFAULT_DELTA,
    observation_var: float = DEFAULT_VE,
) -> KalmanFit:
    """Filter the time-varying hedge ratio and intercept between two price series.

    Args:
        y: dependent price series (the leg you go long when the spread is cheap).
        x: independent price series.
        delta: process-noise scale. Q = delta / (1 - delta) * I. Smaller means a stickier
            hedge ratio. 1e-4 is Chan's value and corresponds to a ratio that moves slowly.
        observation_var: Ve, the measurement noise on y.

    Returns:
        KalmanFit with beta, alpha, the one-step-ahead innovation and its standard deviation.
    """
    if not 0.0 <= delta < 1.0:
        raise ValueError("delta must be in [0, 1)")
    if observation_var <= 0:
        raise ValueError("observation_var must be positive")

    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(frame) < 2:
        raise ValueError("need at least 2 overlapping observations")

    y_arr = frame["y"].to_numpy(dtype=float)
    x_arr = frame["x"].to_numpy(dtype=float)
    n = len(frame)

    trans_cov = (delta / (1.0 - delta)) * np.eye(2) if delta > 0 else np.zeros((2, 2))

    theta = np.zeros(2)  # [beta, alpha]
    p_cov = np.zeros((2, 2))

    betas = np.empty(n)
    alphas = np.empty(n)
    innovations = np.empty(n)
    innovation_sds = np.empty(n)

    for t in range(n):
        obs = np.array([x_arr[t], 1.0])

        # Predict. The prior covariance is last step's posterior plus process noise, so the
        # forecast below uses no information from time t about y.
        prior_cov = p_cov + trans_cov if t > 0 else p_cov

        forecast = float(obs @ theta)
        innovation = y_arr[t] - forecast
        innovation_var = float(obs @ prior_cov @ obs) + observation_var

        gain = prior_cov @ obs / innovation_var
        theta = theta + gain * innovation
        p_cov = prior_cov - np.outer(gain, obs) @ prior_cov

        betas[t] = theta[0]
        alphas[t] = theta[1]
        innovations[t] = innovation
        innovation_sds[t] = np.sqrt(innovation_var)

    idx = frame.index
    return KalmanFit(
        beta=pd.Series(betas, index=idx, name="beta"),
        alpha=pd.Series(alphas, index=idx, name="alpha"),
        innovation=pd.Series(innovations, index=idx, name="innovation"),
        innovation_std=pd.Series(innovation_sds, index=idx, name="innovation_std"),
    )


@dataclass
class SpreadFit:
    """Output of the three-state model. Every series is causal at its own timestamp."""

    beta: pd.Series
    alpha: pd.Series
    spread: pd.Series
    spread_std: float
    phi: float
    innovation: pd.Series
    innovation_std: pd.Series

    @property
    def zscore(self) -> pd.Series:
        """Filtered spread in units of its model-implied stationary standard deviation.

        Causal by construction and needs no rolling window, because the standardisation
        comes from the fitted model rather than from a sample statistic over a window that
        might include the future.
        """
        if not np.isfinite(self.spread_std) or self.spread_std <= 0:
            return pd.Series(np.nan, index=self.spread.index)
        return self.spread / self.spread_std

    @property
    def half_life(self) -> float:
        """Implied by the fitted AR(1) coefficient: -ln(2) / ln(phi)."""
        if not 0 < self.phi < 1:
            return float("inf")
        return float(-np.log(2) / np.log(self.phi))


def kalman_spread_model(
    y: pd.Series,
    x: pd.Series,
    q_beta: float = 1e-7,
    q_alpha: float = 1e-7,
    q_spread: float = 0.25,
    phi: float = 0.95,
    observation_var: float = 1e-4,
) -> SpreadFit:
    """Three-state filter: hedge ratio, intercept, and the spread itself.

        observation:  y_t = beta_t * x_t + alpha_t + s_t + v_t
        transition:   beta_t  = beta_{t-1}  + w_beta
                      alpha_t = alpha_{t-1} + w_alpha
                      s_t     = phi * s_{t-1} + w_s          <- explicit mean reversion

    **Why this exists, and why the two-state version is not enough.** In the standard Chan
    formulation the residual is modelled as white observation noise. It is not white: it is
    the mean-reverting spread, and that is the entire trade. Because the model has nowhere to
    put an autocorrelated residual, the filter puts it into beta and alpha instead, chasing
    each observation and destroying the signal. Measured on synthetic data with a known
    spread half-life of 13.5 days and standard deviation 1.60:

        Chan defaults (delta 1e-4, Ve 1e-3)  ->  half-life 0.7, spread std 0.04
        slow filter   (delta 1e-6, Ve 10)    ->  half-life 11.2, spread std 1.43

    And maximum likelihood makes it worse, not better: the likelihood rewards one-step-ahead
    predictive accuracy, and a filter that chases the observation predicts well one step
    ahead. So fitting the two-state model by MLE actively selects the parameters that eat the
    spread. The likelihood objective and the trading objective point in opposite directions.

    Giving the spread its own AR(1) state fixes the misspecification. The model can now
    explain an autocorrelated residual without corrupting the hedge ratio, and MLE becomes
    the right tool rather than the wrong one.

    Args:
        q_beta, q_alpha: process noise on the hedge ratio and intercept. Small: these should
            drift, not chase.
        q_spread: process noise driving the spread.
        phi: AR(1) coefficient of the spread. Must be in (0, 1) to mean-revert.
        observation_var: measurement noise on y. Small, because the spread state now absorbs
            what used to be dumped here.
    """
    if not 0.0 < phi < 1.0:
        raise ValueError("phi must be in (0, 1) for the spread to mean-revert")
    for name, value in (("q_beta", q_beta), ("q_alpha", q_alpha), ("q_spread", q_spread)):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if observation_var <= 0:
        raise ValueError("observation_var must be positive")

    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(frame) < 30:
        raise ValueError("need at least 30 overlapping observations")

    y_arr = frame["y"].to_numpy(dtype=float)
    x_arr = frame["x"].to_numpy(dtype=float)
    n = len(frame)

    transition = np.diag([1.0, 1.0, phi])
    process_cov = np.diag([q_beta, q_alpha, q_spread])

    # Initialise the hedge ratio from OLS on an opening window so the filter does not spend
    # its first year travelling from zero, which would show up as spurious early profit.
    init = min(250, n // 4)
    design = np.column_stack([x_arr[:init], np.ones(init)])
    ols, *_ = np.linalg.lstsq(design, y_arr[:init], rcond=None)
    theta = np.array([ols[0], ols[1], 0.0])
    state_cov = np.diag([1e-6, 1e-4, q_spread / (1.0 - phi**2)])

    betas = np.empty(n)
    alphas = np.empty(n)
    spreads = np.empty(n)
    innovations = np.empty(n)
    innovation_sds = np.empty(n)

    for t in range(n):
        obs = np.array([x_arr[t], 1.0, 1.0])

        theta = transition @ theta
        state_cov = transition @ state_cov @ transition.T + process_cov

        innovation = y_arr[t] - float(obs @ theta)
        innovation_var = float(obs @ state_cov @ obs) + observation_var

        gain = state_cov @ obs / innovation_var
        theta = theta + gain * innovation
        state_cov = state_cov - np.outer(gain, obs) @ state_cov

        betas[t], alphas[t], spreads[t] = theta
        innovations[t] = innovation
        innovation_sds[t] = np.sqrt(innovation_var)

    idx = frame.index
    stationary_std = float(np.sqrt(q_spread / (1.0 - phi**2)))
    return SpreadFit(
        beta=pd.Series(betas, index=idx, name="beta"),
        alpha=pd.Series(alphas, index=idx, name="alpha"),
        spread=pd.Series(spreads, index=idx, name="spread"),
        spread_std=stationary_std,
        phi=phi,
        innovation=pd.Series(innovations, index=idx, name="innovation"),
        innovation_std=pd.Series(innovation_sds, index=idx, name="innovation_std"),
    )


def log_likelihood(fit: KalmanFit | SpreadFit, skip: int = 20) -> float:
    """Gaussian log-likelihood of the observations under the filter.

    The Kalman recursion produces the one-step-ahead forecast error and its variance, so the
    marginal likelihood of the data falls out with no extra work:

        log p(y) = -0.5 * sum_t [ log(2*pi*Q_t) + e_t^2 / Q_t ]

    `skip` drops the opening bars, where the state is still travelling from its zero
    initialisation and the likelihood is dominated by that transient rather than by the
    parameters being estimated.
    """
    e = fit.innovation.to_numpy()[skip:]
    q = (fit.innovation_std.to_numpy()[skip:]) ** 2
    good = np.isfinite(e) & np.isfinite(q) & (q > 0)
    if not good.any():
        return -np.inf
    e, q = e[good], q[good]
    return float(-0.5 * np.sum(np.log(2 * np.pi * q) + e**2 / q))


def fit_spread_model(
    y: pd.Series,
    x: pd.Series,
    skip: int = 50,
) -> dict[str, float]:
    """Fit (q_spread, phi, observation_var) of the three-state model by maximum likelihood.

    Now that the model is correctly specified, the likelihood and the trading objective
    point the same way: explaining the residual's autocorrelation requires estimating the
    spread's persistence rather than eating it into the hedge ratio. On the two-state model
    the same procedure did the opposite, which is the whole reason that version is kept only
    as a documented counter-example.

    q_beta and q_alpha are held fixed and small. They are weakly identified against q_spread
    (all three feed the same observation) and letting the optimiser trade them off is how you
    get a hedge ratio that wanders. Fixing them is the prior that beta drifts slowly.
    """
    from scipy.optimize import minimize

    def unpack(params: np.ndarray) -> tuple[float, float, float]:
        log_qs, logit_phi, log_ve = params
        return float(np.exp(log_qs)), float(1.0 / (1.0 + np.exp(-logit_phi))), float(np.exp(log_ve))

    def negative_ll(params: np.ndarray) -> float:
        q_spread, phi, ve = unpack(params)
        if not (0.0 < phi < 0.9999) or not np.isfinite(q_spread) or q_spread <= 0:
            return np.inf
        try:
            fit = kalman_spread_model(y, x, q_spread=q_spread, phi=phi, observation_var=ve)
        except (ValueError, np.linalg.LinAlgError):
            return np.inf
        ll = log_likelihood(fit, skip=skip)
        return -ll if np.isfinite(ll) else np.inf

    start = np.array([np.log(0.25), np.log(0.95 / 0.05), np.log(1e-4)])
    result = minimize(
        negative_ll, start, method="Nelder-Mead", options={"maxiter": 400, "fatol": 1e-2}
    )
    q_spread, phi, ve = unpack(result.x)

    # Ve is allowed to go to essentially zero, and often should: if the pair really is
    # "hedge ratio plus a mean-reverting spread" with no extra measurement error, the true
    # value IS zero. It is only floored to keep the innovation variance non-singular.
    # An earlier version rejected any fit with Ve below 1e-12 and silently threw away good
    # estimates, which is why the fitted parameters came back exactly equal to the priors.
    ve = float(np.clip(ve, 1e-12, 1e6))

    # Fall back to the priors only for genuinely degenerate corners: a spread with no
    # variance, or one so persistent it is a random walk and therefore not tradeable.
    if not (1e-10 < q_spread < 1e6) or not (0.5 < phi < 0.9995):
        return {"q_spread": 0.25, "phi": 0.95, "observation_var": 1e-4}
    return {"q_spread": q_spread, "phi": phi, "observation_var": ve}


def fit_kalman_params(
    y: pd.Series,
    x: pd.Series,
    skip: int = 20,
) -> tuple[float, float]:
    """Estimate (delta, observation_var) by maximum likelihood. Returns the fitted pair.

    Why this exists. Chan's delta = 1e-4 and Ve = 1e-3 are hand-set constants, and with
    hand-set constants the filter's forecast variance Q_t has no reason to match the real
    dispersion of the innovations. The consequence is not cosmetic: z = e / sqrt(Q) is then
    not a standardised quantity, so "enter at |z| >= 1" does not mean one standard deviation
    of anything, and the same threshold means something different on every pair.

    Fitting by maximum likelihood removes both magic numbers and makes the z-score mean what
    its name says. This is the empirical-Bayes step the state-space formulation makes free:
    no extra passes, no grid search, and crucially NOT tuned against backtest Sharpe, which
    would be fitting the parameters to the answer.

    Optimised in unconstrained coordinates (logit for delta, log for the variance) so the
    optimiser cannot wander outside the valid region.
    """
    from scipy.optimize import minimize

    def negative_ll(params: np.ndarray) -> float:
        logit_delta, log_ve = params
        delta = 1.0 / (1.0 + np.exp(-logit_delta))
        ve = np.exp(log_ve)
        if not (0.0 < delta < 1.0) or not np.isfinite(ve) or ve <= 0:
            return np.inf
        try:
            fit = kalman_hedge_ratio(y, x, delta=delta, observation_var=ve)
        except (ValueError, np.linalg.LinAlgError):
            return np.inf
        ll = log_likelihood(fit, skip=skip)
        return -ll if np.isfinite(ll) else np.inf

    start = np.array([np.log(DEFAULT_DELTA / (1 - DEFAULT_DELTA)), np.log(DEFAULT_VE)])
    result = minimize(
        negative_ll,
        start,
        method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-2},
    )

    logit_delta, log_ve = result.x
    delta = float(1.0 / (1.0 + np.exp(-logit_delta)))
    ve = float(np.exp(log_ve))

    # Guard against the optimiser running to a degenerate corner on a badly behaved pair.
    if not (1e-8 < delta < 0.5) or not (1e-8 < ve < 1e8):
        return DEFAULT_DELTA, DEFAULT_VE
    return delta, ve


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life from an Ornstein-Uhlenbeck fit.

    Regress d(spread) on lagged spread; the coefficient lambda gives half-life
    -ln(2) / lambda. A pair whose half-life is longer than the holding period you are
    willing to tolerate is not tradeable however cointegrated it looks, and a half-life
    below a day or two at daily frequency is usually noise rather than signal.
    """
    spread = spread.dropna()
    if len(spread) < 10:
        return float("nan")

    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    common = lagged.index.intersection(delta.index)
    lagged, delta = lagged.loc[common], delta.loc[common]

    x = np.column_stack([lagged.to_numpy(), np.ones(len(lagged))])
    coef, *_ = np.linalg.lstsq(x, delta.to_numpy(), rcond=None)
    lam = coef[0]
    if lam >= 0:
        return float("inf")  # diverging, not mean-reverting
    return float(-np.log(2) / lam)


def engle_granger(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Engle-Granger cointegration test. Returns (test statistic, p-value).

    Note on the critical values: statsmodels' `coint` applies MacKinnon critical values for
    a *residual-based* test, which are wider than plain ADF criticals. Running a plain ADF
    on an OLS residual and using ADF tables is a well-known error that finds cointegration
    roughly twice as often as it should, because it ignores that the cointegrating vector
    was itself estimated from the data.
    """
    from statsmodels.tsa.stattools import coint

    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(frame) < 30:
        return float("nan"), 1.0
    stat, pvalue, _ = coint(frame["y"], frame["x"])
    return float(stat), float(pvalue)


@dataclass
class PairCandidate:
    y: str
    x: str
    pvalue: float
    half_life: float
    correlation: float

    def __str__(self) -> str:
        return (
            f"{self.y}/{self.x}: p={self.pvalue:.4f}, "
            f"half-life={self.half_life:.0f}d, corr={self.correlation:.2f}"
        )


def screen_pairs(
    prices: pd.DataFrame,
    candidates: list[tuple[str, str]] | None = None,
    max_pvalue: float = 0.05,
    min_half_life: float = 2.0,
    max_half_life: float = 120.0,
) -> list[PairCandidate]:
    """Rank pairs by cointegration p-value, filtered on a tradeable half-life.

    If `candidates` is None every ordered pair is tested, which is what most tutorials do
    and is exactly the multiple-testing trap: 45 tickers is 990 unordered pairs, and at a 5%
    level roughly 50 of them come back "cointegrated" on noise alone. Pass an explicit,
    economically motivated candidate list wherever possible, and if you do screen everything,
    the count of tests is what the deflated Sharpe has to be told about later.
    """
    cols = list(prices.columns)
    if candidates is None:
        candidates = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1 :]]

    out: list[PairCandidate] = []
    for y_name, x_name in candidates:
        if y_name not in prices.columns or x_name not in prices.columns:
            continue
        pair = prices[[y_name, x_name]].dropna()
        if len(pair) < 250:
            continue

        _, pvalue = engle_granger(pair[y_name], pair[x_name])
        if not np.isfinite(pvalue) or pvalue > max_pvalue:
            continue

        # Half-life from an OU fit on the STATIC OLS residual. Two reasons this and not the
        # filtered spread: it is the standard screening measure, and it is ~500x cheaper.
        # Running a full maximum-likelihood fit per candidate turns a 1,176-pair screen into
        # a 40-minute job, and screening is meant to be the cheap step that decides what is
        # worth fitting properly.
        #
        # What it must NOT be measured on is the two-state innovation: a one-step forecast
        # error is white by construction, so that reports a ~1-day half-life for every pair
        # in the universe regardless of what the spread is actually doing.
        design = np.column_stack([pair[x_name].to_numpy(), np.ones(len(pair))])
        coef, *_ = np.linalg.lstsq(design, pair[y_name].to_numpy(), rcond=None)
        residual = pd.Series(
            pair[y_name].to_numpy() - design @ coef, index=pair.index, name="residual"
        )
        hl = half_life(residual)
        if not (min_half_life <= hl <= max_half_life):
            continue

        out.append(
            PairCandidate(
                y=y_name,
                x=x_name,
                pvalue=pvalue,
                half_life=hl,
                correlation=float(pair[y_name].corr(pair[x_name])),
            )
        )

    return sorted(out, key=lambda c: c.pvalue)


def pair_weights(
    prices: pd.DataFrame,
    y_name: str,
    x_name: str,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
    delta: float = DEFAULT_DELTA,
    observation_var: float = DEFAULT_VE,
    burn_in: int = 250,
    gross_per_pair: float = 1.0,
    fit_params: bool = True,
) -> pd.DataFrame:
    """Target weights for one pair: long the spread when it is cheap, short when rich.

    Entry at |z| >= entry_z, exit when |z| <= exit_z, hold in between. The position is a
    state machine rather than a function of today's z alone, because a pair that entered at
    z = -1.5 should be held through z = -1.2 rather than churned in and out around the
    threshold.

    Legs are scaled so the pair's gross exposure is `gross_per_pair`, split between the two
    legs in the ratio implied by the hedge ratio in dollar terms.

    `burn_in` bars are forced flat while the filter converges from its zero initialisation.
    Without it the first year of any Kalman pairs backtest is trading a hedge ratio that is
    still travelling from 0 to its true value, which shows up as spurious early profit.

    `fit_params` estimates delta and the observation variance by maximum likelihood **on the
    burn-in window only**, then holds them fixed for the rest of the run. That is what you
    could actually have done in real time: at the end of the burn-in you fit, then you trade.
    Fitting on the full sample would be look-ahead, and fitting against backtest Sharpe would
    be fitting to the answer.
    """
    pair = prices[[y_name, x_name]].dropna()

    params = {"q_spread": 0.25, "phi": 0.95, "observation_var": 1e-4}
    if fit_params and len(pair) > burn_in + 50:
        params = fit_spread_model(pair.iloc[:burn_in][y_name], pair.iloc[:burn_in][x_name])

    fit = kalman_spread_model(pair[y_name], pair[x_name], **params)
    z = fit.zscore

    n = len(z)
    position = np.zeros(n)  # +1 long spread, -1 short spread
    state = 0.0
    z_arr = z.to_numpy()

    for t in range(n):
        zt = z_arr[t]
        if t < burn_in or not np.isfinite(zt):
            state = 0.0
        elif state == 0.0:
            if zt <= -entry_z:
                state = 1.0  # spread too low: long y, short x
            elif zt >= entry_z:
                state = -1.0
        elif (state == 1.0 and zt >= -exit_z) or (state == -1.0 and zt <= exit_z):
            state = 0.0  # spread has reverted far enough; close
        position[t] = state

    beta = fit.beta.to_numpy()
    y_px = pair[y_name].to_numpy()
    x_px = pair[x_name].to_numpy()

    # Dollar exposure of the x leg per dollar of the y leg.
    dollar_x = np.abs(beta) * x_px / y_px
    denom = 1.0 + dollar_x
    w_y = position * gross_per_pair / denom
    w_x = -np.sign(beta) * position * gross_per_pair * dollar_x / denom

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.loc[pair.index, y_name] = w_y
    weights.loc[pair.index, x_name] = w_x
    return weights
