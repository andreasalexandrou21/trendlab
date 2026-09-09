"""Pairs-trading checks.

Built around synthetic data where the true answer is known: if the filter cannot recover a
hedge ratio you planted yourself, nothing it says about real prices is worth reading.
"""

import numpy as np
import pandas as pd
import pytest

from trendlab.pairs import (
    engle_granger,
    fit_kalman_params,
    fit_spread_model,
    half_life,
    kalman_hedge_ratio,
    kalman_spread_model,
    log_likelihood,
    pair_weights,
    screen_pairs,
)


def _cointegrated(n=1500, beta=2.0, alpha=5.0, noise=0.5, phi=0.95, seed=0):
    """y = beta*x + alpha + stationary noise. Cointegrated by construction."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    x = 50 + np.cumsum(rng.normal(0, 0.5, n))
    resid = np.zeros(n)
    for t in range(1, n):  # AR(1), mean-reverting
        resid[t] = phi * resid[t - 1] + rng.normal(0, noise)
    y = beta * x + alpha + resid
    return pd.DataFrame({"Y": y, "X": x}, index=idx)


def _independent(n=1500, seed=1):
    """Two unrelated random walks. Must NOT test as cointegrated."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"Y": 100 + np.cumsum(rng.normal(0, 1, n)), "X": 100 + np.cumsum(rng.normal(0, 1, n))},
        index=idx,
    )


def test_kalman_recovers_a_planted_hedge_ratio():
    """The filter must converge to the beta used to build the data."""
    data = _cointegrated(beta=2.0, alpha=5.0)
    fit = kalman_hedge_ratio(data["Y"], data["X"])

    assert fit.beta.iloc[-1] == pytest.approx(2.0, abs=0.15)
    # Asserted on the fitted LINE, not on beta and alpha separately. Over a price range that
    # does not straddle zero the two are only jointly identified: a small beta error trades
    # off against a large alpha error and reproduces the same line. Testing them
    # individually tests the identification, not the filter.
    x_last = data["X"].iloc[-1]
    fitted = fit.beta.iloc[-1] * x_last + fit.alpha.iloc[-1]
    assert fitted == pytest.approx(2.0 * x_last + 5.0, rel=0.02)


def test_kalman_tracks_a_hedge_ratio_that_moves():
    """The whole reason to filter rather than run OLS: beta is allowed to change."""
    n = 3000
    rng = np.random.default_rng(4)
    idx = pd.date_range("2010-01-01", periods=n, freq="B", tz="UTC")
    x = 50 + np.cumsum(rng.normal(0, 0.5, n))
    beta_true = np.where(np.arange(n) < n // 2, 1.0, 3.0)
    y = beta_true * x + rng.normal(0, 0.5, n)
    data = pd.DataFrame({"Y": y, "X": x}, index=idx)

    fit = kalman_hedge_ratio(data["Y"], data["X"], delta=1e-3)

    assert fit.beta.iloc[n // 2 - 50] == pytest.approx(1.0, abs=0.3)
    assert fit.beta.iloc[-1] == pytest.approx(3.0, abs=0.3)


def test_kalman_is_causal():
    """Appending future data must not change any past estimate.

    A forward filter passes this. An RTS smoother, which is the tempting upgrade and gives
    visibly nicer betas, fails it, because the smoothed state at t conditions on data after
    t. That is the most common look-ahead error in published Kalman trading results.
    """
    data = _cointegrated(n=2000)
    full = kalman_hedge_ratio(data["Y"], data["X"]).beta.iloc[:1200]
    partial = kalman_hedge_ratio(data["Y"].iloc[:1200], data["X"].iloc[:1200]).beta

    assert np.allclose(full.to_numpy(), partial.to_numpy(), rtol=1e-12)


def test_hand_set_parameters_do_not_give_a_real_zscore():
    """Chan's constants leave z badly scaled, which is why fitting them matters.

    This is a documented weakness, asserted rather than assumed: with delta and Ve hand-set,
    the filter's forecast variance has no reason to match the real innovation dispersion, so
    "enter at |z| >= 1" does not mean one standard deviation of anything.
    """
    data = _cointegrated(n=3000, noise=0.5)
    z = kalman_hedge_ratio(data["Y"], data["X"]).zscore.iloc[500:]

    assert z.std() > 1.5  # nowhere near unit variance


def test_two_state_mle_selects_parameters_that_destroy_the_spread():
    """The headline negative result, asserted rather than described.

    Fitting the two-state model by maximum likelihood does NOT rescue it. The likelihood
    rewards one-step-ahead prediction, and a filter that chases the observation predicts well
    one step ahead, so the fit drives the innovation towards white noise. The recovered
    half-life collapses far below the true one.
    """
    data = _cointegrated(n=3000, noise=0.5)  # true spread half-life 13.5
    delta, ve = fit_kalman_params(data["Y"].iloc[:1000], data["X"].iloc[:1000])
    fit = kalman_hedge_ratio(data["Y"], data["X"], delta=delta, observation_var=ve)

    assert half_life(fit.innovation.iloc[500:]) < 5.0


def test_spread_model_recovers_the_true_spread():
    """The three-state model must reconstruct a spread it was never shown."""
    data = _cointegrated(n=2000, beta=2.0, alpha=5.0, noise=0.5)
    truth = pd.Series(data["Y"].to_numpy() - 2.0 * data["X"].to_numpy() - 5.0, index=data.index)

    params = fit_spread_model(data["Y"].iloc[:1000], data["X"].iloc[:1000])
    fit = kalman_spread_model(data["Y"], data["X"], **params)

    assert fit.spread.iloc[250:].corr(truth.iloc[250:]) > 0.9
    assert fit.beta.iloc[-1] == pytest.approx(2.0, abs=0.2)
    assert 0.6 < fit.zscore.iloc[250:].std() < 1.6


def test_spread_model_recovers_the_mean_reversion_speed():
    """phi, and therefore the half-life, must come back close to the planted value."""
    for phi_true in (0.85, 0.90, 0.95):
        data = _cointegrated(n=2000, phi=phi_true, noise=0.5)
        params = fit_spread_model(data["Y"].iloc[:1000], data["X"].iloc[:1000])
        assert params["phi"] == pytest.approx(phi_true, abs=0.06)


def test_spread_model_rejects_a_non_reverting_phi():
    data = _cointegrated(n=500)
    for bad in (0.0, 1.0, 1.5, -0.5):
        with pytest.raises(ValueError, match="phi"):
            kalman_spread_model(data["Y"], data["X"], phi=bad)


def test_spread_model_is_causal():
    """Appending future data must not change any past state estimate."""
    data = _cointegrated(n=2000)
    full = kalman_spread_model(data["Y"], data["X"]).spread.iloc[:1200]
    partial = kalman_spread_model(data["Y"].iloc[:1200], data["X"].iloc[:1200]).spread

    assert np.allclose(full.to_numpy(), partial.to_numpy(), rtol=1e-10)


def test_mle_beats_the_defaults_on_likelihood():
    """The fit must actually improve the objective it claims to optimise."""
    data = _cointegrated(n=2000)
    delta, ve = fit_kalman_params(data["Y"], data["X"])

    fitted = log_likelihood(
        kalman_hedge_ratio(data["Y"], data["X"], delta=delta, observation_var=ve)
    )
    default = log_likelihood(kalman_hedge_ratio(data["Y"], data["X"]))
    assert fitted > default


def test_parameter_fitting_is_causal_in_pair_weights():
    """Parameters are fitted on the burn-in window, so later data cannot change them."""
    data = _cointegrated(n=2000)
    full = pair_weights(data, "Y", "X", burn_in=400)
    partial = pair_weights(data.iloc[:1200], "Y", "X", burn_in=400)

    common = full.index.intersection(partial.index)
    assert len(common) > 700
    assert np.allclose(full.loc[common].to_numpy(), partial.loc[common].to_numpy(), atol=1e-12)


def test_delta_zero_is_static_regression():
    """delta = 0 means no process noise, so the hedge ratio must stop moving."""
    data = _cointegrated(n=1500)
    fit = kalman_hedge_ratio(data["Y"], data["X"], delta=0.0)
    late = fit.beta.iloc[-200:]

    assert late.std() < 0.02


def test_delta_validation():
    data = _cointegrated(n=300)
    for bad in (-0.1, 1.0, 2.0):
        with pytest.raises(ValueError, match="delta"):
            kalman_hedge_ratio(data["Y"], data["X"], delta=bad)
    with pytest.raises(ValueError, match="observation_var"):
        kalman_hedge_ratio(data["Y"], data["X"], observation_var=0.0)


def test_cointegration_test_separates_real_from_spurious():
    """Positive and negative control. If both come back significant, the test is useless."""
    _, p_real = engle_granger(_cointegrated()["Y"], _cointegrated()["X"])
    indep = _independent()
    _, p_fake = engle_granger(indep["Y"], indep["X"])

    assert p_real < 0.05
    assert p_fake > 0.05


def test_half_life_matches_a_known_ar1():
    """An AR(1) with phi = 0.95 has half-life ln(2)/ln(1/0.95) = 13.5 periods."""
    rng = np.random.default_rng(7)
    n = 5000
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.95 * z[t - 1] + rng.normal(0, 1)
    series = pd.Series(z, index=pd.date_range("2010-01-01", periods=n, freq="B", tz="UTC"))

    assert half_life(series) == pytest.approx(13.5, rel=0.2)


def test_half_life_is_infinite_for_a_random_walk():
    rng = np.random.default_rng(8)
    walk = pd.Series(
        np.cumsum(rng.normal(0, 1, 3000)),
        index=pd.date_range("2010-01-01", periods=3000, freq="B", tz="UTC"),
    )
    assert half_life(walk) > 100


def test_screen_finds_the_planted_pair_and_rejects_the_noise():
    data = _cointegrated(n=2000)
    noise = _independent(n=2000)
    prices = pd.DataFrame(
        {"Y": data["Y"], "X": data["X"], "N1": noise["Y"].to_numpy(), "N2": noise["X"].to_numpy()},
        index=data.index,
    )

    found = screen_pairs(prices, max_pvalue=0.05, max_half_life=200.0)
    names = {(c.y, c.x) for c in found}

    assert ("Y", "X") in names
    assert ("N1", "N2") not in names


def test_pair_weights_are_dollar_hedged_and_opposite_signed():
    data = _cointegrated(n=1500)
    w = pair_weights(data, "Y", "X", burn_in=250, gross_per_pair=1.0)

    live = w[(w != 0).any(axis=1)]
    assert len(live) > 0
    # The two legs always point opposite ways.
    assert (np.sign(live["Y"]) == -np.sign(live["X"])).all()
    # Gross exposure is the requested size.
    assert np.allclose(live.abs().sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_pair_weights_respect_the_burn_in():
    """No trading while the filter is still travelling from its zero initialisation."""
    data = _cointegrated(n=1500)
    w = pair_weights(data, "Y", "X", burn_in=300)

    assert (w.iloc[:300].to_numpy() == 0.0).all()


def test_position_is_held_between_entry_and_exit_thresholds():
    """A pair entered at z = -1.5 must not be churned out at z = -1.2."""
    data = _cointegrated(n=2000)
    w = pair_weights(data, "Y", "X", entry_z=2.0, exit_z=0.0, burn_in=250)

    gross = w.abs().sum(axis=1)
    changes = (gross.diff().abs() > 1e-12).sum()
    # Far fewer state changes than bars: it holds rather than flipping every day.
    assert changes < len(w) / 20


def test_entry_direction_is_correct():
    """Spread below its forecast means y is cheap relative to x, so go long y."""
    data = _cointegrated(n=1500)
    from trendlab.pairs import kalman_hedge_ratio as kf

    z = kf(data["Y"], data["X"]).zscore
    w = pair_weights(data, "Y", "X", entry_z=1.0, exit_z=0.0, burn_in=250)

    entries = w.index[(w["Y"] != 0) & (w["Y"].shift(1) == 0)]
    entries = [t for t in entries if t in z.index][:20]
    assert len(entries) > 3
    for t in entries:
        assert np.sign(w.loc[t, "Y"]) == -np.sign(z.loc[t])
