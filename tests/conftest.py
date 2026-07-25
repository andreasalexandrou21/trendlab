import numpy as np
import pandas as pd
import pytest


def make_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")


@pytest.fixture
def flat_prices() -> pd.DataFrame:
    """Two assets, price pinned at 100. Isolates cost accounting from market moves."""
    idx = make_index(100)
    return pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)


@pytest.fixture
def random_walk() -> pd.DataFrame:
    """Deterministic geometric random walk, three assets."""
    rng = np.random.default_rng(20260725)
    n, k = 1000, 3
    rets = rng.normal(0.0002, 0.03, size=(n, k))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=make_index(n), columns=["A", "B", "C"])


@pytest.fixture
def single_asset() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 500
    rets = rng.normal(0.001, 0.02, size=n)
    return pd.DataFrame({"A": 100 * np.exp(np.cumsum(rets))}, index=make_index(n))
