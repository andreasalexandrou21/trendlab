"""Transaction and financing costs.

Two independent charges, kept separate because they answer different questions:

- `trade_cost` scales with turnover. It is what kills high-frequency retail strategies.
- `financing_cost` scales with borrowed notional and with *time*, not turnover. It is the
  term that quietly empties a leveraged account even when the strategy is flat.

The second one is the whole reason this module exists as a first-class part of the engine
rather than a constant subtracted at the end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DAYS_PER_YEAR = 365.0  # crypto trades 24/7; no business-day calendar


@dataclass(frozen=True)
class CostModel:
    """Costs in basis points of traded notional, plus an annualised financing rate.

    Defaults are Kraken spot taker (0.40%) plus half a tick of slippage, then doubled.
    Carver's rule: reject any system that is vulnerable to costs rather than trying to
    trade it cheaply. So model them pessimistically and see if the edge survives.
    """

    commission_bps: float = 40.0
    slippage_bps: float = 5.0
    multiplier: float = 2.0

    # Benchmark (ESTR-ish) plus a retail broker spread. CFD and margin desks charge this
    # on the full notional, not on your equity.
    benchmark_rate: float = 0.02
    financing_spread: float = 0.03

    # Interest earned on cash not deployed. Default 0.0 because that is the TRUTH for a
    # small account: IBKR pays nothing on balances under USD 10,000 for accounts below
    # USD 100k NAV. It matters at scale, though, and a trend book running 0.25x gross
    # leaves ~75% of equity idle, so omitting it silently penalises the strategy by more
    # than a percent a year on a larger account. Set it to model that case honestly.
    cash_rate: float = 0.0

    # ponytail: square-root market impact is off by default. At EUR 2,000 notional in BTC
    # you are ~1e-6 of daily volume, so impact is a rounding error and modelling it would
    # be theatre. Set impact_coef > 0 (typical 0.5-1.0) if this ever runs size where
    # participation is measurable.
    impact_coef: float = 0.0

    @property
    def round_trip_bps(self) -> float:
        """One-way cost in bps, after the pessimism multiplier."""
        return (self.commission_bps + self.slippage_bps) * self.multiplier

    @property
    def financing_rate(self) -> float:
        return self.benchmark_rate + self.financing_spread

    def trade_cost(
        self,
        traded_notional: float | np.ndarray,
        volatility: float | np.ndarray | None = None,
        daily_volume: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Cost of trading `traded_notional` of currency value.

        `volatility` (daily, fractional) and `daily_volume` (currency) are only used when
        `impact_coef` is non-zero, in which case the square-root law adds
        `coef * sigma * sqrt(participation)` on top of spread and commission.
        """
        cost = np.abs(traded_notional) * self.round_trip_bps / 1e4

        if self.impact_coef and volatility is not None and daily_volume is not None:
            participation = np.divide(
                np.abs(traded_notional),
                daily_volume,
                out=np.zeros_like(np.asarray(traded_notional, dtype=float)),
                where=np.asarray(daily_volume) > 0,
            )
            impact = self.impact_coef * volatility * np.sqrt(participation)
            cost = cost + np.abs(traded_notional) * impact * self.multiplier

        return cost

    def financing_cost(
        self, gross_notional: float | np.ndarray, equity: float | np.ndarray, days: float = 1.0
    ) -> float | np.ndarray:
        """Financing on the borrowed portion only: max(0, gross notional - equity).

        Unlevered books (gross <= equity) pay nothing, which is exactly the point of
        reporting levered and unlevered curves side by side.
        """
        borrowed = np.maximum(0.0, np.asarray(gross_notional) - np.asarray(equity))
        cost = borrowed * self.financing_rate * days / DAYS_PER_YEAR

        if self.cash_rate:
            idle = np.maximum(0.0, np.asarray(equity) - np.asarray(gross_notional))
            cost = cost - idle * self.cash_rate * days / DAYS_PER_YEAR
        return cost


#: Zero-cost model. Only for tests that need to isolate the accounting from the costs.
FRICTIONLESS = CostModel(
    commission_bps=0.0, slippage_bps=0.0, multiplier=0.0, benchmark_rate=0.0, financing_spread=0.0
)
