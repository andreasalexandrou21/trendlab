"""Grid trading: the most widely sold retail bot strategy.

Place buy orders on a ladder below the current price and sell orders above it. Every time
price oscillates across a rung you bank a small profit. In a sideways market it prints a
steady stream of tiny wins, which is exactly why it sells so well: the equity curve looks
like a straight line right up until it doesn't.

**What it actually is.** A grid bot is short volatility with an unbounded tail. Each rung
crossed is a small realised gain; each unit of trend against you is an unrealised loss on an
inventory that only grows. The payoff is the mirror image of a straddle: you are collecting
premium for absorbing other people's directional flow, without being paid the premium a
market maker earns for the same service.

Two properties follow, and both are visible in the implementation below rather than argued:

1. **Inventory grows monotonically against a trend.** Price falling through every rung
   leaves you long the entire ladder at prices above the market. There is no exit rule that
   fires; the strategy's only response to being wrong is to buy more.
2. **The win rate is near 100% and means nothing.** Almost every closed round trip is a
   winner. The losses live in the open inventory, so a bot can show a 95% win rate and a
   deeply negative mark-to-market simultaneously. This is the number vendors quote.

It is implemented here honestly and tested against the same instrument as everything else,
because "the most popular strategy" deserves to be measured rather than dismissed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GridConfig:
    """Parameters of a single-instrument grid.

    `upper` and `lower` bound the ladder. Real bots either ask the user to pick them or set
    them from a trailing range; both amount to a bet that price stays inside, which is the
    bet the strategy is actually making.
    """

    levels: int = 20
    spacing_pct: float = 0.01
    position_per_level: float = 0.05
    allow_short: bool = False

    def __post_init__(self) -> None:
        if self.levels < 2:
            raise ValueError("levels must be at least 2")
        if self.spacing_pct <= 0:
            raise ValueError("spacing_pct must be positive")
        if self.position_per_level <= 0:
            raise ValueError("position_per_level must be positive")


def grid_weights(
    prices: pd.DataFrame,
    symbol: str,
    config: GridConfig | None = None,
    anchor_window: int = 250,
) -> pd.DataFrame:
    """Target weights for a grid bot on one instrument.

    The ladder is anchored ONCE, at the mean of the first `anchor_window` bars, and then
    held fixed. That is what a real grid bot does: the user sets upper and lower bounds when
    they start it and those bounds do not move. Position size is a step function of how many
    rungs below the anchor price currently sits, so the further price falls, the more you
    hold.

    That single sentence is the whole risk profile. There is no stop, and the response to an
    adverse move is to increase exposure.

    An earlier version re-anchored on a rolling mean, which let the ladder drift down with a
    falling market and quietly halved the measured drawdown. It also made the strategy
    something no vendor actually ships. Modelling a kinder variant than the one being sold
    would have understated exactly the risk this module exists to expose.
    """
    config = config or GridConfig()

    px = prices[symbol].dropna()
    if len(px) <= anchor_window:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    anchor_value = float(px.iloc[:anchor_window].mean())
    anchor = pd.Series(anchor_value, index=px.index)
    anchor.iloc[:anchor_window] = np.nan  # not trading before the grid exists

    # How many rungs below (positive) or above (negative) the anchor we are.
    deviation = (anchor - px) / (anchor * config.spacing_pct)
    rungs = np.floor(deviation).clip(lower=0 if not config.allow_short else -config.levels)
    rungs = rungs.clip(upper=config.levels)

    exposure = rungs * config.position_per_level
    exposure = exposure.where(anchor.notna(), 0.0).fillna(0.0)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.loc[exposure.index, symbol] = exposure.to_numpy()
    return weights


def inventory_stats(result_weights: pd.DataFrame, symbol: str) -> dict[str, float]:
    """Diagnostics that make the hidden risk visible.

    A grid bot's marketing number is its win rate. The numbers that matter are how big the
    inventory got and how long it stayed underwater, because that is where the losses are
    parked while the win rate stays beautiful.
    """
    exposure = result_weights[symbol]
    live = exposure[exposure != 0]
    return {
        "max_exposure": float(exposure.max()),
        "mean_exposure": float(live.mean()) if len(live) else 0.0,
        "pct_bars_invested": float((exposure != 0).mean()),
        "max_consecutive_invested": float(
            (exposure != 0).astype(int).groupby((exposure == 0).cumsum()).sum().max()
            if len(exposure)
            else 0.0
        ),
    }
