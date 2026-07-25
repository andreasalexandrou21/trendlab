"""OHLCV fetch, parquet cache, and integrity checks.

Venue-agnostic by design: the venue decision is deferred until a strategy survives
validation, so every exchange goes through ccxt's unified interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Research history only. Binance has the deepest USDT history (2017-08) but WITHDREW FROM
# THE EU on 1 Jul 2026, so it can never be the execution venue. Measured depth for BTC/USDT:
# binance 2017-08, bybit 2021-07, kraken 721 bars only (its OHLC endpoint ignores `since`).
# Consequence: the paper-trading phase MUST re-validate on the execution venue's own bars,
# because prices, listing dates and liquidity all differ between venues.
DEFAULT_RESEARCH_EXCHANGE = "binance"

# Majors that are tradeable on an EU-licensed venue post-MiCA. Scoped deliberately:
# research must never drift onto something that cannot actually be traded.
#
# KNOWN BIAS, unfixed as of Batch 1: this list is today's majors, chosen because they
# survived. Backtesting it from 2017 is survivorship bias and will flatter trend following,
# because the 2018 major list also contained EOS, TRX, XLM, IOTA and NEO. The fix is a
# point-in-time universe reconstructed from rolling dollar volume, which is Batch 2 work.
# Until then, treat every number produced on this universe as an upper bound, not an estimate.
UNIVERSE = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "LINK/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
]

_TIMEFRAME_MS = {"1d": 86_400_000, "1h": 3_600_000, "4h": 14_400_000}


class DataIntegrityError(Exception):
    """Raised when cached or fetched bars fail a structural check."""


@dataclass(frozen=True)
class IntegrityReport:
    symbol: str
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    missing_bars: int

    def __str__(self) -> str:
        return (
            f"{self.symbol}: {self.rows} bars, {self.start.date()} to {self.end.date()}, "
            f"{self.missing_bars} missing"
        )


def _fetch_paginated(exchange, symbol: str, timeframe: str, since_ms: int) -> list[list]:
    """Page forward through fetch_ohlcv until the exchange stops returning new bars.

    Exchanges disagree about `since` (Kraken's OHLC endpoint ignores it and always returns
    the most recent 720 candles). The cursor-advance check below terminates either way, and
    the caller sees the truncated range in the integrity report.
    """
    step = _TIMEFRAME_MS[timeframe]
    limit = min(getattr(exchange, "max_ohlcv_limit", 1000) or 1000, 1000)
    cursor = since_ms
    now_ms = exchange.milliseconds()
    rows: list[list] = []

    while cursor < now_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        # Drop anything at or before the last bar we already hold.
        if rows:
            batch = [b for b in batch if b[0] > rows[-1][0]]
            if not batch:
                break
        rows.extend(batch)
        next_cursor = rows[-1][0] + step
        if next_cursor <= cursor:  # exchange is not advancing; stop rather than spin
            break
        cursor = next_cursor

    return rows


def fetch(
    symbols: list[str] | None = None,
    timeframe: str = "1d",
    since: str = "2017-01-01",
    exchange_id: str = DEFAULT_RESEARCH_EXCHANGE,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for each symbol. Returns {symbol: DataFrame indexed by UTC timestamp}.

    Default venue is chosen for history depth, not for execution. Bybit and OKX paginate
    backwards properly; Kraken's public OHLC endpoint caps at 720 bars and will silently
    give you two years of daily data instead of eight.
    """
    import ccxt

    symbols = symbols or UNIVERSE
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    since_ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)

    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        log.info("fetching %s %s from %s", symbol, timeframe, since)
        rows = _fetch_paginated(exchange, symbol, timeframe, since_ms)
        if not rows:
            log.warning("no bars returned for %s, skipping", symbol)
            continue
        frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        frame["ts"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
        out[symbol] = frame.set_index("ts").sort_index()
    return out


def check(frame: pd.DataFrame, symbol: str, timeframe: str = "1d") -> IntegrityReport:
    """Structural checks on a single symbol's bars. Raises on anything that would
    silently corrupt a backtest; reports gaps rather than raising, since crypto venues
    do have genuine outages."""
    if frame.empty:
        raise DataIntegrityError(f"{symbol}: no rows")
    if not frame.index.is_monotonic_increasing:
        raise DataIntegrityError(f"{symbol}: index is not monotonic increasing")
    if frame.index.has_duplicates:
        dupes = frame.index[frame.index.duplicated()].tolist()[:5]
        raise DataIntegrityError(f"{symbol}: duplicate timestamps, e.g. {dupes}")

    ohlc = frame[["open", "high", "low", "close"]]
    if ohlc.isna().to_numpy().any():
        raise DataIntegrityError(f"{symbol}: NaN in OHLC")
    if (ohlc <= 0).to_numpy().any():
        raise DataIntegrityError(f"{symbol}: non-positive price")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise DataIntegrityError(f"{symbol}: high below open/close")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise DataIntegrityError(f"{symbol}: low above open/close")

    step = pd.Timedelta(milliseconds=_TIMEFRAME_MS[timeframe])
    expected = int((frame.index[-1] - frame.index[0]) / step) + 1
    return IntegrityReport(
        symbol=symbol,
        rows=len(frame),
        start=frame.index[0],
        end=frame.index[-1],
        missing_bars=expected - len(frame),
    )


def load(
    symbols: list[str] | None = None,
    timeframe: str = "1d",
    since: str = "2017-01-01",
    exchange_id: str = DEFAULT_RESEARCH_EXCHANGE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load close prices for the universe as a single DataFrame (index=UTC date, cols=symbols).

    Caches raw OHLCV per symbol as parquet. Every symbol is integrity-checked on the way
    through, whether it came from the network or from cache.
    """
    symbols = symbols or UNIVERSE
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    missing = []
    for symbol in symbols:
        path = DATA_DIR / f"{symbol.replace('/', '-')}_{timeframe}.parquet"
        if path.exists() and not refresh:
            frames[symbol] = pd.read_parquet(path)
        else:
            missing.append(symbol)

    if missing:
        fetched = fetch(missing, timeframe=timeframe, since=since, exchange_id=exchange_id)
        for symbol, frame in fetched.items():
            path = DATA_DIR / f"{symbol.replace('/', '-')}_{timeframe}.parquet"
            frame.to_parquet(path)
            frames[symbol] = frame

    for symbol, frame in frames.items():
        log.info("%s", check(frame, symbol, timeframe))

    closes = pd.DataFrame({s: f["close"] for s, f in frames.items()})
    return closes.sort_index()
