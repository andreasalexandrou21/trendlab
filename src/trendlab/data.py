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
# SURVIVORSHIP-BIASED. This is today's majors, chosen because they survived, and it is kept
# only as the deliberately-biased baseline to measure the bias against. For research use
# CANDIDATES with `point_in_time_universe`.
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

# Broad candidate pool for point-in-time selection. Deliberately includes coins that were
# top-ten once and then collapsed (EOS, TRX, XLM, IOTA, NEO, ZEC, DASH, WAVES, ONT, QTUM,
# OMG, ZIL) so that ranking by trailing liquidity can pick losers when they were winners.
# Symbols the exchange no longer serves are skipped at fetch time rather than faked.
CANDIDATES = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "SOL/USDT",
    "DOT/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "LINK/USDT",
    "XLM/USDT",
    "TRX/USDT",
    "EOS/USDT",
    "XMR/USDT",
    "ETC/USDT",
    "ATOM/USDT",
    "ALGO/USDT",
    "VET/USDT",
    "FIL/USDT",
    "NEAR/USDT",
    "APT/USDT",
    "ARB/USDT",
    "OP/USDT",
    "AVAX/USDT",
    "UNI/USDT",
    "AAVE/USDT",
    "MKR/USDT",
    "SAND/USDT",
    "MANA/USDT",
    "AXS/USDT",
    "NEO/USDT",
    "IOTA/USDT",
    "ZEC/USDT",
    "DASH/USDT",
    "QTUM/USDT",
    "OMG/USDT",
    "ZIL/USDT",
    "ONT/USDT",
    "WAVES/USDT",
    "THETA/USDT",
    "HBAR/USDT",
    "EGLD/USDT",
    "XTZ/USDT",
    "CHZ/USDT",
    "ENJ/USDT",
    "CRV/USDT",
    "COMP/USDT",
    "SNX/USDT",
    "GRT/USDT",
    "BAT/USDT",
    "ZRX/USDT",
    "RUNE/USDT",
    "KAVA/USDT",
    "IOST/USDT",
    "ONE/USDT",
    "MATIC/USDT",
    "FTM/USDT",
    "ICP/USDT",
    "SUI/USDT",
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


def load_frames(
    symbols: list[str] | None = None,
    timeframe: str = "1d",
    since: str = "2017-01-01",
    exchange_id: str = DEFAULT_RESEARCH_EXCHANGE,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load full OHLCV per symbol, fetching and caching anything not already on disk.

    Every symbol is integrity-checked on the way through, whether it came from the network
    or from cache. Symbols the exchange does not serve are skipped, not faked.
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
    return frames


def load(
    symbols: list[str] | None = None,
    timeframe: str = "1d",
    since: str = "2017-01-01",
    exchange_id: str = DEFAULT_RESEARCH_EXCHANGE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Close prices as a single DataFrame (index = UTC date, columns = symbols)."""
    frames = load_frames(symbols, timeframe, since, exchange_id, refresh)
    return pd.DataFrame({s: f["close"] for s, f in frames.items()}).sort_index()


def dollar_volume(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Quote-currency volume per bar, the liquidity proxy used to rank the universe."""
    return pd.DataFrame({s: f["close"] * f["volume"] for s, f in frames.items()}).sort_index()


def point_in_time_universe(
    dollar_vol: pd.DataFrame,
    n: int = 10,
    lookback: int = 90,
    min_history: int = 250,
) -> pd.DataFrame:
    """Boolean mask: which instruments were top-`n` by liquidity, judged only on the past.

    This is the fix for the biggest bias in Batch 1. Selecting the universe once, today,
    means selecting instruments *because they survived*, which hands a trend strategy free
    money it could never have earned in real time.

    Here the rank is recomputed from a trailing `lookback`-day median of dollar volume, so
    on any given date the selection uses only information that existed on that date. An
    instrument must also have `min_history` bars before it is eligible, so a newly listed
    coin cannot be bought on its debut spike.

    Caveat that this does NOT fix: instruments delisted so long ago that the exchange no
    longer serves their bars are absent from `frames` entirely and cannot be ranked. The
    candidate list mitigates that by including coins that fell hard but still trade
    (EOS, TRX, XLM, IOTA, NEO, ZEC, DASH). Residual bias is smaller, not zero.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    liquidity = dollar_vol.rolling(lookback, min_periods=lookback).median()
    eligible = dollar_vol.notna().cumsum() >= min_history
    liquidity = liquidity.where(eligible)

    # rank 1 = most liquid; ties broken deterministically by column order
    ranks = liquidity.rank(axis=1, ascending=False, method="first")
    return (ranks <= n).fillna(False)


# Cross-asset universe. The point is genuinely different return drivers, which is exactly
# what the crypto universe did not have: effective breadth there was 1.7 against a nominal
# 10, because every coin was one factor wearing a different ticker.
#
# ETF proxies rather than futures, because futures need EUR 30-50k of margin to hold a
# diversified book and the whole question here is whether breadth fixes the result at all.
# If it does, the same signals port to micro futures once capital exists.
CROSS_ASSET = {
    "SPY": "US large cap",
    "QQQ": "US tech",
    "IWM": "US small cap",
    "EFA": "developed ex-US",
    "EEM": "emerging markets",
    "VNQ": "US REITs",
    "TLT": "US 20y+ treasuries",
    "IEF": "US 7-10y treasuries",
    "LQD": "investment grade credit",
    "HYG": "high yield credit",
    "GLD": "gold",
    "SLV": "silver",
    "DBC": "broad commodities",
    "USO": "crude oil",
    "UUP": "US dollar index",
}


def load_cross_asset(
    tickers: list[str] | None = None,
    start: str = "2007-01-01",
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily adjusted closes for the cross-asset ETF universe, cached to parquet.

    Uses yfinance, which scrapes an unofficial Yahoo endpoint and breaks periodically. Fine
    for research, never for production. `auto_adjust=True` matters: unadjusted prices make
    every dividend look like a one-day crash and manufacture fake trend signals.
    """
    import os
    import tempfile

    tickers = tickers or list(CROSS_ASSET)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"cross_asset_{start}.parquet"

    if path.exists() and not refresh:
        frame = pd.read_parquet(path)
    else:
        os.environ.setdefault("YF_CACHE_DIR", tempfile.mkdtemp())
        import yfinance as yf

        raw = yf.download(tickers, start=start, progress=False, auto_adjust=True, threads=False)
        frame = raw["Close"].dropna(how="all")
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        frame.to_parquet(path)

    frame = frame.sort_index()
    log.info(
        "cross-asset: %s bars, %s to %s", len(frame), frame.index[0].date(), frame.index[-1].date()
    )
    return frame
