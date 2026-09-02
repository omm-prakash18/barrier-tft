"""
synthetic_data.py
─────────────────
Point-in-time cross-sectional market & news data generator.

Guarantees:
  • Tickers include delisted / failed names (survivorship-bias free).
  • News events carry `article_public_ts` — the only timestamp used downstream.
  • Bar timestamps are deterministic UTC minutes so joins are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional

# ── reproducibility ───────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── universe definition ───────────────────────────────────────────────────────
SECTORS = ["Tech", "Finance", "Healthcare", "Energy", "Consumer"]

TICKER_DEFS: list[dict] = [
    # (ticker, sector, delisted_after_date_or_None)
    {"ticker": "AAPL",  "sector": "Tech",        "delisted": None},
    {"ticker": "MSFT",  "sector": "Tech",        "delisted": None},
    {"ticker": "GOOGL", "sector": "Tech",        "delisted": None},
    {"ticker": "JPM",   "sector": "Finance",     "delisted": None},
    {"ticker": "GS",    "sector": "Finance",     "delisted": None},
    {"ticker": "JNJ",   "sector": "Healthcare",  "delisted": None},
    {"ticker": "PFE",   "sector": "Healthcare",  "delisted": None},
    {"ticker": "XOM",   "sector": "Energy",      "delisted": None},
    {"ticker": "WMT",   "sector": "Consumer",    "delisted": None},
    # Delisted names — present only for dates up to their delisting
    {"ticker": "ENRN",  "sector": "Energy",      "delisted": "2002-12-01"},  # Enron
    {"ticker": "LEHM",  "sector": "Finance",     "delisted": "2008-09-15"},  # Lehman
]


def _simulate_price_series(
    n_bars: int,
    start_price: float = 100.0,
    mu: float = 0.0,
    sigma: float = 0.012,
    regime_shift_prob: float = 0.005,
) -> np.ndarray:
    """Geometric Brownian motion with random regime-vol shifts."""
    prices = np.empty(n_bars)
    prices[0] = start_price
    current_sigma = sigma
    for i in range(1, n_bars):
        if RNG.random() < regime_shift_prob:
            current_sigma = sigma * RNG.uniform(0.5, 3.0)
        ret = RNG.normal(mu / n_bars, current_sigma)
        prices[i] = prices[i - 1] * np.exp(ret)
    return prices


def generate_market_data(
    start: str = "2020-01-02",
    end: str = "2023-12-31",
    bar_freq: str = "5min",
    trading_hours: tuple[str, str] = ("09:30", "16:00"),
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        timestamp, ticker, sector, open, high, low, close, volume, is_active

    `is_active` is False for delisted tickers after their delisting date —
    this allows the caller to filter correctly for point-in-time universe
    membership without silently dropping history.
    """
    # Build minute-bar timestamp grid (market hours only)
    all_timestamps = pd.date_range(start=start, end=end, freq=bar_freq, tz="UTC")
    open_time  = pd.Timestamp("2000-01-01 " + trading_hours[0], tz="UTC").time()
    close_time = pd.Timestamp("2000-01-01 " + trading_hours[1], tz="UTC").time()
    mask = (
        (all_timestamps.weekday < 5)  # Mon-Fri
        & (all_timestamps.time >= open_time)
        & (all_timestamps.time < close_time)
    )
    timestamps = all_timestamps[mask]
    n_bars = len(timestamps)

    records: list[pd.DataFrame] = []
    for td in TICKER_DEFS:
        ticker  = td["ticker"]
        sector  = td["sector"]
        delisted = pd.Timestamp(td["delisted"], tz="UTC") if td["delisted"] else pd.NaT

        start_price = RNG.uniform(20.0, 500.0)
        sigma       = RNG.uniform(0.008, 0.025)

        close = _simulate_price_series(n_bars, start_price=start_price, sigma=sigma)
        # Simulate intrabar OHLV strictly adhering to high >= max(open, close) >= min(open, close) >= low
        open_  = close * RNG.uniform(0.997, 1.003, n_bars)
        high_bump = RNG.uniform(1.0, 1.006, n_bars)
        low_dip   = RNG.uniform(0.994, 1.0, n_bars)
        high   = np.maximum(open_, close) * high_bump
        low    = np.minimum(open_, close) * low_dip
        volume = (RNG.lognormal(mean=10.0, sigma=1.2, size=n_bars) * 100).astype(int)

        is_active = (pd.isna(delisted) | (timestamps < delisted)).astype(bool)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "ticker":    ticker,
            "sector":    sector,
            "open":      open_,
            "high":      high,
            "low":       low,
            "close":     close,
            "volume":    volume,
            "is_active": is_active,
        })
        records.append(df)

    market_df = pd.concat(records, ignore_index=True)
    market_df.sort_values(["timestamp", "ticker"], inplace=True)
    market_df.reset_index(drop=True, inplace=True)
    return market_df


def generate_news_data(
    market_df: pd.DataFrame,
    n_events: int = 5_000,
    latency_buffer_sec: int = 45,
) -> pd.DataFrame:
    """
    Generates a synthetic news event stream.

    Columns:
        article_public_ts   — the only timestamp used downstream (LEAKAGE RULE)
        ticker
        headline_sentiment  — float in [-1, 1]  (FinBERT proxy)
        credibility         — publisher credibility weight in (0, 1]
        sentiment_embedding — np.ndarray shape (8,) simulating a FinBERT CLS vector

    `latency_buffer_sec` represents the mandatory delay a trading system must
    observe before reacting to a news event (see leakage_guard.py).
    """
    active_tickers = market_df["ticker"].unique().tolist()
    min_ts = market_df["timestamp"].min()
    max_ts = market_df["timestamp"].max()
    ts_range_sec = int((max_ts - min_ts).total_seconds())

    offsets_sec = RNG.integers(0, ts_range_sec, size=n_events)
    public_ts   = min_ts + pd.to_timedelta(offsets_sec, unit="s")

    # The "usable after" timestamp — models must NOT use a bar whose timestamp
    # falls before this value.
    usable_after_ts = public_ts + pd.Timedelta(seconds=latency_buffer_sec)

    tickers_chosen    = RNG.choice(active_tickers, size=n_events)
    sentiments        = RNG.uniform(-1.0, 1.0, size=n_events)
    credibilities     = RNG.beta(a=2, b=1, size=n_events).clip(0.1, 1.0)
    # Simulate 8-dim FinBERT CLS embeddings (unit-normed)
    raw_embeddings    = RNG.standard_normal((n_events, 8))
    norms             = np.linalg.norm(raw_embeddings, axis=1, keepdims=True).clip(1e-8)
    embeddings        = (raw_embeddings / norms).tolist()

    news_df = pd.DataFrame({
        "article_public_ts":  public_ts,
        "usable_after_ts":    usable_after_ts,  # derived, stored for guard checks
        "ticker":             tickers_chosen,
        "headline_sentiment": sentiments,
        "credibility":        credibilities,
        "sentiment_embedding": embeddings,
    })
    news_df.sort_values("article_public_ts", inplace=True)
    news_df.reset_index(drop=True, inplace=True)
    return news_df


if __name__ == "__main__":
    mdf = generate_market_data(start="2022-01-03", end="2022-06-30")
    ndf = generate_news_data(mdf, n_events=500)
    print("Market bars:", len(mdf), "| News events:", len(ndf))
    print(mdf.head(3))
    print(ndf.head(3))
