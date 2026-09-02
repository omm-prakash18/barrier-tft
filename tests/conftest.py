"""
conftest.py
───────────
Shared pytest fixtures for the TFT + FinBERT test suite.

All tests import these fixtures automatically via pytest's fixture discovery.
Using shared fixtures eliminates repeated setup logic across test files and
ensures consistent synthetic data across the suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Date / time constants
# ─────────────────────────────────────────────────────────────────────────────

START_TS = pd.Timestamp("2021-01-04 09:30:00", tz="UTC")
BAR_FREQ  = pd.Timedelta(minutes=5)
N_BARS    = 120    # 2 trading sessions worth of 5-min bars
TICKERS   = ["AAPL", "MSFT", "GOOG"]


# ─────────────────────────────────────────────────────────────────────────────
# Market data
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_market_df() -> pd.DataFrame:
    """
    Minimal multi-ticker OHLCV DataFrame.
    Scope=session: built once and reused by all tests.
    """
    rng = np.random.default_rng(42)
    rows = []
    for ticker in TICKERS:
        price = 100.0
        for i in range(N_BARS):
            ret    = rng.normal(0, 0.002)
            price  = price * np.exp(ret)
            ts     = START_TS + i * BAR_FREQ
            volume = float(rng.integers(10_000, 200_000))
            rows.append({
                "timestamp": ts,
                "ticker":    ticker,
                "open":      price * (1 + rng.uniform(-0.001, 0.001)),
                "high":      price * (1 + abs(rng.uniform(0, 0.003))),
                "low":       price * (1 - abs(rng.uniform(0, 0.003))),
                "close":     price,
                "volume":    volume,
                "is_active": True,
                "sector":    "Technology",
            })
    df = pd.DataFrame(rows)
    df["open"]  = df[["open",  "close"]].max(axis=1) * 0.5 + df["close"] * 0.5
    df["high"]  = df[["high",  "close", "open"]].max(axis=1)
    df["low"]   = df[["low",   "close", "open"]].min(axis=1)
    return df.reset_index(drop=True)


@pytest.fixture(scope="session")
def tiny_news_df(tiny_market_df: pd.DataFrame) -> pd.DataFrame:
    """Small news events DataFrame aligned to tiny_market_df tickers."""
    rng  = np.random.default_rng(99)
    rows = []
    timestamps = tiny_market_df["timestamp"].unique()
    for ticker in TICKERS:
        for _ in range(8):
            pub_ts   = pd.Timestamp(rng.choice(timestamps))
            latency  = int(rng.integers(30, 90))
            rows.append({
                "ticker":             ticker,
                "article_public_ts":  pub_ts,
                "usable_after_ts":    pub_ts + pd.Timedelta(seconds=latency),
                "headline_sentiment": float(rng.uniform(-1, 1)),
                "credibility":        float(rng.uniform(0.3, 1.0)),
            })
    return pd.DataFrame(rows).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature / label DataFrames
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_feature_df(tiny_market_df: pd.DataFrame, tiny_news_df: pd.DataFrame) -> pd.DataFrame:
    """Full feature matrix built from tiny_market_df."""
    from src.data.features import build_feature_matrix
    return build_feature_matrix(tiny_market_df, tiny_news_df, latency_buffer_sec=45)


@pytest.fixture(scope="session")
def tiny_label_df(tiny_feature_df: pd.DataFrame) -> pd.DataFrame:
    """Triple-barrier labels on tiny_feature_df."""
    from src.labeling.triple_barrier import label_universe
    df = label_universe(tiny_feature_df, pt=2.0, sl=2.0, max_holding=10, event_spacing=3)
    assert len(df) > 0, "label_universe returned empty — check synthetic data size"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Model fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_tft():
    """Smallest valid TFT instance for shape / gradient tests."""
    from src.models.tft import TFT
    return TFT(
        n_tickers=3, n_sectors=2, n_regimes=3,
        n_enc_features=8, embed_dim=8, hidden_dim=16,
        n_heads=2, n_lstm_layers=1, dropout=0.0,
    )


@pytest.fixture()
def random_batch():
    """Random tensors simulating one TFT mini-batch. Function-scoped (fresh each test)."""
    B, T, F = 4, 20, 8
    return {
        "x_enc":      torch.randn(B, T, F),
        "ticker_ids": torch.randint(0, 3, (B,)),
        "sector_ids": torch.randint(0, 2, (B,)),
        "regime_ids": torch.randint(0, 3, (B,)),
        "y_target":   torch.randn(B),
        "weight":     torch.ones(B),
    }
