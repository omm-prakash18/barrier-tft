"""
schemas.py
──────────
Pandera DataFrameSchema definitions for all pipeline boundaries.

Each schema is a point-in-time contract: validating at schema boundaries
catches silent data corruption (wrong types, null leakage, timezone
stripping, ticker-symbol mismatches) before they propagate downstream.

Usage:
    from src.data.schemas import validate_market_df, validate_label_df
    validate_market_df(df)    # raises SchemaError on violation
"""

from __future__ import annotations

try:
    import pandera as pa
    from pandera import Column, Check, DataFrameSchema
    _PANDERA_AVAILABLE = True
except ImportError:
    _PANDERA_AVAILABLE = False

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Graceful fallback if pandera is not installed
# ─────────────────────────────────────────────────────────────────────────────

def _require_pandera() -> None:
    if not _PANDERA_AVAILABLE:
        raise ImportError(
            "pandera is required for schema validation. "
            "Install it with: pip install pandera"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Schema definitions
# ─────────────────────────────────────────────────────────────────────────────

def _market_schema() -> "DataFrameSchema":
    return DataFrameSchema(
        {
            "timestamp": Column(
                pa.Timestamp,
                checks=[Check(lambda s: s.dt.tz is not None, error="timestamp must be tz-aware")],
                nullable=False,
            ),
            "ticker":  Column(str,   nullable=False),
            "open":    Column(float, checks=Check.gt(0),   nullable=False),
            "high":    Column(float, checks=Check.gt(0),   nullable=False),
            "low":     Column(float, checks=Check.gt(0),   nullable=False),
            "close":   Column(float, checks=Check.gt(0),   nullable=False),
            "volume":  Column(float, checks=Check.ge(0),   nullable=False),
        },
        checks=[
            Check(
                lambda df: (df["high"] >= df["low"]).all(),
                error="high must be >= low",
            ),
            Check(
                lambda df: (df["high"] >= df["close"]).all(),
                error="high must be >= close",
            ),
        ],
        coerce=False,
        strict=False,  # allow extra columns
    )


def _news_schema() -> "DataFrameSchema":
    return DataFrameSchema(
        {
            "ticker":             Column(str,   nullable=False),
            "article_public_ts":  Column(pa.Timestamp, nullable=False),
            "usable_after_ts":    Column(pa.Timestamp, nullable=False),
            "headline_sentiment": Column(float, checks=Check.in_range(-1.0, 1.0), nullable=False),
            "credibility":        Column(float, checks=Check.in_range(0.0, 1.0),  nullable=False),
        },
        checks=[
            Check(
                lambda df: (df["usable_after_ts"] > df["article_public_ts"]).all(),
                error="usable_after_ts must be strictly after article_public_ts",
            ),
        ],
        coerce=False,
        strict=False,
    )


def _feature_schema() -> "DataFrameSchema":
    return DataFrameSchema(
        {
            "timestamp":         Column(pa.Timestamp, nullable=False),
            "ticker":            Column(str,          nullable=False),
            "close":             Column(float,        checks=Check.gt(0),           nullable=False),
            "log_return":        Column(float,        nullable=True),
            "realized_vol":      Column(float,        checks=Check.ge(0),           nullable=True),
            "vol_norm_return":   Column(float,        nullable=True),
            "momentum_5":        Column(float,        nullable=True),
            "momentum_20":       Column(float,        nullable=True),
            "momentum_60":       Column(float,        nullable=True),
            "rsi_14":            Column(float,        checks=Check.in_range(0, 100), nullable=True),
            "regime":            Column(int,          checks=Check.ge(0),           nullable=False),
            "weighted_sentiment":Column(float,        checks=Check.in_range(-1, 1), nullable=False),
        },
        coerce=False,
        strict=False,
    )


def _label_schema() -> "DataFrameSchema":
    return DataFrameSchema(
        {
            "t0":              Column(pa.Timestamp, nullable=False),
            "t1":              Column(pa.Timestamp, nullable=False),
            "ticker":          Column(str,          nullable=False),
            "label":           Column(int,          checks=Check.isin([-1, 0, 1]),  nullable=False),
            "return_at_touch": Column(float,        nullable=False),
            "sample_weight":   Column(float,        checks=Check.gt(0),             nullable=False),
        },
        checks=[
            Check(
                lambda df: (df["t1"] > df["t0"]).all(),
                error="t1 must be strictly after t0 (label horizon must be positive)",
            ),
        ],
        coerce=False,
        strict=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public validation functions
# ─────────────────────────────────────────────────────────────────────────────

def validate_market_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw market OHLCV DataFrame."""
    if _PANDERA_AVAILABLE:
        return _market_schema().validate(df, lazy=True)
    
    # Native Fallback Checks
    req_cols = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Market DataFrame missing required column: {c}")
    assert (df["high"] >= df["low"]).all(), "Market validation failed: high < low found"
    assert (df["high"] >= df["close"]).all(), "Market validation failed: high < close found"
    assert (df["open"] > 0).all() and (df["close"] > 0).all(), "Prices must be positive"
    return df


def validate_news_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate news events DataFrame."""
    if _PANDERA_AVAILABLE:
        return _news_schema().validate(df, lazy=True)

    req_cols = ["ticker", "article_public_ts", "usable_after_ts", "headline_sentiment", "credibility"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"News DataFrame missing required column: {c}")
    assert (df["usable_after_ts"] > df["article_public_ts"]).all(), "usable_after_ts must be > article_public_ts"
    assert df["headline_sentiment"].between(-1.0, 1.0).all(), "headline_sentiment out of [-1, 1]"
    assert df["credibility"].between(0.0, 1.0).all(), "credibility out of [0, 1]"
    return df


def validate_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate engineered feature matrix."""
    if _PANDERA_AVAILABLE:
        return _feature_schema().validate(df, lazy=True)

    req_cols = ["timestamp", "ticker", "close", "realized_vol", "momentum_60", "regime"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Feature DataFrame missing required column: {c}")
    assert (df["close"] > 0).all(), "Feature prices must be positive"
    return df


def validate_label_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate triple-barrier label DataFrame."""
    if _PANDERA_AVAILABLE:
        return _label_schema().validate(df, lazy=True)

    req_cols = ["t0", "t1", "ticker", "label", "return_at_touch", "sample_weight"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Label DataFrame missing required column: {c}")
    assert (df["t1"] > df["t0"]).all(), "t1 must be strictly after t0"
    assert df["label"].isin([-1, 0, 1]).all(), "label must be in {-1, 0, 1}"
    assert (df["sample_weight"] > 0).all(), "sample_weight must be strictly positive"
    return df

