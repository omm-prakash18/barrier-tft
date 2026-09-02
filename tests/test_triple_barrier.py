"""Tests for triple_barrier.py"""
import pytest
import numpy as np
import pandas as pd
from src.labeling.triple_barrier import (
    get_barrier_labels,
    get_concurrency_weights,
    compute_sample_weights,
    label_universe,
)


def _make_price_series(n=100, start=100.0):
    rng   = np.random.default_rng(0)
    ts    = pd.date_range("2021-01-01 09:30", periods=n, freq="5min", tz="UTC")
    rets  = rng.normal(0, 0.01, n)
    close = start * np.exp(np.cumsum(rets))
    return pd.Series(close, index=ts)


def test_label_output_columns():
    prices = _make_price_series(100)
    events = pd.DataFrame(index=prices.index[25:30])
    labels = get_barrier_labels(prices, events, pt=2.0, sl=2.0, max_holding=10)
    assert not labels.empty
    for col in ["t1", "label", "return_at_touch", "barrier_type", "sigma_t0", "raw_weight"]:
        assert col in labels.columns, f"Missing column: {col}"


def test_labels_valid_values():
    prices = _make_price_series(200)
    events = pd.DataFrame(index=prices.index[30:50])
    labels = get_barrier_labels(prices, events, pt=2.0, sl=2.0, max_holding=15)
    assert labels["label"].isin([-1, 0, 1]).all(), "Labels must be -1, 0, or +1"
    assert (labels["t1"] > labels.index).all(), "t1 must be strictly after t0"


def test_barrier_types():
    prices = _make_price_series(200)
    events = pd.DataFrame(index=prices.index[30:80])
    labels = get_barrier_labels(prices, events, pt=2.0, sl=2.0, max_holding=10)
    assert labels["barrier_type"].isin(["upper", "lower", "vertical"]).all()


def test_upper_barrier_label_positive():
    """Crafted uptrend: should hit upper barrier first → label +1."""
    n   = 50
    ts  = pd.date_range("2021-01-01 09:30", periods=n, freq="5min", tz="UTC")
    # Sharp uptrend: +5% every bar
    close = 100.0 * np.cumprod(np.concatenate([[1.0], np.ones(n-1) * 1.05]))
    prices = pd.Series(close, index=ts)
    events = pd.DataFrame(index=[ts[10]])
    labels = get_barrier_labels(prices, events, pt=1.0, sl=10.0, max_holding=20)
    assert labels.iloc[0]["label"] == 1
    assert labels.iloc[0]["barrier_type"] == "upper"


def test_lower_barrier_label_negative():
    """Crafted downtrend: should hit lower barrier → label -1."""
    n   = 50
    ts  = pd.date_range("2021-01-01 09:30", periods=n, freq="5min", tz="UTC")
    close = 100.0 * np.cumprod(np.concatenate([[1.0], np.ones(n-1) * 0.95]))
    prices = pd.Series(close, index=ts)
    events = pd.DataFrame(index=[ts[10]])
    labels = get_barrier_labels(prices, events, pt=10.0, sl=1.0, max_holding=20)
    assert labels.iloc[0]["label"] == -1
    assert labels.iloc[0]["barrier_type"] == "lower"


def test_concurrency_weights_non_negative():
    """Uniqueness weights must be in (0, 1]."""
    prices = _make_price_series(200)
    events = pd.DataFrame(index=prices.index[30:80])
    labels = get_barrier_labels(prices, events, pt=2.0, sl=2.0, max_holding=15)
    labels = labels.reset_index()
    labels.rename(columns={"t0": "t0"}, inplace=True)
    labels = labels.set_index("t0")

    uniq = get_concurrency_weights(labels)
    assert (uniq > 0).all(), "Uniqueness weights must be positive"
    assert (uniq <= 1.0 + 1e-9).all(), "Uniqueness weights must be <= 1"


def test_sample_weights_positive():
    prices = _make_price_series(200)
    events = pd.DataFrame(index=prices.index[30:80])
    labels = get_barrier_labels(prices, events, pt=2.0, sl=2.0, max_holding=15)
    labels_with_idx = labels.reset_index()
    labels_with_idx = labels_with_idx.set_index("t0")
    weighted = compute_sample_weights(labels, t1_col="t1", normalize=True)
    assert (weighted["sample_weight"] >= 0).all()
    # Normalized weights should sum to approximately 1
    assert abs(weighted["sample_weight"].sum() - 1.0) < 1e-6


def test_label_universe_cross_sectional():
    """label_universe should produce labels for multiple tickers."""
    from src.data.synthetic_data import generate_market_data
    mdf = generate_market_data(start="2022-01-03", end="2022-03-31")
    # Minimal features needed
    mdf["realized_vol"] = 0.01
    mdf["momentum_60"]  = 0.0
    mdf = mdf[mdf["is_active"]].copy()

    ldf = label_universe(mdf, pt=2.0, sl=2.0, max_holding=10, event_spacing=20)
    assert len(ldf) > 0, "Should produce at least one label"
    assert "ticker" in ldf.columns
    assert ldf["ticker"].nunique() > 1, "Should label multiple tickers"
    assert ldf["label"].isin([-1, 0, 1]).all()
