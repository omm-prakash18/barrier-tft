"""Tests for leakage_guard.py"""
import pytest
import pandas as pd
import numpy as np
from src.data.leakage_guard import LeakageError, assert_no_leakage, check_no_future_features


def _make_bar_df(n=50):
    ts = pd.date_range("2021-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "ticker": "AAPL", "close": 100.0})


def test_label_horizon_sanity_bad():
    """t1 <= t0 should raise LeakageError."""
    t0 = pd.Timestamp("2021-01-01 09:30", tz="UTC")
    t1 = t0  # bad: t1 == t0
    feat = pd.DataFrame({"timestamp": [t0], "ticker": ["AAPL"], "close": [100.0]})
    label = pd.DataFrame({"t0": [t0], "t1": [t1]})
    with pytest.raises(LeakageError, match="t1 <= t0"):
        assert_no_leakage(feat, label)


def test_label_horizon_sanity_good():
    """Valid t0 < t1 should pass."""
    t0 = pd.Timestamp("2021-01-01 09:30", tz="UTC")
    t1 = t0 + pd.Timedelta(minutes=30)
    feat  = pd.DataFrame({"timestamp": [t0], "ticker": ["AAPL"]})
    label = pd.DataFrame({"t0": [t0], "t1": [t1]})
    assert_no_leakage(feat, label)  # should not raise


def test_feature_future_leak_via_event_id():
    """Feature timestamp after t0 with event_id should raise LeakageError."""
    t0 = pd.Timestamp("2021-01-01 10:00", tz="UTC")
    t1 = t0 + pd.Timedelta(hours=1)
    # Feature is "from" 1 hour in the future
    feat = pd.DataFrame({
        "event_id": [1],
        "timestamp": [t0 + pd.Timedelta(hours=1)],  # AFTER t0
    })
    label = pd.DataFrame({
        "event_id": [1],
        "t0": [t0],
        "t1": [t1],
    })
    with pytest.raises(LeakageError):
        assert_no_leakage(feat, label)


def test_feature_timestamp_order():
    """Unsorted feature matrix should raise LeakageError."""
    ts = pd.date_range("2021-01-01", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts[::-1]})   # reversed order
    with pytest.raises(LeakageError, match="not sorted"):
        check_no_future_features(df)


def test_feature_timestamp_sorted_ok():
    ts = pd.date_range("2021-01-01", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts})
    check_no_future_features(df)  # should not raise


def test_news_latency_buffer():
    """Bar that falls inside forbidden window [pub_ts, usable_after) should raise LeakageError."""
    latency = 45
    news_ts  = pd.Timestamp("2021-01-01 09:30:10", tz="UTC")
    usable   = news_ts + pd.Timedelta(seconds=latency)         # 09:30:55
    # Bar at 09:30:30: AFTER article published, BEFORE latency expires → forbidden
    bar_ts   = pd.Timestamp("2021-01-01 09:30:30", tz="UTC")
    assert news_ts < bar_ts < usable, "Test setup error: bar not in forbidden window"

    bar_df  = pd.DataFrame({"timestamp": [bar_ts], "ticker": ["AAPL"]})
    news_df = pd.DataFrame({
        "article_public_ts": [news_ts],
        "usable_after_ts":   [usable],
        "ticker": ["AAPL"],
    })
    label_df = pd.DataFrame({
        "t0": [usable + pd.Timedelta(minutes=1)],
        "t1": [usable + pd.Timedelta(minutes=30)],
    })
    with pytest.raises(LeakageError, match="LATENCY"):
        assert_no_leakage(
            bar_df, label_df,
            latency_buffer_sec=latency,
            news_public_ts_col="article_public_ts",
            news_df=news_df,
        )

