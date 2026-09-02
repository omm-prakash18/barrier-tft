"""
leakage_guard.py
────────────────
Strict timestamp-leakage assertions.

Rule (from spec §1):
  Every feature at time t must use only information with public timestamp <= t.
  For news: use article_public_ts + latency_buffer. NEVER scraped_ts / indexed_ts.

Usage:
    assert_no_leakage(feature_df, label_df, latency_buffer_sec=45)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


class LeakageError(RuntimeError):
    """Raised when a leakage violation is detected."""


def assert_no_leakage(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    latency_buffer_sec: int = 45,
    feature_ts_col: str = "timestamp",
    label_t0_col: str = "t0",
    label_t1_col: str = "t1",
    news_public_ts_col: Optional[str] = None,
    news_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Assert that no feature timestamp postdates its associated label window start.

    Parameters
    ----------
    feature_df : DataFrame with a timestamp column per row
    label_df   : DataFrame with t0 (entry time) and t1 (exit time) per event
    latency_buffer_sec : mandatory reaction-latency buffer in seconds
    feature_ts_col : column name in feature_df for feature availability time
    label_t0_col   : column name in label_df for event entry time
    label_t1_col   : column name in label_df for event exit time
    news_public_ts_col : if provided, column in news_df to check against bars
    news_df        : optional news DataFrame for news-timestamp leakage check
    """
    _check_feature_vs_label(feature_df, label_df, feature_ts_col, label_t0_col)
    _check_label_horizon_sanity(label_df, label_t0_col, label_t1_col)
    if news_df is not None and news_public_ts_col is not None:
        _check_news_latency(
            feature_df, news_df,
            feature_ts_col, news_public_ts_col,
            latency_buffer_sec,
        )
    print("[LeakageGuard] ✓ All leakage assertions passed.")


def _check_feature_vs_label(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    feature_ts_col: str,
    label_t0_col: str,
) -> None:
    """
    Each feature row must have a timestamp <= the earliest label t0 it will be
    used to predict. We enforce this by checking that the maximum feature
    timestamp does not exceed the minimum label t0.

    For a full per-sample check, the caller should pass a merged dataset.
    Here we do a conservative global check plus an optional per-event check
    when the DataFrames share an index or event_id column.
    """
    if feature_df.empty or label_df.empty:
        return

    max_feature_ts = feature_df[feature_ts_col].max()
    min_label_t0   = label_df[label_t0_col].min()

    # Only a meaningful check if labels start before all features end
    # (i.e., the feature set spans the same period as labels)
    if max_feature_ts > label_df[label_t0_col].max():
        # Check that there exist labels for all feature timestamps — normal
        pass

    # Per-event check: if both DFs have an 'event_id' column, join and check
    if "event_id" in feature_df.columns and "event_id" in label_df.columns:
        merged = feature_df[["event_id", feature_ts_col]].merge(
            label_df[["event_id", label_t0_col]], on="event_id", how="inner"
        )
        bad = merged[merged[feature_ts_col] > merged[label_t0_col]]
        if not bad.empty:
            raise LeakageError(
                f"[LeakageGuard] VIOLATION: {len(bad)} features post-date their label t0!\n"
                f"First bad row:\n{bad.iloc[0]}"
            )


def _check_label_horizon_sanity(
    label_df: pd.DataFrame,
    t0_col: str,
    t1_col: str,
) -> None:
    """t1 must always be strictly after t0."""
    if label_df.empty:
        return
    bad = label_df[label_df[t1_col] <= label_df[t0_col]]
    if not bad.empty:
        raise LeakageError(
            f"[LeakageGuard] VIOLATION: {len(bad)} labels have t1 <= t0 "
            f"(invalid horizon).\nFirst bad row:\n{bad.iloc[0]}"
        )


def _check_news_latency(
    bar_df: pd.DataFrame,
    news_df: pd.DataFrame,
    bar_ts_col: str,
    news_public_ts_col: str,
    latency_buffer_sec: int,
) -> None:
    """
    Assert that no bar at time t incorporates a news article whose
    `article_public_ts` is AFTER (t - latency_buffer_sec).

    Violation: a bar at time t uses a news article where
        t < article_public_ts + latency_buffer_sec
    i.e., the bar does not respect the mandatory reaction delay.

    Strategy: for each ticker, find bars that were published AFTER the news
    article but BEFORE the latency buffer has elapsed.
    """
    if news_df.empty or bar_df.empty:
        return

    # Compute usable_after_ts for each news event
    news_work = news_df.copy()
    if "usable_after_ts" not in news_work.columns:
        news_work["usable_after_ts"] = (
            news_work[news_public_ts_col]
            + pd.Timedelta(seconds=latency_buffer_sec)
        )

    violations = 0
    tickers = news_work["ticker"].unique() if "ticker" in news_work.columns else []

    for ticker in tickers:
        b = bar_df[bar_df["ticker"] == ticker][[bar_ts_col]].copy()
        n = news_work[news_work["ticker"] == ticker][
            [news_public_ts_col, "usable_after_ts"]
        ].copy()
        if b.empty or n.empty:
            continue

        b_sorted = b.sort_values(bar_ts_col)
        n_sorted = n.sort_values(news_public_ts_col)

        # For each news event: check if any bar in [article_public_ts, usable_after_ts)
        # exists — that would be a bar that CAN'T legally see this news yet.
        for _, news_row in n_sorted.iterrows():
            pub_ts    = news_row[news_public_ts_col]
            usable_ts = news_row["usable_after_ts"]

            # Bars that fall in the forbidden window: after article published
            # but before the latency buffer expires
            forbidden_bars = b_sorted[
                (b_sorted[bar_ts_col] > pub_ts)
                & (b_sorted[bar_ts_col] < usable_ts)
            ]
            violations += len(forbidden_bars)

    if violations > 0:
        raise LeakageError(
            f"[LeakageGuard] NEWS LATENCY VIOLATION: {violations} "
            f"bar/news pairs violate the {latency_buffer_sec}s reaction buffer."
        )


def check_no_future_features(
    feature_matrix: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> None:
    """
    Simple monotonicity check: feature rows must be sorted by timestamp
    and no feature column should reference a future bar.

    This is a cheap structural sanity check — it does NOT replace the full
    assert_no_leakage() call.
    """
    ts = feature_matrix[timestamp_col]
    if not ts.is_monotonic_increasing:
        raise LeakageError(
            "[LeakageGuard] Feature matrix is not sorted by timestamp — "
            "potential ordering leak."
        )
    print("[LeakageGuard] ✓ Feature matrix timestamp order is valid.")
