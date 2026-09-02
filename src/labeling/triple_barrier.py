"""
triple_barrier.py
─────────────────
Lopez de Prado triple-barrier labeling with:
  • Upper barrier  (take-profit): p0 * (1 + pt * sigma_t0)
  • Lower barrier  (stop-loss):   p0 * (1 - sl * sigma_t0)
  • Vertical barrier: t0 + max_holding_bars

Label assignment (whichever barrier is touched first):
  +1  → upper
  -1  → lower
   0  → vertical (with reduced weight — weaker signal)

Sample weighting:
  1. Uniqueness weight: 1 / avg_concurrency  (down-weights overlapping events)
  2. Return-magnitude weight: |realized_return|
  Final weight = uniqueness_weight * return_magnitude_weight (normalized)

Public API:
  get_barrier_labels(prices, events, pt, sl, max_holding, vol_window)
  get_concurrency_weights(label_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core labeling
# ─────────────────────────────────────────────────────────────────────────────

def get_barrier_labels(
    prices: pd.Series,
    events: pd.DataFrame,
    pt: float = 2.0,
    sl: float = 2.0,
    max_holding: int = 20,
    vol_window: int = 20,
    vol_floor: float = 1e-4,
    vertical_weight_scale: float = 0.5,
) -> pd.DataFrame:
    """
    Compute triple-barrier labels for each candidate entry event.

    Parameters
    ----------
    prices       : pd.Series indexed by timestamp, values = close price for ONE ticker
    events       : pd.DataFrame with DatetimeIndex (= t0 timestamps)
                   Required column: none beyond index.
                   Optional column: 'sigma' to override per-event vol estimate.
    pt           : take-profit barrier multiplier (in realized-vol units)
    sl           : stop-loss barrier multiplier   (in realized-vol units)
    max_holding  : maximum holding period in bars (vertical barrier)
    vol_window   : lookback for rolling realized vol estimate at t0
    vol_floor    : minimum vol to avoid divide-by-zero in tiny price series
    vertical_weight_scale : weight reduction factor for vertical-barrier touches

    Returns
    -------
    pd.DataFrame with columns:
        t0                : event entry time (index copy)
        t1                : event exit time  (barrier-touch or vertical)
        label             : +1, -1, or 0 (sign of return for vertical)
        return_at_touch   : realized return at touch
        barrier_type      : 'upper', 'lower', 'vertical'
        raw_weight        : |return_at_touch| * vertical_scale_if_applicable
    """
    prices   = prices.sort_index()
    out_rows = []

    # Pre-compute rolling realized vol on the prices series (causal)
    log_rets      = np.log(prices / prices.shift(1))
    rolling_vol   = log_rets.shift(1).rolling(vol_window, min_periods=5).std()
    rolling_vol   = rolling_vol.clip(lower=vol_floor)

    price_index   = prices.index
    price_array   = prices.values

    for t0 in events.index:
        if t0 not in price_index:
            continue

        iloc_t0 = price_index.get_loc(t0)
        p0      = price_array[iloc_t0]

        # ── sigma at t0 ───────────────────────────────────────────────────
        if "sigma" in events.columns:
            sigma_t0 = events.loc[t0, "sigma"]
        else:
            sigma_t0 = rolling_vol.iloc[iloc_t0] if iloc_t0 < len(rolling_vol) else vol_floor
        sigma_t0 = max(sigma_t0, vol_floor)

        # ── barrier levels ────────────────────────────────────────────────
        upper = p0 * (1.0 + pt * sigma_t0)
        lower = p0 * (1.0 - sl * sigma_t0)

        # ── scan forward up to max_holding bars ───────────────────────────
        end_iloc = min(iloc_t0 + max_holding + 1, len(price_array))
        scan_prices = price_array[iloc_t0 + 1 : end_iloc]
        scan_index  = price_index[iloc_t0 + 1 : end_iloc]

        label        = 0
        t1           = price_index[min(iloc_t0 + max_holding, len(price_index) - 1)]
        barrier_type = "vertical"
        return_at_touch = (price_array[min(iloc_t0 + max_holding, len(price_array) - 1)] - p0) / p0

        for k, (ts_k, px_k) in enumerate(zip(scan_index, scan_prices)):
            if px_k >= upper:
                label        = +1
                t1           = ts_k
                barrier_type = "upper"
                return_at_touch = (px_k - p0) / p0
                break
            elif px_k <= lower:
                label        = -1
                t1           = ts_k
                barrier_type = "lower"
                return_at_touch = (px_k - p0) / p0
                break
        else:
            # Vertical barrier: label by sign of return
            label = int(np.sign(return_at_touch)) if return_at_touch != 0 else 0

        # Raw weight = |return|; vertical touches get a penalty
        scale      = vertical_weight_scale if barrier_type == "vertical" else 1.0
        raw_weight = abs(return_at_touch) * scale

        out_rows.append({
            "t0":             t0,
            "t1":             t1,
            "label":          label,
            "return_at_touch": return_at_touch,
            "barrier_type":   barrier_type,
            "sigma_t0":       sigma_t0,
            "raw_weight":     raw_weight,
        })

    result = pd.DataFrame(out_rows)
    if result.empty:
        return result

    result.set_index("t0", inplace=True)
    result.index.name = "t0"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency / uniqueness weighting
# ─────────────────────────────────────────────────────────────────────────────

def get_concurrency_weights(
    label_df: pd.DataFrame,
    t0_col: str = None,   # use index if None
    t1_col: str = "t1",
) -> pd.Series:
    """
    Compute uniqueness weights (1 / avg_concurrency) for each labeled event.

    Algorithm (Lopez de Prado, Ch. 4):
      For each event i with interval [t0_i, t1_i], count how many other events
      j have an overlapping interval [t0_j, t1_j]. The uniqueness weight is the
      reciprocal of (1 + number of overlapping events).

    Returns
    -------
    pd.Series of uniqueness weights, indexed the same as label_df.
    """
    df = label_df.copy()
    if t0_col is None:
        t0s = df.index.to_series().values
    else:
        t0s = df[t0_col].values
    t1s = df[t1_col].values

    # Normalize to int64 so datetime64 and int64 can be compared uniformly
    t0s = _to_comparable_int(t0s)
    t1s = _to_comparable_int(t1s)

    n = len(df)
    concurrency = np.ones(n, dtype=float)

    for i in range(n):
        # Count how many windows [t0_j, t1_j] overlap with [t0_i, t1_i]
        overlaps = np.sum(
            (t0s <= t1s[i]) & (t1s >= t0s[i])
        ) - 1  # exclude self
        concurrency[i] = max(1.0, 1.0 + overlaps)

    uniqueness = 1.0 / concurrency
    return pd.Series(uniqueness, index=df.index, name="uniqueness_weight")


def _to_comparable_int(arr: np.ndarray) -> np.ndarray:
    """Convert datetime64 or Timestamp arrays to int64 ordinals for comparison."""
    if arr.dtype.kind == "M":          # datetime64
        return arr.astype(np.int64)
    if len(arr) > 0 and hasattr(arr[0], "value"):  # pd.Timestamp
        return np.array([x.value for x in arr], dtype=np.int64)
    return arr.astype(np.int64)



def compute_sample_weights(
    label_df: pd.DataFrame,
    t1_col: str = "t1",
    raw_weight_col: str = "raw_weight",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Combine uniqueness weight and return-magnitude weight into a single
    sample weight used for training.

    Steps:
      1. uniqueness_weight = get_concurrency_weights(label_df)
      2. magnitude_weight  = |return_at_touch|  (already in raw_weight)
      3. weight = uniqueness_weight * raw_weight
      4. Optionally normalize to sum-to-1

    Returns label_df with added columns:
        uniqueness_weight, sample_weight
    """
    uniq = get_concurrency_weights(label_df, t1_col=t1_col)
    label_df = label_df.copy()
    label_df["uniqueness_weight"] = uniq.values
    label_df["sample_weight"]     = label_df["uniqueness_weight"] * label_df[raw_weight_col]

    if normalize and label_df["sample_weight"].sum() > 1e-12:
        label_df["sample_weight"] /= label_df["sample_weight"].sum()

    return label_df


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional labeling wrapper
# ─────────────────────────────────────────────────────────────────────────────

def label_universe(
    feature_df: pd.DataFrame,
    pt: float = 2.0,
    sl: float = 2.0,
    max_holding: int = 20,
    vol_window: int = 20,
    min_events_per_ticker: int = 10,
    event_spacing: int = 1,   # sample every N-th bar as a candidate entry
) -> pd.DataFrame:
    """
    Apply triple-barrier labeling across all tickers in feature_df.

    Returns a combined label_df with columns:
        t0, t1, ticker, label, return_at_touch, barrier_type,
        sigma_t0, raw_weight, uniqueness_weight, sample_weight
    """
    feature_df = feature_df.sort_values(["ticker", "timestamp"])
    all_labels = []

    for ticker, grp in feature_df.groupby("ticker"):
        if len(grp) < min_events_per_ticker + vol_window:
            continue

        prices = grp.set_index("timestamp")["close"]

        # Candidate entries: every event_spacing-th bar (skip warm-up period)
        entry_times = prices.index[vol_window::event_spacing]
        events      = pd.DataFrame(index=entry_times)

        labels = get_barrier_labels(
            prices, events,
            pt=pt, sl=sl,
            max_holding=max_holding,
            vol_window=vol_window,
        )
        if labels.empty:
            continue

        labels["ticker"] = ticker
        all_labels.append(labels)

    if not all_labels:
        return pd.DataFrame()

    combined = pd.concat(all_labels)
    combined = combined.reset_index()          # t0 → column
    combined = compute_sample_weights(combined, t1_col="t1", raw_weight_col="raw_weight")
    combined.sort_values("t0", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


if __name__ == "__main__":
    from src.data.synthetic_data import generate_market_data
    from src.data.features import build_feature_matrix

    mdf = generate_market_data(start="2022-01-03", end="2022-06-30")
    fdf = build_feature_matrix(mdf)

    label_df = label_universe(fdf, pt=2.0, sl=2.0, max_holding=20, event_spacing=5)
    print("Labels shape:", label_df.shape)
    print(label_df[["t0", "t1", "ticker", "label", "barrier_type", "sample_weight"]].head(10))
    print("Label distribution:\n", label_df["label"].value_counts())
    print("Barrier type distribution:\n", label_df["barrier_type"].value_counts())
