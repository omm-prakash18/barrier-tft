"""
features.py
───────────
Feature engineering pipeline:

  1. Volatility-normalized returns  (return / rolling_realized_vol)
  2. Technical features             (momentum, spread, ADV)
  3. Market regime state            (rolling vol+corr clustering → integer label)
  4. Credibility-weighted FinBERT   sentiment merged with latency buffer

All operations are causal (no look-ahead). Rolling windows use .shift(1) on the
right edge so that the window at bar t only contains bars [t-window, t-1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Optional

# ── constants ─────────────────────────────────────────────────────────────────
REALIZED_VOL_WINDOW  = 20     # bars for rolling realized vol
MOMENTUM_WINDOWS     = [5, 20, 60]
ADV_WINDOW           = 20     # bars for average daily volume proxy
REGIME_CORR_WINDOW   = 60     # bars for cross-sectional correlation for regime
N_REGIMES            = 3      # {low-vol, normal, high-vol / crisis}
SENTIMENT_DECAY_BARS = 12     # how many bars a sentiment signal decays over


# ─────────────────────────────────────────────────────────────────────────────
# 1. Volatility-normalized returns
# ─────────────────────────────────────────────────────────────────────────────

def add_vol_normalized_returns(
    df: pd.DataFrame,
    price_col: str = "close",
    vol_window: int = REALIZED_VOL_WINDOW,
    min_periods: int = 5,
) -> pd.DataFrame:
    """
    Adds columns:
        log_return          — log(close_t / close_{t-1})
        realized_vol        — rolling std of log_return (annualized proxy)
        vol_norm_return     — log_return / realized_vol  (the prediction target)

    Uses .groupby(ticker) so each ticker has its own history.
    All windows use shift(1) to prevent bar-t data leaking into bar-t features.
    """
    df = df.copy()
    df.sort_values(["ticker", "timestamp"], inplace=True)

    grp = df.groupby("ticker", group_keys=False)

    # Log returns (already causal: uses close_{t-1})
    df["log_return"] = grp[price_col].transform(
        lambda x: np.log(x / x.shift(1))
    )

    # Realized vol: rolling std of PAST returns (shift(1) to exclude current bar)
    df["realized_vol"] = grp["log_return"].transform(
        lambda x: x.shift(1).rolling(vol_window, min_periods=min_periods).std()
    )

    # Vol-normalized return (the actual model target)
    df["vol_norm_return"] = df["log_return"] / df["realized_vol"].clip(lower=1e-8)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Technical features
# ─────────────────────────────────────────────────────────────────────────────

def add_technical_features(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Adds causal technical features per ticker:
        momentum_{n}    — log return over last n bars (shifted 1)
        adv             — average volume over ADV_WINDOW bars
        hl_spread       — (high - low) / close, proxy for intrabar vol
        rsi_14          — Wilder RSI (causal)
        z_price         — rolling z-score of close (60-bar window)
    """
    df = df.copy()
    df.sort_values(["ticker", "timestamp"], inplace=True)
    grp = df.groupby("ticker", group_keys=False)

    for w in MOMENTUM_WINDOWS:
        col = f"momentum_{w}"
        df[col] = grp[price_col].transform(
            lambda x, w=w: np.log(x.shift(1) / x.shift(w + 1))
        )

    df["adv"] = grp["volume"].transform(
        lambda x: x.shift(1).rolling(ADV_WINDOW, min_periods=5).mean()
    )

    df["hl_spread"] = (df["high"] - df["low"]) / df["close"].clip(lower=1e-8)

    # RSI-14 (causal)
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.shift(1).diff()
        gain  = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
        rs    = gain / loss.clip(lower=1e-8)
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = grp[price_col].transform(_rsi)

    # Rolling z-score of price (de-trends level but still causal)
    def _zscale(x: pd.Series, w: int = 60) -> pd.Series:
        mu  = x.shift(1).rolling(w, min_periods=10).mean()
        std = x.shift(1).rolling(w, min_periods=10).std().clip(lower=1e-8)
        return (x - mu) / std

    df["z_price"] = grp[price_col].transform(_zscale)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Regime clustering
# ─────────────────────────────────────────────────────────────────────────────

def compute_regime_states(
    df: pd.DataFrame,
    n_regimes: int = N_REGIMES,
    vol_window: int = REGIME_CORR_WINDOW,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Computes a market-wide regime integer label {0, 1, ..., n_regimes-1}.

    Method:
      - For each timestamp, compute cross-sectional mean realized_vol
        and cross-sectional mean pairwise correlation (from rolling returns).
      - Fit a GaussianMixture on (mean_vol, mean_corr) using ONLY past data
        up to each timestamp (walk-forward fit in expanding windows).
      - The GMM is re-fit every `refit_every` bars to remain point-in-time.

    Returns df with new column `regime` (int).
    """
    if "realized_vol" not in df.columns:
        raise ValueError("Call add_vol_normalized_returns() before compute_regime_states().")

    df = df.copy()
    df.sort_values(["timestamp", "ticker"], inplace=True)

    # Build pivot: rows=timestamp, cols=ticker
    vol_pivot  = df.pivot_table(index="timestamp", columns="ticker", values="realized_vol")
    ret_pivot  = df.pivot_table(index="timestamp", columns="ticker", values="log_return")

    timestamps = vol_pivot.index
    n          = len(timestamps)

    # pandas 3.0 removed rolling(axis=0); rolling is now always row-wise per column.
    # Compute rolling mean of realized vol for each ticker column, then average cross-sectionally.
    mean_vol = (
        vol_pivot.rolling(vol_window, min_periods=min_periods)
        .mean()
        .mean(axis=1)
    )
    # Rolling std of log returns as cross-sectional correlation proxy
    mean_corr_proxy = (
        ret_pivot.rolling(vol_window, min_periods=min_periods)
        .std()
        .mean(axis=1)
    )

    regime_features = pd.DataFrame({
        "mean_vol":        mean_vol,
        "mean_corr_proxy": mean_corr_proxy,
    }, index=timestamps).dropna()

    # Walk-forward GMM fit: fit periodically on expanding historical data
    refit_every = max(200, len(regime_features) // 20)
    regimes_arr = np.zeros(len(regime_features), dtype=int)
    
    scaler = StandardScaler()
    gmm = None
    
    # Process in chunks to vectorize predict while preserving point-in-time causality
    for chunk_start in range(0, len(regime_features), refit_every):
        chunk_end = min(chunk_start + refit_every, len(regime_features))
        fit_idx = max(min_periods, chunk_start)
        train_data = regime_features.iloc[:fit_idx].values
        
        if len(train_data) >= min_periods:
            X_train = scaler.fit_transform(train_data)
            gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=1)
            gmm.fit(X_train)
            
            chunk_data = regime_features.iloc[chunk_start:chunk_end].values
            X_chunk = scaler.transform(chunk_data)
            regimes_arr[chunk_start:chunk_end] = gmm.predict(X_chunk)
        else:
            regimes_arr[chunk_start:chunk_end] = 0

    regime_series = pd.Series(regimes_arr, index=regime_features.index)
    regimes = regime_series.reindex(timestamps).ffill().fillna(0).astype(int)

    # Map back to bar-level dataframe
    regime_map = regimes.to_dict()
    df["regime"] = df["timestamp"].map(regime_map).fillna(0).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Credibility-weighted FinBERT sentiment
# ─────────────────────────────────────────────────────────────────────────────

def merge_sentiment_features(
    bar_df: pd.DataFrame,
    news_df: pd.DataFrame,
    latency_buffer_sec: int = 45,
    decay_bars: int = SENTIMENT_DECAY_BARS,
    bar_ts_col: str = "timestamp",
    news_public_ts_col: str = "article_public_ts",
    credibility_col: str = "credibility",
    sentiment_col: str = "headline_sentiment",
) -> pd.DataFrame:
    """
    Merges credibility-weighted FinBERT sentiment into bar_df.
    High-performance implementation: iterates over news events (O(N_news))
    using searchsorted instead of looping over every bar (O(N_bars)).
    """
    bar_df  = bar_df.copy().sort_values(["ticker", bar_ts_col])
    news_df = news_df.copy()

    news_df["usable_after"] = (
        news_df[news_public_ts_col]
        + pd.Timedelta(seconds=latency_buffer_sec)
    )

    bar_df["weighted_sentiment"] = 0.0
    bar_df["sentiment_count"]    = 0
    bar_df["max_credibility"]    = 0.0

    bar_freq_seconds = _infer_bar_freq_seconds(bar_df, bar_ts_col)
    decay_window_td  = pd.Timedelta(seconds=bar_freq_seconds * decay_bars)
    max_age_sec      = max(bar_freq_seconds * decay_bars, 1.0)

    for ticker, t_bars in bar_df.groupby("ticker"):
        t_news = news_df[news_df["ticker"] == ticker].copy()
        if t_news.empty:
            continue

        t_bars_sorted = t_bars.sort_values(bar_ts_col)
        bar_timestamps = t_bars_sorted[bar_ts_col].values
        n_bars = len(bar_timestamps)

        ws_numer = np.zeros(n_bars, dtype=np.float64)
        ws_denom = np.zeros(n_bars, dtype=np.float64)
        cnt_vals = np.zeros(n_bars, dtype=np.int32)
        mc_vals  = np.zeros(n_bars, dtype=np.float64)

        for _, news_row in t_news.iterrows():
            u_after = news_row["usable_after"]
            u_after_ns = u_after.value
            u_after_dt64 = np.datetime64(u_after_ns, 'ns')
            w_end_dt64   = np.datetime64((u_after + decay_window_td).value, 'ns')


            # Find bars in range [usable_after, usable_after + decay_window]
            idx_start = np.searchsorted(bar_timestamps, u_after_dt64, side="left")
            idx_end   = np.searchsorted(bar_timestamps, w_end_dt64, side="right")

            if idx_start >= n_bars or idx_start >= idx_end:
                continue

            bar_slice_ns = bar_timestamps[idx_start:idx_end].astype("datetime64[ns]").astype(np.int64)
            ages_sec     = (bar_slice_ns - u_after_ns) / 1e9
            decay_w      = np.exp(-3.0 * ages_sec / max_age_sec)


            cred = float(news_row[credibility_col])
            sent = float(news_row[sentiment_col])
            combined_w = decay_w * cred

            ws_numer[idx_start:idx_end] += combined_w * sent
            ws_denom[idx_start:idx_end] += combined_w
            cnt_vals[idx_start:idx_end] += 1
            mc_vals[idx_start:idx_end] = np.maximum(mc_vals[idx_start:idx_end], cred)

        # Compute final weighted sentiment
        has_news = ws_denom > 1e-10
        ws_final = np.zeros(n_bars, dtype=np.float64)
        ws_final[has_news] = ws_numer[has_news] / ws_denom[has_news]

        bar_df.loc[t_bars_sorted.index, "weighted_sentiment"] = ws_final
        bar_df.loc[t_bars_sorted.index, "sentiment_count"]    = cnt_vals
        bar_df.loc[t_bars_sorted.index, "max_credibility"]    = mc_vals

    return bar_df



def _infer_bar_freq_seconds(bar_df: pd.DataFrame, ts_col: str) -> float:
    """Estimate bar frequency in seconds from the first ticker's data."""
    sample = bar_df.groupby("ticker")[ts_col].apply(
        lambda x: x.sort_values().diff().dropna().median().total_seconds()
        if len(x) > 1 else 300.0
    )
    return float(sample.median()) if not sample.empty else 300.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full feature pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    market_df: pd.DataFrame,
    news_df: Optional[pd.DataFrame] = None,
    latency_buffer_sec: int = 45,
    n_regimes: int = N_REGIMES,
    vol_window: int = REALIZED_VOL_WINDOW,
) -> pd.DataFrame:
    """
    Full causal feature pipeline:
      1. Vol-normalized returns
      2. Technical features
      3. Regime states
      4. Sentiment (if news_df provided)

    Returns a merged DataFrame ready for labeling.
    """
    df = add_vol_normalized_returns(market_df, vol_window=vol_window)
    df = add_technical_features(df)
    df = compute_regime_states(df, n_regimes=n_regimes, vol_window=max(vol_window * 3, 60))

    if news_df is not None:
        df = merge_sentiment_features(df, news_df, latency_buffer_sec=latency_buffer_sec)
    else:
        df["weighted_sentiment"] = 0.0
        df["sentiment_count"]    = 0
        df["max_credibility"]    = 0.0

    # Drop rows with all-NaN features (warm-up period)
    feature_cols = [
        "vol_norm_return", "realized_vol", "log_return",
        "momentum_5", "momentum_20", "momentum_60",
        "adv", "hl_spread", "rsi_14", "z_price",
        "regime", "weighted_sentiment",
    ]
    df.dropna(subset=["realized_vol", "momentum_60"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == "__main__":
    from src.data.synthetic_data import generate_market_data, generate_news_data

    mdf = generate_market_data(start="2022-01-03", end="2022-03-31")
    ndf = generate_news_data(mdf, n_events=300)
    fdf = build_feature_matrix(mdf, ndf)
    print("Feature matrix shape:", fdf.shape)
    print(fdf[["timestamp", "ticker", "vol_norm_return", "regime", "weighted_sentiment"]].head(10))
