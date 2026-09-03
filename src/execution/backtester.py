"""
backtester.py
─────────────
Cross-sectional event-driven backtesting engine (spec §8).

Simulates holding positions opened at signal time t0, closed at t1,
tracks P&L, positions, turnover, slippage, and costs.

Key design choices:
  • Positions are sized per the execution-aware PositionSizer output.
  • Entry and exit prices are simulated with spread slippage.
  • No look-ahead: exit price known only AFTER t1 (uses actual barrier price).
  • Concurrent position limits enforced (max_concurrent_positions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.execution.position_sizer import PositionSizer


class BacktestResult:
    """Container for backtest output."""

    def __init__(
        self,
        equity_curve: pd.Series,
        trade_log: pd.DataFrame,
        daily_returns: pd.Series,
    ):
        self.equity_curve  = equity_curve
        self.trade_log     = trade_log
        self.daily_returns = daily_returns

    def sharpe_ratio(self, annualize: int = 252) -> float:
        r = self.daily_returns.dropna()
        if r.std() < 1e-10:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(annualize))

    def max_drawdown(self) -> float:
        ec  = self.equity_curve
        roll_max = ec.cummax()
        dd   = (ec - roll_max) / roll_max.clip(lower=1e-8)
        return float(dd.min())

    def sortino_ratio(self, annualize: int = 252) -> float:
        r    = self.daily_returns.dropna()
        neg  = r[r < 0]
        down_std = neg.std() if len(neg) > 1 else 1e-10
        return float(r.mean() / down_std * np.sqrt(annualize)) if down_std > 1e-10 else 0.0

    def calmar_ratio(self) -> float:
        ann_ret = self.daily_returns.mean() * 252
        mdd     = abs(self.max_drawdown())
        return float(ann_ret / mdd) if mdd > 1e-10 else 0.0

    def win_rate(self) -> float:
        pnl = self.trade_log["net_pnl"]
        return float((pnl > 0).mean())

    def summary(self) -> dict:
        return {
            "sharpe_ratio":   self.sharpe_ratio(),
            "sortino_ratio":  self.sortino_ratio(),
            "calmar_ratio":   self.calmar_ratio(),
            "max_drawdown":   self.max_drawdown(),
            "win_rate":       self.win_rate(),
            "n_trades":       len(self.trade_log),
            "total_return":   (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1,
        }


class Backtester:
    """
    Cross-sectional event-driven backtester.

    Parameters
    ----------
    initial_capital      : Starting portfolio value (dollars).
    max_concurrent_pos   : Maximum number of open positions at any time.
    spread_bps           : Half-spread slippage per side (basis points).
    cost_bps             : Fixed cost per trade (basis points).
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        max_concurrent_pos: int = 20,
        spread_bps: float = 3.0,
        cost_bps: float = 5.0,
    ):
        self.initial_capital     = initial_capital
        self.max_concurrent_pos  = max_concurrent_pos
        self.spread_bps          = spread_bps / 10_000.0
        self.cost_bps            = cost_bps / 10_000.0

    def run(
        self,
        signals_df: pd.DataFrame,
        price_df: pd.DataFrame,
        sizer: Optional[PositionSizer] = None,
    ) -> BacktestResult:
        """
        Execute a cross-sectional event-driven backtest.

        Parameters
        ----------
        signals_df : Must have columns: t0, t1, ticker, direction (±1),
                     final_size_frac, final_size_dollars, return_at_touch.
                     Output of PositionSizer.compute_sizes() joined with label_df.
        price_df   : Full bar-level price DataFrame for actual P&L lookup.
        sizer      : PositionSizer instance (used for reference, not re-run here).

        Returns
        -------
        BacktestResult with equity curve, trade log, and daily returns.
        """
        # Only take "acted" signals
        acted = signals_df[signals_df["final_size_frac"] > 0].copy()
        if acted.empty:
            start_ts = price_df["timestamp"].min() if not price_df.empty else pd.Timestamp.now(tz="UTC")
            end_ts   = price_df["timestamp"].max() if not price_df.empty else pd.Timestamp.now(tz="UTC")
            daily_idx = pd.date_range(start=start_ts.date(), end=end_ts.date(), freq="D", tz="UTC")
            return BacktestResult(
                equity_curve=pd.Series(self.initial_capital, index=daily_idx),
                trade_log=pd.DataFrame(),
                daily_returns=pd.Series(0.0, index=daily_idx),
                initial_capital=self.initial_capital,
            )

        acted.sort_values("t0", inplace=True)

        capital      = self.initial_capital
        equity_curve = {}
        trade_records = []

        # Track open positions by a running count
        open_positions: list[dict] = []

        all_timestamps = pd.date_range(
            start=acted["t0"].min().date(),
            end=acted["t1"].max().date(),
            freq="D",
        )


        # Pre-index price arrays per ticker for fast searchsorted lookups
        price_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if not price_df.empty:
            for ticker, grp in price_df.sort_values("timestamp").groupby("ticker"):
                price_index[str(ticker)] = (
                    pd.to_datetime(grp["timestamp"]).values,
                    grp["close"].to_numpy(dtype=float),
                )

        def _fast_entry_price(ticker: str, t0: pd.Timestamp) -> Optional[float]:
            if ticker not in price_index:
                return None
            ts_arr, close_arr = price_index[ticker]
            target_ts = np.datetime64(pd.Timestamp(t0).tz_convert("UTC").tz_localize(None) if pd.Timestamp(t0).tz is not None else pd.Timestamp(t0))
            idx = np.searchsorted(ts_arr.astype("datetime64[ns]"), target_ts)
            if idx < len(close_arr):
                return float(close_arr[idx])
            return None

        def _fast_exit_price(ticker: str, t1: pd.Timestamp) -> Optional[float]:
            if ticker not in price_index:
                return None
            ts_arr, close_arr = price_index[ticker]
            target_ts = np.datetime64(pd.Timestamp(t1).tz_convert("UTC").tz_localize(None) if pd.Timestamp(t1).tz is not None else pd.Timestamp(t1))
            idx = np.searchsorted(ts_arr.astype("datetime64[ns]"), target_ts, side="right") - 1
            if 0 <= idx < len(close_arr):
                return float(close_arr[idx])
            return None

        for ts in all_timestamps:
            ts = pd.Timestamp(ts, tz="UTC")
            ts_end = ts + pd.Timedelta(days=1)

            # Close positions that expired by today
            still_open = []
            for pos in open_positions:
                if pos["t1"] <= ts_end:
                    # Position closed at t1
                    close_price = _fast_exit_price(pos["ticker"], pos["t1"])
                    if close_price is None:
                        # Use return_at_touch proxy
                        close_price = pos["entry_price"] * (1.0 + pos["return_at_touch"])

                    raw_pnl     = (close_price - pos["entry_price"]) * pos["direction"] * pos["shares"]
                    cost_pnl    = -self.cost_bps * abs(pos["notional"])  # exit cost
                    spread_pnl  = -self.spread_bps * abs(pos["notional"])  # exit spread
                    net_pnl     = raw_pnl + cost_pnl + spread_pnl
                    capital    += net_pnl

                    trade_records.append({
                        "t0":             pos["t0"],
                        "t1":             pos["t1"],
                        "ticker":         pos["ticker"],
                        "direction":      pos["direction"],
                        "entry_price":    pos["entry_price"],
                        "exit_price":     close_price,
                        "shares":         pos["shares"],
                        "notional":       pos["notional"],
                        "gross_pnl":      raw_pnl,
                        "cost_pnl":       cost_pnl + spread_pnl,
                        "net_pnl":        net_pnl,
                        "return_at_touch": pos["return_at_touch"],
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

            # Open new signals starting today
            new_today = acted[(acted["t0"] >= ts) & (acted["t0"] < ts_end)]
            for _, sig in new_today.iterrows():
                if len(open_positions) >= self.max_concurrent_pos:
                    break
                if capital <= 0:
                    break

                entry_price = _fast_entry_price(sig["ticker"], sig["t0"])
                if entry_price is None or entry_price <= 0:
                    continue

                notional = sig["final_size_dollars"]
                shares   = notional / entry_price

                # Entry slippage
                spread_cost = -self.spread_bps * notional
                entry_cost  = -self.cost_bps * notional
                capital    += spread_cost + entry_cost

                open_positions.append({
                    "t0":             sig["t0"],
                    "t1":             sig["t1"],
                    "ticker":         sig["ticker"],
                    "direction":      sig["direction"],
                    "entry_price":    entry_price,
                    "notional":       notional,
                    "shares":         shares,
                    "return_at_touch": sig.get("return_at_touch", 0.0),
                })

            equity_curve[ts] = capital

        eq_series  = pd.Series(equity_curve).sort_index()
        trade_log  = pd.DataFrame(trade_records)
        daily_rets = eq_series.pct_change().dropna()

        return BacktestResult(eq_series, trade_log, daily_rets)

    # ── Price lookup helpers ──────────────────────────────────────────────

    def _lookup_entry_price(
        self,
        price_df: pd.DataFrame,
        ticker: str,
        t0: pd.Timestamp,
    ) -> Optional[float]:
        """Find the closing price at or just after t0 for the given ticker."""
        ticker_bars = price_df[price_df["ticker"] == ticker]
        future = ticker_bars[ticker_bars["timestamp"] >= t0]
        if future.empty:
            return None
        return float(future.iloc[0]["close"])

    def _lookup_exit_price(
        self,
        price_df: pd.DataFrame,
        ticker: str,
        t1: pd.Timestamp,
    ) -> Optional[float]:
        """Find the closing price at or just before t1."""
        ticker_bars = price_df[price_df["ticker"] == ticker]
        before = ticker_bars[ticker_bars["timestamp"] <= t1]
        if before.empty:
            return None
        return float(before.iloc[-1]["close"])


def run_regime_split_analysis(
    result: BacktestResult,
    label_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Break down performance by market regime.

    Returns a DataFrame with per-regime Sharpe, win-rate, and n_trades.
    """
    trade_log = result.trade_log.copy()
    if "regime" not in trade_log.columns:
        # Try to join regime from label_df
        if "regime" in label_df.columns:
            regime_map = label_df.set_index(["ticker", "t0"])["regime"].to_dict()
            trade_log["regime"] = trade_log.apply(
                lambda r: regime_map.get((r["ticker"], r["t0"]), -1), axis=1
            )
        else:
            trade_log["regime"] = -1

    rows = []
    for regime, grp in trade_log.groupby("regime"):
        ret    = grp["net_pnl"] / grp["notional"].clip(lower=1.0)
        sr     = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 1e-10 else 0.0
        rows.append({
            "regime":    regime,
            "n_trades":  len(grp),
            "win_rate":  (grp["net_pnl"] > 0).mean(),
            "mean_ret":  ret.mean(),
            "sharpe":    sr,
        })

    return pd.DataFrame(rows).sort_values("regime")


if __name__ == "__main__":
    from src.data.synthetic_data import generate_market_data, generate_news_data
    from src.data.features import build_feature_matrix
    from src.labeling.triple_barrier import label_universe
    from src.execution.position_sizer import PositionSizer

    print("Generating synthetic data for backtest smoke test…")
    mdf  = generate_market_data(start="2021-01-04", end="2022-12-31")
    fdf  = build_feature_matrix(mdf)
    ldf  = label_universe(fdf, pt=2.0, sl=2.0, max_holding=20, event_spacing=10)

    # Create a dummy signals DataFrame
    signals = ldf.merge(fdf[["timestamp", "ticker", "close", "adv", "realized_vol"]],
                        left_on=["t0", "ticker"], right_on=["timestamp", "ticker"], how="left")
    signals["p50"]          = signals["return_at_touch"] * 0.7 + np.random.randn(len(signals)) * 0.01
    signals["band_width"]   = np.random.uniform(0.5, 3.0, len(signals))
    signals["meta_decision"]= 1
    sizer  = PositionSizer()
    signals = sizer.compute_sizes(signals, portfolio_value=1_000_000)

    bt    = Backtester(initial_capital=1_000_000, max_concurrent_pos=10)
    result = bt.run(signals, mdf, sizer)

    print("── Backtest Summary ──")
    for k, v in result.summary().items():
        print(f"  {k}: {v:.4f}")
    print(f"  Trades: {len(result.trade_log)}")
