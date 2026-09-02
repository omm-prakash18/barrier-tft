"""
position_sizer.py
─────────────────
Execution-aware position sizing (spec §6).

Final position size = f(confidence) adjusted for:
  1. Calibrated confidence: inverse of band_width (P90_adj - P10_adj)
  2. Bid-ask spread cost
  3. Market impact (square-root model based on ADV fraction)
  4. Fixed cost floor assumption

A high-confidence signal on an illiquid name sizes down or is skipped.

Public API:
    sizer = PositionSizer(...)
    sizes = sizer.compute_sizes(signals_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class PositionSizer:
    """
    Execution-aware position sizer.

    Parameters
    ----------
    max_position_frac   : max fraction of portfolio per position
    min_confidence      : minimum confidence (1/band_width) to trade at all
    cost_bps            : assumed fixed one-way transaction cost (basis points)
    spread_bps          : assumed typical half-spread (basis points)
    impact_coeff        : market impact coefficient k in sqrt model:
                              impact = k * sigma * sqrt(Q / ADV)
                          where Q = order size, ADV = avg daily volume
    max_adv_frac        : maximum fraction of ADV per order (liquidity limit)
    confidence_floor    : band_width above which confidence = 0 (skip signal)
    """

    def __init__(
        self,
        max_position_frac: float = 0.05,
        min_confidence: float = 0.1,
        cost_bps: float = 5.0,
        spread_bps: float = 3.0,
        impact_coeff: float = 0.1,
        max_adv_frac: float = 0.01,
        confidence_floor_band: float = 10.0,
    ):
        self.max_position_frac      = max_position_frac
        self.min_confidence         = min_confidence
        self.cost_bps               = cost_bps / 10_000.0
        self.spread_bps             = spread_bps / 10_000.0
        self.impact_coeff           = impact_coeff
        self.max_adv_frac           = max_adv_frac
        self.confidence_floor_band  = confidence_floor_band

    def compute_sizes(
        self,
        signals_df: pd.DataFrame,
        portfolio_value: float = 1_000_000.0,
    ) -> pd.DataFrame:
        """
        Compute position sizes for a batch of signals.

        Expected columns in signals_df:
            p50         : point forecast (vol-normalized return)
            band_width  : calibrated interval width (P90_adj - P10_adj)
            adv         : average daily volume in shares (or dollar ADV)
            realized_vol: rolling realized volatility at signal time
            meta_decision: 0 or 1 from meta-labeling (1 = act, 0 = skip)
            [close]     : last close price (for dollar sizing, optional)

        Returns signals_df with added columns:
            confidence        : 1 / band_width (clipped)
            total_cost_est    : spread + fixed_cost (fraction)
            net_expected_ret  : |p50| - total_cost_est
            adv_size_limit    : max $ size based on ADV fraction
            raw_size_frac     : confidence-scaled fraction (before ADV cap)
            final_size_frac   : final portfolio fraction (0 if skip/below min)
            final_size_dollars: final_size_frac * portfolio_value
            direction         : +1 (long) or -1 (short) based on p50 sign
        """
        df = signals_df.copy()

        # ── 1. Confidence from calibrated band width ──────────────────────
        bw               = df["band_width"].clip(lower=1e-4)
        df["confidence"] = (1.0 / bw).clip(upper=1.0 / 1e-4)
        # Normalize confidence to [0, 1] relative to band floor
        max_conf         = 1.0 / 1e-4
        df["confidence"] = (df["confidence"] / max_conf).clip(0.0, 1.0)

        # ── 2. Total cost estimate ────────────────────────────────────────
        df["total_cost_est"] = self.spread_bps + self.cost_bps

        # ── 3. Net expected return ────────────────────────────────────────
        df["net_expected_ret"] = df["p50"].abs() - df["total_cost_est"]

        # ── 4. ADV-based liquidity size limit ─────────────────────────────
        # adv is in shares; close is price. ADV_dollar = adv * close
        if "close" in df.columns:
            adv_dollar = df["adv"] * df["close"]
        else:
            adv_dollar = df["adv"]   # assume ADV is already in dollars

        df["adv_size_limit"] = adv_dollar * self.max_adv_frac / portfolio_value

        # ── 5. Market impact estimate (sqrt model) ─────────────────────────
        # Assume order = raw_size_frac * portfolio_value dollars
        # impact = k * sigma * sqrt(order_dollars / adv_dollars)
        # We iterate: first compute impact at max_adv_frac to get a cost floor
        sigma_t = df.get("realized_vol", pd.Series(np.ones(len(df)) * 0.01, index=df.index))
        raw_order_frac    = df["confidence"] * self.max_position_frac
        order_as_adv_frac = raw_order_frac / df["adv_size_limit"].clip(lower=1e-8)
        impact_cost       = (
            self.impact_coeff
            * sigma_t.values
            * np.sqrt(order_as_adv_frac.values.clip(0))
        )
        df["impact_cost_est"] = impact_cost

        # Adjusted total cost
        total_cost_adj = df["total_cost_est"] + df["impact_cost_est"]

        # ── 6. Raw size (confidence-scaled, before ADV cap) ───────────────
        df["raw_size_frac"] = df["confidence"] * self.max_position_frac

        # ── 7. Apply filters ──────────────────────────────────────────────
        skip_mask = (
            (df.get("meta_decision", pd.Series(1, index=df.index)) == 0)     # meta skip
            | (df["confidence"] < self.min_confidence)                         # too uncertain
            | (df["net_expected_ret"] < total_cost_adj)                        # not profitable
            | (bw > self.confidence_floor_band)                                # band too wide
        )

        df["final_size_frac"]    = df["raw_size_frac"].clip(upper=df["adv_size_limit"])
        df.loc[skip_mask, "final_size_frac"] = 0.0

        df["direction"]          = np.sign(df["p50"]).astype(int)
        df["final_size_dollars"] = df["final_size_frac"] * portfolio_value

        return df

    def summary(self, sized_df: pd.DataFrame) -> dict:
        """Compute sizing summary statistics."""
        acted     = sized_df[sized_df["final_size_frac"] > 0]
        skipped   = sized_df[sized_df["final_size_frac"] == 0]
        return {
            "total_signals":  len(sized_df),
            "acted":          len(acted),
            "skipped":        len(skipped),
            "act_rate":       len(acted) / max(len(sized_df), 1),
            "mean_size_frac": acted["final_size_frac"].mean() if len(acted) > 0 else 0.0,
            "mean_confidence": sized_df["confidence"].mean(),
        }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n   = 100
    df  = pd.DataFrame({
        "p50":        rng.normal(0, 0.5, n),
        "band_width": rng.uniform(0.2, 5.0, n),
        "adv":        rng.lognormal(10, 1, n),
        "close":      rng.uniform(20, 500, n),
        "realized_vol": rng.uniform(0.005, 0.03, n),
        "meta_decision": rng.integers(0, 2, n),
    })

    sizer  = PositionSizer()
    result = sizer.compute_sizes(df, portfolio_value=1_000_000)
    stats  = sizer.summary(result)
    print("Sizing summary:", stats)
    print(result[["p50", "confidence", "final_size_frac", "final_size_dollars", "direction"]].head(10))
    # Verify: skipped signals have size = 0
    assert (result.loc[result["meta_decision"] == 0, "final_size_frac"] == 0).all()
    print("✓ PositionSizer test passed.")
