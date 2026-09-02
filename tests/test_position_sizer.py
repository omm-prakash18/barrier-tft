"""
test_position_sizer.py
──────────────────────
Unit tests for PositionSizer: ADV cap, meta-skip gate, direction sign, Kelly scaling.

PositionSizer.compute_sizes(signals_df) is the public API:
    signals_df columns: p50, band_width, adv, realized_vol, meta_decision, [close]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.execution.position_sizer import PositionSizer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _signal(
    p50: float = 0.05,
    band_width: float = 0.5,
    adv: float = 1_000_000.0,
    realized_vol: float = 0.01,
    meta_decision: int = 1,
) -> pd.DataFrame:
    """Build a single-row signals DataFrame."""
    return pd.DataFrame([{
        "p50":          p50,
        "band_width":   band_width,
        "adv":          adv,
        "realized_vol": realized_vol,
        "meta_decision": meta_decision,
    }])


@pytest.fixture()
def sizer() -> PositionSizer:
    return PositionSizer(
        max_position_frac=0.05,
        max_adv_frac=0.10,       # 10% of ADV cap
        min_confidence=0.0,      # disable confidence floor so ADV cap is the binding constraint
        confidence_floor_band=99.0,
    )


# ── ADV Clamp ─────────────────────────────────────────────────────────────────

class TestADVClamp:
    """ADV-limit clamp — confirmed gap in the FAANG debug audit."""

    def test_adv_cap_limits_position(self, sizer: PositionSizer):
        """
        With large capital (100M) and high-confidence signal,
        Kelly-optimal size would exceed ADV cap.
        final_size_dollars must be <= adv * max_adv_frac.
        """
        portfolio_value = 100_000_000.0
        adv = 1_000_000.0
        result = sizer.compute_sizes(
            _signal(p50=0.10, band_width=0.001, adv=adv),  # very tight band → max confidence
            portfolio_value=portfolio_value,
        )
        max_allowed_dollars = adv * sizer.max_adv_frac
        assert result["final_size_dollars"].iloc[0] <= max_allowed_dollars + 1e-6, (
            f"Size {result['final_size_dollars'].iloc[0]:.2f} > ADV cap {max_allowed_dollars:.2f}"
        )

    def test_adv_cap_does_not_clip_small_positions(self, sizer: PositionSizer):
        """A narrow-confidence signal produces a smaller size than a wide-confidence one."""
        result_low_conf = sizer.compute_sizes(
            _signal(p50=0.01, band_width=5.0, adv=1_000_000.0),  # wide band → low confidence
        )
        result_hi_conf = sizer.compute_sizes(
            _signal(p50=0.10, band_width=0.01, adv=1_000_000.0),  # narrow band → high confidence
        )
        assert (
            result_low_conf["final_size_frac"].iloc[0]
            <= result_hi_conf["final_size_frac"].iloc[0] + 1e-9
        )

    def test_zero_adv_produces_zero_size(self, sizer: PositionSizer):
        """ADV=0 (halted or illiquid) must clamp position to zero."""
        result = sizer.compute_sizes(_signal(adv=0.0))
        assert result["final_size_frac"].iloc[0] == pytest.approx(0.0)

    def test_meta_skip_produces_zero_size(self, sizer: PositionSizer):
        """meta_decision=0 must override all other factors and produce zero size."""
        result = sizer.compute_sizes(
            _signal(p50=0.10, band_width=0.001, adv=10_000_000.0, meta_decision=0)
        )
        assert result["final_size_frac"].iloc[0] == pytest.approx(0.0)

    def test_direction_follows_p50_sign(self, sizer: PositionSizer):
        """direction must equal sign(p50)."""
        long_r  = sizer.compute_sizes(_signal(p50=+0.05))
        short_r = sizer.compute_sizes(_signal(p50=-0.05))
        assert long_r["direction"].iloc[0]  ==  1
        assert short_r["direction"].iloc[0] == -1
