"""Tests for conformal.py and backtest components."""
import pytest
import numpy as np
import pandas as pd
from src.calibration.conformal import SplitConformalCalibrator


class TestSplitConformal:
    """Tests for split conformal prediction calibrator."""

    def _make_data(self, n=300, noise=0.3, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        y      = rng.standard_normal(n)
        p50    = y + rng.normal(0, noise, n)
        p10    = p50 - 1.0    # initial interval half-width
        p90    = p50 + 1.0
        return p10, p50, p90, y

    def test_fit_returns_self(self):
        cal = SplitConformalCalibrator(alpha=0.10)
        p10, _, p90, y = self._make_data(200)
        result = cal.fit(p10, p90, y)
        assert result is cal

    def test_q_hat_set_after_fit(self):
        cal = SplitConformalCalibrator(alpha=0.10)
        p10, _, p90, y = self._make_data(200)
        assert cal.q_hat is None
        cal.fit(p10, p90, y)
        assert cal.q_hat is not None

    def test_transform_widens_interval(self):
        cal = SplitConformalCalibrator(alpha=0.10)
        p10, _, p90, y = self._make_data(200)
        cal.fit(p10, p90, y)
        p10_adj, p90_adj = cal.transform(p10, p90)
        assert (p10_adj <= p10).all(), "Adjusted P10 should be <= original P10"
        assert (p90_adj >= p90).all(), "Adjusted P90 should be >= original P90"

    def test_empirical_coverage_achieves_target(self):
        """After calibration, OOS coverage should be >= (1 - alpha)."""
        rng  = np.random.default_rng(7)
        p10_cal, _, p90_cal, y_cal = self._make_data(500, rng=rng)
        p10_te,  _, p90_te,  y_te  = self._make_data(300, rng=rng)

        cal = SplitConformalCalibrator(alpha=0.10)
        cal.fit(p10_cal, p90_cal, y_cal)
        p10_adj, p90_adj = cal.transform(p10_te, p90_te)

        coverage = cal.empirical_coverage(p10_adj, p90_adj, y_te)
        assert coverage >= 0.85, (
            f"Coverage {coverage:.3f} is below acceptable threshold 0.85 "
            f"(target: {1 - cal.alpha:.0%})"
        )

    def test_band_width_positive(self):
        cal = SplitConformalCalibrator(alpha=0.10)
        p10, _, p90, y = self._make_data(200)
        cal.fit(p10, p90, y)
        p10_adj, p90_adj = cal.transform(p10, p90)
        bw = cal.band_width(p10_adj, p90_adj)
        assert (bw > 0).all(), "Band widths must be positive"

    def test_transform_before_fit_raises(self):
        cal = SplitConformalCalibrator(alpha=0.10)
        with pytest.raises(RuntimeError, match="fit"):
            cal.transform(np.array([0.0]), np.array([1.0]))

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            SplitConformalCalibrator(alpha=1.5)
        with pytest.raises(ValueError):
            SplitConformalCalibrator(alpha=0.0)

    def test_mismatched_lengths_raise(self):
        cal = SplitConformalCalibrator(alpha=0.10)
        with pytest.raises(ValueError):
            cal.fit(np.array([0.0, 1.0]), np.array([1.0]), np.array([0.5, 0.5]))

    def test_fit_transform_convenience(self):
        rng  = np.random.default_rng(1)
        p10c, _, p90c, yc = self._make_data(200, rng=rng)
        p10t, _, p90t, yt = self._make_data(100, rng=rng)
        cal = SplitConformalCalibrator(alpha=0.10)
        p10_adj, p90_adj = cal.fit_transform(p10c, p90c, yc, p10t, p90t)
        coverage = cal.empirical_coverage(p10_adj, p90_adj, yt)
        assert coverage >= 0.80


class TestPositionSizer:
    def test_meta_skip_produces_zero_size(self):
        from src.execution.position_sizer import PositionSizer
        rng = np.random.default_rng(0)
        n   = 20
        df  = pd.DataFrame({
            "p50":          rng.normal(0, 0.5, n),
            "band_width":   rng.uniform(0.2, 2.0, n),
            "adv":          rng.lognormal(10, 1, n),
            "close":        rng.uniform(50, 300, n),
            "realized_vol": rng.uniform(0.005, 0.02, n),
            "meta_decision": [0] * n,   # all skip
        })
        sizer  = PositionSizer()
        result = sizer.compute_sizes(df)
        assert (result["final_size_frac"] == 0.0).all(), "Meta=0 should produce zero size"

    def test_high_confidence_larger_size(self):
        from src.execution.position_sizer import PositionSizer
        df = pd.DataFrame({
            "p50":          [0.5, 0.5],
            "band_width":   [0.1, 5.0],   # narrow vs wide
            "adv":          [1e6, 1e6],
            "close":        [100.0, 100.0],
            "realized_vol": [0.01, 0.01],
            "meta_decision": [1, 1],
        })
        sizer  = PositionSizer()
        result = sizer.compute_sizes(df)
        # Narrow band (high confidence) should produce >= size than wide band
        assert result.iloc[0]["final_size_frac"] >= result.iloc[1]["final_size_frac"]

    def test_direction_matches_p50_sign(self):
        from src.execution.position_sizer import PositionSizer
        df = pd.DataFrame({
            "p50":          [0.5, -0.5],
            "band_width":   [0.5, 0.5],
            "adv":          [1e6, 1e6],
            "close":        [100.0, 100.0],
            "realized_vol": [0.01, 0.01],
            "meta_decision": [1, 1],
        })
        sizer  = PositionSizer()
        result = sizer.compute_sizes(df)
        assert result.iloc[0]["direction"] == 1,  "Positive p50 → long"
        assert result.iloc[1]["direction"] == -1, "Negative p50 → short"
