"""
conformal.py
────────────
Split conformal prediction calibrator for TFT quantile outputs.

Spec §4 requirement:
  "Wrap outputs in split conformal prediction using a held-out calibration
   fold (itself purged/embargoed from training) to guarantee empirical
   coverage of the P10–P90 band."

Method (split conformal, Papadopoulos et al. / Angelopoulos et al.):
  1. On a held-out calibration set (purged/embargoed from training), compute
     the nonconformity score for each sample:
         s_i = max(P10_i - y_i, y_i - P90_i)
     Positive when y falls OUTSIDE [P10, P90]; negative when inside.
  2. Compute the empirical (1 - alpha) quantile of the scores:
         q_hat = quantile(scores, (1 - alpha) * (1 + 1/n))
  3. At test time, expand the interval:
         P10_adj = P10 - q_hat
         P90_adj = P90 + q_hat
  This guarantees marginal coverage P(y in [P10_adj, P90_adj]) >= 1 - alpha.

Usage:
    cal = SplitConformalCalibrator(alpha=0.10)
    cal.fit(p10_cal, p90_cal, y_cal)           # calibrate on held-out fold
    p10_adj, p90_adj = cal.transform(p10, p90) # widen intervals at test time
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class SplitConformalCalibrator:
    """
    Split conformal prediction interval calibrator.

    Parameters
    ----------
    alpha : miscoverage level. Default 0.10 → target 90% coverage.
    """

    def __init__(self, alpha: float = 0.10):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        self.alpha   = alpha
        self.q_hat: Optional[float] = None
        self._n_cal: int = 0

    def fit(
        self,
        p10_cal: np.ndarray,
        p90_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "SplitConformalCalibrator":
        """
        Compute the calibration quantile from held-out predictions and targets.

        Parameters
        ----------
        p10_cal : (N,) — lower quantile predictions on calibration set
        p90_cal : (N,) — upper quantile predictions on calibration set
        y_cal   : (N,) — true labels on calibration set

        Nonconformity score:
            s_i = max(p10_cal_i - y_i, y_i - p90_cal_i)

        s_i > 0  ↔  y_i is outside [p10, p90]
        s_i <= 0 ↔  y_i is inside  [p10, p90]
        """
        p10_cal = np.asarray(p10_cal, dtype=np.float64).ravel()
        p90_cal = np.asarray(p90_cal, dtype=np.float64).ravel()
        y_cal   = np.asarray(y_cal,   dtype=np.float64).ravel()

        if not (len(p10_cal) == len(p90_cal) == len(y_cal)):
            raise ValueError("p10_cal, p90_cal, and y_cal must have the same length.")

        scores = np.maximum(p10_cal - y_cal, y_cal - p90_cal)

        n            = len(scores)
        self._n_cal  = n
        level        = np.ceil((1.0 - self.alpha) * (n + 1)) / n
        level        = min(level, 1.0)
        # Clamp q_hat to >= 0: negative means raw intervals already have sufficient
        # coverage. We never shrink intervals below the raw model output.
        self.q_hat   = max(0.0, float(np.quantile(scores, level)))

        # Report empirical coverage before calibration
        raw_coverage = np.mean((y_cal >= p10_cal) & (y_cal <= p90_cal))
        adj_coverage = np.mean(
            (y_cal >= p10_cal - self.q_hat) & (y_cal <= p90_cal + self.q_hat)
        )
        print(
            f"[Conformal] Calibrated on {n} samples | "
            f"Raw coverage: {raw_coverage:.3f} | "
            f"Adjusted coverage (target {1 - self.alpha:.0%}): {adj_coverage:.3f} | "
            f"q_hat: {self.q_hat:.6f}"
        )
        return self

    def transform(
        self,
        p10: np.ndarray,
        p90: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Expand prediction intervals by q_hat.

        Returns
        -------
        p10_adj : p10 - q_hat
        p90_adj : p90 + q_hat
        """
        if self.q_hat is None:
            raise RuntimeError("Call fit() before transform().")
        p10 = np.asarray(p10, dtype=np.float64)
        p90 = np.asarray(p90, dtype=np.float64)
        return p10 - self.q_hat, p90 + self.q_hat

    def fit_transform(
        self,
        p10_cal: np.ndarray,
        p90_cal: np.ndarray,
        y_cal: np.ndarray,
        p10_test: np.ndarray,
        p90_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: fit on calibration set, transform test set."""
        self.fit(p10_cal, p90_cal, y_cal)
        return self.transform(p10_test, p90_test)

    def empirical_coverage(
        self,
        p10_adj: np.ndarray,
        p90_adj: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Compute empirical coverage rate of adjusted intervals."""
        y  = np.asarray(y_true).ravel()
        lo = np.asarray(p10_adj).ravel()
        hi = np.asarray(p90_adj).ravel()
        return float(np.mean((y >= lo) & (y <= hi)))

    def band_width(self, p10_adj: np.ndarray, p90_adj: np.ndarray) -> np.ndarray:
        """
        Returns the interval width (P90_adj - P10_adj).
        Used downstream as inverse-confidence for position sizing.
        """
        return np.asarray(p90_adj) - np.asarray(p10_adj)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Simulate calibration set: true labels from N(0,1), raw intervals slightly narrow
    n_cal = 500
    y_cal   = rng.standard_normal(n_cal)
    p50_cal = y_cal + rng.normal(0, 0.3, n_cal)    # biased predictions
    p10_cal = p50_cal - 0.8   # too narrow
    p90_cal = p50_cal + 0.8

    cal = SplitConformalCalibrator(alpha=0.10)
    cal.fit(p10_cal, p90_cal, y_cal)

    # Test set
    n_test  = 200
    y_test  = rng.standard_normal(n_test)
    p50_t   = y_test + rng.normal(0, 0.3, n_test)
    p10_t   = p50_t - 0.8
    p90_t   = p50_t + 0.8
    p10_adj, p90_adj = cal.transform(p10_t, p90_t)

    cov = cal.empirical_coverage(p10_adj, p90_adj, y_test)
    print(f"Test coverage: {cov:.3f}  (target: {1 - cal.alpha:.0%})")
    assert cov >= 0.85, f"Coverage too low: {cov}"
    print("[OK] SplitConformalCalibrator test passed.")

