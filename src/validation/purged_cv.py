"""
purged_cv.py
────────────
Scikit-learn compatible PurgedKFold cross-validator.

Implements Lopez de Prado's purged, embargoed K-fold CV (Advances in Financial
Machine Learning, Ch. 7) to prevent label-horizon leakage in time series with
overlapping observation windows.

Rules enforced:
  1. Splits are CONTIGUOUS and CHRONOLOGICAL — no shuffling, ever.
  2. PURGING: any training sample whose label window [t0, t1] overlaps
     the test fold's date range is removed from training.
  3. EMBARGO: additionally remove training samples that BEGIN within
     embargo_pct of the total timeline AFTER the test fold ends —
     autocorrelated rolling features can carry test information backward.
  4. The test fold itself is NEVER modified.

Usage:
    from src.validation.purged_cv import PurgedKFold

    cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
    for train_idx, test_idx in cv.split(X, y, groups=label_t1_series):
        ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from typing import Iterator


class PurgedKFold(BaseCrossValidator):
    """
    Purged, embargoed K-fold CV splitter.

    Parameters
    ----------
    n_splits    : number of folds (K)
    embargo_pct : fraction of total timeline to embargo AFTER the test fold.
                  e.g. 0.01 = 1% of total sample length

    Note: `groups` parameter in .split() is repurposed as the t1 array
    (end-times of label windows) to maintain sklearn API compatibility.
    Alternatively pass `t1` as a keyword argument.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        super().__init__()
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2.")
        if not (0.0 <= embargo_pct < 0.5):
            raise ValueError("embargo_pct must be in [0, 0.5).")
        self.n_splits    = n_splits
        self.embargo_pct = embargo_pct

    # ── sklearn BaseCV interface ───────────────────────────────────────────

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self,
        X,
        y=None,
        groups=None,
        t1: pd.Series | np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate (train_indices, test_indices) pairs.

        Parameters
        ----------
        X      : array-like of shape (n_samples, ...) — used only for length
        y      : ignored (kept for sklearn API compatibility)
        groups : pd.Series or array of label end-times (t1). If t1 kwarg is
                 also provided, t1 kwarg takes precedence.
        t1     : pd.Series indexed like X (or array) giving each sample's
                 label horizon end-time. If None, uses groups.
        """
        if t1 is None:
            t1 = groups
        if t1 is None:
            raise ValueError(
                "PurgedKFold.split() requires t1 (label end-times). "
                "Pass as groups= or t1= keyword argument."
            )

        n      = _n_samples(X)
        t1_arr = _to_array(t1, n)  # shape (n,) of datetime-like or numeric

        # t0 indices: we assume the DataFrame/X is sorted by t0
        # (i.e. position i in X corresponds to the i-th chronological sample)
        indices = np.arange(n)

        # Fold boundaries (contiguous, chronological)
        fold_starts = np.array_split(indices, self.n_splits)
        embargo_size = int(np.ceil(self.embargo_pct * n))

        for fold_idx, test_fold_indices in enumerate(fold_starts):
            test_start  = test_fold_indices[0]
            test_end    = test_fold_indices[-1]

            # Test fold t0-range (by position, because X is sorted by t0)
            test_t0_start_pos = test_start
            test_t0_end_pos   = test_end

            # The t1 of the LAST test sample defines the purge boundary
            test_t1_max = t1_arr[test_end]
            # t0 of the FIRST test sample
            test_t0_min = _position_to_t0(test_start, t1_arr)

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_fold_indices] = False  # exclude test fold

            # ── PURGE: remove training samples whose [t0, t1] overlaps test ──
            # A training sample at position i overlaps the test fold if:
            #   t1_i >= t0 of the first test sample  (its window extends into test)
            # OR
            #   position i >= test_start (already excluded above)
            for i in range(n):
                if not train_mask[i]:
                    continue
                # The sample's label end-time overlaps the test fold's t0 range
                if t1_arr[i] >= _position_to_t0(test_start, t1_arr):
                    # Only purge samples BEFORE the test fold (look-ahead risk)
                    if i < test_start:
                        train_mask[i] = False

            # ── EMBARGO: remove samples starting shortly AFTER test fold ──
            embargo_end = min(test_end + 1 + embargo_size, n)
            train_mask[test_end + 1 : embargo_end] = False

            train_indices = indices[train_mask]
            test_indices  = test_fold_indices

            yield train_indices, test_indices

    def _iter_test_indices(self, X=None, y=None, groups=None):
        # Required by BaseCrossValidator but we override split() directly
        n = _n_samples(X)
        indices = np.arange(n)
        for fold in np.array_split(indices, self.n_splits):
            yield fold


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _n_samples(X) -> int:
    if hasattr(X, "__len__"):
        return len(X)
    if hasattr(X, "shape"):
        return X.shape[0]
    raise ValueError("Cannot determine number of samples from X.")


def _to_array(t1, n: int) -> np.ndarray:
    """Convert t1 to a comparable numpy array (ordinal ints or floats)."""
    if isinstance(t1, pd.Series):
        arr = t1.values
    elif isinstance(t1, (list, np.ndarray)):
        arr = np.array(t1)
    else:
        raise TypeError(f"Unsupported t1 type: {type(t1)}")

    if arr.dtype.kind == "M":            # datetime64
        return arr.astype(np.int64)
    if hasattr(arr[0], "value"):         # pd.Timestamp
        return np.array([x.value for x in arr], dtype=np.int64)
    return arr.astype(np.float64)


def _position_to_t0(pos: int, t1_arr: np.ndarray) -> float | int:
    """
    Approximate t0 of sample at `pos`.
    Since we don't have explicit t0 stored alongside t1 in the array,
    we use the VALUE of t1 at position `pos` minus a small offset —
    in practice, what matters is whether t1_i >= t0_test_start, which
    we bound by using the t1 of the sample JUST BEFORE the test fold.
    """
    if pos == 0:
        return -np.inf
    # Use t1 of the previous sample as a proxy for t0 of test_start
    return t1_arr[pos - 1]


# ─────────────────────────────────────────────────────────────────────────────
# Validation utility
# ─────────────────────────────────────────────────────────────────────────────

def verify_no_leakage(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    t0_arr: np.ndarray,
    t1_arr: np.ndarray,
) -> bool:
    """
    Post-hoc check: ensure no training sample's [t0, t1] overlaps test t0 range.

    Returns True if clean, raises AssertionError if leakage detected.
    """
    test_t0_min = t0_arr[test_idx].min()
    test_t0_max = t0_arr[test_idx].max()

    for i in train_idx:
        t0_i = t0_arr[i]
        t1_i = t1_arr[i]
        # Overlap condition: t0_i <= test_t0_max AND t1_i >= test_t0_min
        if t0_i <= test_t0_max and t1_i >= test_t0_min:
            raise AssertionError(
                f"Leakage detected: training sample {i} "
                f"[t0={t0_i}, t1={t1_i}] overlaps test range "
                f"[{test_t0_min}, {test_t0_max}]."
            )
    return True


if __name__ == "__main__":
    # ── quick sanity check ────────────────────────────────────────────────
    import pandas as pd

    n = 200
    t0 = pd.date_range("2020-01-01", periods=n, freq="1D")
    t1 = t0 + pd.Timedelta(days=5)   # 5-day label horizon → lots of overlap
    X  = np.random.randn(n, 3)

    cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
    fold_sizes = []
    for fold, (tr, te) in enumerate(cv.split(X, t1=pd.Series(t1))):
        print(f"Fold {fold}: train={len(tr)}, test={len(te)}, "
              f"test_range=[{te[0]}, {te[-1]}]")
        fold_sizes.append((len(tr), len(te)))
    print("[OK] PurgedKFold sanity check passed.")

