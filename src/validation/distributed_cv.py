"""
distributed_cv.py
─────────────────
Distributed & Parallel Cross-Sectional Combinatorial Purged Cross-Validation.

In institutional quantitative finance, evaluating models across thousands of
assets with combinatorial purged CV (CPCV) requires parallelizing fold evaluation
across multiple cores/nodes while maintaining strict isolation:
  - Zero-copy data sharing across worker processes where possible.
  - Independent fold training, inference, and metric logging.
  - Out-of-fold prediction assembly and cross-sectional aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.validation.purged_cv import PurgedKFold


@dataclass
class FoldResult:
    """Evaluation result from a single cross-validation fold."""
    fold_idx: int
    train_size: int
    test_size: int
    purged_count: int
    metrics: Dict[str, float]
    oof_indices: np.ndarray
    oof_predictions: np.ndarray
    oof_targets: np.ndarray


def _execute_single_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    train_eval_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[Dict[str, float], np.ndarray]],
    total_samples: int,
) -> FoldResult:
    """Worker task evaluating one CV fold in an isolated worker process."""
    purged_count = total_samples - len(train_idx) - len(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    metrics, oof_preds = train_eval_fn(X_train, y_train, X_test, y_test)

    return FoldResult(
        fold_idx=fold_idx,
        train_size=len(train_idx),
        test_size=len(test_idx),
        purged_count=purged_count,
        metrics=metrics,
        oof_indices=test_idx,
        oof_predictions=oof_preds,
        oof_targets=y_test,
    )


class DistributedPurgedCV:
    """
    Parallel Combinatorial Purged Cross-Validation Coordinator.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.splitter = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        t1: Union[np.ndarray, pd.Series],
        train_eval_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[Dict[str, float], np.ndarray]],
    ) -> Dict[str, Any]:
        """
        Execute parallel cross-validation across all folds.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        t1_arr = np.asarray(t1)

        splits = list(self.splitter.split(X_arr, y_arr, t1=t1_arr))
        total_samples = len(X_arr)

        # Dispatch parallel fold jobs
        fold_results: List[FoldResult] = Parallel(
            n_jobs=self.n_jobs,
            prefer="processes",
            verbose=self.verbose,
        )(
            delayed(_execute_single_fold)(
                fold_idx=i,
                train_idx=train_idx,
                test_idx=test_idx,
                X=X_arr,
                y=y_arr,
                train_eval_fn=train_eval_fn,
                total_samples=total_samples,
            )
            for i, (train_idx, test_idx) in enumerate(splits)
        )

        # Aggregate metrics
        all_metrics: Dict[str, List[float]] = {}
        for fr in fold_results:
            for k, v in fr.metrics.items():
                all_metrics.setdefault(k, []).append(v)

        summary_metrics = {
            f"{k}_mean": float(np.mean(vals)) for k, vals in all_metrics.items()
        }
        summary_metrics.update({
            f"{k}_std": float(np.std(vals)) for k, vals in all_metrics.items()
        })

        # Assemble Out-of-Fold (OOF) predictions
        oof_preds = np.zeros_like(y_arr, dtype=float)
        oof_mask = np.zeros(len(y_arr), dtype=bool)
        for fr in fold_results:
            oof_preds[fr.oof_indices] = fr.oof_predictions
            oof_mask[fr.oof_indices] = True

        return {
            "summary_metrics": summary_metrics,
            "fold_results": fold_results,
            "oof_predictions": oof_preds,
            "oof_mask": oof_mask,
            "n_folds": len(fold_results),
        }
