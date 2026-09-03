"""
test_distributed_cv.py
──────────────────────
Unit tests for Distributed Purged Cross-Validation and DDP training helpers.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.validation.distributed_cv import DistributedPurgedCV
from src.models.distributed_trainer import (
    DDPConfig,
    setup_distributed_environment,
    wrap_model_ddp,
)


def _dummy_eval_fn(X_train, y_train, X_test, y_test):
    # Dummy linear model: mean of training target
    mean_val = float(np.mean(y_train))
    preds = np.full(len(y_test), mean_val)
    mse = float(np.mean((y_test - preds) ** 2))
    return {"mse": mse}, preds


def test_distributed_purged_cv_execution():
    """Test parallel evaluation across folds with metric aggregation."""
    n_samples = 100
    X = np.random.randn(n_samples, 5)
    y = np.random.randn(n_samples)
    t1 = np.arange(n_samples) + 2  # overlapping 2-step labels

    dist_cv = DistributedPurgedCV(n_splits=3, embargo_pct=0.02, n_jobs=2)
    results = dist_cv.evaluate(X, y, t1, _dummy_eval_fn)

    assert results["n_folds"] == 3
    assert "summary_metrics" in results
    assert "mse_mean" in results["summary_metrics"]
    assert len(results["fold_results"]) == 3
    assert len(results["oof_predictions"]) == n_samples
    assert np.any(results["oof_mask"])


def test_ddp_single_device_fallback():
    """Test DDP helper behaves correctly in standard non-distributed single-device runs."""
    config = setup_distributed_environment()
    assert not config.is_distributed

    linear = nn.Linear(10, 2)
    wrapped = wrap_model_ddp(linear, config)
    assert wrapped is linear  # No-op when not distributed
