"""
baseline.py
───────────
GBDT baseline model evaluated with PurgedKFold.

Purpose (spec §4, build order step 4):
  "If a simple model can't beat random on properly-validated data,
   the TFT won't either."

This module:
  1. Trains LightGBM classifiers (3-class: -1, 0, +1) on hand features.
  2. Uses PurgedKFold for all CV — no standard K-fold, ever.
  3. Reports OOS accuracy, log-loss, and balanced accuracy per fold.
  4. Outputs a flat score array for meta-labeling / downstream usage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
)
from sklearn.preprocessing import LabelEncoder
from typing import Optional

from src.validation.purged_cv import PurgedKFold


# ── hand-feature columns used by baseline ────────────────────────────────────
BASELINE_FEATURE_COLS = [
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "rsi_14",
    "hl_spread",
    "z_price",
    "realized_vol",
    "weighted_sentiment",
    "sentiment_count",
    "max_credibility",
    "regime",
    "adv",
]


class GBDTBaseline:
    """
    LightGBM baseline trained with purged K-fold CV.

    Attributes
    ----------
    models   : list of fitted LGBMClassifier (one per fold)
    oof_preds: out-of-fold class probabilities (n_samples × 3)
    fold_metrics: list of dicts with per-fold metrics
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        lgbm_params: Optional[dict] = None,
        n_estimators: int = 300,
        random_state: int = 42,
    ):
        self.n_splits     = n_splits
        self.embargo_pct  = embargo_pct
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.lgbm_params  = lgbm_params or {}
        self.models: list  = []
        self.oof_preds: Optional[np.ndarray] = None
        self.fold_metrics: list = []
        self._le = LabelEncoder()

    # ─────────────────────────────────────────────────────────────────────
    def fit(
        self,
        feature_df: pd.DataFrame,
        label_df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        label_col: str = "label",
        weight_col: str = "sample_weight",
    ) -> "GBDTBaseline":
        """
        Fit the baseline with PurgedKFold cross-validation.

        Parameters
        ----------
        feature_df  : Full feature matrix (must be sorted by timestamp).
        label_df    : Output of label_universe() — must have t0, t1, label, sample_weight.
        feature_cols: Columns to use as features (defaults to BASELINE_FEATURE_COLS).
        label_col   : Column in label_df with {-1, 0, +1} labels.
        weight_col  : Column in label_df with sample weights.

        Returns self.
        """
        if feature_cols is None:
            feature_cols = [c for c in BASELINE_FEATURE_COLS if c in feature_df.columns]

        # ── join features to labels on (ticker, t0) ───────────────────────
        merged = _join_features_labels(feature_df, label_df, feature_cols)
        if merged.empty:
            raise ValueError("No samples after joining features to labels.")

        X        = merged[feature_cols].values.astype(np.float32)
        y_raw    = merged[label_col].values
        weights  = merged[weight_col].values if weight_col in merged.columns else None
        t1_series = merged["t1"]

        # Encode labels: {-1→0, 0→1, +1→2}
        self._le.fit([-1, 0, 1])
        y = self._le.transform(y_raw)

        n = len(X)
        self.oof_preds = np.zeros((n, len(self._le.classes_)), dtype=np.float32)

        cv = PurgedKFold(n_splits=self.n_splits, embargo_pct=self.embargo_pct)

        for fold_idx, (train_idx, test_idx) in enumerate(
            cv.split(X, y, t1=t1_series)
        ):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            w_tr = weights[train_idx] if weights is not None else None

            params = {
                "n_estimators":   self.n_estimators,
                "learning_rate":  0.05,
                "num_leaves":     31,
                "min_child_samples": 20,
                "subsample":      0.8,
                "colsample_bytree": 0.8,
                "class_weight":   "balanced",
                "random_state":   self.random_state,
                "n_jobs":         -1,
                "verbose":        -1,
                **self.lgbm_params,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_te, y_te)],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            self.models.append(model)

            proba = model.predict_proba(X_te)
            self.oof_preds[test_idx] = proba

            y_pred = proba.argmax(axis=1)
            metrics = {
                "fold":          fold_idx,
                "n_train":       len(train_idx),
                "n_test":        len(test_idx),
                "accuracy":      accuracy_score(y_te, y_pred),
                "bal_accuracy":  balanced_accuracy_score(y_te, y_pred),
                "log_loss":      log_loss(y_te, proba, labels=list(range(len(self._le.classes_)))),
            }
            self.fold_metrics.append(metrics)
            print(
                f"  [Baseline] Fold {fold_idx}: "
                f"bal_acc={metrics['bal_accuracy']:.4f}  "
                f"log_loss={metrics['log_loss']:.4f}  "
                f"(train={len(train_idx)}, test={len(test_idx)})"
            )

        return self

    # ─────────────────────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average ensemble prediction across all fold models."""
        probas = np.stack([m.predict_proba(X) for m in self.models], axis=0)
        return probas.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return decoded class labels {-1, 0, +1}."""
        proba  = self.predict_proba(X)
        enc    = proba.argmax(axis=1)
        return self._le.inverse_transform(enc)

    # ─────────────────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        """Return per-fold metrics as a DataFrame."""
        if not self.fold_metrics:
            raise RuntimeError("Call fit() first.")
        df = pd.DataFrame(self.fold_metrics)
        mean_row = df.mean(numeric_only=True).to_dict()
        mean_row["fold"] = "mean"
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        return df

    def feature_importance(self, feature_cols: list[str]) -> pd.DataFrame:
        """Average feature importance across folds."""
        if not self.models:
            raise RuntimeError("Call fit() first.")
        imps = np.stack([m.feature_importances_ for m in self.models], axis=0)
        mean_imp = imps.mean(axis=0)
        df = pd.DataFrame({"feature": feature_cols, "importance": mean_imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _join_features_labels(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Left-join label_df on feature_df using (ticker, timestamp == t0).
    Only keeps rows where all feature columns are non-null.
    """
    feat = feature_df[["timestamp", "ticker"] + feature_cols].copy()
    labels = label_df[["t0", "t1", "ticker", "label", "sample_weight",
                        "return_at_touch", "barrier_type"]].copy()
    labels.rename(columns={"t0": "timestamp"}, inplace=True)

    merged = labels.merge(feat, on=["timestamp", "ticker"], how="left")
    merged.rename(columns={"timestamp": "t0"}, inplace=True)
    merged.dropna(subset=feature_cols, inplace=True)
    merged.sort_values("t0", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


if __name__ == "__main__":
    from src.data.synthetic_data import generate_market_data, generate_news_data
    from src.data.features import build_feature_matrix
    from src.labeling.triple_barrier import label_universe

    print("Generating data…")
    mdf = generate_market_data(start="2021-01-04", end="2022-06-30")
    ndf = generate_news_data(mdf, n_events=1000)
    fdf = build_feature_matrix(mdf, ndf)
    ldf = label_universe(fdf, pt=2.0, sl=2.0, max_holding=20, event_spacing=5)

    print(f"Training baseline on {len(ldf)} labeled events…")
    baseline = GBDTBaseline(n_splits=5, embargo_pct=0.01)
    baseline.fit(fdf, ldf)

    print("\n── Baseline CV Summary ──")
    print(baseline.summary().to_string(index=False))
