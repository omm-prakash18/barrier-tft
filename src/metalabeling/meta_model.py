"""
meta_model.py
─────────────
Meta-labeling classifier (spec §5).

Architecture:
  Secondary LightGBM binary classifier trained ON TOP of TFT outputs.
  Input features:
    - p50     : TFT point forecast (vol-normalized return)
    - p10     : lower quantile
    - p90     : upper quantile
    - band_width : p90_adj - p10_adj  (calibrated uncertainty)
    - regime  : current market regime label
    - recent_hit_rate : rolling fraction of correct directional calls (last N events)

  Target:
    1 → primary model's directional call is profitable net of costs
    0 → skip (cost-adjusted loss)

Validated with same PurgedKFold as primary model — no exceptions.

Usage:
    meta = MetaLabelClassifier(n_splits=5, embargo_pct=0.01)
    meta.fit(meta_df)
    decisions = meta.predict(new_meta_df)  # "act" or "skip"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from typing import Optional

from src.validation.purged_cv import PurgedKFold


# ── meta features produced by the primary TFT ────────────────────────────────
META_FEATURE_COLS = [
    "p50",
    "p10",
    "p90",
    "band_width",
    "regime",
    "recent_hit_rate",
    "vol_norm_return_lag1",   # lagged realized vol-norm return as context
]


def build_meta_labels(
    label_df: pd.DataFrame,
    tft_preds: dict[str, np.ndarray],
    cost_bps: float = 5.0,
    min_bps_per_trade: float = 2.0,
) -> pd.DataFrame:
    """
    Build meta-label DataFrame from primary model predictions and realized returns.

    Meta target (binary):
      1 if: sign(p50) == sign(return_at_touch) AND |return_at_touch| > cost_bps / 10000
      0 otherwise (wrong direction OR unprofitable after costs)

    Parameters
    ----------
    label_df   : Output of label_universe() with realized return_at_touch.
    tft_preds  : Dict with 'p10', 'p50', 'p90', 'p10_adj', 'p90_adj' arrays
                 aligned to label_df index.
    cost_bps   : One-way transaction cost estimate in basis points.
    min_bps_per_trade : Minimum net return (bp) for a trade to count as "act".

    Returns
    -------
    meta_df: DataFrame with all meta features and 'meta_label' (0/1).
    """
    df = label_df.copy()

    n = len(df)
    for k, v in tft_preds.items():
        if len(v) == n:
            df[k] = v

    # Compute band width (calibrated)
    if "p10_adj" in df.columns and "p90_adj" in df.columns:
        df["band_width"] = df["p90_adj"] - df["p10_adj"]
    elif "p10" in df.columns and "p90" in df.columns:
        df["band_width"] = df["p90"] - df["p10"]
    else:
        df["band_width"] = 1.0

    # Directionally correct AND profitable net of costs
    ret       = df["return_at_touch"].values
    p50       = df["p50"].values
    cost_ret  = cost_bps / 10_000.0
    min_ret   = min_bps_per_trade / 10_000.0

    meta_label = (
        (np.sign(p50) == np.sign(ret))         # right direction
        & (np.abs(ret) > cost_ret + min_ret)   # beats cost floor
    ).astype(int)

    df["meta_label"] = meta_label

    # Rolling hit-rate feature (past 20 events per ticker)
    df.sort_values(["ticker", "t0"], inplace=True)
    df["recent_hit_rate"] = (
        df.groupby("ticker")["meta_label"]
        .transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
        .fillna(0.5)
    )

    # Lagged vol-norm return as context feature
    if "vol_norm_return" in df.columns:
        df["vol_norm_return_lag1"] = (
            df.groupby("ticker")["vol_norm_return"]
            .transform(lambda x: x.shift(1))
            .fillna(0.0)
        )
    else:
        df["vol_norm_return_lag1"] = 0.0

    df.sort_values("t0", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


class MetaLabelClassifier:
    """
    Secondary GBDT binary classifier for act-vs-skip decisions.

    Validated with PurgedKFold — same splitter as primary model.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        n_estimators: int = 200,
        random_state: int = 42,
        lgbm_params: Optional[dict] = None,
    ):
        self.n_splits    = n_splits
        self.embargo_pct = embargo_pct
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.lgbm_params  = lgbm_params or {}
        self.models: list = []
        self.oof_proba: Optional[np.ndarray] = None
        self.fold_metrics: list = []

    def fit(
        self,
        meta_df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        label_col: str = "meta_label",
        t1_col: str = "t1",
        weight_col: str = "sample_weight",
    ) -> "MetaLabelClassifier":
        """
        Train meta-labeling classifier on TFT-derived features.

        Parameters
        ----------
        meta_df      : Output of build_meta_labels().
        feature_cols : Override default META_FEATURE_COLS.
        label_col    : Binary target column.
        t1_col       : Label end-time column for purged CV.
        weight_col   : Sample weight column.
        """
        if feature_cols is None:
            feature_cols = [c for c in META_FEATURE_COLS if c in meta_df.columns]

        df = meta_df.dropna(subset=feature_cols + [label_col]).copy()
        df.sort_values("t0", inplace=True)
        df.reset_index(drop=True, inplace=True)

        X       = df[feature_cols].values.astype(np.float32)
        y       = df[label_col].values.astype(int)
        weights = df[weight_col].values if weight_col in df.columns else np.ones(len(df))
        t1      = df[t1_col]

        n = len(X)
        self.oof_proba = np.zeros(n, dtype=np.float32)

        cv = PurgedKFold(n_splits=self.n_splits, embargo_pct=self.embargo_pct)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, t1=t1)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            w_tr       = weights[train_idx]

            params = {
                "n_estimators":      self.n_estimators,
                "learning_rate":     0.05,
                "num_leaves":        15,
                "min_child_samples": 20,
                "subsample":         0.8,
                "colsample_bytree":  0.8,
                "scale_pos_weight":  max(1.0, (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)),
                "random_state":      self.random_state,
                "n_jobs":            -1,
                "verbose":           -1,
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

            proba = model.predict_proba(X_te)[:, 1]
            self.oof_proba[test_idx] = proba

            y_pred = (proba >= 0.5).astype(int)
            auc    = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else np.nan

            metrics = {
                "fold":      fold_idx,
                "n_train":   len(train_idx),
                "n_test":    len(test_idx),
                "accuracy":  accuracy_score(y_te, y_pred),
                "precision": precision_score(y_te, y_pred, zero_division=0),
                "recall":    recall_score(y_te, y_pred, zero_division=0),
                "f1":        f1_score(y_te, y_pred, zero_division=0),
                "auc_roc":   auc,
                "act_rate":  y_pred.mean(),
            }
            self.fold_metrics.append(metrics)
            print(
                f"  [MetaLabel] Fold {fold_idx}: "
                f"F1={metrics['f1']:.4f}  AUC={metrics['auc_roc']:.4f}  "
                f"act_rate={metrics['act_rate']:.3f}"
            )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Ensemble probability (act=1) across fold models."""
        probas = np.stack([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return probas.mean(axis=0)

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return binary act (1) / skip (0) decisions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def summary(self) -> pd.DataFrame:
        if not self.fold_metrics:
            raise RuntimeError("Call fit() first.")
        df = pd.DataFrame(self.fold_metrics)
        mean_row = df.mean(numeric_only=True).to_dict()
        mean_row["fold"] = "mean"
        return pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)


if __name__ == "__main__":
    # Synthetic smoke test
    rng = np.random.default_rng(1)
    n   = 500
    t0  = pd.date_range("2021-01-01", periods=n, freq="1D")
    t1  = t0 + pd.Timedelta(days=5)

    meta_df = pd.DataFrame({
        "t0":    t0,
        "t1":    t1,
        "ticker": rng.choice(["AAPL", "MSFT"], size=n),
        "p50":   rng.normal(0, 0.5, n),
        "p10":   rng.normal(-0.8, 0.2, n),
        "p90":   rng.normal(0.8, 0.2, n),
        "band_width": rng.uniform(0.5, 2.0, n),
        "regime": rng.integers(0, 3, n),
        "recent_hit_rate": rng.uniform(0.3, 0.7, n),
        "vol_norm_return_lag1": rng.normal(0, 1, n),
        "return_at_touch": rng.normal(0, 0.02, n),
        "meta_label": rng.integers(0, 2, n),
        "sample_weight": np.ones(n),
    })

    clf = MetaLabelClassifier(n_splits=3, embargo_pct=0.01)
    clf.fit(meta_df)
    print("\n── Meta-Label CV Summary ──")
    print(clf.summary().to_string(index=False))
