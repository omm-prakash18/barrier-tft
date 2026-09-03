"""
main.py
───────
Master orchestrator for the Cross-Sectional TFT + FinBERT Price Forecaster.

Production features:
  - Validated dataclass configuration (src.config.PipelineConfig)
  - Structured JSON & colored console logging (src.logger.get_logger)
  - Pandera data validation schemas at all pipeline boundaries
  - Automatic model checkpointing (state_dict, optimizer, kwargs)
  - Optional MLflow experiment tracking with metric/param logging
  - Leakage-proof purged cross-validation & deflated Sharpe evaluation

Run: python main.py
"""

from __future__ import annotations

import pathlib
import time
import warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Infrastructure & Config ──────────────────────────────────────────────────
from src.config import PipelineConfig
from src.logger import get_logger
from src.data.schemas import (
    validate_market_df,
    validate_news_df,
    validate_feature_df,
    validate_label_df,
)

# ── Project modules ───────────────────────────────────────────────────────────
from src.data.synthetic_data import generate_market_data, generate_news_data
from src.data.features import build_feature_matrix
from src.data.leakage_guard import assert_no_leakage, check_no_future_features

from src.labeling.triple_barrier import label_universe
from src.validation.purged_cv import PurgedKFold

from src.models.baseline import GBDTBaseline, BASELINE_FEATURE_COLS
from src.models.tft import (
    TFT, TFTDataset, train_tft_epoch, predict_tft, save_checkpoint
)

from src.calibration.conformal import SplitConformalCalibrator
from src.metalabeling.meta_model import (
    MetaLabelClassifier, build_meta_labels, META_FEATURE_COLS
)
from src.execution.position_sizer import PositionSizer
from src.execution.backtester import Backtester

from src.data.universe import PointInTimeUniverse, create_sample_pit_universe
from src.data.partitioned_store import PartitionedTimeSeriesStore
from src.validation.distributed_cv import DistributedPurgedCV
from src.models.exporter import export_tft_to_onnx
from src.models.inference_engine import TFTInferenceEngine
from src.evaluation.metrics import generate_evaluation_report, print_report

# Initialize structured logger
log = get_logger("orchestrator")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Data pipeline
# ─────────────────────────────────────────────────────────────────────────────

def step1_data_pipeline(cfg: PipelineConfig):
    log.info("step.start", step=1, name="Data Pipeline")
    t0 = time.time()

    market_df = generate_market_data(
        start=cfg.start, end=cfg.end, bar_freq=cfg.bar_freq
    )
    validate_market_df(market_df)

    news_df = generate_news_data(
        market_df,
        n_events=cfg.n_news_events,
        latency_buffer_sec=cfg.latency_buffer_sec,
    )
    validate_news_df(news_df)

    feature_df = build_feature_matrix(
        market_df, news_df, latency_buffer_sec=cfg.latency_buffer_sec
    )
    validate_feature_df(feature_df)

    # Monotonicity & Leakage check
    check_no_future_features(feature_df, timestamp_col="timestamp")
    
    elapsed = time.time() - t0
    log.info(
        "step.complete",
        step=1,
        bars=len(market_df),
        news=len(news_df),
        feature_rows=len(feature_df),
        tickers=feature_df["ticker"].nunique(),
        elapsed_sec=round(elapsed, 2),
    )
    return market_df, news_df, feature_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Triple-barrier labeling
# ─────────────────────────────────────────────────────────────────────────────

def step2_labeling(feature_df: pd.DataFrame, cfg: PipelineConfig):
    log.info("step.start", step=2, name="Triple-Barrier Labeling")
    t0 = time.time()

    label_df = label_universe(
        feature_df,
        pt=cfg.pt,
        sl=cfg.sl,
        max_holding=cfg.max_holding,
        event_spacing=cfg.event_spacing,
    )
    validate_label_df(label_df)

    # Leakage assertion: feature timestamps must not post-date their labels
    assert_no_leakage(
        feature_df,
        label_df,
        latency_buffer_sec=cfg.latency_buffer_sec,
    )

    elapsed = time.time() - t0
    log.info(
        "step.complete",
        step=2,
        labels_total=len(label_df),
        pos_labels=int((label_df["label"] == 1).sum()),
        neg_labels=int((label_df["label"] == -1).sum()),
        zero_labels=int((label_df["label"] == 0).sum()),
        elapsed_sec=round(elapsed, 2),
    )
    return label_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Purged CV validation
# ─────────────────────────────────────────────────────────────────────────────

def step3_validate_cv(label_df: pd.DataFrame, cfg: PipelineConfig):
    log.info("step.start", step=3, name="Purged & Embargoed CV")
    X_dummy = np.zeros((len(label_df), 1))
    t1 = label_df["t1"]

    cv = PurgedKFold(n_splits=cfg.n_splits, embargo_pct=cfg.embargo_pct)
    for fold_idx, (tr, te) in enumerate(cv.split(X_dummy, t1=t1)):
        n_purged = len(label_df) - len(tr) - len(te)
        log.info("cv.fold", fold=fold_idx, train_size=len(tr), test_size=len(te), purged=n_purged)

    log.info("step.complete", step=3, status="validated")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: GBDT Baseline
# ─────────────────────────────────────────────────────────────────────────────

def step4_baseline(feature_df: pd.DataFrame, label_df: pd.DataFrame, cfg: PipelineConfig):
    log.info("step.start", step=4, name="GBDT Baseline")
    baseline = GBDTBaseline(
        n_splits=cfg.n_splits,
        embargo_pct=cfg.embargo_pct,
        n_estimators=cfg.baseline_estimators,
    )
    baseline.fit(feature_df, label_df)
    summary = baseline.summary()

    feat_cols = [c for c in BASELINE_FEATURE_COLS if c in feature_df.columns]
    fi = baseline.feature_importance(feat_cols)
    log.info("baseline.complete", top_features=fi.head(3).to_dict(orient="records"))
    return baseline


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 + 6: TFT training + conformal calibration
# ─────────────────────────────────────────────────────────────────────────────

def step56_tft_and_conformal(feature_df: pd.DataFrame, label_df: pd.DataFrame, cfg: PipelineConfig):
    log.info("step.start", step=5, name="TFT Training & Conformal Calibration")

    ticker_list = sorted(feature_df["ticker"].unique().tolist())
    sector_list = sorted(feature_df["sector"].unique().tolist())
    ticker2id   = {t: i for i, t in enumerate(ticker_list)}
    sector2id   = {s: i for i, s in enumerate(sector_list)}

    seq_len = cfg.tft_seq_len
    enc_features = [
        "vol_norm_return", "log_return", "realized_vol",
        "momentum_5", "momentum_20", "momentum_60",
        "rsi_14", "hl_spread", "weighted_sentiment",
    ]
    enc_features = [c for c in enc_features if c in feature_df.columns]
    n_enc_feat   = len(enc_features)

    # Pre-extract arrays per ticker for instant searchsorted slicing
    ticker_features: dict[str, np.ndarray] = {}
    ticker_times: dict[str, np.ndarray] = {}
    ticker_sectors: dict[str, int] = {}

    for tk, grp in feature_df.groupby("ticker"):
        grp_sorted = grp.sort_values("timestamp")
        ticker_times[tk] = grp_sorted["timestamp"].values
        ticker_features[tk] = grp_sorted[enc_features].values.astype(np.float32)
        ticker_sectors[tk] = sector2id.get(grp_sorted["sector"].iloc[0], 0)

    sequences, ticker_ids, sector_ids, regime_ids, y_targets, weights = [], [], [], [], [], []

    for _, row in label_df.iterrows():
        tk = row["ticker"]
        t0 = row["t0"]

        if tk not in ticker_features:
            continue

        times = ticker_times[tk]
        feat_mat = ticker_features[tk]
        t0_dt64 = np.datetime64(t0.value, 'ns') if hasattr(t0, 'value') else np.datetime64(t0)
        idx = np.searchsorted(times, t0_dt64, side="left")

        if idx < seq_len // 2:
            continue

        start_idx = max(0, idx - seq_len)
        slice_arr = feat_mat[start_idx:idx]
        if len(slice_arr) < seq_len:
            pad = np.zeros((seq_len - len(slice_arr), n_enc_feat), dtype=np.float32)
            slice_arr = np.vstack([pad, slice_arr])

        sequences.append(slice_arr)
        ticker_ids.append(ticker2id.get(tk, 0))
        sector_ids.append(ticker_sectors.get(tk, 0))
        regime_ids.append(int(row.get("regime", 0)) if "regime" in label_df.columns else 0)
        y_targets.append(float(row.get("vol_norm_return", row["return_at_touch"])))
        weights.append(float(row["sample_weight"]))

    if len(sequences) == 0:
        log.warning("tft.warning", msg="No sequences built - skipping TFT")
        return None, None, None


    X_seq  = np.stack(sequences)
    t_ids  = np.array(ticker_ids)
    s_ids  = np.array(sector_ids)
    r_ids  = np.array(regime_ids)
    Y      = np.array(y_targets)
    W      = np.array(weights)
    n      = len(X_seq)

    cv = PurgedKFold(n_splits=cfg.n_splits, embargo_pct=cfg.embargo_pct)
    t1_arr = label_df["t1"].iloc[:n].reset_index(drop=True)
    all_splits = list(cv.split(X_seq, t1=t1_arr))

    train_idx, test_idx = all_splits[-1]
    _, cal_idx = all_splits[-2]
    cal_idx  = cal_idx[cal_idx < n]
    test_idx = test_idx[test_idx < n]
    train_idx = train_idx[train_idx < n]
    train_idx = np.setdiff1d(train_idx, cal_idx)

    model_kwargs = {
        "n_tickers": len(ticker_list),
        "n_sectors": len(sector_list),
        "n_regimes": 3,
        "n_enc_features": n_enc_feat,
        "embed_dim": cfg.tft_embed_dim,
        "hidden_dim": cfg.tft_hidden,
        "n_heads": cfg.tft_heads,
        "n_lstm_layers": cfg.tft_lstm_layers,
        "dropout": cfg.tft_dropout,
    }
    model = TFT(**model_kwargs).to(DEVICE)

    def _make_loader(idx, shuffle=False):
        ds = TFTDataset(X_seq[idx], t_ids[idx], s_ids[idx], r_ids[idx], Y[idx], W[idx])
        return torch.utils.data.DataLoader(ds, batch_size=cfg.tft_batch_size, shuffle=shuffle)

    tr_loader  = _make_loader(train_idx, shuffle=True)
    cal_loader = _make_loader(cal_idx)
    te_loader  = _make_loader(test_idx)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.tft_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.tft_epochs)

    final_loss = 0.0
    for epoch in range(cfg.tft_epochs):
        loss = train_tft_epoch(model, tr_loader, optimizer, DEVICE, clip_grad_norm=cfg.tft_clip_grad)
        scheduler.step()
        final_loss = loss
        if (epoch + 1) % max(1, cfg.tft_epochs // 5) == 0:
            log.info("tft.epoch", epoch=epoch + 1, total_epochs=cfg.tft_epochs, loss=round(loss, 6))

    # Save model checkpoint
    ckpt_path = pathlib.Path(cfg.checkpoint_dir) / "tft_latest.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=cfg.tft_epochs,
        loss=final_loss,
        model_kwargs=model_kwargs,
        path=ckpt_path,
    )
    log.info("tft.checkpoint_saved", path=str(ckpt_path))

    # Export to ONNX Engine & Benchmark Inference Latency
    onnx_path = pathlib.Path(cfg.checkpoint_dir) / "tft_production.onnx"
    export_tft_to_onnx(model, onnx_path, sample_seq_len=cfg.tft_seq_len)
    engine = TFTInferenceEngine(onnx_path)
    perf_stats = engine.benchmark_latency(
        n_iterations=50,
        batch_size=min(32, len(test_idx)),
        seq_len=cfg.tft_seq_len,
        n_features=n_enc_feat,
    )
    log.info(
        "inference_engine.benchmarked",
        p50_us=round(perf_stats["p50_us"], 2),
        p95_us=round(perf_stats["p95_us"], 2),
        throughput_hz=round(perf_stats["throughput_samples_per_sec"], 1),
    )

    # Conformal calibration
    cal_preds = predict_tft(model, cal_loader, DEVICE)
    te_preds  = predict_tft(model, te_loader,  DEVICE)

    calibrator = SplitConformalCalibrator(alpha=cfg.conformal_alpha)
    calibrator.fit(cal_preds["p10"], cal_preds["p90"], Y[cal_idx])
    p10_adj, p90_adj = calibrator.transform(te_preds["p10"], te_preds["p90"])

    cov = calibrator.empirical_coverage(p10_adj, p90_adj, Y[test_idx])
    log.info("conformal.coverage", oos_coverage=round(cov, 4), target=1 - cfg.conformal_alpha)

    test_preds_dict = {
        "p10":     te_preds["p10"],
        "p50":     te_preds["p50"],
        "p90":     te_preds["p90"],
        "p10_adj": p10_adj,
        "p90_adj": p90_adj,
    }
    return model, calibrator, (test_idx, test_preds_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Meta-labeling
# ─────────────────────────────────────────────────────────────────────────────

def step7_meta_labeling(
    label_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    test_idx: np.ndarray,
    tft_preds: dict,
    cfg: PipelineConfig,
):
    log.info("step.start", step=7, name="Meta-Labeling")
    test_labels = label_df.iloc[:len(label_df)].iloc[test_idx].copy().reset_index(drop=True)

    if "regime" not in test_labels.columns and "regime" in feature_df.columns:
        reg_lookup = feature_df.set_index(["ticker", "timestamp"])["regime"].to_dict()
        test_labels["regime"] = test_labels.apply(
            lambda r: reg_lookup.get((r["ticker"], r["t0"]), 0), axis=1
        )

    n = min(len(test_labels), len(tft_preds["p50"]))
    test_labels = test_labels.iloc[:n].copy()
    aligned_preds = {k: v[:n] for k, v in tft_preds.items()}

    vnr_map = feature_df.set_index(["ticker", "timestamp"])["vol_norm_return"].to_dict()
    test_labels["vol_norm_return"] = test_labels.apply(
        lambda r: vnr_map.get((r["ticker"], r["t0"]), 0.0), axis=1
    )

    meta_df = build_meta_labels(test_labels, aligned_preds, cost_bps=cfg.cost_bps)

    clf = MetaLabelClassifier(
        n_splits    = min(3, cfg.n_splits),
        embargo_pct = cfg.embargo_pct,
        n_estimators = cfg.meta_estimators,
    )
    clf.fit(meta_df)
    log.info("meta_labeling.complete", summary=clf.summary().to_dict(orient="records"))
    return clf, meta_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Execution-aware backtest
# ─────────────────────────────────────────────────────────────────────────────

def step8_backtest(
    meta_df: pd.DataFrame,
    meta_clf: MetaLabelClassifier,
    feature_df: pd.DataFrame,
    market_df: pd.DataFrame,
    cfg: PipelineConfig,
):
    log.info("step.start", step=8, name="Backtesting")

    feat_cols = [c for c in META_FEATURE_COLS if c in meta_df.columns]
    X_meta    = meta_df[feat_cols].fillna(0).values.astype(np.float32)
    meta_df["meta_decision"] = meta_clf.predict(X_meta)

    signals = meta_df.merge(
        feature_df[["timestamp", "ticker", "close", "adv", "realized_vol"]],
        left_on=["t0", "ticker"],
        right_on=["timestamp", "ticker"],
        how="left",
    ).drop(columns=["timestamp"], errors="ignore")

    sizer = PositionSizer(
        max_position_frac = 0.05,
        cost_bps          = cfg.cost_bps,
        spread_bps        = cfg.spread_bps,
    )
    signals = sizer.compute_sizes(signals, portfolio_value=cfg.initial_capital)

    bt = Backtester(
        initial_capital    = cfg.initial_capital,
        max_concurrent_pos = cfg.max_concurrent,
        spread_bps         = cfg.spread_bps,
        cost_bps           = cfg.cost_bps,
    )
    result = bt.run(signals, market_df, sizer)
    log.info("backtest.summary", **{k: round(v, 4) if isinstance(v, float) else v for k, v in result.summary().items()})
    return result, signals


# ─────────────────────────────────────────────────────────────────────────────
# Step 9: Evaluation report
# ─────────────────────────────────────────────────────────────────────────────

def step9_evaluation(result, label_df: pd.DataFrame, cfg: PipelineConfig):
    log.info("step.start", step=9, name="Deflated Sharpe & PBO Evaluation")

    daily_rets = result.daily_returns.dropna().values
    rng = np.random.default_rng(0)

    n_trials_sim = 6
    T = len(daily_rets)
    returns_matrix = np.column_stack([
        daily_rets,
        *[daily_rets * rng.uniform(0.7, 1.3) + rng.normal(0, 0.001, T)
          for _ in range(n_trials_sim - 1)]
    ])

    report = generate_evaluation_report(
        result,
        n_trials       = cfg.n_trials,
        returns_matrix = returns_matrix,
        label_df       = label_df,
    )
    print_report(report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def main(config: PipelineConfig | None = None):
    cfg = config or PipelineConfig()
    log.info("pipeline.init", device=str(DEVICE), config=cfg.to_dict())

    # Step 1
    market_df, news_df, feature_df = step1_data_pipeline(cfg)

    # Step 2
    label_df = step2_labeling(feature_df, cfg)
    if label_df.empty:
        log.error("pipeline.error", msg="No labels produced.")
        return

    # Step 3
    step3_validate_cv(label_df, cfg)

    # Step 4
    step4_baseline(feature_df, label_df, cfg)

    # Step 5+6
    model, calibrator, tft_out = step56_tft_and_conformal(feature_df, label_df, cfg)

    if tft_out is not None:
        test_idx, tft_preds = tft_out
        # Step 7
        meta_clf, meta_df = step7_meta_labeling(label_df, feature_df, test_idx, tft_preds, cfg)
        # Step 8
        result, _ = step8_backtest(meta_df, meta_clf, feature_df, market_df, cfg)
        # Step 9
        step9_evaluation(result, label_df, cfg)

    log.info("pipeline.complete", status="success")


if __name__ == "__main__":
    main()
