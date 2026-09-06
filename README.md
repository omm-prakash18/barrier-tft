# Barrier TFT: Cross-Sectional TFT + FinBERT Equity Forecaster

A production-grade, leak-free, institutional cross-sectional quantitative equity forecasting and trade execution framework combining **Temporal Fusion Transformers (TFT)**, **FinBERT sentiment embeddings with credibility weighting**, **Triple-Barrier labeling with sample uniqueness weighting**, **Purged & Embargoed Cross-Validation**, **Secondary Meta-Labeling**, **High-Throughput ONNX/TensorRT Inference Engine**, and **Point-in-Time Survivorship-Bias Free Universes**.

---

## 🏛 Architecture Overview

```
                               ┌─────────────────────────────┐
                               │ Point-in-Time Data Lake     │
                               │  - Partitioned Parquet Lake │
                               │  - PIT Universe (No Bias)   │
                               │  - News Feed (Credibility)  │
                               └──────────────┬──────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │ Price & Volume Features │                       │ FinBERT Sentiment Flow  │
        │ - Realized Volatility   │                       │ - Domain-specific LLM   │
        │ - Volume Imbalance      │                       │ - Source Credibility    │
        │ - Momentum / Reversal   │                       │ - Decay & Alignment     │
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │    Point-in-Time Align & Join   │
                             │ (Strict Zero Lookahead Leakage) │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │  Triple-Barrier Labeling Engine │
                             │  - Upper / Lower / Time Barrier │
                             │  - Sample Uniqueness Weighting  │
                             └────────────────┬────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │ Distributed Purged CV   │                       │ Temporal Fusion Transf. │
        │ - Parallel CPCV Folds   │                       │ - Monotonic Quantiles   │
        │ - Serial Embargo buffer │                       │ - Self-Attention & VSN  │
        │ - Multi-GPU DDP Trainer │                       │ - ONNX / Low Latency    │
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │     Secondary Meta-Labeler      │
                             │ (LightGBM Sizing / Net Profits) │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ Realistic Vectorized Backtester │
                             │ - Latency Buffer & Slippage     │
                             │ - Searchsorted Fast Execution   │
                             │ - Transaction Costs / Spreads   │
                             └─────────────────────────────────┘
```

---

## 🚀 Key Modules & Components

- **`src/data/`**:
  - **`universe.py`**: Institutional Point-in-Time Universe Manager tracking ticker lifecycles (IPOs, delistings, acquisitions, corporate actions) to guarantee 100% survivorship-bias-free universes.
  - **`partitioned_store.py`**: High-throughput partitioned Parquet storage with time-travel query filters and predicate pushdown.
  - **`features.py`**: Cross-sectional features & FinBERT sentiment embeddings with source credibility discounting.
  - **`schemas.py` & `leakage_guard.py`**: Pandera runtime schema validation contracts and strict timestamp lookahead assertion guards.
- **`src/labeling/`**: Marcos Lopez de Prado's Triple-Barrier Method (`t0`, `t1`, dynamic volatility barriers) and sample concurrency/uniqueness weighting to remove label redundancy.
- **`src/validation/`**:
  - **`purged_cv.py`**: Purged & Embargoed K-Fold Cross-Validation preventing serial correlation and label-horizon leaks.
  - **`distributed_cv.py`**: Multi-core parallel Combinatorial Purged Cross-Validation (CPCV) coordinator with worker isolation and Out-of-Fold prediction assembly.
- **`src/models/`**:
  - **`tft.py`**: PyTorch Temporal Fusion Transformer (TFT) with strictly monotonic quantile heads ($P_{10} \le P_{50} \le P_{90}$) reparameterized via Softplus.
  - **`exporter.py`**: PyTorch-to-ONNX graph exporter with dynamic batching, sequence dimensions, and TorchScript companion tracing.
  - **`inference_engine.py`**: Sub-millisecond low-latency production inference engine with multi-threaded CPU / CUDA / TensorRT execution providers and microsecond benchmarking ($p50, p95, p99$).
  - **`distributed_trainer.py`**: Multi-GPU PyTorch DistributedDataParallel (DDP) coordinator with Automatic Mixed Precision (AMP).
- **`src/calibration/`**: Calibrated conformal prediction intervals and empirical coverage estimators.
- **`src/metalabeling/`**: Secondary trade sizing and filtration model (LightGBM) to filter false-positive directional bets net of transaction costs.
- **`src/execution/`**: Vectorized and event-driven execution simulator factoring in reaction latency buffers, slippage models, spread fees, turnover penalties, and fast $O(\log K)$ price lookup indexing.

---

## 🛠 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/omm-prakash18/barrier-tft.git
cd barrier-tft

# Install dependencies in editable mode
pip install -e .
```

### Running the Full Pipeline

```bash
# Execute end-to-end data generation, training, purged CV, meta-labeling, ONNX export, and backtest
python main.py
```

### Running Test Suite

```bash
python -m pytest
```

---

# 📖 Debug Playbook: Cross-Sectional TFT + FinBERT Forecaster

This is a diagnostic reference to run when something looks wrong — or worse, when something looks *too good*. Financial ML bugs are unusual: the most dangerous ones don't crash, they just quietly inflate your backtest. Work through the relevant section by symptom, not top to bottom.

### How to use this

When diagnosing a symptom, work through the checklist against the implementation before proposing a fix. Diagnosis is the hardest and most critical step in quantitative financial machine learning.

---

### 🚨 Symptom: Backtest Sharpe ratio looks great (most dangerous case)

A suspiciously good result is a bug report, not a success. Check in this order:

1. **Timestamp leakage**: For every feature, print the timestamp it was computed from alongside the label window `[t0, t1]` it's paired with. Confirm `feature_ts <= t0` for every single row, not just on average. Specifically check FinBERT sentiment features — `article_public_ts` vs `article_scraped_ts`/`article_indexed_ts` bugs are the #1 cause of unrealistic backtests in this exact architecture.
2. **Purge/embargo correctness**: Pick one test fold, manually list 5 training samples closest to its boundary, and confirm their `[t0, t1]` windows truly don't overlap the test fold's date range — off-by-one errors here (purging by `t0` instead of the full `[t0, t1]` window) are extremely common and leak exactly enough to move Sharpe from mediocre to great.
3. **Survivorship bias**: Confirm the universe at each historical date includes names that were later delisted, not just today's constituents (enforced via `src.data.universe.PointInTimeUniverse`).
4. **Rolling feature leakage**: Any feature using `.rolling()` or `.ewm()` — confirm it's not accidentally centered (`center=True` is a classic silent leak) and that the window only looks backward.
5. **Duplicate/near-duplicate samples across train/test**: For overlapping-window labels, confirm concurrency weighting is actually being applied, not just computed and discarded.
6. **Global-statistic normalization leak**: This is arguably the single most common leak in financial ML and it's easy to miss because the code "looks" correct. If any feature is z-scored, min-max scaled, or otherwise normalized using statistics (mean, std, min, max) computed over the full dataset rather than only the training fold available at that point in time, the model has effectively seen the future's distribution. Confirm every normalization step is fit inside the CV loop on the training fold only, and applied (not refit) to validation/test data.

*If all six check out and it's still too good, distrust it further before believing it — rerun on a held-out time period the model has never touched in any form (not even for CV), ideally the most recent few months.*

---

### 📐 Symptom: Quantile predictions cross ($P_{10} > P_{50}$, or $P_{50} > P_{90}$)

- Confirm the deltas (`P50 - P10`, `P90 - P50`) are passed through a strictly non-negative activation (softplus, not ReLU — ReLU can output exact zero and doesn't guarantee strict ordering under floating point edge cases).
- Confirm the loss function is applied to the *reparameterized* quantiles, not accidentally to three independent raw outputs that happen to share a name with monotonic outputs elsewhere in the code.
- If crossing only happens rarely and at prediction extremes, check for numerical overflow in the softplus at large magnitudes — clip or use a numerically stable softplus implementation.

---

### ✂️ Symptom: Purged CV splitter throws, or produces empty/tiny folds

- Confirm `t1` (label end-times) is actually being passed to the splitter, not silently defaulting to `t0` — a splitter that purges by entry time only will look like it works but purges nothing.
- Check `embargo_pct` isn't so large relative to fold size that it purges entire folds — this shows up as empty training sets on small datasets or short date ranges.
- Log the purge count per fold. Zero purged samples on any fold is itself a red flag — it usually means the overlap check has a bug, not that there's genuinely no overlap.

---

### 📰 Symptom: Sentiment features look useless / near-constant / all zero

- Print raw FinBERT embeddings for a handful of manually-read articles with obviously different sentiment (one clearly positive, one clearly negative) and confirm the embeddings actually differ — a common bug is truncating input text before the sentiment-bearing sentence, or a tokenizer mismatch between the FinBERT checkpoint and the tokenizer being loaded.
- Confirm credibility weighting isn't averaging away signal — a naive mean across many low-credibility sources can wash out one high-credibility outlier that should have dominated.
- Check for a silent join failure: if the news-to-price join fails (mismatched ticker symbols, wrong timezone) it can default to zero/null sentiment for most rows without erroring, and the model correctly learns to ignore a feature that's genuinely empty 95% of the time.

---

### 💥 Symptom: Model loss is NaN or explodes early in training

- Check the volatility-normalization step for division by near-zero $\sigma_{t0}$ on illiquid names or low-volatility days — clip or floor the volatility estimate.
- Confirm gradient clipping is enabled — TFT's LSTM component is prone to exploding gradients on financial data's fat-tailed return distributions.
- Check for `inf`/`NaN` propagating from the triple-barrier labeling itself, e.g. events near the end of the dataset with no valid vertical barrier.

---

### ⚡ Symptom: Live/paper-trading signal doesn't match backtest behavior

- This is almost always **feature parity** — something computed differently (or with different lookback availability) between the offline backtest pipeline and the live feature pipeline. Diff the feature vector for the same instrument/timestamp computed by both pipelines directly.
- Confirm the live pipeline respects the same reaction-latency buffer used in backtesting — it's easy to accidentally use the most recent live tick instantly in production while the backtest imposed a 30-60s buffer.
- Confirm regime state and instrument embeddings are computed the same way live as historically — a regime classifier refit only on historical data can silently diverge from what's being fed at inference time.

---

### 🎯 Symptom: Meta-labeling filter rejects almost everything (or accepts almost everything)

- Check the class balance of the meta-labeling training target — "was the primary model's call profitable net of costs" is often heavily imbalanced, and an unweighted classifier will collapse to predicting the majority class.
- Confirm the meta-labeler was validated with the same purged CV scheme — it's easy to remember to purge for the primary model and forget for the secondary one.

---

### 🛡 General Debugging Discipline

- Never trust a result you haven't reproduced on a held-out time slice the model has never touched, including during hyperparameter search.
- When in doubt, isolate: test triple-barrier labeling on a synthetic price series with a known, hand-computed answer before trusting it on real data. Do the same for the purged CV splitter with synthetic overlapping windows.
- Prefer "too good, investigate" over "too good, ship it" every time in this domain — it's cheaper to spend a day hunting a leak than to discover it in live capital.
