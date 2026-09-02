"""
config.py
─────────
Typed, validated pipeline configuration using Python dataclasses.

Replaces the raw CFG dict in main.py. All fields have types and defaults.
Validation is performed at instantiation via __post_init__.

Usage:
    from src.config import PipelineConfig
    cfg = PipelineConfig()              # defaults
    cfg = PipelineConfig(tft_epochs=50) # override
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import pathlib


@dataclass
class PipelineConfig:
    # ── Data ─────────────────────────────────────────────────────────────────
    start: str              = "2020-01-02"
    end: str                = "2023-06-30"
    bar_freq: str           = "5min"
    n_news_events: int      = 3_000
    latency_buffer_sec: int = 45

    # ── Labeling ──────────────────────────────────────────────────────────────
    pt: float               = 2.0   # take-profit multiplier (vol units)
    sl: float               = 2.0   # stop-loss multiplier   (vol units)
    max_holding: int        = 20    # max bars in position
    event_spacing: int      = 20    # sample every N-th bar as entry candidate (~39k events)

    # ── Cross-validation ──────────────────────────────────────────────────────
    n_splits: int           = 5
    embargo_pct: float      = 0.01

    # ── Baseline GBDT ─────────────────────────────────────────────────────────
    baseline_estimators: int = 200

    # ── TFT ───────────────────────────────────────────────────────────────────
    tft_hidden: int         = 64
    tft_heads: int          = 4
    tft_seq_len: int        = 30
    tft_epochs: int         = 3     # 3 epochs for fast CPU execution; increase for production GPU runs
    tft_batch_size: int     = 64

    tft_lr: float           = 1e-3
    tft_embed_dim: int      = 16
    tft_lstm_layers: int    = 2
    tft_dropout: float      = 0.1
    tft_clip_grad: float    = 1.0

    # ── Calibration ───────────────────────────────────────────────────────────
    conformal_alpha: float  = 0.10  # target miscoverage rate

    # ── Meta-labeling ─────────────────────────────────────────────────────────
    meta_estimators: int    = 200
    cost_bps: float         = 5.0

    # ── Backtesting ───────────────────────────────────────────────────────────
    initial_capital: float  = 1_000_000.0
    max_concurrent: int     = 15
    spread_bps: float       = 3.0

    # ── Evaluation ────────────────────────────────────────────────────────────
    n_trials: int           = 50    # DSR: number of strategy trials to correct for
    pbo_n_subsets: int      = 16    # CSCV subsets (16 → C(16,8)=12,870 combos)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir: str     = "checkpoints"
    mlflow_experiment: str  = "tft_finbert_forecaster"

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.pt <= 0 or self.sl <= 0:
            errors.append("pt and sl must be positive.")
        if not (0 < self.embargo_pct < 0.5):
            errors.append("embargo_pct must be in (0, 0.5).")
        if self.tft_heads <= 0 or self.tft_hidden % self.tft_heads != 0:
            errors.append(
                f"tft_hidden ({self.tft_hidden}) must be divisible by "
                f"tft_heads ({self.tft_heads})."
            )
        if self.conformal_alpha <= 0 or self.conformal_alpha >= 1:
            errors.append("conformal_alpha must be in (0, 1).")
        if errors:
            raise ValueError("PipelineConfig validation errors:\n  " + "\n  ".join(errors))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | pathlib.Path) -> None:
        """Persist config alongside model checkpoint for full reproducibility."""
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "PipelineConfig":
        data = json.loads(pathlib.Path(path).read_text())
        return cls(**data)
