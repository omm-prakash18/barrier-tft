"""
test_inference_engine.py
────────────────────────
Unit tests for TFT ONNX / TorchScript model exporter and high-throughput inference engine.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.tft import TemporalFusionTransformer
from src.models.exporter import export_tft_to_onnx
from src.models.inference_engine import TFTInferenceEngine


@pytest.fixture
def tiny_tft_model():
    return TemporalFusionTransformer(
        n_tickers=5,
        n_sectors=3,
        n_regimes=2,
        n_enc_features=4,
        embed_dim=8,
        hidden_dim=16,
        n_heads=2,
        n_lstm_layers=1,
    )


def test_tft_export_and_inference(tiny_tft_model):
    """Test exporting TFT model and running inference with shape parity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = Path(tmp_dir) / "tft_model.onnx"
        exported_path = export_tft_to_onnx(tiny_tft_model, onnx_file, sample_seq_len=10)

        assert exported_path.exists()

        # Initialize Inference Engine
        engine = TFTInferenceEngine(exported_path)

        batch_size = 4
        ticker_ids = np.array([0, 1, 2, 3], dtype=np.int64)
        sector_ids = np.array([0, 1, 0, 2], dtype=np.int64)
        regime_ids = np.array([0, 1, 1, 0], dtype=np.int64)
        x_enc = np.random.randn(batch_size, 10, 4).astype(np.float32)

        preds = engine.predict(ticker_ids, sector_ids, regime_ids, x_enc)

        assert "p10" in preds and "p50" in preds and "p90" in preds
        assert preds["p10"].shape == (batch_size,)
        assert preds["p50"].shape == (batch_size,)
        assert preds["p90"].shape == (batch_size,)

        # Monotonicity check on output
        assert np.all(preds["p10"] <= preds["p50"] + 1e-5)
        assert np.all(preds["p50"] <= preds["p90"] + 1e-5)


def test_latency_benchmark(tiny_tft_model):
    """Test microsecond latency benchmark execution."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = Path(tmp_dir) / "tft_benchmark.onnx"
        exported_path = export_tft_to_onnx(tiny_tft_model, onnx_file, sample_seq_len=10)

        engine = TFTInferenceEngine(exported_path)
        stats = engine.benchmark_latency(n_iterations=20, batch_size=8, seq_len=10, n_features=4)

        assert "p50_us" in stats
        assert "p95_us" in stats
        assert "throughput_samples_per_sec" in stats
        assert stats["p50_us"] > 0
        assert stats["throughput_samples_per_sec"] > 0
