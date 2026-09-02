"""Tests for TFT model — especially monotonic quantile property."""
import pytest
import torch
import numpy as np
from src.models.tft import (
    TFT,
    MonotonicQuantileHead,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    pinball_loss,
)


DEVICE = torch.device("cpu")


def _make_tft(n_tickers=5, n_sectors=3, n_regimes=3, n_features=8, hidden=32):
    return TFT(
        n_tickers=n_tickers,
        n_sectors=n_sectors,
        n_regimes=n_regimes,
        n_enc_features=n_features,
        embed_dim=8,
        hidden_dim=hidden,
        n_heads=2,
        n_lstm_layers=1,
        dropout=0.0,
    ).to(DEVICE)


def _make_batch(B=4, T=10, F=8, n_tickers=5, n_sectors=3, n_regimes=3):
    return {
        "x_enc":      torch.randn(B, T, F),
        "ticker_ids": torch.randint(0, n_tickers, (B,)),
        "sector_ids": torch.randint(0, n_sectors, (B,)),
        "regime_ids": torch.randint(0, n_regimes, (B,)),
    }


class TestMonotonicQuantileHead:
    def test_monotonicity_guaranteed(self):
        """P10 <= P50 <= P90 must hold for ANY input by construction."""
        head = MonotonicQuantileHead(hidden_dim=16)
        # Test with 1000 random inputs
        h = torch.randn(1000, 16)
        p10, p50, p90 = head(h)
        assert (p10 <= p50).all(), "P10 > P50 violation!"
        assert (p50 <= p90).all(), "P50 > P90 violation!"

    def test_output_shapes(self):
        head = MonotonicQuantileHead(hidden_dim=32)
        h = torch.randn(8, 32)
        p10, p50, p90 = head(h)
        assert p10.shape == (8, 1)
        assert p50.shape == (8, 1)
        assert p90.shape == (8, 1)

    def test_p50_is_not_bounded(self):
        """P50 can be negative, positive, or zero."""
        head = MonotonicQuantileHead(hidden_dim=8)
        h_pos = torch.ones(4, 8) * 5.0
        h_neg = torch.ones(4, 8) * -5.0
        _, p50_pos, _ = head(h_pos)
        _, p50_neg, _ = head(h_neg)
        # P50 values from opposite inputs should differ
        assert not torch.allclose(p50_pos, p50_neg)


class TestGRN:
    def test_output_shape(self):
        grn = GatedResidualNetwork(input_dim=8, hidden_dim=16, output_dim=8)
        x   = torch.randn(4, 8)
        out = grn(x)
        assert out.shape == (4, 8)

    def test_with_context(self):
        grn = GatedResidualNetwork(input_dim=8, hidden_dim=16, output_dim=8, context_dim=4)
        x   = torch.randn(4, 8)
        ctx = torch.randn(4, 4)
        out = grn(x, ctx)
        assert out.shape == (4, 8)

    def test_residual_when_dims_differ(self):
        grn = GatedResidualNetwork(input_dim=4, hidden_dim=16, output_dim=8)
        x   = torch.randn(2, 4)
        out = grn(x)
        assert out.shape == (2, 8)


class TestVSN:
    def test_output_shape(self):
        vsn = VariableSelectionNetwork(n_vars=5, var_dim=4, hidden_dim=16)
        x   = torch.randn(3, 5 * 4)
        out, weights = vsn(x)
        assert out.shape     == (3, 16)
        assert weights.shape == (3, 5)

    def test_weights_sum_to_one(self):
        vsn = VariableSelectionNetwork(n_vars=4, var_dim=3, hidden_dim=8)
        x   = torch.randn(10, 4 * 3)
        _, weights = vsn(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5), "VSN weights don't sum to 1"


class TestTFT:
    def test_forward_output_shapes(self):
        model  = _make_tft()
        batch  = _make_batch()
        out    = model(**batch)
        B = batch["x_enc"].shape[0]
        assert out["p10"].shape == (B,)
        assert out["p50"].shape == (B,)
        assert out["p90"].shape == (B,)

    def test_monotonicity_end_to_end(self):
        """Full model forward pass must satisfy P10 <= P50 <= P90 always."""
        model = _make_tft()
        model.eval()
        with torch.no_grad():
            # 200 random batches
            for _ in range(200):
                batch = _make_batch(B=8, T=15, F=8)
                out   = model(**batch)
                assert (out["p10"] <= out["p50"]).all(), "E2E: P10 > P50"
                assert (out["p50"] <= out["p90"]).all(), "E2E: P50 > P90"

    def test_gradient_flows(self):
        """Backward pass should not error and produce non-None gradients."""
        model = _make_tft()
        batch = _make_batch()
        out   = model(**batch)
        y     = torch.randn(batch["x_enc"].shape[0])
        loss  = pinball_loss(y, out["p10"], out["p50"], out["p90"])
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestPinballLoss:
    def test_perfect_prediction_low_loss(self):
        """When P50 == y, loss should be near-zero for median term."""
        y    = torch.tensor([0.5, -0.3, 1.0])
        p50  = y.clone()
        p10  = y - 0.5
        p90  = y + 0.5
        loss = pinball_loss(y, p10, p50, p90)
        assert loss.item() >= 0
        assert loss.item() < 0.5

    def test_loss_non_negative(self):
        """Pinball loss must always be non-negative."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            n   = 20
            y   = torch.tensor(rng.standard_normal(n), dtype=torch.float32)
            p50 = torch.tensor(rng.standard_normal(n), dtype=torch.float32)
            head = MonotonicQuantileHead(8)
            h   = torch.randn(n, 8)
            p10, p50_m, p90 = head(h)
            loss = pinball_loss(y, p10.squeeze(), p50_m.squeeze(), p90.squeeze())
            assert loss.item() >= 0, f"Negative loss: {loss.item()}"
