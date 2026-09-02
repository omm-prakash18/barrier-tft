"""
tft.py
──────
Regime-conditioned Temporal Fusion Transformer (PyTorch).

Architecture (spec §4):
  Static inputs  : ticker embedding, sector embedding, regime state
  Time-varying   : price/volume features, credibility-weighted FinBERT sentiment
  Core           : Variable Selection Network → LSTM encoder → Multi-Head Self-Attention
  Output head    : Monotonic Quantile Head (P10, P50, P90)
                   P10 = P50 - softplus(delta1)   ← guaranteed no crossing
                   P90 = P50 + softplus(delta2)

Loss: Pinball loss at quantiles 0.10, 0.50, 0.90.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    """
    GRN: core building block of TFT.
    Projects input → output with a gated residual skip connection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.fc1      = nn.Linear(input_dim, hidden_dim)
        self.fc2      = nn.Linear(hidden_dim, output_dim)
        self.gate     = nn.Linear(hidden_dim, output_dim)
        self.ln       = nn.LayerNorm(output_dim)
        self.dropout  = nn.Dropout(dropout)

        # Optional context injection (for static covariate conditioning)
        self.ctx_proj = nn.Linear(context_dim, hidden_dim) if context_dim else None

        # Skip connection adapter (if dims differ)
        self.skip = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        if context is not None and self.ctx_proj is not None:
            h = h + self.ctx_proj(context)
        h = self.dropout(h)
        gate  = torch.sigmoid(self.gate(h))
        out   = gate * self.fc2(h)
        return self.ln(out + self.skip(x))


class VariableSelectionNetwork(nn.Module):
    """
    VSN: learns soft-attention weights over input variables.
    Each variable is separately projected then gated.
    """

    def __init__(
        self,
        n_vars: int,
        var_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_vars   = n_vars
        self.var_dim  = var_dim

        # Per-variable GRNs
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(var_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(n_vars)
        ])

        # Selection GRN: maps flat input to per-variable weights
        self.select_grn = GatedResidualNetwork(
            n_vars * var_dim, hidden_dim, n_vars, dropout,
            context_dim=context_dim,
        )

    def forward(
        self,
        x: torch.Tensor,                  # (batch, n_vars * var_dim)  or  (batch, T, n_vars * var_dim)
        context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            out      : (batch [,T], hidden_dim)  weighted combination
            weights  : (batch [,T], n_vars)      selection weights (softmax)
        """
        is_temporal = x.dim() == 3
        if is_temporal:
            B, T, D = x.shape
            x_flat = x                                   # (B, T, D)
        else:
            B, D = x.shape
            x_flat = x                                   # (B, D)

        # Variable weights
        weights = F.softmax(self.select_grn(x_flat, context), dim=-1)  # (B[,T], n_vars)

        # Per-variable projections
        var_inputs = x_flat.view(*x_flat.shape[:-1], self.n_vars, self.var_dim)  # (B[,T], n_vars, var_dim)
        processed  = torch.stack(
            [grn(var_inputs[..., i, :]) for i, grn in enumerate(self.var_grns)],
            dim=-2,
        )  # (B[,T], n_vars, hidden_dim)

        weights_exp = weights.unsqueeze(-1)               # (B[,T], n_vars, 1)
        out = (processed * weights_exp).sum(dim=-2)        # (B[,T], hidden_dim)
        return out, weights


class StaticCovariateEncoder(nn.Module):
    """
    Encodes static inputs (ticker, sector, regime) into 4 context vectors:
        c_s  → enriches temporal features
        c_e  → LSTM initial hidden state
        c_c  → LSTM initial cell state
        c_h  → enriches attention
    """

    def __init__(
        self,
        n_tickers: int,
        n_sectors: int,
        n_regimes: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ticker_emb  = nn.Embedding(n_tickers, embed_dim)
        self.sector_emb  = nn.Embedding(n_sectors, embed_dim)
        self.regime_emb  = nn.Embedding(n_regimes, embed_dim)

        static_dim = 3 * embed_dim
        self.vsn   = VariableSelectionNetwork(
            n_vars=3, var_dim=embed_dim,
            hidden_dim=hidden_dim, dropout=dropout,
        )
        self.c_s   = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.c_e   = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.c_c   = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.c_h   = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

    def forward(
        self,
        ticker_ids: torch.Tensor,   # (B,)
        sector_ids: torch.Tensor,   # (B,)
        regime_ids: torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        te = self.ticker_emb(ticker_ids)  # (B, embed_dim)
        se = self.sector_emb(sector_ids)
        re = self.regime_emb(regime_ids)

        flat = torch.cat([te, se, re], dim=-1)   # (B, 3*embed_dim)
        out, _ = self.vsn(flat)                  # (B, hidden_dim)

        return self.c_s(out), self.c_e(out), self.c_c(out), self.c_h(out)


class InterpretableMultiHeadAttention(nn.Module):
    """
    TFT-style multi-head attention that shares value projections across heads
    for interpretability (each head attends differently but values are shared).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = math.sqrt(self.d_head)

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, self.d_head)  # shared across heads
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)
        self.ln       = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, d_model)
        Returns:
            out        : (B, T, d_model)
            attn_weights: (B, n_heads, T, T)
        """
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x)  # (B, T, d_head) — shared

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        # Causal mask: lower-triangular
        mask   = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        # Shared V: expand to (B, H, T, d_head)
        V_exp  = V.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        ctx    = (attn @ V_exp)                          # (B, H, T, d_head)
        ctx    = ctx.transpose(1, 2).reshape(B, T, -1)   # (B, T, H*d_head = d_model — No, d_model=H*d_head)
        # Note: H * d_head might differ from d_model when shared V. Reproject:
        ctx    = self.out_proj(ctx[..., :self.n_heads * self.d_head])

        return self.ln(ctx + x), attn


class MonotonicQuantileHead(nn.Module):
    """
    Output head that guarantees P10 <= P50 <= P90.

    Predicts:
      p50    = linear(h)
      delta1 = softplus(linear(h))  → always non-negative
      delta2 = softplus(linear(h))  → always non-negative

    Then:
      P10 = P50 - delta1
      P90 = P50 + delta2
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.p50    = nn.Linear(hidden_dim, 1)
        self.delta1 = nn.Linear(hidden_dim, 1)
        self.delta2 = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h: (B, hidden_dim)
        Returns:
            p10: (B, 1)
            p50: (B, 1)
            p90: (B, 1)
        """
        p50    = self.p50(h)
        delta1 = F.softplus(self.delta1(h))
        delta2 = F.softplus(self.delta2(h))
        p10    = p50 - delta1
        p90    = p50 + delta2
        return p10, p50, p90


# ─────────────────────────────────────────────────────────────────────────────
# Full TFT Model
# ─────────────────────────────────────────────────────────────────────────────

class TFT(nn.Module):
    """
    Cross-sectional, regime-conditioned Temporal Fusion Transformer.

    Inputs per forward pass:
      Static:
        ticker_ids  : (B,)       int — ticker index
        sector_ids  : (B,)       int — sector index
        regime_ids  : (B,)       int — regime cluster label
      Time-varying (encoder window):
        x_enc       : (B, T, n_enc_features)  — price/vol + sentiment
      Static context is injected into LSTM init state and VSN.

    Output:
        p10, p50, p90 : each (B, 1)  — monotonic quantile predictions
        attn_weights  : (B, H, T, T) — for interpretability
    """

    def __init__(
        self,
        n_tickers: int,
        n_sectors: int,
        n_regimes: int,
        n_enc_features: int,       # number of time-varying input features
        embed_dim: int    = 16,
        hidden_dim: int   = 64,
        n_heads: int      = 4,
        n_lstm_layers: int = 2,
        dropout: float    = 0.1,
    ):
        super().__init__()
        self.hidden_dim     = hidden_dim
        self.n_enc_features = n_enc_features

        # ── Static encoder ────────────────────────────────────────────────
        self.static_encoder = StaticCovariateEncoder(
            n_tickers=n_tickers,
            n_sectors=n_sectors,
            n_regimes=n_regimes,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ── Per-feature projections (scalar → hidden_dim) + learned VSN weighting ──
        # Each of the n_enc_features scalars is independently projected to hidden_dim,
        # then weighted by a context-conditioned selection network.
        self.feature_proj = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(n_enc_features)
        ])

        # ── LSTM sequence encoder ─────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        # Project static context to LSTM init state dimensions
        self.h0_proj = nn.Linear(hidden_dim, n_lstm_layers * hidden_dim)
        self.c0_proj = nn.Linear(hidden_dim, n_lstm_layers * hidden_dim)

        # ── Gated skip connection after LSTM ──────────────────────────────
        self.post_lstm_gate = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.post_lstm_ln   = nn.LayerNorm(hidden_dim)

        # ── Interpretable Multi-Head Self-Attention ───────────────────────
        self.attn = InterpretableMultiHeadAttention(hidden_dim, n_heads, dropout)

        # ── Post-attention GRN + LN ──────────────────────────────────────────────
        # c_h enriches the post-attention representations (TFT paper, §4.2)
        self.post_attn_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout,
            context_dim=hidden_dim,   # receives c_h from static encoder
        )
        self.post_attn_ln  = nn.LayerNorm(hidden_dim)

        # ── Monotonic Quantile Head ───────────────────────────────────────
        self.quantile_head = MonotonicQuantileHead(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        ticker_ids: torch.Tensor,   # (B,)
        sector_ids: torch.Tensor,   # (B,)
        regime_ids: torch.Tensor,   # (B,)
        x_enc: torch.Tensor,        # (B, T, n_enc_features)
    ) -> dict[str, torch.Tensor]:
        B, T, n_feat = x_enc.shape

        # ── Static context ────────────────────────────────────────────────
        c_s, c_e, c_c, c_h = self.static_encoder(ticker_ids, sector_ids, regime_ids)

        # ── Project each feature scalar → hidden_dim, then stack ──────────
        feat_list = [
            self.feature_proj[i](x_enc[:, :, i:i+1])   # (B, T, hidden_dim)
            for i in range(self.n_enc_features)
        ]
        # Stack → (B, T, n_features, hidden_dim), then weighted sum
        feat_stack   = torch.stack(feat_list, dim=2)    # (B, T, n_features, hidden_dim)
        vsn_weights  = torch.nn.functional.softmax(
            self._temporal_vsn_weights(x_enc, c_s), dim=-1
        ).unsqueeze(-1)                                   # (B, T, n_features, 1)
        lstm_input   = (feat_stack * vsn_weights).sum(dim=2)  # (B, T, hidden_dim)

        # ── LSTM with static init ─────────────────────────────────────────
        n_layers = self.lstm.num_layers
        h0 = self.h0_proj(c_e).view(B, n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = self.c0_proj(c_c).view(B, n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        lstm_out, _ = self.lstm(lstm_input, (h0, c0))  # (B, T, hidden_dim)

        # Gated residual skip from temporal VSN input to LSTM output
        gated = self.post_lstm_gate(lstm_out.reshape(B * T, -1)).view(B, T, -1)
        gated = self.post_lstm_ln(gated + lstm_input)

        # ── Self-Attention ────────────────────────────────────────────────
        attn_out, attn_weights = self.attn(gated)      # (B, T, hidden_dim)

        # Post-attention GRN conditioned on c_h (static attention context)
        # Expand c_h to match temporal dimension: (B, hidden_dim) -> (B*T, hidden_dim)
        c_h_exp = c_h.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        post = self.post_attn_grn(
            attn_out.reshape(B * T, -1), context=c_h_exp
        ).view(B, T, -1)
        post = self.post_attn_ln(post + gated)

        # ── Quantile head on last timestep ────────────────────────────────
        h_last = post[:, -1, :]   # (B, hidden_dim)
        p10, p50, p90 = self.quantile_head(h_last)

        return {
            "p10": p10.squeeze(-1),   # (B,)
            "p50": p50.squeeze(-1),
            "p90": p90.squeeze(-1),
            "attn_weights": attn_weights,
        }

    def _temporal_vsn_weights(
        self, x_enc: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Simple gated selection over raw feature magnitudes.
        Returns (B, T, n_enc_features) logits.
        """
        B, T, n_feat = x_enc.shape
        ctx_exp = context.unsqueeze(1).expand(B, T, -1)  # (B, T, hidden_dim)
        # Use a learned linear combination of features conditioned on context
        if not hasattr(self, "_vsn_select"):
            self._vsn_select = nn.Linear(self.hidden_dim + n_feat, n_feat).to(x_enc.device)
        combined = torch.cat([ctx_exp, x_enc], dim=-1)   # (B, T, hidden_dim+n_feat)
        return self._vsn_select(combined)                 # (B, T, n_feat) logits


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def pinball_loss(
    y_true: torch.Tensor,
    p10: torch.Tensor,
    p50: torch.Tensor,
    p90: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Multi-quantile pinball (quantile regression) loss.

    L_q(y, q_hat) = q * max(y - q_hat, 0) + (1-q) * max(q_hat - y, 0)

    Averages across quantiles {0.10, 0.50, 0.90}.
    """
    def _pinball(pred: torch.Tensor, q: float) -> torch.Tensor:
        err = y_true - pred
        loss = torch.where(err >= 0, q * err, (q - 1.0) * err)
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss.mean()

    return (_pinball(p10, 0.10) + _pinball(p50, 0.50) + _pinball(p90, 0.90)) / 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

class TFTDataset(torch.utils.data.Dataset):
    """
    Dataset wrapping pre-built sequence tensors for TFT training.

    Each item:
        x_enc       : (T, n_enc_features)  float32
        ticker_id   : int
        sector_id   : int
        regime_id   : int
        y_target    : float (vol-normalized return)
        weight      : float (sample weight)
    """

    def __init__(
        self,
        x_enc: np.ndarray,       # (N, T, F)
        ticker_ids: np.ndarray,   # (N,) int
        sector_ids: np.ndarray,   # (N,) int
        regime_ids: np.ndarray,   # (N,) int
        y_targets: np.ndarray,    # (N,) float
        weights: np.ndarray,      # (N,) float
    ):
        self.x_enc      = torch.tensor(x_enc,      dtype=torch.float32)
        self.ticker_ids = torch.tensor(ticker_ids,  dtype=torch.long)
        self.sector_ids = torch.tensor(sector_ids,  dtype=torch.long)
        self.regime_ids = torch.tensor(regime_ids,  dtype=torch.long)
        self.y_targets  = torch.tensor(y_targets,   dtype=torch.float32)
        self.weights    = torch.tensor(weights,      dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y_targets)

    def __getitem__(self, idx: int):
        return {
            "x_enc":      self.x_enc[idx],
            "ticker_id":  self.ticker_ids[idx],
            "sector_id":  self.sector_ids[idx],
            "regime_id":  self.regime_ids[idx],
            "y_target":   self.y_targets[idx],
            "weight":     self.weights[idx],
        }


def train_tft_epoch(
    model: TFT,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad_norm: float = 1.0,
) -> float:
    """One training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        x_enc      = batch["x_enc"].to(device)
        ticker_ids = batch["ticker_id"].to(device)
        sector_ids = batch["sector_id"].to(device)
        regime_ids = batch["regime_id"].to(device)
        y          = batch["y_target"].to(device)
        w          = batch["weight"].to(device)

        optimizer.zero_grad()
        out  = model(ticker_ids, sector_ids, regime_ids, x_enc)
        loss = pinball_loss(y, out["p10"], out["p50"], out["p90"], w)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def predict_tft(
    model: TFT,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run inference, return dict with 'p10', 'p50', 'p90' arrays."""
    model.eval()
    p10s, p50s, p90s = [], [], []
    for batch in loader:
        x_enc      = batch["x_enc"].to(device)
        ticker_ids = batch["ticker_id"].to(device)
        sector_ids = batch["sector_id"].to(device)
        regime_ids = batch["regime_id"].to(device)
        out = model(ticker_ids, sector_ids, regime_ids, x_enc)
        p10s.append(out["p10"].cpu().numpy())
        p50s.append(out["p50"].cpu().numpy())
        p90s.append(out["p90"].cpu().numpy())

    return {
        "p10": np.concatenate(p10s),
        "p50": np.concatenate(p50s),
        "p90": np.concatenate(p90s),
    }


if __name__ == "__main__":
    # Quick smoke test — random tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TFT(
        n_tickers=10, n_sectors=5, n_regimes=3,
        n_enc_features=8, embed_dim=8, hidden_dim=32, n_heads=2, n_lstm_layers=1,
    ).to(device)

    B, T, F = 4, 20, 8
    ticker_ids = torch.randint(0, 10, (B,)).to(device)
    sector_ids = torch.randint(0, 5,  (B,)).to(device)
    regime_ids = torch.randint(0, 3,  (B,)).to(device)
    x_enc      = torch.randn(B, T, F).to(device)

    out = model(ticker_ids, sector_ids, regime_ids, x_enc)
    p10, p50, p90 = out["p10"], out["p50"], out["p90"]

    assert (p10 <= p50).all(), "Monotonicity violated: P10 > P50"
    assert (p50 <= p90).all(), "Monotonicity violated: P50 > P90"
    print(f"✓ TFT smoke test passed. P10={p10[0]:.4f}  P50={p50[0]:.4f}  P90={p90[0]:.4f}")
    print(f"  Attn shape: {out['attn_weights'].shape}")
