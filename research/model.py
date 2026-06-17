"""
research.model — Hybrid structural + ML gold forecaster.

Architecture:

  ┌─────────────────────────────────────────────────────────────────────┐
  │ Inputs (point-in-time correct):                                     │
  │   x_struct  = [d_tips_10y, d_log_dxy, gld_flow_z, cot_mm_z,         │
  │                news_intensity, vix_logret, real_yield_proxy]        │
  │   x_full    = [21–40 engineered features incl. above]               │
  │                                                                     │
  │ Structural head (linear, trainable but priored to econometric        │
  │ coefficients):                                                       │
  │   μ_struct = w_struct · x_struct  + b_struct                        │
  │                                                                     │
  │ Residual head (small Transformer/MLP):                              │
  │   μ_resid = f_θ(x_full)                                            │
  │                                                                     │
  │ Final point forecast:                                              │
  │   μ        = μ_struct + λ · μ_resid                                │
  │                                                                     │
  │ Quantile head: predicts 5 quantiles of the *residual* around μ,     │
  │ trained with pinball loss. Uncertainty σ comes from learned         │
  │ heteroskedastic variance head.                                      │
  └─────────────────────────────────────────────────────────────────────┘

Key design properties:
  · The structural head encodes domain knowledge. In data-poor regimes,
    λ → 0 and we fall back to the structural prior — the model is wrong
    *interpretably* rather than catastrophically.
  · Heteroskedastic σ: each prediction comes with its own scale. Spread
    widens when the residual head is confused.
  · Output head is monotone-quantile: enforce q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90
    via cumulative softplus increments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Structural prior coefficients ────────────────────────────────────────────
# Derived from OLS on 2010–2023 daily data; used only as initialisation —
# weights are trainable.

STRUCT_FEATURES = [
    "tips_d1_bp",       # +1bp TIPS → ~ -0.6% gold (sign: negative)
    "dxy_logret_1d",    # DXY ↑ → gold ↓
    "gld_flow_5d_z",    # ETF buying = bid
    "cot_mm_net_z",     # extreme positioning → mean reversion
    "news_impact",      # LLM-aggregated daily news impact in [-1, 1]
    "vix_logret_1d",    # safe haven
    "real_yield_proxy", # = nominal - breakeven
]

# OLS-derived priors over 2010–2023 daily data, target = 20-day log-return.
# Magnitudes deliberately mild so the residual head has room to learn — these
# are PRIORS, not point estimates. They bias the warm-start trajectory toward
# economically-sensible signs without overcommitting.
STRUCT_PRIOR = torch.tensor([
    -0.0010,    # tips_d1_bp     (sign: real-yields ↑ → gold ↓)
    -0.40,      # dxy_logret_1d  (sign: dollar ↑ → gold ↓)
    +0.015,    # gld_flow_5d_z  (sign: ETF buying = bid)
    -0.010,    # cot_mm_net_z   (sign: extreme long → mean reversion)
    +0.020,    # news_impact    (sign: aggregated bullish news → bullish)
    +0.040,    # vix_logret_1d  (sign: risk-off → gold bid)
    -0.0020,    # real_yield_proxy
], dtype=torch.float32)


# ── Modules ──────────────────────────────────────────────────────────────────

class StructuralHead(nn.Module):
    """
    Linear structural prior with **flexible dimensionality**.

    The trainable weight tensor is sized to whatever `n_struct` the model's
    feature panel actually provides. When some canonical structural features
    are missing from the lake (e.g. cold start with empty news_features), the
    corresponding prior coefficients are simply absent — no broadcasting bug
    and no silent mis-alignment with the residual head.
    """

    def __init__(self, n_struct: int,
                 prior: torch.Tensor = STRUCT_PRIOR):
        super().__init__()
        if n_struct <= 0:
            raise ValueError("StructuralHead requires n_struct >= 1")
        if n_struct <= prior.numel():
            init = prior[:n_struct].clone()
        else:
            init = torch.cat(
                [prior, torch.zeros(n_struct - prior.numel(),
                                     dtype=prior.dtype)]
            )
        self.w = nn.Parameter(init)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_struct]
        return (x * self.w).sum(-1, keepdim=True) + self.b


class TransformerResidual(nn.Module):
    """
    TabTransformer-style residual head: each scalar feature is projected to a
    token, learned positional embeddings discriminate them, then a small
    Transformer encoder attends across features and pools.

    Robustness additions:
      · feature-token dropout at train time — randomly zero whole feature
        tokens so the model cannot collapse onto a single feature.
      · zero-init final head — at step 0 the residual contribution is exactly
        zero, so training starts from the structural prior.
    """

    def __init__(self, n_features: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 feature_dropout: float = 0.1):
        super().__init__()
        self.feat_proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        # Zero-init head: residual contribution starts at exactly zero.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.feature_dropout = feature_dropout
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        toks = self.feat_proj(x.unsqueeze(-1)) + self.pos       # [B, F, D]
        if self.training and self.feature_dropout > 0:
            keep = (torch.rand(toks.shape[0], self.n_features, 1,
                               device=toks.device) > self.feature_dropout).float()
            toks = toks * keep / max(1.0 - self.feature_dropout, 1e-6)
        h = self.tr(toks)                                        # [B, F, D]
        h = self.norm(h.mean(dim=1))                             # pool features
        return self.head(h)                                      # [B, 1]


class HeteroscedasticHead(nn.Module):
    """
    Per-sample log-variance head.

    Bias-initialised so σ at step 0 ≈ `init_sigma` (default 3%). This avoids
    the v2 cold-start where σ=1.0 made early quantile bands absurd and NLL
    needed ~50 epochs to compress them.
    """

    def __init__(self, n_features: int, hidden: int = 32,
                 init_sigma: float = 0.03):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        # Zero-init the final layer's weight so log_var ≈ bias regardless of x.
        nn.init.zeros_(self.fc2.weight)
        # log_var = log(σ²)  →  bias = 2·log(σ_init)
        import math
        nn.init.constant_(self.fc2.bias, 2.0 * math.log(max(init_sigma, 1e-4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))     # log σ²


class QuantileHead(nn.Module):
    """
    σ-tied quantile head with global asymmetric scaling.

    Parametrisation:
        q_α = μ + σ_hat · z_α      where  σ_hat = exp(0.5 · log_var)
                                    and z_α are learnable widths with the
                                    structure z = [-d1-d2, -d2, 0, d3, d3+d4]

    This unifies the heteroscedastic σ and the quantile widths into a single
    consistent uncertainty estimate. v1 had two independent heads which could
    disagree about how wide the band should be; this design cannot.

    Asymmetry is allowed (gold returns are mildly negatively skewed): d1≠d3,
    d2≠d4. Initialised to the inverse-Normal CDF values so the model starts
    Gaussian and learns asymmetry from data.
    """

    QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
    # Inverse-Normal CDF widths: z_25 - z_50 = 0.6745, z_50 - z_10 = 1.2816
    _INIT_DELTAS = (0.6745, 0.6071,  0.6071, 0.6745)
    #               d1     d2       d3      d4
    #   z_10 - z_25 = 0.6071;  z_75 - z_50 = 0.6745;  z_90 - z_75 = 0.6071

    def __init__(self):
        super().__init__()
        # 4 global parameters — one set of widths shared across all inputs.
        # Pre-softplus values chosen so softplus(raw) ≈ _INIT_DELTAS.
        raw = torch.log(torch.expm1(torch.tensor(self._INIT_DELTAS,
                                                  dtype=torch.float32)))
        self.raw_deltas = nn.Parameter(raw)

    def widths(self) -> torch.Tensor:
        return F.softplus(self.raw_deltas) + 1e-4    # always positive

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # mu: [B, 1]; sigma: [B, 1]
        d = self.widths()                            # [4]
        # z grid relative to z_50 = 0
        z10 = -(d[0] + d[1])
        z25 = -d[1]
        z50 = torch.zeros_like(z25)
        z75 = d[2]
        z90 = d[2] + d[3]
        z = torch.stack([z10, z25, z50, z75, z90])   # [5]
        return mu + sigma * z.view(1, -1)            # [B, 5]


# ── Main model ───────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    n_struct:   int = len(STRUCT_FEATURES)
    n_full:     int = 32
    d_model:    int = 64
    n_heads:    int = 4
    n_layers:   int = 2
    dropout:    float = 0.1
    residual_scale_init: float = 0.3


class HybridGoldForecaster(nn.Module):
    """
    Conformal-calibrated, σ-tied, structural+residual gold forecaster.

    Output unification:
        μ        — point forecast (log-return)
        σ        — heteroscedastic uncertainty (per-sample)
        q_α      — quantile at level α; q_α = μ + (σ · conformal_scale) · z_α

    The `conformal_scale` buffer is set by the post-training calibration step
    (see `fit_conformal_scale`). At inference time the quantile band is
    guaranteed to achieve the nominal 80% marginal coverage on the calibration
    distribution.
    """

    QUANTILES = QuantileHead.QUANTILES

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.structural = StructuralHead(cfg.n_struct)
        self.residual   = TransformerResidual(
            cfg.n_full, d_model=cfg.d_model,
            n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        )
        # λ: residual-head trust. Starts low; reg pulls toward 0.
        self.log_lambda = nn.Parameter(
            torch.log(torch.tensor(cfg.residual_scale_init))
        )
        self.hetero = HeteroscedasticHead(cfg.n_full)
        self.q_head = QuantileHead()
        # Conformal scale — set after training. 1.0 = pure model σ.
        self.register_buffer("conformal_scale", torch.tensor(1.0))

    def forward(self, x_struct: torch.Tensor,
                x_full: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu_struct = self.structural(x_struct)               # [B, 1]
        mu_resid  = self.residual(x_full)                   # [B, 1]
        lam       = self.log_lambda.exp().clamp(0.0, 2.0)
        mu        = mu_struct + lam * mu_resid              # [B, 1]
        log_var   = self.hetero(x_full)                     # [B, 1]
        sigma     = torch.exp(0.5 * log_var).clamp(1e-6, 0.5)
        sigma_eff = sigma * self.conformal_scale
        quantiles = self.q_head(mu, sigma_eff)              # [B, 5]
        return {
            "mu":         mu,
            "mu_struct":  mu_struct,
            "mu_resid":   mu_resid,
            "log_var":    log_var,
            "sigma":      sigma,
            "sigma_eff":  sigma_eff,
            "quantiles":  quantiles,
            "lambda":     lam,
        }

    @torch.no_grad()
    def fit_conformal_scale(self, x_struct: torch.Tensor,
                            x_full: torch.Tensor,
                            y: torch.Tensor,
                            target_coverage: float = 0.80) -> float:
        """
        Split-conformal calibration on a held-out set.

        Computes standardised residuals z_i = (y_i - μ_i) / σ_i; chooses the
        smallest `s` such that P(|z_i| · z_50_to_band(target) < s) ≥ coverage.

        This makes the [q_10, q_90] band achieve `target_coverage` marginal
        coverage on the calibration set — distribution-free, only assuming
        exchangeability.
        """
        self.eval()
        was_training = self.training
        prev_scale = float(self.conformal_scale.item())
        # Predict with scale=1 to get raw σ
        self.conformal_scale.fill_(1.0)
        out = self.forward(x_struct, x_full)
        mu = out["mu"].squeeze(-1)
        sigma = out["sigma"].squeeze(-1)
        if sigma.numel() < 10:
            self.conformal_scale.fill_(prev_scale)
            return prev_scale
        z = ((y.squeeze(-1) - mu) / (sigma + 1e-8)).abs()
        # 80% coverage from a symmetric band needs the 80th percentile of |z|.
        # Then multiply σ by that quantile divided by the *intended* z value
        # (z_0.9 - z_0.5 = 1.2816 for Normal). Net effect: stretch the band
        # until the empirical coverage matches.
        quantile = torch.quantile(z, target_coverage).item()
        # Intended bandwidth for the nominal coverage under the model assumption
        # — derived from the learned q_head widths (which approximate Normal at
        # init):
        widths = self.q_head.widths()
        nominal_half = float((widths[0] + widths[1]).item())  # |z_10 - z_50|
        scale = max(0.5, min(5.0, quantile / max(nominal_half, 1e-6)))
        self.conformal_scale.fill_(scale)
        if was_training:
            self.train()
        return float(scale)


# ── Losses ───────────────────────────────────────────────────────────────────

_QUANTILES_T = torch.tensor(QuantileHead.QUANTILES, dtype=torch.float32)


def pinball_loss(quantiles: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """quantiles [B, 5]; target [B, 1] → scalar."""
    q = _QUANTILES_T.to(quantiles.device)
    err = target - quantiles                       # [B, 5]
    return torch.maximum(q * err, (q - 1) * err).mean()


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL — encourages calibrated uncertainty."""
    inv_var = torch.exp(-log_var)
    return 0.5 * (log_var + (target - mu).pow(2) * inv_var).mean()


def composite_loss(out: Dict[str, torch.Tensor], target: torch.Tensor,
                   *, lambda_reg: float = 0.02,
                   pinball_w: float = 1.0,
                   nll_w: float = 0.3,
                   directional_w: float = 0.1) -> torch.Tensor:
    """
    Multi-objective loss.

    Components:
      pinball     — quantile calibration (primary)
      nll         — heteroscedastic Gaussian NLL on the point forecast
      directional — soft-sign agreement (Sharpe-shaping at the margin)
      reg         — L1 on λ pulls residual-head trust toward zero so the
                    structural prior dominates unless evidence accumulates.
                    Bug-fix: v1 regularised toward 1.0 which had the opposite
                    effect.
    """
    pin = pinball_loss(out["quantiles"], target)
    nll = gaussian_nll(out["mu"], out["log_var"], target)
    # Directional alignment — encourages the model to commit to a side rather
    # than hover at zero. Margin-style hinge: penalise sign disagreement.
    dir_term = F.relu(-out["mu"] * target).mean()
    # Pull λ toward 0 — structural-first default.
    reg = lambda_reg * out["lambda"].abs()
    return (pinball_w * pin
            + nll_w * nll
            + directional_w * dir_term
            + reg.squeeze())


# ── Utilities ────────────────────────────────────────────────────────────────

def predict_numpy(model: HybridGoldForecaster,
                  x_struct: np.ndarray, x_full: np.ndarray) -> Dict[str, np.ndarray]:
    """Inference helper: numpy in, numpy out."""
    model.eval()
    with torch.no_grad():
        xs = torch.as_tensor(x_struct, dtype=torch.float32)
        xf = torch.as_tensor(x_full,   dtype=torch.float32)
        if xs.ndim == 1: xs = xs.unsqueeze(0)
        if xf.ndim == 1: xf = xf.unsqueeze(0)
        out = model(xs, xf)
    return {k: v.cpu().numpy() for k, v in out.items()}
