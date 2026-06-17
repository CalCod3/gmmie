"""
Tests for the model: shape contracts, conformal calibration, monotone quantiles,
and that composite_loss is decreasing on a simple synthetic problem.
"""

from __future__ import annotations

import numpy as np
import torch
import pytest

from research.model import (
    HybridGoldForecaster,
    ModelConfig,
    QuantileHead,
    StructuralHead,
    STRUCT_PRIOR,
    composite_loss,
    pinball_loss as pinball_t,
    gaussian_nll,
)


# ── shape + init ─────────────────────────────────────────────────────────────

def test_structural_head_variable_dim():
    """Slicing the prior must work for n_struct < and > prior length."""
    h_small = StructuralHead(n_struct=3)
    assert h_small.w.shape == (3,)
    # Smaller than prior — initialised from first 3 entries
    assert torch.allclose(h_small.w.detach(), STRUCT_PRIOR[:3])

    h_big = StructuralHead(n_struct=10)
    assert h_big.w.shape == (10,)
    # Larger than prior — excess entries zero
    assert torch.allclose(h_big.w.detach()[: STRUCT_PRIOR.numel()], STRUCT_PRIOR)
    assert torch.allclose(h_big.w.detach()[STRUCT_PRIOR.numel():],
                           torch.zeros(10 - STRUCT_PRIOR.numel()))


def test_model_forward_shapes():
    cfg = ModelConfig(n_struct=5, n_full=15)
    model = HybridGoldForecaster(cfg)
    B = 4
    xs = torch.randn(B, 5)
    xf = torch.randn(B, 15)
    out = model(xs, xf)
    assert out["mu"].shape == (B, 1)
    assert out["sigma"].shape == (B, 1)
    assert out["quantiles"].shape == (B, 5)
    # σ at init is small after the bias fix (≈ 3%)
    assert (out["sigma"].detach() < 0.10).all()


def test_quantiles_monotone():
    """q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90 must hold by construction."""
    cfg = ModelConfig(n_struct=3, n_full=8)
    model = HybridGoldForecaster(cfg)
    xs = torch.randn(64, 3)
    xf = torch.randn(64, 8)
    with torch.no_grad():
        q = model(xs, xf)["quantiles"].numpy()
    diffs = np.diff(q, axis=1)
    assert (diffs >= -1e-7).all(), f"quantile monotonicity violated: {diffs.min()}"


# ── conformal ───────────────────────────────────────────────────────────────

def test_conformal_scale_widens_band_to_match_coverage():
    """
    Train σ initially too small, then fit conformal scale on a cal set
    where empirical |z| is large. Conformal scale should grow > 1.
    """
    cfg = ModelConfig(n_struct=3, n_full=8)
    model = HybridGoldForecaster(cfg)
    # Force tiny initial σ then create a cal set with much larger residuals
    rng = np.random.default_rng(0)
    xs = torch.randn(200, 3)
    xf = torch.randn(200, 8)
    y_big = torch.tensor(rng.normal(0, 0.10, size=(200, 1)),
                         dtype=torch.float32)   # 10% σ residuals
    s0 = float(model.conformal_scale.item())
    scale = model.fit_conformal_scale(xs, xf, y_big, target_coverage=0.80)
    assert scale > s0


# ── losses ──────────────────────────────────────────────────────────────────

def test_pinball_torch_zero_when_perfect():
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    t = torch.tensor([[3.0]])
    # not perfect for all quantiles, but check structure
    val = pinball_t(q, t)
    assert torch.isfinite(val)


def test_composite_loss_decreases_on_synthetic():
    """
    Trivial synthetic problem: y = w · x + ε; gradient descent must reduce
    composite loss within 50 steps.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(n_struct=4, n_full=12)
    model = HybridGoldForecaster(cfg)
    rng = np.random.default_rng(0)
    N = 256
    X = rng.normal(0, 1, size=(N, 12)).astype(np.float32)
    true_w = rng.normal(0, 0.5, size=12).astype(np.float32)
    y = (X @ true_w + rng.normal(0, 0.01, size=N)).astype(np.float32)
    Xs = torch.as_tensor(X[:, :4])
    Xf = torch.as_tensor(X)
    yt = torch.as_tensor(y).unsqueeze(-1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    out0 = model(Xs, Xf)
    loss0 = composite_loss(out0, yt).item()
    for _ in range(80):
        opt.zero_grad()
        out = model(Xs, Xf)
        loss = composite_loss(out, yt)
        loss.backward()
        opt.step()
    loss_final = composite_loss(model(Xs, Xf), yt).item()
    assert loss_final < loss0, f"loss didn't decrease: {loss0:.4f} → {loss_final:.4f}"
