"""
Tests for the disciplinary core: walk-forward splits, Sharpe estimators,
pinball loss, quantile coverage. If any of these is wrong the entire system
silently makes biased decisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.backtest import (
    walkforward_splits,
    sharpe,
    sharpe_newey_west,
    stationary_bootstrap_sharpe,
    pinball_loss,
    quantile_coverage,
    directional_hit_rate,
)


# ── walkforward_splits ───────────────────────────────────────────────────────

def test_walkforward_splits_strictly_causal():
    splits = walkforward_splits(n=1000, train_days=500, test_days=50,
                                embargo_days=5, label_horizon_days=20)
    assert splits, "should produce at least one fold"
    for tr, te in splits:
        # Train always BEFORE test (with embargo + label-window gap)
        assert tr.max() < te.min()
        assert te.min() - tr.max() >= 5 + 20      # embargo + label horizon
        # Test windows are contiguous size N
        assert te.max() - te.min() + 1 == 50


def test_walkforward_splits_advance_one_window():
    splits = walkforward_splits(n=2000, train_days=1000, test_days=100,
                                embargo_days=5, label_horizon_days=20)
    # Each successive fold advances by exactly test_days
    for (_, te0), (_, te1) in zip(splits[:-1], splits[1:]):
        assert te1.min() - te0.min() == 100


# ── sharpe estimators ────────────────────────────────────────────────────────

def test_sharpe_iid_matches_closed_form():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=10_000)
    s = sharpe(r, periods_per_year=252)
    # Expected: 0.001/0.01 × √252 ≈ 1.587
    assert 1.4 < s < 1.8


def test_sharpe_nw_matches_naive_when_iid():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=5000)
    s_iid = sharpe(r)
    s_nw  = sharpe_newey_west(r, lag=20)
    # With true i.i.d., HAC adjustment should be within ~10%
    assert abs(s_iid - s_nw) / max(abs(s_iid), 1e-6) < 0.15


def test_sharpe_nw_corrects_overlap_inflation():
    """
    Synthetic overlapping-return test: build daily-sampled 20-day moving sums
    of i.i.d. shocks. Naive Sharpe over-states ~√20 because of positive
    autocorrelation; NW should be much closer to the truth.
    """
    rng = np.random.default_rng(1)
    eps = rng.normal(0.0, 0.01, size=2000)
    # 20-day forward sum (overlapping)
    h = 20
    r_over = np.convolve(eps, np.ones(h), mode="valid")
    s_naive = sharpe(r_over)
    s_nw    = sharpe_newey_west(r_over, lag=h)
    # The NW variance is multiplied by ~h relative to iid, so SE ≈ sqrt(h)
    # larger and Sharpe ≈ s_naive / sqrt(h). For h=20: s_nw ≈ s_naive / 4.5
    assert abs(s_nw) < abs(s_naive) * 0.6, \
        f"HAC should shrink overlapping-Sharpe; got naive={s_naive:.3f} nw={s_nw:.3f}"


def test_stationary_bootstrap_returns_finite_ci():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0005, 0.012, size=2000)
    med, lo, hi = stationary_bootstrap_sharpe(r, n_boot=200, p=0.1)
    assert np.isfinite(med) and np.isfinite(lo) and np.isfinite(hi)
    assert lo <= med <= hi


# ── pinball + coverage ──────────────────────────────────────────────────────

def test_pinball_zero_when_perfect():
    y = np.array([1.0, 2.0, 3.0])
    q = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, q, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_pinball_asymmetric():
    # quantile τ=0.9 penalises over-prediction less than under-prediction
    y = np.array([1.0])
    over  = pinball_loss(y, np.array([2.0]), 0.9)   # pred 2, actual 1
    under = pinball_loss(y, np.array([0.0]), 0.9)   # pred 0, actual 1
    assert under > over


def test_coverage_perfect_band():
    y = np.array([0.0, 0.5, 1.0])
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([ 1.5,  1.5,  1.5])
    assert quantile_coverage(y, lo, hi) == pytest.approx(1.0)


def test_coverage_no_band():
    y  = np.array([0.0, 0.5, 1.0])
    lo = np.array([2.0, 2.0, 2.0])
    hi = np.array([3.0, 3.0, 3.0])
    assert quantile_coverage(y, lo, hi) == pytest.approx(0.0)


# ── directional hit ─────────────────────────────────────────────────────────

def test_hit_rate_perfect_signs():
    y    = np.array([0.1, -0.2, 0.3, -0.4])
    pred = np.array([0.5, -0.5, 0.5, -0.5])
    assert directional_hit_rate(y, pred) == pytest.approx(1.0)


def test_hit_rate_random_is_half():
    rng = np.random.default_rng(3)
    y    = rng.normal(0, 1, size=10_000)
    pred = rng.normal(0, 1, size=10_000)
    # ~ Bernoulli(0.5), 95% CI ~ ±0.01
    h = directional_hit_rate(y, pred)
    assert 0.47 < h < 0.53
