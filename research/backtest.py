"""
research.backtest — purged walk-forward cross-validation with embargo.

The single most important file in the repo. If this is wrong, every model
decision downstream is biased. Implements:

  · purged k-fold (Lopez de Prado, AFML §7) — removes train samples whose
    label window overlaps the test window
  · embargo — additional purge gap to defeat serial correlation leakage
  · walk-forward — strictly causal: train on past, test on future, slide
  · stationary bootstrap (Politis & Romano 1994) — CI on Sharpe / hit rate
  · pinball + Brier scoring for quantile / probabilistic outputs

Inputs:
  panel: DataFrame indexed by date with feature columns and label `y`.
  fit_predict_fn: (X_train, y_train, X_test) -> y_pred (ndarray or DataFrame
                   for quantile outputs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Splits ────────────────────────────────────────────────────────────────────

def walkforward_splits(
    n: int,
    *,
    train_days: int = 252 * 8,        # 8 years
    test_days:  int = 63,             # 1 quarter
    embargo_days: int = 5,
    label_horizon_days: int = 20,     # purge label window
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, test_idx) tuples for strictly walk-forward CV.
    The first `train_days` rows are the initial training window; each fold
    advances by `test_days`. `embargo_days` are dropped around the test set
    to prevent leakage via serial correlation.
    """
    splits = []
    start_test = train_days
    while start_test + test_days <= n:
        train_end = start_test - embargo_days - label_horizon_days
        if train_end <= 0:
            start_test += test_days
            continue
        train_idx = np.arange(0, train_end)
        test_idx  = np.arange(start_test, start_test + test_days)
        splits.append((train_idx, test_idx))
        start_test += test_days
    return splits


# ── Metrics ──────────────────────────────────────────────────────────────────

def directional_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def sharpe(returns: np.ndarray, *, periods_per_year: float = 252.0) -> float:
    r = returns[np.isfinite(returns)]
    if r.std(ddof=0) < 1e-12:
        return 0.0
    return float(r.mean() / r.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_newey_west(returns: np.ndarray, *, periods_per_year: float = 252.0,
                       lag: int = 20) -> float:
    """
    Newey-West HAC-corrected Sharpe.

    When `returns` represent OVERLAPPING horizon outcomes (e.g., daily-sampled
    20-day forward log-returns), the i.i.d. variance estimate understates the
    true variance by ~horizon-fold. This estimator inflates σ to match.

    Reference: Lo, A.W. (2002), "The Statistics of Sharpe Ratios."
    """
    r = returns[np.isfinite(returns)].astype(np.float64)
    n = len(r)
    if n < lag + 2 or r.std(ddof=0) < 1e-12:
        return 0.0
    mu = r.mean()
    e  = r - mu
    var_iid = float((e * e).sum()) / n
    # Newey-West kernel with Bartlett weights
    var_hac = var_iid
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = float((e[k:] * e[:-k]).sum()) / n
        var_hac += 2.0 * w * gamma_k
    var_hac = max(var_hac, 1e-12)
    return float(mu / np.sqrt(var_hac) * np.sqrt(periods_per_year))


def stationary_bootstrap_sharpe(
    returns: np.ndarray, *, n_boot: int = 1000, p: float = 1 / 10,
    periods_per_year: int = 252, seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Stationary-bootstrap Sharpe with 95% CI (Politis & Romano, 1994).

    Vectorised: blocks are sampled in numpy, then `n_boot` rows of length `n`
    are stacked and Sharpe is computed in one matrix op. ~50× faster than
    the per-step Python loop in v1 for n=2000, n_boot=1000.
    """
    rng = np.random.default_rng(seed)
    r = returns[np.isfinite(returns)]
    n = len(r)
    if n < 30:
        s = sharpe(r, periods_per_year=periods_per_year)
        return (s, s, s)

    indices = _stationary_bootstrap_indices(rng, n=n, n_boot=n_boot, p=p)
    samples = r[indices]                          # [n_boot, n]
    mu = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=0)
    sharpes = np.where(sd > 1e-12, mu / sd, 0.0) * np.sqrt(periods_per_year)
    return (
        float(np.median(sharpes)),
        float(np.quantile(sharpes, 0.025)),
        float(np.quantile(sharpes, 0.975)),
    )


def _stationary_bootstrap_indices(
    rng: np.random.Generator, *, n: int, n_boot: int, p: float,
) -> np.ndarray:
    """
    Generate an [n_boot, n] index matrix following the stationary bootstrap.

    Block lengths ~ Geom(p), starts uniform on [0, n). We sample blocks until
    each row reaches length n. With expected block length 1/p, the inner
    Python loop runs ~ n*p iterations per bootstrap — O(n × p × n_boot) total
    vs O(n × n_boot) for the per-step approach.
    """
    out = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        pos = 0
        while pos < n:
            start = int(rng.integers(0, n))
            L = int(rng.geometric(p))
            L = min(L, n - pos)
            # block = (start + 0..L-1) mod n  (vectorised)
            out[b, pos:pos + L] = (start + np.arange(L)) % n
            pos += L
    return out


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray,
                 quantile: float) -> float:
    err = y_true - q_pred
    return float(np.mean(np.maximum(quantile * err, (quantile - 1) * err)))


def quantile_coverage(y_true: np.ndarray, q_low: np.ndarray,
                      q_high: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(q_low) & np.isfinite(q_high)
    if not mask.any():
        return float("nan")
    inside = (y_true[mask] >= q_low[mask]) & (y_true[mask] <= q_high[mask])
    return float(inside.mean())


# ── Run ──────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold:        int
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    n_test:      int
    hit_rate:    float
    pinball_q50: float
    sharpe:      float
    sharpe_nw:   float
    raw_returns: np.ndarray = field(repr=False)
    regime_metrics: Optional[Dict[str, Dict[str, float]]] = None

@dataclass
class BacktestReport:
    folds:        List[FoldResult]
    overall_hit:  float
    overall_pin:  float
    sharpe_med:   float
    sharpe_lo:    float
    sharpe_hi:    float
    sharpe_nw:    Optional[float] = None
    sharpe_nonoverlap: Optional[float] = None
    coverage_80:  Optional[float] = None
    cost_bps:     float = 0.0
    threshold:    float = 0.0
    horizon_d:    int   = 1
    by_regime:    Optional[Dict[str, Dict[str, float]]] = None

    def __str__(self) -> str:
        parts = [
            f"Backtest h={self.horizon_d}d cost={self.cost_bps:.1f}bp "
            f"thr={self.threshold:.4g} | n_folds={len(self.folds)} | "
            f"hit={self.overall_hit:.3f} pin={self.overall_pin:.4f}",
            f"Sharpe(naive)={self.sharpe_med:.2f} "
            f"[{self.sharpe_lo:.2f}, {self.sharpe_hi:.2f}]",
        ]
        if self.sharpe_nw is not None:
            parts.append(f"Sharpe(NW-HAC)={self.sharpe_nw:.2f}")
        if self.sharpe_nonoverlap is not None:
            parts.append(f"Sharpe(non-overlap)={self.sharpe_nonoverlap:.2f}")
        if self.coverage_80 is not None:
            parts.append(f"cov80={self.coverage_80:.2f}")
        return "  ".join(parts)


def run(
    panel: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    label_col: str,
    fit_predict_fn: Callable[..., np.ndarray],
    quantile_fit_predict_fn: Optional[Callable] = None,
    train_days: int = 252 * 8,
    test_days:  int = 63,
    embargo_days: int = 5,
    label_horizon_days: int = 20,
    cost_bps: float = 1.0,
    position_threshold: Optional[float] = None,
    threshold_percentile: float = 30.0,
    regime_col: Optional[str] = "vix_close",
) -> BacktestReport:
    """
    `fit_predict_fn` and `quantile_fit_predict_fn` callable signatures:
        fn(X_train, y_train, X_test) -> ndarray[N_test]            (point)
        fn(X_train, y_train, X_test) -> ndarray[N_test, 5]         (quantiles)

    or, optionally (preferred for the per-fold position threshold):
        fn(X_train, y_train, X_test) -> (test_preds, train_preds)

    When the closure returns the tuple form, the threshold is derived from
    the **train**-side prediction distribution — no test-set leakage. When
    it returns the bare array form, the test-side fallback is used and the
    threshold reported in the BacktestReport is a soft upper bound.
    """
    """
    Run a purged walk-forward backtest.

    `panel` must be chronologically sorted. A `date` column is required (either
    as a true column or as the index — we normalise both paths).

    If `quantile_fit_predict_fn` is provided, it must return a 2-D ndarray
    `[n_test, 5]` (the q10/q25/q50/q75/q90 columns).
    """
    panel = panel.copy()
    # Normalise: ensure `date` is a column, not the index, and dtype is datetime
    if "date" not in panel.columns:
        panel = panel.reset_index()
        # After reset, the old index column might be named "index"
        if "date" not in panel.columns and "index" in panel.columns:
            panel = panel.rename(columns={"index": "date"})
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)
    panel = panel.dropna(subset=[label_col])
    panel = panel.reset_index(drop=True)
    n = len(panel)
    if n < train_days + test_days:
        raise RuntimeError(
            f"Not enough data: {n} rows < train_days({train_days}) + test_days({test_days})"
        )
    splits = walkforward_splits(
        n,
        train_days=train_days, test_days=test_days,
        embargo_days=embargo_days, label_horizon_days=label_horizon_days,
    )
    if not splits:
        raise RuntimeError("Not enough data for a single fold")

    folds: List[FoldResult] = []
    all_returns: List[np.ndarray] = []
    all_dates:   List[np.ndarray] = []
    all_mid:     List[np.ndarray] = []
    all_y:       List[np.ndarray] = []
    all_regime:  List[np.ndarray] = []
    all_cov: List[float] = []
    cost = float(cost_bps) * 1e-4

    for k, (tr_idx, te_idx) in enumerate(splits):
        tr = panel.iloc[tr_idx]
        te = panel.iloc[te_idx]
        X_tr = tr[feature_cols]
        y_tr = tr[label_col]
        X_te = te[feature_cols]
        y_te = te[label_col].values

        train_mid = None
        if quantile_fit_predict_fn is not None:
            result = quantile_fit_predict_fn(X_tr, y_tr, X_te)
            if isinstance(result, tuple):
                q_pred, train_q = result
                train_mid = train_q[:, 2]
            else:
                q_pred = result
            mid = q_pred[:, 2]                    # q50
            pin = pinball_loss(y_te, mid, 0.5)
            cov = quantile_coverage(y_te, q_pred[:, 0], q_pred[:, 4])
            all_cov.append(cov)
        else:
            result = fit_predict_fn(X_tr, y_tr, X_te)
            if isinstance(result, tuple):
                mid, train_mid = result
            else:
                mid = result
            pin = pinball_loss(y_te, mid, 0.5)

        # Threshold derived strictly from the TRAIN-side prediction distribution
        # when the closure provided one — no test-set leakage. Falls back to
        # the test side when no train-side preds were returned.
        if position_threshold is None:
            ref = train_mid if train_mid is not None else mid
            thr = float(np.quantile(np.abs(ref),
                                     threshold_percentile / 100.0))
        else:
            thr = float(position_threshold)

        pos = np.where(np.abs(mid) < thr, 0.0, np.sign(mid)).astype(np.float64)

        # Transaction cost: charge `cost` whenever position changes
        d_pos = np.zeros_like(pos)
        d_pos[1:] = np.abs(pos[1:] - pos[:-1])
        # Initial entry: charge full cost
        d_pos[0] = np.abs(pos[0])
        ret = pos * y_te - d_pos * cost

        hit = directional_hit_rate(y_te[pos != 0], mid[pos != 0])
        s_fold = sharpe(ret)
        s_nw   = sharpe_newey_west(ret, lag=label_horizon_days)

        # Per-regime breakdown using the chosen regime column
        regime_metrics: Optional[Dict[str, Dict[str, float]]] = None
        if regime_col and regime_col in te.columns:
            reg_vals = te[regime_col].to_numpy()
            regime_metrics = _per_regime(reg_vals, ret, mid, y_te)
            all_regime.append(reg_vals)
        else:
            all_regime.append(np.full(len(te), np.nan))

        folds.append(FoldResult(
            fold=k,
            test_start=pd.Timestamp(te["date"].iloc[0]),
            test_end=pd.Timestamp(te["date"].iloc[-1]),
            n_test=len(te),
            hit_rate=hit,
            pinball_q50=pin,
            sharpe=s_fold,
            sharpe_nw=s_nw,
            raw_returns=ret,
            regime_metrics=regime_metrics,
        ))
        all_returns.append(ret)
        all_dates.append(te["date"].to_numpy())
        all_mid.append(mid)
        all_y.append(y_te)

        logger.info(
            "fold %2d %s → %s n=%d hit=%.3f pin=%.4f Sh=%.2f Sh-NW=%.2f thr=%.4g",
            k, te["date"].iloc[0], te["date"].iloc[-1], len(te),
            hit, pin, s_fold, s_nw, thr,
        )

    all_ret = np.concatenate(all_returns)
    med, lo, hi = stationary_bootstrap_sharpe(all_ret)
    sharpe_nw = sharpe_newey_west(all_ret, lag=label_horizon_days)

    # Non-overlapping evaluation: subsample every `label_horizon_days` rows,
    # which gives ~independent observations. The correct annualisation factor
    # is √(252/horizon) — properly conservative when label_horizon ≫ 1.
    if label_horizon_days >= 2:
        non_ov = all_ret[::label_horizon_days]
        ppy_no = 252.0 / max(label_horizon_days, 1)
        sharpe_no = sharpe(non_ov, periods_per_year=ppy_no)
    else:
        sharpe_no = None

    by_regime = None
    if any(rm is not None for rm in (f.regime_metrics for f in folds)):
        # Aggregate by binned regime across all folds
        all_reg = np.concatenate(all_regime)
        by_regime = _per_regime(all_reg, all_ret,
                                np.concatenate(all_mid),
                                np.concatenate(all_y))

    report = BacktestReport(
        folds=folds,
        overall_hit=float(np.nanmean([f.hit_rate    for f in folds])),
        overall_pin=float(np.nanmean([f.pinball_q50 for f in folds])),
        sharpe_med=med, sharpe_lo=lo, sharpe_hi=hi,
        sharpe_nw=sharpe_nw,
        sharpe_nonoverlap=sharpe_no,
        coverage_80=(float(np.nanmean(all_cov)) if all_cov else None),
        cost_bps=cost_bps,
        threshold=(position_threshold if position_threshold is not None
                   else float(np.quantile(np.abs(np.concatenate(all_mid)),
                                          threshold_percentile / 100.0))),
        horizon_d=label_horizon_days,
        by_regime=by_regime,
    )
    logger.info("%s", report)
    return report


def _per_regime(regime_vals: np.ndarray, ret: np.ndarray,
                mid: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Bucket by regime tertile (low/mid/high) and report metrics each."""
    mask = np.isfinite(regime_vals)
    if not mask.any():
        return {}
    rv = regime_vals[mask]
    t33, t67 = np.quantile(rv, [1 / 3, 2 / 3])
    out: Dict[str, Dict[str, float]] = {}
    for name, sel in [("low",  regime_vals <= t33),
                       ("mid",  (regime_vals > t33) & (regime_vals <= t67)),
                       ("high", regime_vals > t67)]:
        sel = sel & mask
        if sel.sum() < 5:
            continue
        out[name] = {
            "n":       int(sel.sum()),
            "hit":     float(directional_hit_rate(y[sel], mid[sel])),
            "sharpe":  float(sharpe(ret[sel])),
            "mean_r":  float(ret[sel].mean()),
        }
    return out
