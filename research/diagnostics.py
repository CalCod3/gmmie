"""
research.diagnostics — model introspection + data drift detection.

Two utilities:

  1. `permutation_importance` — for each feature, shuffle it in the holdout
     set and measure the increase in pinball loss. Big delta = the model
     relies on that feature. Tiny delta = the feature is decoration.

  2. `feature_drift` — Kolmogorov-Smirnov two-sample test comparing recent
     feature distributions to the training-window distribution. Returns
     per-feature D-statistic and a hard alarm if max(D) > threshold.

These run in seconds and produce JSON suitable for the daily strategist
dossier ("model is currently driven by tips_10y; feature dxy_logret_1d has
drifted since training").
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .model import HybridGoldForecaster, ModelConfig

logger = logging.getLogger(__name__)


# ── permutation importance ───────────────────────────────────────────────────

@dataclass
class FeatureImportance:
    feature: str
    delta_pinball: float
    relative: float       # delta / baseline


def _pinball_q50(y: np.ndarray, mu: np.ndarray) -> float:
    err = y - mu
    return float(np.mean(np.maximum(0.5 * err, -0.5 * err)))


@torch.no_grad()
def permutation_importance(
    model: HybridGoldForecaster,
    X: np.ndarray, y: np.ndarray,
    *, feature_cols: Sequence[str], n_struct: int,
    n_repeats: int = 5, seed: int = 0,
) -> List[FeatureImportance]:
    """
    For each feature, randomly permute its column in X and measure how much
    the median forecast's pinball loss increases. Returns sorted descending.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    Xs = torch.as_tensor(X[:, :n_struct], dtype=torch.float32)
    Xf = torch.as_tensor(X,               dtype=torch.float32)
    base = model(Xs, Xf)["mu"].squeeze(-1).cpu().numpy()
    baseline = _pinball_q50(y, base) + 1e-9

    results: List[FeatureImportance] = []
    for j, feat in enumerate(feature_cols):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            Xs_p = torch.as_tensor(Xp[:, :n_struct], dtype=torch.float32)
            Xf_p = torch.as_tensor(Xp,               dtype=torch.float32)
            mu_p = model(Xs_p, Xf_p)["mu"].squeeze(-1).cpu().numpy()
            deltas.append(_pinball_q50(y, mu_p) - baseline)
        avg = float(np.mean(deltas))
        results.append(FeatureImportance(
            feature=feat,
            delta_pinball=avg,
            relative=avg / baseline,
        ))
    results.sort(key=lambda r: r.delta_pinball, reverse=True)
    return results


# ── KS drift monitor ─────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    feature:  str
    d_stat:   float
    p_value:  float
    drifted:  bool


def feature_drift(
    train_panel: pd.DataFrame,
    recent_panel: pd.DataFrame,
    *, feature_cols: Sequence[str],
    p_threshold: float = 0.01,
    d_threshold: float = 0.15,
) -> List[DriftReport]:
    """
    Two-sample KS test: train_panel[col] vs recent_panel[col].

    A feature is flagged as drifted if the KS p-value < p_threshold OR the
    D-statistic > d_threshold. Both conditions catch different failure modes:
    p-value catches subtle persistent shifts; D-stat catches large local
    distribution differences even when n is large.
    """
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        logger.error("scipy required for drift monitor")
        return []

    reports: List[DriftReport] = []
    for c in feature_cols:
        if c not in train_panel.columns or c not in recent_panel.columns:
            continue
        a = train_panel[c].dropna().to_numpy(dtype=np.float64)
        b = recent_panel[c].dropna().to_numpy(dtype=np.float64)
        if len(a) < 30 or len(b) < 20:
            continue
        d, p = ks_2samp(a, b)
        reports.append(DriftReport(
            feature=c,
            d_stat=float(d),
            p_value=float(p),
            drifted=bool(p < p_threshold or d > d_threshold),
        ))
    reports.sort(key=lambda r: r.d_stat, reverse=True)
    return reports


# ── Wrapper: run both and dump to checkpoint dir ─────────────────────────────

def write_diagnostics(
    ckpt_dir: Path,
    model: HybridGoldForecaster,
    X_val: np.ndarray, y_val: np.ndarray,
    *, feature_cols: Sequence[str], n_struct: int,
    train_panel: Optional[pd.DataFrame] = None,
    recent_panel: Optional[pd.DataFrame] = None,
) -> dict:
    out: dict = {}

    if len(X_val) >= 50:
        try:
            imp = permutation_importance(
                model, X_val, y_val.reshape(-1),
                feature_cols=feature_cols, n_struct=n_struct,
            )
            out["importance"] = [asdict(r) for r in imp]
            top = ", ".join(r.feature for r in imp[:5])
            logger.info("top-5 features by permutation Δ: %s", top)
        except Exception as exc:
            logger.warning("importance failed: %s", exc)

    if train_panel is not None and recent_panel is not None:
        try:
            drift = feature_drift(train_panel, recent_panel,
                                   feature_cols=list(feature_cols))
            out["drift"] = [asdict(r) for r in drift]
            n_drift = sum(1 for r in drift if r.drifted)
            if n_drift:
                logger.warning(
                    "DRIFT ALARM: %d/%d features drifted (top: %s)",
                    n_drift, len(drift),
                    ", ".join(f"{r.feature}(D={r.d_stat:.2f})"
                              for r in drift[:3] if r.drifted),
                )
        except Exception as exc:
            logger.warning("drift failed: %s", exc)

    (ckpt_dir / "diagnostics.json").write_text(json.dumps(out, indent=2))
    return out
