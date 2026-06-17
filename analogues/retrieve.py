"""
analogues.retrieve — top-K analogue retrieval with optional regime filtering.

Used by the strategist to ground its thesis in actual historical episodes.
The expected-value statistic over retrieved analogues is also useful as a
non-parametric forecast prior — when the model and the analogues agree, raise
confidence; when they disagree, lower it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .index import AnalogueIndex

logger = logging.getLogger(__name__)


@dataclass
class AnalogueHit:
    date:             str
    similarity:       float
    ref_price:        float
    realised_logret:  float
    vix:              Optional[float] = None
    cot_mm_net_z:     Optional[float] = None
    gld_flow_5d_z:    Optional[float] = None


def retrieve(
    idx: AnalogueIndex,
    query_features: pd.Series,
    *,
    k: int = 10,
    min_date_gap_days: int = 30,
    regime_constraints: Optional[dict] = None,
) -> List[AnalogueHit]:
    """
    Find top-K most-similar historical days.

    `min_date_gap_days` excludes near-identical recent days so we retrieve
    actual analogues, not yesterday's data. `regime_constraints` is a dict
    of {col: (min, max)} to filter by regime band before similarity ranking
    (e.g. {"vix_close": (15.0, 25.0)} for "comparable VIX environment").
    """
    # Build query vector with the same scaler
    q = (query_features.reindex(idx.feature_cols).fillna(0.0)
                       .to_numpy(dtype=np.float32))
    q_n = np.clip((q - idx.med) / idx.mad, -8.0, 8.0).astype(np.float32)
    norm = float(np.linalg.norm(q_n) + 1e-12)
    q_n = q_n / norm

    # Optional regime filter — boolean mask over the index
    mask = np.ones(len(idx.vectors), dtype=bool)
    if regime_constraints:
        for col, (lo, hi) in regime_constraints.items():
            if col not in idx.meta.columns:
                continue
            v = pd.to_numeric(idx.meta[col], errors="coerce").to_numpy()
            mask &= np.isfinite(v) & (v >= lo) & (v <= hi)

    # Recency exclusion
    if min_date_gap_days > 0 and "date" in idx.meta.columns:
        dates = pd.to_datetime(idx.meta["date"])
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(
            days=min_date_gap_days)
        mask &= (dates <= cutoff).to_numpy()

    if not mask.any():
        return []

    # Score
    scores = idx.vectors @ q_n           # cosine via inner product (L2-normed)
    scores = np.where(mask, scores, -np.inf)
    # Top-K
    K = min(k, int(mask.sum()))
    top = np.argpartition(-scores, K - 1)[:K]
    top = top[np.argsort(-scores[top])]

    hits: List[AnalogueHit] = []
    for i in top:
        row = idx.meta.iloc[int(i)]
        hits.append(AnalogueHit(
            date=str(pd.to_datetime(row["date"]).date()),
            similarity=float(scores[i]),
            ref_price=float(row.get("gold_close", 0.0)) if "gold_close" in row else 0.0,
            realised_logret=float(row["target_logret"]),
            vix=_maybe_float(row.get("vix_close")),
            cot_mm_net_z=_maybe_float(row.get("cot_mm_net_z")),
            gld_flow_5d_z=_maybe_float(row.get("gld_flow_5d_z")),
        ))
    return hits


def _maybe_float(x):
    try:
        v = float(x)
        return v if v == v else None     # NaN → None
    except (TypeError, ValueError):
        return None


def analogue_summary(hits: List[AnalogueHit]) -> dict:
    """Distribution stats over retrieved analogues' realised outcomes."""
    if not hits:
        return {"n": 0}
    r = np.array([h.realised_logret for h in hits], dtype=np.float64)
    return {
        "n":          len(hits),
        "mean":       float(r.mean()),
        "median":     float(np.median(r)),
        "std":        float(r.std()),
        "p_up":       float((r > 0).mean()),
        "q10":        float(np.quantile(r, 0.10)),
        "q90":        float(np.quantile(r, 0.90)),
        "avg_sim":    float(np.mean([h.similarity for h in hits])),
    }
