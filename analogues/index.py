"""
analogues.index — build a Faiss / numpy index over historical state vectors.

Each historical trading day becomes one vector in the index. The vector is
the model's *state representation*: a normalised, dimension-reduced feature
snapshot. Retrieval finds the K most-similar historical days; their realised
forward log-returns become the empirical prior for current conditions.

Why this is more useful than the v1 PatternMemory:
  · The vector is the model's actual feature representation, not random embeddings.
  · Each vector has a real labelled outcome (forward log-return) — not 0.0.
  · The index supports "regime-conditioned retrieval": filter by VIX bucket
    or COT-Z bucket so we retrieve only economically-comparable periods.

Storage:
  data/analogues/<horizon>/
      vectors.npy      — [N, D]   normalised feature vectors
      meta.parquet     — N rows: date, ref_price, target_logret_<h>d,
                          vix, cot_mm_net_z, gld_flow_5d_z, regime_bucket
      scaler.npz       — med, mad used to normalise vectors
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path("data") / "analogues"


@dataclass
class AnalogueIndex:
    horizon: str
    feature_cols: List[str]
    vectors: np.ndarray         # [N, D] float32, L2-normalised
    meta: pd.DataFrame
    med: np.ndarray
    mad: np.ndarray
    _faiss: object = field(default=None, repr=False)

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])


# ── build ────────────────────────────────────────────────────────────────────

def build_index(panel: pd.DataFrame, *, horizon: str,
                feature_cols: Sequence[str],
                label_col: str,
                regime_cols: Sequence[str] = ("vix_close",
                                                 "cot_mm_net_z",
                                                 "gld_flow_5d_z"),
                ) -> AnalogueIndex:
    """
    Build an analogue index over the given panel.

    Only rows with a complete label are indexed (no future leakage at query
    time — the label is what we'll *retrieve*, not the query vector).
    """
    df = panel.copy()
    df = df.dropna(subset=[label_col])
    df = df.dropna(subset=list(feature_cols), how="all")

    X = df[list(feature_cols)].fillna(0.0).to_numpy(dtype=np.float32)
    # Median / MAD scaling — same family as the model scaler.
    med = np.median(X, axis=0).astype(np.float32)
    mad = (np.median(np.abs(X - med), axis=0) * 1.4826).astype(np.float32)
    mad = np.where(mad < 1e-8, 1.0, mad).astype(np.float32)

    Xn = np.clip((X - med) / mad, -8.0, 8.0).astype(np.float32)
    # L2-normalise so inner product = cosine similarity
    norms = np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
    Xn = (Xn / norms).astype(np.float32)

    meta_cols = ["date", "gold_close", label_col] + [c for c in regime_cols
                                                       if c in df.columns]
    meta = df[meta_cols].reset_index(drop=True)
    meta = meta.rename(columns={label_col: "target_logret"})

    out_dir = _ROOT / horizon
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vectors.npy", Xn)
    np.savez(out_dir / "scaler.npz", med=med, mad=mad)
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    (out_dir / "feature_cols.json").write_text(json.dumps(list(feature_cols)))

    logger.info("analogue index built: %s (n=%d, d=%d)",
                horizon, Xn.shape[0], Xn.shape[1])
    return AnalogueIndex(
        horizon=horizon, feature_cols=list(feature_cols),
        vectors=Xn, meta=meta, med=med, mad=mad,
        _faiss=_maybe_build_faiss(Xn),
    )


def _maybe_build_faiss(X: np.ndarray):
    try:
        import faiss
        idx = faiss.IndexFlatIP(X.shape[1])
        idx.add(X)
        return idx
    except ImportError:
        return None


# ── load ─────────────────────────────────────────────────────────────────────

def load_index(horizon: str) -> Optional[AnalogueIndex]:
    p = _ROOT / horizon
    if not (p / "vectors.npy").exists():
        return None
    vectors = np.load(p / "vectors.npy")
    scaler  = np.load(p / "scaler.npz")
    meta    = pd.read_parquet(p / "meta.parquet")
    fcols   = json.loads((p / "feature_cols.json").read_text())
    return AnalogueIndex(
        horizon=horizon,
        feature_cols=fcols,
        vectors=vectors,
        meta=meta,
        med=scaler["med"], mad=scaler["mad"],
        _faiss=_maybe_build_faiss(vectors),
    )
