"""
research.serve — load + cache the active trained checkpoint for live use.

The runtime engine should consume the offline model rather than its old
random-projection forecast head. This module is the integration seam.

Two layers:
  · `ModelServer` holds a loaded checkpoint plus its scaler & winsorizer.
  · `forecast_from_panel_row(server, row)` produces the same μ/σ/quantile
    payload that the live engine's IntelligencePacket would expose, given a
    single feature row (as produced by `research.features.build_panel`).

Thread-safe: forward() is wrapped in torch.no_grad(); the model is held in
eval mode after load.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .predict import load_active

logger = logging.getLogger(__name__)


@dataclass
class ServingPrediction:
    mu:        float
    sigma:     float
    lambda_:   float
    quantiles: Dict[str, float]
    conformal_scale: float


class ModelServer:
    """One server per horizon. Lazy-loads on first use, then reuses."""

    _lock = threading.RLock()
    _cache: Dict[str, "ModelServer"] = {}

    @classmethod
    def get(cls, horizon: str) -> Optional["ModelServer"]:
        with cls._lock:
            srv = cls._cache.get(horizon)
            if srv is not None:
                return srv
            loaded = load_active(horizon)
            if not loaded:
                return None
            srv = ModelServer(horizon, *loaded)
            cls._cache[horizon] = srv
            return srv

    @classmethod
    def reload(cls, horizon: Optional[str] = None) -> None:
        """Drop cached models. Next get() reloads from disk."""
        with cls._lock:
            if horizon is None:
                cls._cache.clear()
            else:
                cls._cache.pop(horizon, None)

    def __init__(self, horizon, model, cfg, feature_cols, scaler_bundle, meta):
        self.horizon = horizon
        self.model = model
        self.cfg = cfg
        self.feature_cols: List[str] = list(feature_cols)
        self.wlo, self.whi, self.med, self.mad = scaler_bundle
        self.meta = meta
        logger.info(
            "ModelServer ready: horizon=%s features=%d lake_hash=%s",
            horizon, len(self.feature_cols),
            meta.get("lake_state_hash", "?"),
        )

    @torch.no_grad()
    def forecast(self, feature_row) -> Optional[ServingPrediction]:
        """
        feature_row: pandas Series indexed by feature name. Missing features
        default to 0.0 (≈ "no signal" after standardisation).
        """
        try:
            x = (feature_row.reindex(self.feature_cols).fillna(0.0)
                            .to_numpy(dtype=np.float32))
        except Exception as exc:
            logger.error("forecast row→numpy failed: %s", exc)
            return None
        # Apply train-time winsorizer then scaler (point-in-time correct)
        x = np.clip(x, self.wlo, self.whi).astype(np.float32)
        x = np.clip((x - self.med) / self.mad, -8.0, 8.0).astype(np.float32)
        xs = torch.as_tensor(x[: self.cfg.n_struct], dtype=torch.float32).unsqueeze(0)
        xf = torch.as_tensor(x,                       dtype=torch.float32).unsqueeze(0)
        out = self.model(xs, xf)
        q   = out["quantiles"].numpy().reshape(-1)
        return ServingPrediction(
            mu=float(out["mu"].item()),
            sigma=float(out["sigma"].item()),
            lambda_=float(out["lambda"].item()),
            quantiles={f"q{int(p * 100):02d}": float(q[i])
                        for i, p in enumerate(self.model.QUANTILES)},
            conformal_scale=float(self.model.conformal_scale.item()),
        )
