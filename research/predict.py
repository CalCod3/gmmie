"""
research.predict — point-in-time inference + outcome settlement.

Two responsibilities:

1. **Predict**: load the active checkpoint for a horizon, build the feature
   panel up to `--asof`, and emit a single prediction into the `predictions`
   table. Used by the strategist and (eventually) the live engine.

2. **Settle**: scan `predictions` for rows whose horizon has elapsed but have
   no matching `outcomes` row, compute realised log-return + pinball loss,
   insert. This is the disciplinary loop — without it the registry/meta
   layers cannot improve.

Idempotent: re-running settle on the same lake is a no-op past the first run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from data_lake.db import LakeDB
from . import registry
from .features import build_panel
from .model import HybridGoldForecaster, ModelConfig

logger = logging.getLogger(__name__)


# ── Predict ───────────────────────────────────────────────────────────────────

def load_active(horizon: str) -> Optional[tuple]:
    """Return (model, cfg, feature_cols, med, mad, meta) for active checkpoint."""
    path = registry.active_path(horizon)
    if path is None:
        logger.error("no active model for horizon=%s", horizon)
        return None

    cfg_d = json.loads((path / "config.json").read_text())
    cfg   = ModelConfig(**cfg_d)
    feature_cols = json.loads((path / "feature_cols.json").read_text())
    meta = json.loads((path / "meta.json").read_text())

    scaler = np.load(path / "scaler.npz")
    med = scaler["med"]
    mad = scaler["mad"]

    # Winsorizer is optional for backward compatibility with v2 checkpoints.
    wpath = path / "winsorizer.npz"
    if wpath.exists():
        w = np.load(wpath)
        wlo, whi = w["lo"], w["hi"]
    else:
        # No winsorizer in checkpoint — degrade safely to ±10σ clip via scaler
        wlo = np.full_like(med, -np.inf)
        whi = np.full_like(med,  np.inf)

    model = HybridGoldForecaster(cfg)
    state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    # Compatibility check: the structural-head weight shape must match cfg.n_struct
    expected_n_struct = state.get("structural.w", torch.empty(cfg.n_struct)).shape[0]
    if expected_n_struct != cfg.n_struct:
        logger.warning(
            "structural-head dim mismatch: state has %d, cfg says %d — using state",
            expected_n_struct, cfg.n_struct,
        )
    model.load_state_dict(state)
    model.eval()

    # Bundle winsorizer alongside scaler for callers
    return model, cfg, feature_cols, (wlo, whi, med, mad), meta


def predict_one(db: LakeDB, horizon: str, asof: dt.date,
                *, model_version: Optional[str] = None,
                dry_run: bool = False) -> Optional[str]:
    """
    Emit a single prediction for `asof`. Returns the prediction id, or None.

    If `dry_run=True`, the prediction is computed but NOT inserted into the
    predictions table — used by the strategist's dossier-time peek so we
    don't pollute the audit log with non-committed thinking forecasts.
    """
    loaded = load_active(horizon)
    if not loaded:
        return None
    model, cfg, feature_cols, (wlo, whi, med, mad), meta = loaded

    panel = build_panel(db, start=(asof - dt.timedelta(days=365 * 3)).isoformat())
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    asof_ts = pd.Timestamp(asof)

    # Most-recent row ≤ asof — strictly point-in-time correct
    row = panel.loc[panel.index <= asof_ts].tail(1)
    if row.empty:
        logger.error("no panel row available for asof=%s", asof)
        return None

    X = row[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    # Apply the train-time winsoriser then the train-time scaler
    X = np.clip(X, wlo, whi).astype(np.float32)
    Xs = np.clip((X - med) / mad, -8.0, 8.0).astype(np.float32)
    ref_price = float(row["gold_close"].iloc[0])

    with torch.no_grad():
        out = model(
            torch.as_tensor(Xs[:, :cfg.n_struct]),
            torch.as_tensor(Xs),
        )
    mu = float(out["mu"].item())
    q  = out["quantiles"].numpy().reshape(-1)            # [q10, q25, q50, q75, q90]
    log_var = float(out["log_var"].item())
    confidence = float(np.exp(-0.5 * log_var))
    confidence = float(np.clip(confidence, 0.05, 0.99))

    pred_id  = uuid.uuid4().hex
    ver      = model_version or f"{horizon}@{Path(registry.active_path(horizon)).name}"
    context = {
        "mu": mu, "log_var": log_var,
        "lambda": float(out["lambda"].item()) if "lambda" in out else None,
        "sigma":  float(out["sigma"].item())  if "sigma"  in out else None,
        "conformal_scale": float(model.conformal_scale.item()),
    }

    if dry_run:
        logger.info(
            "[dry_run] prediction %s @ %s | ref=%.2f mu=%+.4f q50=%+.4f",
            pred_id[:8], asof, ref_price, mu, float(q[2]),
        )
        return pred_id

    db.upsert(
        "predictions",
        ["id", "ts", "model_version", "horizon_d", "ref_price",
         "q10", "q25", "q50", "q75", "q90",
         "regime", "confidence", "context_json"],
        [(pred_id, dt.datetime.combine(asof, dt.time(16, 0)),  # NY close-ish
          ver, int(meta["horizon_d"]), ref_price,
          float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]),
          None, confidence, json.dumps(context))],
        conflict_key=("id",),
    )
    logger.info(
        "prediction %s @ %s | ref=%.2f mu=%+.4f q50=%+.4f conf=%.2f",
        pred_id[:8], asof, ref_price, mu, float(q[2]), confidence,
    )
    return pred_id


# ── Settle ────────────────────────────────────────────────────────────────────

def _pinball_q50(q50_logret: float, realised_logret: float) -> float:
    err = realised_logret - q50_logret
    return float(max(0.5 * err, -0.5 * err))


def settle_outcomes(db: LakeDB, *, today: Optional[dt.date] = None) -> int:
    """For every prediction whose horizon has elapsed and no outcome exists,
    compute the realised outcome and write it to `outcomes`.
    """
    today = today or dt.date.today()
    # Pending: predictions with no outcome row AND horizon elapsed
    pending = db.df(
        """
        SELECT p.id, p.ts, p.horizon_d, p.ref_price, p.q50
        FROM predictions p
        LEFT JOIN outcomes o ON o.prediction_id = p.id
        WHERE o.prediction_id IS NULL
          AND CAST(p.ts AS DATE) + INTERVAL (p.horizon_d) DAY <= ?
        """,
        [today],
    )
    if pending.empty:
        logger.info("no predictions to settle")
        return 0

    settled = 0
    rows = []
    for _, p in pending.iterrows():
        pred_id = p["id"]
        settle_date = (pd.Timestamp(p["ts"]).date()
                        + dt.timedelta(days=int(p["horizon_d"])))
        realised = db.scalar(
            "SELECT close FROM prices_d WHERE symbol='GC=F' AND date>=? "
            "ORDER BY date ASC LIMIT 1",
            [settle_date],
        )
        if realised is None or p["ref_price"] in (None, 0):
            continue
        realised = float(realised)
        ref = float(p["ref_price"])
        if ref <= 0 or realised <= 0:
            continue
        log_ret = float(np.log(realised / ref))
        pin = _pinball_q50(float(p["q50"]), log_ret)
        rows.append((
            pred_id,
            dt.datetime.combine(settle_date, dt.time(16, 0)),
            realised, log_ret, pin,
        ))
        settled += 1

    if rows:
        db.upsert(
            "outcomes",
            ["prediction_id", "settled_at", "realised_price",
             "realised_logret", "pinball_loss"],
            rows,
            conflict_key=("prediction_id",),
        )
    logger.info("settled %d outcomes", settled)
    return settled


def recent_skill(db: LakeDB, horizon: str, *, lookback_days: int = 30) -> dict:
    """Recent forecaster skill: hit rate + mean pinball + count."""
    df = db.df(
        """
        SELECT p.q50 AS q50, o.realised_logret AS y
        FROM predictions p JOIN outcomes o ON o.prediction_id = p.id
        WHERE p.horizon_d = (SELECT horizon_d FROM predictions
                              WHERE model_version LIKE ? LIMIT 1)
          AND o.settled_at >= ?
        """,
        [f"{horizon}@%", dt.datetime.utcnow() - dt.timedelta(days=lookback_days)],
    )
    if df.empty:
        return {"n": 0}
    hit = float(np.mean(np.sign(df["q50"]) == np.sign(df["y"])))
    pin = float(np.mean(np.maximum(0.5 * (df["y"] - df["q50"]),
                                    -0.5 * (df["y"] - df["q50"]))))
    return {
        "n": int(len(df)),
        "hit_rate": round(hit, 3),
        "mean_pinball_q50": round(pin, 5),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    p = argparse.ArgumentParser(description="Inference + outcome settlement")
    p.add_argument("--horizon", default="20d")
    p.add_argument("--asof", default=None,
                   help="YYYY-MM-DD; default = today (US market days only)")
    p.add_argument("--predict", action="store_true",
                   help="emit a prediction for --asof")
    p.add_argument("--settle", action="store_true",
                   help="settle any predictions whose horizon has elapsed")
    p.add_argument("--skill", action="store_true",
                   help="print recent skill stats")
    args = p.parse_args(argv)

    if not (args.predict or args.settle or args.skill):
        p.print_help()
        return 1

    asof = dt.date.fromisoformat(args.asof) if args.asof else dt.date.today()
    with LakeDB() as db:
        if args.predict:
            predict_one(db, args.horizon, asof)
        if args.settle:
            settle_outcomes(db, today=asof)
        if args.skill:
            print(json.dumps(recent_skill(db, args.horizon), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
