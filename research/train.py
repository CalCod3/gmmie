"""
research.train — train HybridGoldForecaster on the data lake.

Usage:
    python -m research.train --horizon 20d --epochs 200 --device cpu

Output:
    data/models/hybrid_<horizon>_<timestamp>/
        weights.pt
        config.json
        train_log.json
        feature_cols.json
        scaler.npz
        backtest_report.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_lake.db import LakeDB
from .backtest import run as run_backtest
from .features import build_panel
from .model import (
    STRUCT_FEATURES,
    HybridGoldForecaster,
    ModelConfig,
    composite_loss,
)

logger = logging.getLogger(__name__)


# ── Feature plumbing ─────────────────────────────────────────────────────────

def _select_feature_cols(panel: pd.DataFrame, n_full: int) -> List[str]:
    """
    Pick the top-`n_full` most-populated numeric columns, excluding label
    columns and the raw gold close. Structural columns are guaranteed
    inclusion if present.
    """
    forbidden = {c for c in panel.columns if c.startswith("target_")}
    forbidden |= {"gold_close"}

    candidates = [c for c in panel.columns
                  if c not in forbidden and panel[c].dtype.kind in "fi"]
    coverage = panel[candidates].notna().mean().sort_values(ascending=False)

    # Always include structural features first when present
    forced = [c for c in STRUCT_FEATURES if c in candidates]
    rest = [c for c in coverage.index if c not in forced]

    selected = forced + rest
    return selected[:max(n_full, len(forced))]


def _to_xy(panel: pd.DataFrame, feature_cols: List[str], label_col: str
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = panel.dropna(subset=[label_col]).copy()
    df[feature_cols] = df[feature_cols].fillna(0.0)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    dates = df["date"].to_numpy() if "date" in df.columns else df.index.to_numpy()
    return X, y, dates


def fit_winsorizer(X_train: np.ndarray, *,
                   lo_q: float = 0.005, hi_q: float = 0.995
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature winsorisation bounds fit on the TRAIN portion only."""
    lo = np.quantile(X_train, lo_q, axis=0).astype(np.float32)
    hi = np.quantile(X_train, hi_q, axis=0).astype(np.float32)
    # Guard against degenerate columns (constant features)
    same = hi <= lo
    if same.any():
        hi = np.where(same, lo + 1e-6, hi)
    return lo, hi


def apply_winsorizer(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(X, lo, hi).astype(np.float32)


# ── Training loop ────────────────────────────────────────────────────────────

def train_one(
    X_raw: np.ndarray, y: np.ndarray,
    *, cfg: ModelConfig, n_struct: int,
    epochs: int, batch_size: int, lr: float, weight_decay: float,
    val_frac: float, device: str, seed: int,
    early_stop_patience: int = 25,
) -> Tuple[HybridGoldForecaster, dict, Tuple[np.ndarray, np.ndarray]]:
    """
    Train a single HybridGoldForecaster on (X_raw, y).

    Critical: the robust median/MAD scaler is fit on the TRAIN PORTION ONLY
    (no lookahead into validation). Returns the fitted scaler so callers can
    apply the same transform at inference time.

    Returns (model, log, (med, mad)).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(X_raw)
    split = max(1, int(n * (1 - val_frac)))
    X_tr_raw, X_va_raw = X_raw[:split], X_raw[split:]
    y_tr,     y_va     = y[:split],     y[split:]

    # Winsorise on TRAIN portion only — no lookahead from val/test into the
    # clip bounds. Apply the same bounds to both partitions.
    wlo, whi = fit_winsorizer(X_tr_raw)
    X_tr_raw = apply_winsorizer(X_tr_raw, wlo, whi)
    X_va_raw = apply_winsorizer(X_va_raw, wlo, whi)

    # Fit scaler on train portion ONLY — eliminates val/test leakage
    med = np.median(X_tr_raw, axis=0)
    mad = np.median(np.abs(X_tr_raw - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-8, 1.0, mad).astype(np.float32)
    med = med.astype(np.float32)

    def _scale(X):
        return np.clip((X - med) / mad, -8.0, 8.0).astype(np.float32)

    X_tr = _scale(X_tr_raw)
    X_va = _scale(X_va_raw)

    Xs_tr = torch.as_tensor(X_tr[:, :n_struct], dtype=torch.float32, device=device)
    Xf_tr = torch.as_tensor(X_tr,               dtype=torch.float32, device=device)
    yt_tr = torch.as_tensor(y_tr,               dtype=torch.float32, device=device)
    Xs_va = torch.as_tensor(X_va[:, :n_struct], dtype=torch.float32, device=device)
    Xf_va = torch.as_tensor(X_va,               dtype=torch.float32, device=device)
    yt_va = torch.as_tensor(y_va,               dtype=torch.float32, device=device)

    model = HybridGoldForecaster(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    best_epoch = 0
    best_state: Dict = {}
    log: Dict = {"train_loss": [], "val_loss": [], "lambda": [],
                 "best_epoch": 0, "best_val": None,
                 "scaler": {"median": med.tolist(), "mad": mad.tolist()}}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xf_tr), device=device)
        losses = []
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            out = model(Xs_tr[idx], Xf_tr[idx])
            loss = composite_loss(out, yt_tr[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        model.eval()
        with torch.no_grad():
            out_v = model(Xs_va, Xf_va)
            val = composite_loss(out_v, yt_va).item()
        log["train_loss"].append(float(np.mean(losses)))
        log["val_loss"].append(float(val))
        log["lambda"].append(float(out_v["lambda"].item()))

        if val < best_val - 1e-6:
            best_val = val
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= early_stop_patience:
            logger.info("early stop @ epoch %d (best %d)", epoch, best_epoch)
            break

        if epoch % 20 == 0 or epoch == epochs - 1:
            logger.info(
                "epoch %3d train=%.5f val=%.5f λ=%.3f (best=%.5f@%d)",
                epoch, np.mean(losses), val, log["lambda"][-1],
                best_val, best_epoch,
            )

    if best_state:
        model.load_state_dict(best_state)
    log["best_epoch"] = best_epoch
    log["best_val"]   = best_val

    # ── Conformal calibration ──
    # Use the held-out validation tensor as the calibration set. After this
    # step the model's quantile bands carry a marginal coverage guarantee.
    if len(Xf_va) >= 30:
        scale = model.fit_conformal_scale(Xs_va, Xf_va, yt_va,
                                          target_coverage=0.80)
        log["conformal_scale"] = float(scale)
        logger.info("conformal_scale fitted: %.3f (target 80%% cov)", scale)
    return model, log, (med, mad)


# ── Backtest adapter ─────────────────────────────────────────────────────────

def _make_predict_fns(cfg: ModelConfig, feature_cols: List[str],
                      epochs: int, batch_size: int, lr: float, wd: float,
                      device: str, seed: int):
    """
    Return fit/predict closures expected by backtest.run().

    Each fold trains a fresh model — scaler is fit inside train_one() on the
    fold's training set, so each fold is fully purged.
    """

    def _train_fold(X_tr_df, y_tr_s):
        X_tr_raw = X_tr_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        y_tr     = y_tr_s.to_numpy(dtype=np.float32).reshape(-1, 1)
        return train_one(
            X_tr_raw, y_tr, cfg=cfg, n_struct=cfg.n_struct,
            epochs=max(40, epochs // 4),
            batch_size=batch_size, lr=lr, weight_decay=wd,
            val_frac=0.1, device=device, seed=seed,
        )

    def _apply(model, X_te_df, med, mad):
        X_te_raw = X_te_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        X_te = np.clip((X_te_raw - med) / mad, -8.0, 8.0).astype(np.float32)
        model.eval()
        with torch.no_grad():
            return model(
                torch.as_tensor(X_te[:, :cfg.n_struct], device=device),
                torch.as_tensor(X_te, device=device),
            )

    def fit_predict(X_tr_df, y_tr_s, X_te_df):
        model, _, (med, mad) = _train_fold(X_tr_df, y_tr_s)
        out_te = _apply(model, X_te_df, med, mad)
        out_tr = _apply(model, X_tr_df, med, mad)
        return (out_te["mu"].cpu().numpy().reshape(-1),
                out_tr["mu"].cpu().numpy().reshape(-1))

    def quantile_fit_predict(X_tr_df, y_tr_s, X_te_df):
        model, _, (med, mad) = _train_fold(X_tr_df, y_tr_s)
        out_te = _apply(model, X_te_df, med, mad)
        out_tr = _apply(model, X_tr_df, med, mad)
        return (out_te["quantiles"].cpu().numpy(),
                out_tr["quantiles"].cpu().numpy())

    return fit_predict, quantile_fit_predict


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    p = argparse.ArgumentParser(description="Train hybrid gold forecaster")
    p.add_argument("--horizon", default="20d",
                   help="label horizon: 5d | 20d | 60d")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-full", type=int, default=32)
    p.add_argument("--backtest", action="store_true",
                   help="run walk-forward backtest after training")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", default="2010-01-01")
    args = p.parse_args(argv)

    from .seed import seed_everywhere
    seed_everywhere(args.seed)

    label_col = f"target_logret_{args.horizon}"
    if not label_col.endswith(("5d", "20d", "60d")):
        logger.error("invalid --horizon %s; expected 5d/20d/60d", args.horizon)
        return 2

    # 1) Build panel + capture lake provenance
    from .features import _lake_state_hash
    with LakeDB(read_only=True) as db:
        lake_hash = _lake_state_hash(db, "GC=F", args.start)
        panel = build_panel(db, start=args.start)
    if "date" not in panel.columns:
        panel = panel.reset_index().rename(columns={"index": "date"})
    panel = panel.sort_values("date").reset_index(drop=True)
    logger.info("panel: %d rows × %d cols (lake_hash=%s)",
                panel.shape[0], panel.shape[1], lake_hash)

    # 2) Pick features
    feature_cols = _select_feature_cols(panel, args.n_full)
    logger.info("features (%d): %s", len(feature_cols), feature_cols)

    cfg = ModelConfig(n_struct=sum(1 for c in STRUCT_FEATURES if c in feature_cols),
                      n_full=len(feature_cols))

    X, y, _ = _to_xy(panel, feature_cols, label_col)
    if len(X) < 1000:
        logger.error("not enough data (%d rows) — build the lake first", len(X))
        return 3

    # 3) Train. train_one() fits the scaler on its OWN train portion only — no
    # lookahead into the val split.
    model, log, (med, mad) = train_one(
        X, y, cfg=cfg, n_struct=cfg.n_struct,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        val_frac=args.val_frac, device=args.device, seed=args.seed,
    )

    # 4) Save artefacts
    horizon_int = int(args.horizon.rstrip("d"))
    out_dir = Path("data") / "models" / (
        f"hybrid_{args.horizon}_{dt.datetime.utcnow():%Y%m%dT%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "weights.pt")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / "train_log.json").write_text(json.dumps(log))
    (out_dir / "feature_cols.json").write_text(json.dumps(feature_cols))
    (out_dir / "meta.json").write_text(json.dumps({
        "horizon_d":       horizon_int,
        "label_col":       label_col,
        "n_train_rows":    int(len(X)),
        "trained_at_utc":  dt.datetime.utcnow().isoformat(),
        "seed":            args.seed,
        "lake_state_hash": lake_hash,
        "arch_version":    "hybrid-v3",
        "panel_start":     args.start,
        "winsorizer":      {"lo": None, "hi": None},   # filled below for live serving
    }, indent=2))
    # Also persist the winsorizer for safe live inference
    wlo, whi = fit_winsorizer(X[: int(len(X) * (1 - args.val_frac))])
    np.savez(out_dir / "winsorizer.npz", lo=wlo, hi=whi)
    np.savez(out_dir / "scaler.npz", med=med, mad=mad)
    logger.info("saved model → %s", out_dir)

    # 4a) Build analogue index over training data — used by strategist + RAG
    try:
        from analogues.index import build_index
        build_index(
            panel.head(len(panel) - 60),       # train portion only
            horizon=args.horizon,
            feature_cols=feature_cols,
            label_col=label_col,
        )
    except Exception as exc:
        logger.warning("analogue index build failed: %s", exc)

    # 4b) Diagnostics on the held-out tail of the training data
    try:
        from .diagnostics import write_diagnostics
        val_n = max(int(len(X) * args.val_frac), 50)
        X_diag = X[-val_n:]
        y_diag = y[-val_n:].reshape(-1)
        X_diag_s = np.clip((X_diag - med) / mad, -8.0, 8.0).astype(np.float32)
        # Recent panel = last 60 trading days for drift comparison
        recent_panel = panel.tail(60)
        train_panel  = panel.head(max(252, len(panel) - 60))
        write_diagnostics(
            out_dir, model, X_diag_s, y_diag,
            feature_cols=feature_cols, n_struct=cfg.n_struct,
            train_panel=train_panel, recent_panel=recent_panel,
        )
    except Exception as exc:
        logger.warning("diagnostics write failed: %s", exc)

    # 5) Optional: backtest
    backtest_metrics = None
    if args.backtest:
        fit_pred, qfit_pred = _make_predict_fns(
            cfg, feature_cols,
            args.epochs, args.batch_size, args.lr, args.weight_decay,
            args.device, args.seed,
        )
        report = run_backtest(
            panel,
            feature_cols=feature_cols,
            label_col=label_col,
            fit_predict_fn=fit_pred,
            quantile_fit_predict_fn=qfit_pred,
            label_horizon_days=horizon_int,
        )
        backtest_metrics = {
            "overall_hit":       report.overall_hit,
            "overall_pin":       report.overall_pin,
            "sharpe_med":        report.sharpe_med,
            "sharpe_lo":         report.sharpe_lo,
            "sharpe_hi":         report.sharpe_hi,
            "sharpe_nw":         report.sharpe_nw,
            "sharpe_nonoverlap": report.sharpe_nonoverlap,
            "coverage_80":       report.coverage_80,
            "cost_bps":          report.cost_bps,
            "threshold":         report.threshold,
            "horizon_d":         report.horizon_d,
            "by_regime":         report.by_regime,
            "n_folds":           len(report.folds),
        }
        (out_dir / "backtest_report.json").write_text(
            json.dumps(backtest_metrics, indent=2))
        logger.info("backtest report → %s", out_dir / "backtest_report.json")

    # 6) Registry promotion — only mark as active if backtest beats incumbent
    try:
        from .registry import maybe_promote
        promoted = maybe_promote(out_dir, horizon=args.horizon,
                                 metrics=backtest_metrics)
        if promoted:
            logger.info("PROMOTED to active model for horizon %s", args.horizon)
        else:
            logger.info("not promoted (existing active model is stronger)")
    except Exception as exc:
        logger.warning("registry promotion failed: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
