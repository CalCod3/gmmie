"""
research.registry — active model pointer + promotion logic.

A trained model lives under `data/models/hybrid_<horizon>_<ts>/`. The registry
maintains, for each horizon, a pointer to the *active* checkpoint: the one
inference (predict.py) and the live engine should load.

Promotion rule: a new checkpoint is promoted iff
    new.sharpe_lo > 0  AND  new.sharpe_med > incumbent.sharpe_med + δ
where δ = 0.05 (5-percent improvement minimum to overcome backtest noise).

If no incumbent exists and `new.sharpe_lo > -0.25` (any non-disastrous result),
the new checkpoint is promoted as a cold-start default.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MODELS_ROOT = Path("data") / "models"
_REGISTRY_FILE = _MODELS_ROOT / "registry.json"


def _load() -> dict:
    if not _REGISTRY_FILE.exists():
        return {}
    try:
        return json.loads(_REGISTRY_FILE.read_text())
    except Exception:
        return {}


def _save(reg: dict) -> None:
    _MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    _REGISTRY_FILE.write_text(json.dumps(reg, indent=2))


def active_path(horizon: str) -> Optional[Path]:
    """Return the absolute path of the active checkpoint for a given horizon."""
    reg = _load()
    entry = reg.get(horizon)
    if not entry:
        return None
    p = Path(entry["path"])
    return p if p.exists() else None


def active_meta(horizon: str) -> Optional[dict]:
    reg = _load()
    return reg.get(horizon)


def maybe_promote(checkpoint_dir: Path, *, horizon: str,
                  metrics: Optional[dict],
                  improvement_threshold: float = 0.05) -> bool:
    """
    Decide whether `checkpoint_dir` should become the new active model
    for `horizon`. Returns True if promoted.
    """
    reg = _load()
    incumbent = reg.get(horizon)

    if metrics is None:
        # No backtest — only promote if we have no incumbent at all
        if incumbent is None:
            _write_entry(reg, horizon, checkpoint_dir, metrics=None,
                         reason="cold_start_no_backtest")
            return True
        return False

    # Promotion uses the CONSERVATIVE Sharpe: Newey-West HAC (overlap-corrected)
    # or non-overlapping when available. Falls back to bootstrap median if
    # neither is set (cold start before the v3 backtester landed).
    def _conservative(m: dict) -> float:
        for k in ("sharpe_nonoverlap", "sharpe_nw", "sharpe_med"):
            v = m.get(k)
            if v is not None:
                return float(v)
        return -float("inf")

    new_sh_lo  = metrics.get("sharpe_lo")  or -float("inf")
    new_cons   = _conservative(metrics)
    new_hit    = metrics.get("overall_hit") or 0.0

    # Cold start
    if incumbent is None or not incumbent.get("metrics"):
        if new_sh_lo > -0.25 and new_cons > -0.10:
            _write_entry(reg, horizon, checkpoint_dir, metrics=metrics,
                         reason=f"cold_start cons={new_cons:.2f} lo={new_sh_lo:.2f}")
            return True
        logger.info("cold-start REJECTED: cons=%.2f lo=%.2f", new_cons, new_sh_lo)
        return False

    inc_cons = _conservative(incumbent["metrics"])
    if new_sh_lo > 0 and new_cons > inc_cons + improvement_threshold:
        _write_entry(
            reg, horizon, checkpoint_dir, metrics=metrics,
            reason=f"cons {new_cons:.2f}>{inc_cons:.2f}+δ hit={new_hit:.3f}",
        )
        return True
    logger.info(
        "NOT promoted: cons %.2f vs incumbent %.2f (δ=%.2f) | lo=%.2f",
        new_cons, inc_cons, improvement_threshold, new_sh_lo,
    )
    return False


def _write_entry(reg: dict, horizon: str, ckpt: Path,
                 metrics: Optional[dict], reason: str) -> None:
    reg[horizon] = {
        "path":      str(ckpt.resolve()),
        "metrics":   metrics,
        "reason":    reason,
        "promoted_at_utc": __import__("datetime").datetime.utcnow().isoformat(),
    }
    _save(reg)
    logger.info("registry[%s] ← %s (%s)", horizon, ckpt.name, reason)


def show() -> dict:
    return _load()
