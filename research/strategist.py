"""
research.strategist — daily LLM thesis generator.

At NY close (or any cron point), this prints a structured macro thesis grounded
in:
  · current world state from the trained model
  · the day's top news items + their LLM-extracted features
  · top-k historical-analogue periods retrieved by feature similarity
  · 30-day rolling prediction performance vs realised

This is the unfair-advantage layer: a daily reasoning step that synthesises
quantitative signals into a falsifiable narrative. Logs to `theses` table for
Brier-scoring over time.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional

from data_lake.db import LakeDB

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an experienced gold-market macro analyst at a small
hedge fund. You write one daily thesis. Your output is a SHORT structured JSON
object — nothing else — with these fields:

{
  "thesis":           string (2-4 sentences, plain English, the core view),
  "direction":        one of ["long", "short", "flat"],
  "horizon_days":     integer 1..90,
  "confidence":       float in [0, 1],
  "primary_driver":   short string (e.g. "real_yields", "dxy_break", "geopolitical"),
  "key_evidence":     list of 3-5 short strings,
  "falsifiable":      list of 2-3 short strings starting with "I am wrong if…",
  "risks":            list of 1-3 short strings,
  "trade_idea":       string (entry, stop, target — or "stand aside")
}

Rules:
- Always be willing to say "flat" / "stand aside" if signals are ambiguous.
- Falsifiable claims must be observable within `horizon_days` and measurable.
- Cite specific numbers from the dossier (TIPS yield, DXY, COT %).
- No hedging language unless it adds information.
"""


@dataclass
class Dossier:
    asof:           dt.date
    spot:           Optional[float]
    tips_10y:       Optional[float]
    dxy_close:      Optional[float]
    vix_close:      Optional[float]
    cot_mm_net_z:   Optional[float]
    gld_flow_5d_z:  Optional[float]
    regime:         Optional[str]
    model_q:        Dict[str, float]            # q10..q90 over horizon
    model_lambda:   Optional[float]
    top_news:       List[Dict]                  # [{title, gold_impact, confidence, event_type}]
    analogues:      List[Dict]                  # [{date, similarity, outcome}]
    recent_skill:   Dict[str, float]            # {hit, brier, calibration}

    def to_text(self) -> str:
        lines = [
            f"DATE: {self.asof.isoformat()}",
            f"SPOT: {self.spot}",
            f"TIPS_10Y: {self.tips_10y} | DXY: {self.dxy_close} | VIX: {self.vix_close}",
            f"COT_MM_NET_Z: {self.cot_mm_net_z} | GLD_FLOW_5D_Z: {self.gld_flow_5d_z}",
            f"REGIME: {self.regime}",
            f"MODEL_QUANTILES (h-day log-ret): {self.model_q}",
            f"MODEL_LAMBDA (residual trust): {self.model_lambda}",
            "",
            "TOP_NEWS_TODAY:",
        ]
        for n in self.top_news[:8]:
            lines.append(
                f"  - [{n.get('event_type','?')}] impact={n.get('gold_impact'):+.2f} "
                f"conf={n.get('confidence'):.2f}: {n.get('title','')[:140]}"
            )
        lines.append("")
        lines.append("HISTORICAL_ANALOGUES (top-k similar regimes):")
        for a in self.analogues[:5]:
            lines.append(
                f"  - {a.get('date')} sim={a.get('similarity'):.2f} → "
                f"realised log-ret over horizon = {a.get('outcome'):+.4f}"
            )
        lines.append("")
        lines.append(f"RECENT_MODEL_SKILL (last 30d): {self.recent_skill}")
        return "\n".join(lines)


# ── Dossier assembly ─────────────────────────────────────────────────────────

def build_dossier(db: LakeDB, asof: dt.date,
                  horizon: str = "20d") -> Dossier:
    """
    Build a decision dossier for `asof`.

    The feature panel is constructed exactly once and re-used by the model
    snapshot, the positioning signals, and the analogue retrieval — eliminating
    the v2 triple-build cost (3 × panel-cache reads + 3 × index scans).
    """
    horizon_days = int(horizon.rstrip("d"))

    # ── Build the panel ONCE and share ──
    from .features import build_panel
    import pandas as pd
    panel = build_panel(db,
                        start=(asof - dt.timedelta(days=365 * 3)).isoformat())
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    today_row = panel.loc[panel.index <= pd.Timestamp(asof)].tail(1)

    # Market snapshot from the panel (no extra SQL)
    spot = _f(today_row["gold_close"].iloc[0]) if "gold_close" in today_row.columns and not today_row.empty else None
    dxy  = _f(today_row["dxy_close"].iloc[0])  if "dxy_close"  in today_row.columns and not today_row.empty else None
    vix  = _f(today_row["vix_close"].iloc[0])  if "vix_close"  in today_row.columns and not today_row.empty else None
    tips = _f(today_row["tips_10y"].iloc[0])   if "tips_10y"   in today_row.columns and not today_row.empty else None

    # Positioning signals straight from the panel — already rolling-z'd
    cot_mm_z   = (_f(today_row["cot_mm_net_z"].iloc[0])
                   if "cot_mm_net_z" in today_row.columns and not today_row.empty else None)
    gld_flow_z = (_f(today_row["gld_flow_5d_z"].iloc[0])
                   if "gld_flow_5d_z" in today_row.columns and not today_row.empty else None)

    # News for the day — single SQL JOIN
    top_news = db.df(
        """
        SELECT n.title, f.gold_impact, f.confidence, f.event_type,
               f.surprise_signed, f.rationale
        FROM news n JOIN news_features f ON n.id = f.news_id
        WHERE CAST(n.ts AS DATE) = ?
        ORDER BY f.confidence * ABS(f.gold_impact) DESC NULLS LAST
        LIMIT 8
        """,
        [asof],
    ).to_dict("records")

    # Model forecast — share `today_row` instead of re-building the panel
    model_q, model_lambda, regime = _model_snapshot_from_row(horizon, today_row)

    # Recent forecaster skill (point-in-time correct: last 30 settled days)
    from .predict import recent_skill
    skill = recent_skill(db, horizon, lookback_days=30)

    # Analogue retrieval — share `today_row`, no extra panel build
    analogues_list = _retrieve_analogues_from_row(horizon, today_row, vix=vix)

    return Dossier(
        asof=asof,
        spot=spot, tips_10y=tips, dxy_close=dxy, vix_close=vix,
        cot_mm_net_z=cot_mm_z,
        gld_flow_5d_z=gld_flow_z,
        regime=regime,
        model_q=model_q,
        model_lambda=model_lambda,
        top_news=top_news,
        analogues=analogues_list,
        recent_skill=skill,
    )


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _retrieve_analogues(db, asof, horizon: str, vix) -> list:
    """Query analogue index for top-K historical comparables."""
    try:
        from analogues import load_index, retrieve, analogue_summary
        from .features import build_panel
    except Exception as exc:
        logger.debug("analogues unavailable: %s", exc)
        return []

    idx = load_index(horizon)
    if idx is None:
        return []
    try:
        # Build query: most-recent feature row at or before asof
        panel = build_panel(db,
                            start=(asof - dt.timedelta(days=365)).isoformat())
        import pandas as pd
        panel.index = pd.to_datetime(panel.index)
        row = panel.loc[panel.index <= pd.Timestamp(asof)].tail(1)
        if row.empty:
            return []

        constraints = {}
        if vix is not None and vix > 0:
            # VIX band: ±35% of current — broad but not absurd
            constraints["vix_close"] = (vix * 0.65, vix * 1.35)

        hits = retrieve(idx, row.iloc[0], k=8,
                        regime_constraints=constraints)
        summary = analogue_summary(hits)
        logger.info("analogues retrieved: n=%d mean=%+.4f p_up=%.2f",
                    summary.get("n", 0),
                    summary.get("mean", 0.0),
                    summary.get("p_up", 0.0))
        return [
            {"date": h.date, "similarity": h.similarity,
             "outcome": h.realised_logret,
             "vix": h.vix, "cot_z": h.cot_mm_net_z}
            for h in hits
        ]
    except Exception as exc:
        logger.warning("analogue retrieval failed: %s", exc)
        return []


def _model_snapshot_from_row(horizon: str, row):
    """Use the shared ModelServer rather than re-loading the model per call."""
    try:
        from .serve import ModelServer
    except Exception as exc:
        logger.debug("serve unavailable: %s", exc)
        return {}, None, None
    if row is None or row.empty:
        return {}, None, None
    srv = ModelServer.get(horizon)
    if srv is None:
        return {}, None, None
    pred = srv.forecast(row.iloc[0])
    if pred is None:
        return {}, None, None
    return pred.quantiles, pred.lambda_, None


def _retrieve_analogues_from_row(horizon: str, row, vix=None) -> list:
    """Analogue retrieval using a pre-built query row (no panel rebuild)."""
    try:
        from analogues import load_index, retrieve, analogue_summary
    except Exception as exc:
        logger.debug("analogues unavailable: %s", exc)
        return []
    if row is None or row.empty:
        return []
    idx = load_index(horizon)
    if idx is None:
        return []
    try:
        constraints = {}
        if vix is not None and vix > 0:
            constraints["vix_close"] = (vix * 0.65, vix * 1.35)
        hits = retrieve(idx, row.iloc[0], k=8, regime_constraints=constraints)
        summary = analogue_summary(hits)
        logger.info("analogues retrieved: n=%d mean=%+.4f p_up=%.2f",
                    summary.get("n", 0), summary.get("mean", 0.0),
                    summary.get("p_up", 0.0))
        return [
            {"date": h.date, "similarity": h.similarity,
             "outcome": h.realised_logret,
             "vix": h.vix, "cot_z": h.cot_mm_net_z}
            for h in hits
        ]
    except Exception as exc:
        logger.warning("analogue retrieval failed: %s", exc)
        return []


# v2 helpers retained for backward compatibility (legacy callers)
def _positioning_signals(db: LakeDB, asof: dt.date):
    """Latest COT mm-net-z and GLD 5d flow z available at or before asof."""
    # Robust median/MAD-z using last 1y window — keeps strategist independent
    # of the offline feature panel.
    cot = db.df(
        """
        SELECT report_date, mm_long - mm_short AS net
        FROM cot_disagg
        WHERE contract='GOLD_COMEX' AND report_date <= ?
        ORDER BY report_date DESC LIMIT 60
        """,
        [asof],
    )
    cot_z = _robust_z_scalar(cot["net"].to_numpy(dtype=float)) if not cot.empty else None

    gld = db.df(
        """
        SELECT date, flow_usd FROM etf_flows
        WHERE ticker='GLD' AND date <= ? AND flow_usd IS NOT NULL
        ORDER BY date DESC LIMIT 260
        """,
        [asof],
    )
    if not gld.empty and len(gld) >= 10:
        flow_5d = gld["flow_usd"].iloc[:5].sum()
        gld_z = _robust_z_scalar(
            gld["flow_usd"].rolling(5).sum().dropna().to_numpy(dtype=float)
        )
    else:
        gld_z = None
    return (
        round(float(cot_z), 2) if cot_z is not None else None,
        round(float(gld_z), 2) if gld_z is not None else None,
    )


def _robust_z_scalar(arr) -> Optional[float]:
    if arr is None or len(arr) < 5:
        return None
    import numpy as np
    a = np.asarray(arr, dtype=float)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    if mad < 1e-12:
        return 0.0
    # z of most recent value
    return (a[0] - med) / mad


def _model_snapshot(db: LakeDB, asof: dt.date, *, horizon: str):
    """Pull the active model and produce a single forecast for asof."""
    try:
        from .predict import predict_one
        from . import registry
        if registry.active_path(horizon) is None:
            return {}, None, None
        # We don't store the new prediction (predict_one would do that); instead
        # we issue an ephemeral one to feed the dossier without polluting the
        # predictions table when this is a "dry" thesis call.
        # For simplicity, we DO store it — it lets us track strategist-time
        # forecasts vs settled outcomes.
        # Dry-run: compute the forecast but DO NOT log it as a committed
        # prediction. The strategist may inspect many candidate forecasts;
        # only the actual end-of-day prediction (via `make predict`) is logged.
        from .predict import load_active
        loaded = load_active(horizon)
        if loaded is None:
            return {}, None, None
        model, cfg, feature_cols, med, mad, _meta = loaded

        from .features import build_panel
        import pandas as pd
        panel = build_panel(db,
                            start=(asof - dt.timedelta(days=365)).isoformat())
        panel.index = pd.to_datetime(panel.index)
        row = panel.loc[panel.index <= pd.Timestamp(asof)].tail(1)
        if row.empty:
            return {}, None, None

        import numpy as np
        import torch
        X = row[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        Xs = np.clip((X - med) / mad, -8.0, 8.0).astype(np.float32)
        with torch.no_grad():
            out = model(
                torch.as_tensor(Xs[:, :cfg.n_struct]),
                torch.as_tensor(Xs),
            )
        q_arr = out["quantiles"].numpy().reshape(-1)
        q = {f"q{int(p*100):02d}": round(float(q_arr[i]), 5)
             for i, p in enumerate(model.QUANTILES)}
        lam = float(out["lambda"].item())
        return q, lam, None
    except Exception as exc:
        logger.warning("model snapshot failed: %s", exc)
        return {}, None, None


# ── LLM call ─────────────────────────────────────────────────────────────────

# Tool definition forces structured JSON output via Anthropic's tool-use schema —
# more reliable than asking nicely for JSON and parsing free-form text.
_THESIS_TOOL = {
    "name": "submit_thesis",
    "description": "Submit your structured daily gold-market thesis.",
    "input_schema": {
        "type": "object",
        "required": ["thesis", "direction", "horizon_days", "confidence",
                     "primary_driver", "key_evidence", "falsifiable",
                     "risks", "trade_idea"],
        "properties": {
            "thesis":         {"type": "string"},
            "direction":      {"type": "string", "enum": ["long", "short", "flat"]},
            "horizon_days":   {"type": "integer", "minimum": 1, "maximum": 90},
            "confidence":     {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "primary_driver": {"type": "string"},
            "key_evidence":   {"type": "array", "items": {"type": "string"},
                                "minItems": 2, "maxItems": 6},
            "falsifiable":    {"type": "array", "items": {"type": "string"},
                                "minItems": 1, "maxItems": 4},
            "risks":          {"type": "array", "items": {"type": "string"},
                                "minItems": 1, "maxItems": 4},
            "trade_idea":     {"type": "string"},
        },
    },
}


def call_claude(dossier_text: str,
                model: str = "claude-haiku-4-5-20251001",
                api_key: Optional[str] = None) -> Optional[Dict]:
    """
    Call Claude with tool-use to guarantee structured output. The model is
    forced to use `submit_thesis` so we get a validated JSON object back
    rather than parsing free-form prose.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.error("anthropic SDK not installed")
        return None

    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=1000,
            system=[
                {"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}},
            ],
            tools=[_THESIS_TOOL],
            tool_choice={"type": "tool", "name": "submit_thesis"},
            messages=[{"role": "user", "content": dossier_text}],
        )
        for block in msg.content:
            if getattr(block, "type", "") == "tool_use" and block.name == "submit_thesis":
                return dict(block.input)
        logger.error("Claude returned no tool_use block")
        return None
    except Exception as exc:
        logger.error("Claude strategist call failed: %s", exc)
        return None


# ── Top-level entry ──────────────────────────────────────────────────────────

def generate_and_store(db: LakeDB, asof: Optional[dt.date] = None,
                       *, horizon: str = "20d",
                       model: str = "claude-haiku-4-5-20251001") -> Optional[Dict]:
    asof = asof or dt.date.today()
    dossier = build_dossier(db, asof, horizon=horizon)
    text = dossier.to_text()
    logger.info("dossier built (%d chars)", len(text))

    thesis = call_claude(text, model=model)
    if not thesis:
        return None

    db.upsert(
        "theses",
        ["date", "model", "thesis", "confidence", "predictions",
         "falsifiable", "brier_score"],
        [(asof, model,
          thesis.get("thesis", ""),
          float(thesis.get("confidence", 0.0)),
          json.dumps({k: thesis.get(k) for k in
                       ("direction", "horizon_days", "primary_driver",
                        "trade_idea", "key_evidence", "risks")}),
          json.dumps(thesis.get("falsifiable", [])),
          None,
         )],
        conflict_key=("date",),
    )
    logger.info("thesis stored for %s", asof)
    return thesis


def main(argv=None) -> int:
    import argparse, sys
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    p = argparse.ArgumentParser(description="Daily LLM strategist thesis")
    p.add_argument("--asof", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--horizon", default="20d")
    args = p.parse_args(argv)

    asof = (dt.date.fromisoformat(args.asof) if args.asof
            else dt.date.today())

    with LakeDB() as db:
        thesis = generate_and_store(db, asof=asof, horizon=args.horizon,
                                    model=args.model)
    if thesis:
        print(textwrap.dedent(json.dumps(thesis, indent=2)))
        return 0
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
