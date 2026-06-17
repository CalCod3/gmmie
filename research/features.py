"""
research.features — offline feature engineering for the hybrid model.

Single function `build_panel()` returns a pandas DataFrame indexed by date with
the full set of features the hybrid model expects. Designed to be the *one*
place where feature definitions live, so the same panel is used at training
time and at decision time (point-in-time correct: no future leakage).

All inputs come from the data lake; nothing is computed from live ingestion.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_lake.db import LakeDB

logger = logging.getLogger(__name__)

_PANEL_CACHE = Path("data") / "cache" / "panels"


def _lake_state_hash(db: LakeDB, target_symbol: str, start: Optional[str]) -> str:
    """
    Fingerprint of the lake's relevant state. Two panels share a hash iff:
      · same target symbol & start date
      · same max(date) across the four tables that contribute to the panel
      · same row counts (cheap proxy for upserts since the last build)

    This lets us safely skip rebuilds across thousands of research calls.
    """
    h = hashlib.sha256()
    h.update(target_symbol.encode())
    h.update(b"|")
    h.update(str(start).encode())
    for tbl, col in [("prices_d", "date"), ("macro_d", "date"),
                      ("cot_disagg", "report_date"), ("etf_flows", "date"),
                      ("news", "ts"), ("news_features", "extracted_at")]:
        try:
            n = db.scalar(f"SELECT COUNT(*) FROM {tbl}") or 0
            mx = db.scalar(f"SELECT MAX({col}) FROM {tbl}") or "none"
            h.update(f"|{tbl}:{n}:{mx}".encode())
        except Exception:
            pass
    return h.hexdigest()[:16]


def build_panel(db: LakeDB, *, target_symbol: str = "GC=F",
                start: Optional[str] = "2010-01-01",
                use_cache: bool = True) -> pd.DataFrame:
    """
    Build a daily feature panel.

    All time series are reindexed to the gold trading-day calendar so every
    feature is point-in-time correct (today's row contains only information
    available at today's NY close).

    Features (selection — all numeric, scale-invariant):
        gold_close, gold_logret_{1,5,20}d
        tips_10y, tips_d1_bp, tips_z, breakeven_10y, real_yield_proxy
        yield_curve_2_10, fedfunds
        dxy/vix/spx/slv/wti/copper/btc/gdx _close, _logret_{1,5,20}d, _z
        gold_silver_ratio, gold_oil_ratio, gold_copper_ratio
        cot_mm_net, cot_mm_net_z, cot_oi_z
        gld_holdings_tonnes, gld_flow_5d_z, gld_flow_intensity
        news_impact            — daily LLM-aggregated signed impact (conf-weighted)
        news_intensity         — count of high-confidence news events that day
        news_dispersion        — std-dev of impact (high = mixed signals)
        target_logret_{5,20,60}d  ← labels

    Cached to `data/cache/panels/<hash>.parquet`; set `use_cache=False` to
    force a rebuild.
    """
    # ── Cache lookup ─────────────────────────────────────────────────────────
    cache_key = None
    if use_cache and not os.environ.get("GMMIE_DISABLE_PANEL_CACHE"):
        cache_key = _lake_state_hash(db, target_symbol, start)
        cache_file = _PANEL_CACHE / f"{cache_key}.parquet"
        if cache_file.exists():
            try:
                logger.info("panel cache hit: %s", cache_key)
                df = pd.read_parquet(cache_file)
                df.index = pd.to_datetime(df.index)
                return df
            except Exception as exc:
                logger.warning("panel cache read failed (%s) — rebuilding", exc)

    # ── Prices ───────────────────────────────────────────────────────────────
    prices_raw = db.df(
        "SELECT date, symbol, close FROM prices_d WHERE date >= ? "
        "ORDER BY date, symbol",
        [start],
    )
    if prices_raw.empty:
        raise RuntimeError("prices_d is empty — run `make lake-yahoo` first")
    prices = prices_raw.pivot(index="date", columns="symbol", values="close")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    if target_symbol not in prices.columns:
        raise RuntimeError(f"target symbol {target_symbol} missing from lake")

    panel = pd.DataFrame(index=prices.index)
    panel.index.name = "date"
    panel["gold_close"] = prices[target_symbol]

    log_gold = np.log(panel["gold_close"])
    for n in (1, 5, 20):
        panel[f"gold_logret_{n}d"] = log_gold.diff(n)

    for sym, name in [
        ("DX-Y.NYB", "dxy"),
        ("^VIX",     "vix"),
        ("^GSPC",    "spx"),
        ("SLV",      "slv"),
        ("CL=F",     "wti"),
        ("HG=F",     "copper"),
        ("BTC-USD",  "btc"),
        ("GDX",      "gdx"),
    ]:
        if sym in prices.columns:
            s = prices[sym].astype(float)
            ls = np.log(s.replace(0.0, np.nan))
            panel[f"{name}_close"]      = s
            panel[f"{name}_logret_1d"]  = ls.diff(1)
            panel[f"{name}_logret_5d"]  = ls.diff(5)
            panel[f"{name}_logret_20d"] = ls.diff(20)
            panel[f"{name}_z"]          = _robust_z(s, 252)

    # Cross-asset ratios — regime tells
    if "SLV" in prices.columns:
        panel["gold_silver_ratio"] = panel["gold_close"] / prices["SLV"]
    if "CL=F" in prices.columns:
        panel["gold_oil_ratio"]    = panel["gold_close"] / prices["CL=F"]
    if "HG=F" in prices.columns:
        panel["gold_copper_ratio"] = panel["gold_close"] / prices["HG=F"]

    # ── Yang-Zhang realised volatility (uses OHLC, not just close) ────────
    # Yang & Zhang (2000) is the most efficient OHLC volatility estimator —
    # ~7× tighter than close-to-close σ. Decomposes total variance into
    # overnight + open-to-close drift + Rogers-Satchell intra-bar.
    ohlc = db.df(
        "SELECT date, symbol, open, high, low, close FROM prices_d "
        "WHERE date >= ? AND symbol IN ('GC=F','DX-Y.NYB','^VIX')",
        [start],
    )
    if not ohlc.empty:
        for sym, name in [("GC=F", "gold"), ("DX-Y.NYB", "dxy"),
                          ("^VIX",  "vix")]:
            sub = ohlc[ohlc["symbol"] == sym]
            if sub.empty:
                continue
            sub = sub.set_index(pd.to_datetime(sub["date"])).sort_index()
            yz_20 = _yang_zhang(sub, window=20)
            yz_5  = _yang_zhang(sub, window=5)
            yz_20 = yz_20.reindex(panel.index).ffill(limit=2)
            yz_5  = yz_5.reindex(panel.index).ffill(limit=2)
            panel[f"{name}_yz_20"]   = yz_20
            panel[f"{name}_yz_5"]    = yz_5
            # Vol-regime indicator: ratio of short-term to long-term YZ vol.
            # > 1 = vol expansion (often gold-bid), < 1 = vol compression.
            panel[f"{name}_vol_regime"] = (yz_5 / yz_20.replace(0, np.nan)
                                            ).clip(0.0, 5.0)

    # ── Macro (FRED) ────────────────────────────────────────────────────────
    macro_raw = db.df(
        "SELECT date, series_id, value FROM macro_d WHERE date >= ? "
        "ORDER BY date, series_id",
        [start],
    )
    if not macro_raw.empty:
        macro = macro_raw.pivot(index="date", columns="series_id", values="value")
        macro.index = pd.to_datetime(macro.index)
        macro = macro.sort_index().reindex(panel.index).ffill(limit=5)

        if "DFII10" in macro.columns:
            panel["tips_10y"]   = macro["DFII10"]
            panel["tips_d1_bp"] = macro["DFII10"].diff() * 100.0
            panel["tips_z"]     = _robust_z(macro["DFII10"], 252)
        if "T10YIE" in macro.columns:
            panel["breakeven_10y"] = macro["T10YIE"]
        if "DGS10" in macro.columns and "T10YIE" in macro.columns:
            panel["real_yield_proxy"] = macro["DGS10"] - macro["T10YIE"]
        if "DGS10" in macro.columns and "DGS2" in macro.columns:
            panel["yield_curve_2_10"] = macro["DGS10"] - macro["DGS2"]
        if "FEDFUNDS" in macro.columns:
            panel["fedfunds"] = macro["FEDFUNDS"]
        if "VIXCLS" in macro.columns and "vix_close" not in panel.columns:
            panel["vix_close"] = macro["VIXCLS"]

    # ── COT (weekly → ffill to daily, capped at 7d staleness) ──────────────
    cot = db.df(
        "SELECT report_date, mm_long, mm_short, open_interest "
        "FROM cot_disagg WHERE contract='GOLD_COMEX' AND report_date >= ?",
        [start],
    )
    if not cot.empty:
        cot["report_date"] = pd.to_datetime(cot["report_date"])
        cot = cot.set_index("report_date").sort_index()
        cot["mm_net"]   = cot["mm_long"] - cot["mm_short"]
        cot["mm_net_z"] = _robust_z(cot["mm_net"], 52)
        cot["oi_z"]     = _robust_z(cot["open_interest"], 52)
        cot_d = cot.reindex(panel.index).ffill(limit=7)
        panel["cot_mm_net"]   = cot_d["mm_net"]
        panel["cot_mm_net_z"] = cot_d["mm_net_z"]
        panel["cot_oi_z"]     = cot_d["oi_z"]

    # ── GLD flows ──────────────────────────────────────────────────────────
    gld = db.df(
        "SELECT date, holdings_tonnes, flow_usd FROM etf_flows "
        "WHERE ticker='GLD' AND date >= ?",
        [start],
    )
    if not gld.empty:
        gld["date"] = pd.to_datetime(gld["date"])
        gld = gld.set_index("date").sort_index().reindex(panel.index).ffill(limit=2)
        panel["gld_holdings_tonnes"] = gld["holdings_tonnes"]
        flow_5d = gld["flow_usd"].rolling(5, min_periods=2).sum()
        panel["gld_flow_5d_z"]        = _robust_z(flow_5d, 252)
        # Intensity = |flow|/recent typical |flow| — magnitude regardless of sign
        flow_abs = gld["flow_usd"].abs().rolling(20, min_periods=5).median()
        panel["gld_flow_intensity"]   = (gld["flow_usd"].abs() /
                                         flow_abs.replace(0, np.nan))

    # ── News (LLM-extracted daily aggregation) ─────────────────────────────
    # IMPORTANT: this column is consumed by the structural head. If news_features
    # is empty we still emit a zero column so the model has a stable schema.
    news_daily = _daily_news_signal(db, start=start)
    if not news_daily.empty:
        news_daily.index = pd.to_datetime(news_daily.index)
        news_daily = news_daily.reindex(panel.index).fillna(0.0)
        panel["news_impact"]     = news_daily["impact_wmean"]
        panel["news_intensity"]  = news_daily["intensity"]
        panel["news_dispersion"] = news_daily["dispersion"]
    else:
        # Schema-stable zero columns — structural head sees no signal but won't crash.
        panel["news_impact"]     = 0.0
        panel["news_intensity"]  = 0.0
        panel["news_dispersion"] = 0.0

    # ── Labels (future) ────────────────────────────────────────────────────
    for h in (5, 20, 60):
        panel[f"target_logret_{h}d"] = log_gold.shift(-h) - log_gold

    # NOTE: winsorisation was previously applied here using FULL-panel
    # quantiles, which is a subtle lookahead leak. It is now applied at
    # train time on the training portion only (see research.train.fit_winsorizer).

    # ── Cache write ──────────────────────────────────────────────────────────
    if cache_key:
        try:
            _PANEL_CACHE.mkdir(parents=True, exist_ok=True)
            panel.to_parquet(_PANEL_CACHE / f"{cache_key}.parquet")
            logger.info("panel cached: %s (%d × %d)",
                        cache_key, *panel.shape)
        except Exception as exc:
            logger.warning("panel cache write failed: %s", exc)

    return panel


def _daily_news_signal(db: LakeDB, *, start: Optional[str]) -> pd.DataFrame:
    """
    Aggregate LLM-extracted per-headline features into daily series.

    Returns DataFrame indexed by date with:
        impact_wmean : confidence-weighted mean of gold_impact
        intensity    : count of (confidence >= 0.5) events
        dispersion   : std-dev of impact (high = conflicting headlines)
    """
    # Group by ET (NY market) calendar date — gold's primary session is COMEX.
    # `n.ts` is stored as UTC TIMESTAMP; convert with DuckDB's tz functions.
    df = db.df(
        """
        SELECT
            CAST((n.ts AT TIME ZONE 'UTC') AT TIME ZONE 'America/New_York'
                 AS DATE)         AS date,
            f.gold_impact         AS impact,
            f.confidence          AS conf,
            COALESCE(f.novel, FALSE) AS novel
        FROM news n JOIN news_features f ON n.id = f.news_id
        WHERE CAST((n.ts AT TIME ZONE 'UTC') AT TIME ZONE 'America/New_York'
                   AS DATE) >= ?
          AND f.gold_impact IS NOT NULL
          AND f.confidence  IS NOT NULL
        ORDER BY n.ts
        """,
        [start],
    )
    if df.empty:
        return pd.DataFrame()

    # Confidence-weighted aggregations per day. All ops are vectorised pandas
    # group-bys — no `.apply` lambda (deprecated in pandas 2.2+).
    df["w"]  = df["conf"].clip(0.0, 1.0)
    df["wi"] = df["impact"] * df["w"]
    df["hi_conf_novel"] = ((df["conf"] >= 0.5) & df["novel"].fillna(False)
                            ).astype(int)
    g = df.groupby("date", sort=True)
    out = pd.DataFrame({
        "impact_wmean": g["wi"].sum() / g["w"].sum().replace(0, np.nan),
        "intensity":    g["hi_conf_novel"].sum(),
        "dispersion":   g["impact"].std(),
    }).fillna(0.0)
    return out


def _robust_z(s: pd.Series, window: int) -> pd.Series:
    """Rolling robust z-score (median / MAD-σ)."""
    med = s.rolling(window, min_periods=max(20, window // 4)).median()
    mad = (s - med).abs().rolling(window, min_periods=max(20, window // 4)).median()
    sigma = mad * 1.4826
    return (s - med) / sigma.replace(0, np.nan)


def _yang_zhang(ohlc: pd.DataFrame, window: int = 20,
                annualization: float = 252.0) -> pd.Series:
    """
    Yang-Zhang volatility (Yang & Zhang, 2000).

    σ²_YZ = σ²_overnight + k·σ²_open_to_close + (1-k)·σ²_RS
    with k = 0.34 / (1.34 + (n+1)/(n-1))  (variance-minimising weight)

    Inputs require columns: open, high, low, close. NaN-tolerant.
    """
    if ohlc.empty or window < 2:
        return pd.Series(dtype=float)
    o = pd.to_numeric(ohlc["open"],  errors="coerce")
    h = pd.to_numeric(ohlc["high"],  errors="coerce")
    l = pd.to_numeric(ohlc["low"],   errors="coerce")
    c = pd.to_numeric(ohlc["close"], errors="coerce")
    o_prev_close = c.shift(1)

    # Overnight return (close_{t-1} → open_t)
    r_on = np.log(o / o_prev_close)
    # Open-to-close return
    r_oc = np.log(c / o)
    # Rogers-Satchell intra-bar variance contribution
    rs = (np.log(h / c) * np.log(h / o)
          + np.log(l / c) * np.log(l / o))

    var_on = r_on.pow(2).rolling(window, min_periods=max(2, window // 4)).mean()
    var_oc = r_oc.pow(2).rolling(window, min_periods=max(2, window // 4)).mean()
    var_rs = rs.rolling(window, min_periods=max(2, window // 4)).mean()

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var_yz = var_on + k * var_oc + (1.0 - k) * var_rs
    var_yz = var_yz.clip(lower=0.0)
    return (var_yz * annualization).pow(0.5)
