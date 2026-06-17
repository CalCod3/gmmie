"""
FRED historical backfill.

Pulls full history for the core gold-pricing macro series. FRED's API has no
hard rate limit for reasonable use; we still throttle to be polite.

Critical series for gold:
    DFII10     — 10Y TIPS yield (THE structural driver)
    T10YIE     — 10Y breakeven inflation
    DGS10      — 10Y nominal yield
    DGS2       — 2Y nominal yield
    DTWEXBGS   — Trade-weighted broad dollar
    FEDFUNDS   — Effective fed funds rate
    CPIAUCSL   — CPI level
    UNRATE     — Unemployment rate
    VIXCLS     — VIX
    DCOILWTICO — WTI crude
    GOLDAMGBD228NLBM — London gold AM fix
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Iterable, List, Optional, Tuple

import httpx

from ..db import LakeDB
from ._http import get_with_retry

logger = logging.getLogger(__name__)

OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

CORE_SERIES: List[str] = [
    "DFII10", "T10YIE", "DGS10", "DGS2", "DTWEXBGS",
    "FEDFUNDS", "CPIAUCSL", "UNRATE",
    "VIXCLS", "DCOILWTICO", "GOLDAMGBD228NLBM",
    "PCEPI", "INDPRO", "PAYEMS", "RRPONTSYD",         # core+supplementary
]


def fetch_series(
    api_key: str,
    series_id: str,
    *,
    start: dt.date = dt.date(1970, 1, 1),
    end: Optional[dt.date] = None,
    client: Optional[httpx.Client] = None,
) -> List[Tuple[dt.date, str, float]]:
    """Return list of (date, series_id, value) for one series."""
    end = end or dt.date.today()
    own_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        params = {
            "series_id": series_id,
            "api_key":   api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end":   end.isoformat(),
        }
        r = get_with_retry(client, OBS_URL, params=params)
        data = r.json()
    finally:
        if own_client:
            client.close()

    rows: List[Tuple[dt.date, str, float]] = []
    for obs in data.get("observations", []):
        v = obs.get("value", ".")
        if v == ".":
            continue
        try:
            value = float(v)
        except ValueError:
            continue
        try:
            d = dt.date.fromisoformat(obs["date"])
        except (KeyError, ValueError):
            continue
        rows.append((d, series_id, value))
    return rows


def backfill(
    db: LakeDB,
    api_key: str,
    series: Iterable[str] = CORE_SERIES,
    *,
    start: dt.date = dt.date(1970, 1, 1),
    throttle_s: float = 0.2,
) -> int:
    total = 0
    with db.run("fred_history") as run_id, httpx.Client(timeout=30.0) as client:
        for sid in series:
            try:
                rows = fetch_series(api_key, sid, start=start, client=client)
            except Exception as exc:
                logger.error("FRED [%s] fetch failed: %s", sid, exc)
                time.sleep(throttle_s)
                continue
            n = db.upsert(
                "macro_d",
                ["date", "series_id", "value"],
                rows,
                conflict_key=("date", "series_id"),
            )
            db.record_rows(run_id, n)
            total += n
            logger.info("FRED [%s] backfilled %d rows", sid, n)
            time.sleep(throttle_s)
    return total
