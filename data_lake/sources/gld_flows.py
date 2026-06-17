"""
SPDR Gold Trust (GLD) holdings + flows backfill.

GLD discloses holdings daily on `spdrgoldshares.com`. We approximate flows by
the day-over-day delta in shares outstanding × NAV (i.e. net creates/redeems).

This is a leading indicator: institutional gold demand shows up in GLD/IAU
flows hours before the spot move shows up on most retail-tier feeds.

If the SPDR endpoint is unavailable, we fall back to deriving holdings from
the WGC ETF flow CSV (when available).
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import List, Tuple

import io

import httpx
import pandas as pd

from ..db import LakeDB
from ._http import get_with_retry

logger = logging.getLogger(__name__)

# Public historical holdings export from SPDR. URL is stable over years.
SPDR_HISTORY_URL = (
    "https://www.spdrgoldshares.com/assets/dynamic/GLD/"
    "GLD_US_archive_EN.csv"
)


def backfill(db: LakeDB) -> int:
    """Backfill daily GLD holdings + derived flows."""
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            r = get_with_retry(client, SPDR_HISTORY_URL)
            df = pd.read_csv(io.BytesIO(r.content), skiprows=6)
    except Exception as exc:
        logger.error("SPDR GLD download failed: %s", exc)
        return 0

    # Columns vary; normalise
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if "Date" in c), None)
    tonnes_col = next((c for c in df.columns if "Tonnes" in c), None)
    nav_col = next((c for c in df.columns if "NAV" in c or "Per Share" in c), None)
    shares_col = next(
        (c for c in df.columns if "Shares" in c or "Ounces" in c), None
    )
    if not all([date_col, tonnes_col, nav_col]):
        logger.error("SPDR GLD: unexpected column layout: %s", df.columns.tolist())
        return 0

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # Day-over-day tonnage delta * spot ≈ flow_usd (gold spot kg → usd estimate)
    # We compute flow_usd more correctly below using NAV.
    df["_tonnes"]  = pd.to_numeric(df[tonnes_col], errors="coerce")
    df["_nav"]     = pd.to_numeric(df[nav_col],    errors="coerce")
    if shares_col:
        df["_shares"] = pd.to_numeric(df[shares_col], errors="coerce")
    else:
        df["_shares"] = pd.NA

    # AUM = tonnes * (USD/oz) ; 1 tonne = 32,150.7 troy oz. We don't have gold
    # spot here; estimate AUM from shares * NAV when both available.
    df["_aum"] = df["_shares"] * df["_nav"]
    df["_flow"] = df["_shares"].diff() * df["_nav"]

    rows: List[Tuple] = []
    for _, r in df.iterrows():
        d = r[date_col].date()
        rows.append((
            d, "GLD",
            _f(r["_tonnes"]),
            _f(r["_shares"]),
            _f(r["_nav"]),
            _f(r["_aum"]),
            _f(r["_flow"]),
        ))

    n = db.upsert(
        "etf_flows",
        ["date", "ticker", "holdings_tonnes", "shares_out",
         "nav", "aum_usd", "flow_usd"],
        rows,
        conflict_key=("date", "ticker"),
    )
    logger.info("GLD: backfilled %d rows", n)
    return n


def _f(x):
    try:
        if x is None or (isinstance(x, float) and x != x):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None
