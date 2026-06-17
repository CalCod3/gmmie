"""
Yahoo Finance historical OHLCV via yfinance.

Symbols of interest for a gold-driven cross-asset model:
    GC=F   COMEX gold front-month future
    GLD    SPDR Gold Trust ETF (proxy for spot, daily flow visibility)
    DX-Y.NYB  US Dollar Index (DXY) — primary gold antagonist
    ^TNX   10Y Treasury yield (×10)
    ^VIX   VIX
    ^GSPC  S&P 500
    SLV    Silver ETF (gold/silver ratio is a regime tell)
    CL=F   WTI crude
    BTC-USD Bitcoin (modern safe-haven competitor)
    GDX    Gold miners ETF
    HG=F   Copper (gold/copper as growth signal)
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable, List, Tuple

from ..db import LakeDB

logger = logging.getLogger(__name__)

CORE_SYMBOLS = [
    "GC=F", "GLD", "DX-Y.NYB", "^TNX", "^VIX", "^GSPC",
    "SLV", "CL=F", "BTC-USD", "GDX", "HG=F",
]


def backfill(
    db: LakeDB,
    symbols: Iterable[str] = CORE_SYMBOLS,
    *,
    start: dt.date = dt.date(1990, 1, 1),
) -> int:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance not installed: pip install yfinance") from exc

    end = dt.date.today()
    total = 0

    with db.run("yahoo_history") as run_id:
        for sym in symbols:
            try:
                df = yf.download(
                    sym,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    progress=False,
                    auto_adjust=False,
                    actions=False,
                )
            except Exception as exc:
                logger.error("Yahoo [%s] failed: %s", sym, exc)
                continue

            if df is None or df.empty:
                logger.warning("Yahoo [%s] empty", sym)
                continue

            # yfinance may return MultiIndex columns when a single symbol is passed;
            # flatten to scalar columns.
            if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
                df.columns = df.columns.get_level_values(0)

            rows: List[Tuple] = []
            for ts, row in df.iterrows():
                rows.append((
                    ts.date(), sym,
                    _f(row.get("Open")),
                    _f(row.get("High")),
                    _f(row.get("Low")),
                    _f(row.get("Close")),
                    _f(row.get("Volume")),
                    "yahoo",
                ))
            n = db.upsert(
                "prices_d",
                ["date", "symbol", "open", "high", "low", "close", "volume", "source"],
                rows,
                conflict_key=("date", "symbol"),
            )
            db.record_rows(run_id, n)
            total += n
            logger.info("Yahoo [%s] backfilled %d rows", sym, n)
    return total


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
