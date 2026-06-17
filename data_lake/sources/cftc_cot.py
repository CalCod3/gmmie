"""
CFTC Disaggregated COT (Commitments of Traders) historical backfill.

Pulls the annual TXT zips from cftc.gov/files/dea/history, extracts the GOLD
contract (CFTC market code 088691), and upserts into `cot_disagg`.

Rewrite notes (v2):
  · Vectorised — no iterrows. Avoids the v1 .iloc bug after filtering.
  · Tolerant column matching — column names mutate slightly across years
    (e.g. `Swap_Positions_Short_All` vs `Swap__Positions_Short_All` with a
    double-underscore typo in a few years' files).
  · 404s on the URL are treated as soft failures (the year may not be
    published yet) — the rest of the backfill continues.

Why this matters: managed-money net long is one of the most reliable
medium-horizon mean-reversion signals on gold. Extreme positioning
(95th/5th percentile) typically gives back ~3–6% in the following 4–8 weeks.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import re
import zipfile
from typing import List, Optional

import httpx
import pandas as pd

from ..db import LakeDB
from ._http import get_with_retry

logger = logging.getLogger(__name__)

ZIP_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
GOLD_CFTC_CODE = "088691"

# Tolerant column-name resolver. Matches any column whose name, after
# normalising (lowercase, strip whitespace, collapse repeated underscores,
# drop trailing spaces), equals one of the candidates.

def _normcol(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"_+", "_", s)
    s = s.replace(" ", "_")
    return s


def _resolve(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first column whose normalised name matches any candidate."""
    targets = {_normcol(c) for c in candidates}
    for c in df.columns:
        if _normcol(c) in targets:
            return c
    return None


def fetch_year(year: int, *, client: httpx.Client) -> List[tuple]:
    url = ZIP_URL.format(year=year)
    # Quick existence probe — don't burn retries on a future year
    head = client.head(url)
    if head.status_code == 404:
        logger.info("COT %d: 404 (not yet published)", year)
        return []
    r = get_with_retry(client, url)

    z = zipfile.ZipFile(io.BytesIO(r.content))
    txt_names = [n for n in z.namelist() if n.lower().endswith(".txt")]
    if not txt_names:
        logger.warning("COT %d: zip contains no .txt", year)
        return []

    with z.open(txt_names[0]) as f:
        df = pd.read_csv(f, low_memory=False, encoding="latin-1")

    code_col = _resolve(df, "CFTC_Contract_Market_Code",
                            "CFTC Contract Market Code")
    if code_col is None:
        logger.error("COT %d: no contract-code column; cols=%s",
                     year, df.columns.tolist()[:8])
        return []

    df = df[df[code_col].astype(str).str.strip() == GOLD_CFTC_CODE].copy()
    if df.empty:
        return []

    date_col = _resolve(df, "Report_Date_as_YYYY-MM-DD",
                           "Report Date as YYYY-MM-DD",
                           "Report_Date_as_MM_DD_YYYY")
    if date_col is None:
        logger.error("COT %d: no date column", year)
        return []

    df["_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df = df.dropna(subset=["_date"])
    if df.empty:
        return []

    def col(*names):
        c = _resolve(df, *names)
        if c is None:
            return pd.Series(dtype=float, index=df.index)
        return pd.to_numeric(df[c], errors="coerce")

    out = pd.DataFrame({
        "report_date":      df["_date"],
        "contract":         "GOLD_COMEX",
        "open_interest":    col("Open_Interest_All"),
        "mm_long":          col("M_Money_Positions_Long_All"),
        "mm_short":         col("M_Money_Positions_Short_All"),
        "mm_spread":        col("M_Money_Positions_Spread_All",
                                "M_Money_Positions_Spread_All"),
        "swap_long":        col("Swap_Positions_Long_All",
                                "Swap__Positions_Long_All"),
        "swap_short":       col("Swap_Positions_Short_All",
                                "Swap__Positions_Short_All"),
        "producer_long":    col("Prod_Merc_Positions_Long_All"),
        "producer_short":   col("Prod_Merc_Positions_Short_All"),
        "other_rep_long":   col("Other_Rept_Positions_Long_All"),
        "other_rep_short":  col("Other_Rept_Positions_Short_All"),
        "nonrep_long":      col("NonRept_Positions_Long_All"),
        "nonrep_short":     col("NonRept_Positions_Short_All"),
    })

    # Convert NaN → None for DuckDB
    out = out.where(pd.notna(out), None)
    return list(out.itertuples(index=False, name=None))


def backfill(db: LakeDB, *, start_year: int = 2010,
             end_year: Optional[int] = None,
             timeout_s: int = 60) -> int:
    end_year = end_year or dt.date.today().year
    total = 0
    cols = ["report_date", "contract", "open_interest",
            "mm_long", "mm_short", "mm_spread",
            "swap_long", "swap_short",
            "producer_long", "producer_short",
            "other_rep_long", "other_rep_short",
            "nonrep_long", "nonrep_short"]
    with db.run("cftc_cot") as run_id, httpx.Client(
            timeout=timeout_s, follow_redirects=True,
            headers={"User-Agent": "GMMIE/0.1 (research)"}) as client:
        for year in range(start_year, end_year + 1):
            try:
                rows = fetch_year(year, client=client)
            except Exception as exc:
                logger.error("COT %d failed: %s", year, exc)
                continue
            if not rows:
                continue
            n = db.upsert(
                "cot_disagg", cols, rows,
                conflict_key=("report_date", "contract"),
            )
            db.record_rows(run_id, n)
            total += n
            logger.info("COT %d: backfilled %d rows", year, n)
    return total
