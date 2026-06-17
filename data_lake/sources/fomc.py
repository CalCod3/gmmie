"""
FOMC statement backfill.

Scrapes statement text from the Federal Reserve press release archive.
For each FOMC meeting we store the statement, plus a *diff* vs the previous
meeting's statement. The diff is the structured signal — markets react to
*changes* in language, not the language itself.

URL pattern:
    https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm  (list)
    https://www.federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm

We keep this lightweight — no headless browser. If a meeting page returns
unexpected HTML, we skip it. The dataset is small (~8 meetings/year × 25
years = 200 rows total), so this can be re-run any time.
"""

from __future__ import annotations

import datetime as dt
import difflib
import logging
import re
from typing import List, Optional, Tuple

import httpx

from ..db import LakeDB

logger = logging.getLogger(__name__)

CAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
STMT_URL = (
    "https://www.federalreserve.gov/newsevents/pressreleases/"
    "monetary{date}a.htm"
)

_DATE_RE = re.compile(r"monetary(\d{8})a\.htm")


def discover_meeting_dates(client: httpx.Client,
                           since: dt.date = dt.date(2000, 1, 1)) -> List[dt.date]:
    """Discover meeting dates by scanning the press release archive index."""
    dates: set = set()
    # We can't reliably enumerate every year from the calendar HTML. Use the
    # press-release-by-year index instead.
    for year in range(since.year, dt.date.today().year + 1):
        url = (f"https://www.federalreserve.gov/newsevents/"
               f"pressreleases/{year}-press.htm")
        try:
            r = client.get(url)
            r.raise_for_status()
        except Exception:
            continue
        for m in _DATE_RE.finditer(r.text):
            try:
                dates.add(dt.datetime.strptime(m.group(1), "%Y%m%d").date())
            except ValueError:
                continue
    return sorted(dates)


def fetch_statement(client: httpx.Client, date: dt.date) -> Optional[str]:
    url = STMT_URL.format(date=date.strftime("%Y%m%d"))
    try:
        r = client.get(url)
        r.raise_for_status()
    except Exception as exc:
        logger.debug("FOMC %s: HTTP error %s", date, exc)
        return None
    # Strip HTML — naive but fine for this use
    text = re.sub(r"<script[\s\S]*?</script>", " ", r.text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Heuristic: a statement is between "Federal Reserve issued the following" and "Voting"
    m = re.search(
        r"(Information received[\s\S]+?)(?:Voting for|For media inquiries)",
        text,
    )
    return (m.group(1).strip() if m else text)[:50_000]


def diff_text(prev: str, cur: str) -> str:
    diff = difflib.unified_diff(
        (prev or "").split(), (cur or "").split(), lineterm="", n=0
    )
    return "\n".join(list(diff)[:200])      # cap


def backfill(db: LakeDB, *, since: dt.date = dt.date(2008, 1, 1)) -> int:
    total = 0
    with db.run("fomc") as run_id, httpx.Client(
            timeout=30.0, follow_redirects=True,
            headers={"User-Agent": "GMMIE/0.1 (research)"}) as client:
        dates = discover_meeting_dates(client, since=since)
        prev_text = ""
        rows: List[Tuple] = []
        for d in dates:
            text = fetch_statement(client, d)
            if not text:
                continue
            diff = diff_text(prev_text, text)
            rows.append((d, text, diff, None))
            prev_text = text
        n = db.upsert(
            "fomc_statements",
            ["date", "statement", "diff_prior", "embedding"],
            rows,
            conflict_key=("date",),
        )
        db.record_rows(run_id, n)
        total += n
    logger.info("FOMC: backfilled %d statements", total)
    return total
