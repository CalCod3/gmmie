"""
build_lake.py — one-shot historical backfill orchestrator.

Usage:
    python -m data_lake.build_lake --all
    python -m data_lake.build_lake --fred --yahoo
    python -m data_lake.build_lake --cot --since 2010
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys

from .db import LakeDB
from .sources import fred_history, yahoo, cftc_cot, gld_flows, fomc

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("build_lake")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="GMMIE data lake backfill")
    p.add_argument("--all",    action="store_true", help="run every source")
    p.add_argument("--fred",   action="store_true")
    p.add_argument("--yahoo",  action="store_true")
    p.add_argument("--cot",    action="store_true")
    p.add_argument("--gld",    action="store_true")
    p.add_argument("--fomc",   action="store_true")
    p.add_argument("--since",  default="1990-01-01",
                   help="ISO date — start of price history")
    p.add_argument("--cot-start-year", type=int, default=2010)
    args = p.parse_args(argv)

    if args.all:
        args.fred = args.yahoo = args.cot = args.gld = args.fomc = True

    if not any([args.fred, args.yahoo, args.cot, args.gld, args.fomc]):
        p.print_help()
        return 1

    since = dt.date.fromisoformat(args.since)

    with LakeDB() as db:
        if args.fred:
            key = os.environ.get("FRED_API_KEY", "")
            if not key:
                logger.error("FRED_API_KEY not set — skipping FRED")
            else:
                fred_history.backfill(db, key, start=since)

        if args.yahoo:
            yahoo.backfill(db, start=since)

        if args.cot:
            cftc_cot.backfill(db, start_year=args.cot_start_year)

        if args.gld:
            gld_flows.backfill(db)

        if args.fomc:
            fomc.backfill(db, since=since)

    logger.info("Backfill complete. Lake: %s", LakeDB().path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
