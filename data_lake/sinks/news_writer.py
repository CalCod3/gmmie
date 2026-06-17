"""
data_lake.sinks.news_writer — persist live news/sentiment events into the lake.

The live ingestion stack (ingestion.py) produces RawEvent instances on an
asyncio.Queue. Until now nothing wrote them to the `news` table — meaning
the offline LLM feature extractor had nothing to chew on, and the structural
model's `news_impact` column was always zero.

This bridge subscribes to a queue, batches inserts (1s window or 64 events
whichever first), and upserts via the LakeDB.

It is import-safe: starting the bridge does NOT require an active asyncio
loop at import time, only at start().
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _news_id(url: str, ts: float, fallback_text: str) -> str:
    """Stable id from url+ts; falls back to text hash when url is empty."""
    if url:
        seed = f"{url}|{int(ts)}"
    else:
        seed = f"{fallback_text[:200]}|{int(ts)}"
    return hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()[:32]


class NewsToLakeBridge:
    """
    Consumes RawEvent objects from a source queue and upserts NEWS/SENTIMENT
    events into the lake's `news` table.

    Usage (from main.py):
        bridge = NewsToLakeBridge(source_queue=engine.ingestion.queue)
        task = asyncio.create_task(bridge.run())
    """

    BATCH_MAX_N    = 64
    BATCH_MAX_WAIT = 1.0     # seconds

    def __init__(self, source_queue: "asyncio.Queue",
                 db_path: Optional[str] = None,
                 also_forward: bool = True):
        from data_lake.db import LakeDB
        self._src = source_queue
        self._db  = LakeDB(db_path)
        self._db.open()
        self._also_forward = also_forward
        self._running: bool = False
        self._n_persisted: int = 0
        self._n_skipped:   int = 0

    async def run(self) -> None:
        self._running = True
        buf: List[Tuple] = []
        last_flush = time.monotonic()
        forwarded: List[Any] = []
        logger.info("NewsToLakeBridge started")
        while self._running:
            try:
                ev = await asyncio.wait_for(self._src.get(), timeout=0.5)
                row = self._event_to_row(ev)
                if row is not None:
                    buf.append(row)
                if self._also_forward:
                    forwarded.append(ev)
            except asyncio.TimeoutError:
                pass
            except Exception as exc:
                logger.error("news bridge get error: %s", exc)

            now = time.monotonic()
            if buf and (len(buf) >= self.BATCH_MAX_N
                        or (now - last_flush) >= self.BATCH_MAX_WAIT):
                self._flush(buf)
                buf.clear()
                last_flush = now

            if self._also_forward and forwarded:
                # The downstream queue is the same object we're reading from
                # when this bridge is wired in "consume only" mode. To allow
                # the engine to also see the event, the caller must wire a
                # tee/fan-out — left to the orchestrator.
                forwarded.clear()

        if buf:
            self._flush(buf)
        logger.info("NewsToLakeBridge stopped (persisted=%d skipped=%d)",
                    self._n_persisted, self._n_skipped)

    def stop(self) -> None:
        self._running = False

    def _flush(self, batch: List[Tuple]) -> None:
        try:
            self._db.upsert(
                "news",
                ["id", "ts", "source", "url", "title", "text", "raw_json"],
                batch,
                conflict_key=("id",),
            )
            self._n_persisted += len(batch)
        except Exception as exc:
            logger.error("news flush failed: %s", exc)
            self._n_skipped += len(batch)

    def _event_to_row(self, ev) -> Optional[Tuple]:
        """Map a RawEvent (or NormalizedEvent) to a `news` row."""
        import datetime as dt
        # Duck-typed: accept both raw and normalised events
        et = getattr(ev, "type", None)
        if hasattr(et, "value"):
            et = et.value
        et = str(et) if et is not None else ""
        if et not in {"news", "sentiment"}:
            return None
        ts = float(getattr(ev, "timestamp", None)
                   or getattr(ev, "wall_time", None) or time.time())
        payload = getattr(ev, "payload", {}) or {}
        url     = (payload.get("url") or "").strip()[:1024]
        title   = (payload.get("title") or "").strip()[:512]
        text    = (payload.get("text") or "").strip()[:8192]
        if not (title or text):
            return None
        source  = getattr(ev, "source", "unknown")
        return (
            _news_id(url, ts, title or text),
            dt.datetime.utcfromtimestamp(ts),
            source[:64], url, title, text,
            json.dumps(payload, default=str)[:32 * 1024],
        )

    @property
    def stats(self) -> dict:
        return {"persisted": self._n_persisted, "skipped": self._n_skipped}
