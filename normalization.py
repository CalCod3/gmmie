"""
normalization.py — Layer 2: Normalization
==========================================
Converts RawEvents → time-aligned NormalizedEvents.
  · Lamport timestamps (distributed consistency)
  · Pydantic-style schema validation (pure dataclass)
  · Rolling z-score scaler for prices (online, no lookahead)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Lamport clock ─────────────────────────────────────────────────────────────

class LamportClock:
    def __init__(self): self._t = 0
    def tick(self)    -> int:
        self._t += 1
        return self._t
    def update(self, received: int) -> int:
        self._t = max(self._t, received) + 1
        return self._t
    @property
    def value(self) -> int: return self._t

_clock = LamportClock()


# ── Unified schema ────────────────────────────────────────────────────────────

@dataclass
class NormalizedEvent:
    wall_time:  float
    logical_t:  int
    type:       str          # market | news | sentiment | macro
    payload:    Dict[str, Any]
    source:     str
    confidence: float = 1.0
    embedding:  Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def price(self)     -> Optional[float]: return self.payload.get("mid") or self.payload.get("close")
    @property
    def is_market(self) -> bool: return self.type == "market"
    @property
    def is_text(self)   -> bool: return self.type in {"news", "sentiment"}


# ── Per-type normalisers ──────────────────────────────────────────────────────

def _norm_market(p: dict) -> dict:
    mid    = float(p.get("mid",    0))
    bid    = float(p.get("bid",    mid))
    ask    = float(p.get("ask",    mid))
    return {
        "symbol":  p.get("symbol",   "XAU/USD"),
        "bid":     round(bid, 5),
        "ask":     round(ask, 5),
        "mid":     round((bid + ask) / 2, 5),
        "spread":  round(ask - bid, 5),
        "volume":  max(0.0, float(p.get("volume", 0) or 0)),
        "exchange": p.get("exchange", ""),
    }

def _norm_news(p: dict) -> dict:
    title = (p.get("title") or "").strip()[:512]
    desc  = (p.get("description") or "").strip()[:1024]
    text  = (p.get("text") or (title + " " + desc)).strip()
    return {
        "title":  title,
        "text":   text[:1024],
        "source": p.get("source", ""),
        "url":    p.get("url",    ""),
        "author": p.get("author", ""),
    }

def _norm_sentiment(p: dict) -> dict:
    title  = (p.get("title") or "").strip()[:512]
    body   = (p.get("text")  or "").strip()[:1024]
    text   = (title + " " + body).strip()
    return {
        "text":      text[:1024],
        "score":     int(p.get("score", 0) or 0),
        "subreddit": p.get("subreddit", ""),
        "likes":     int(p.get("likes",     0) or 0),
        "retweets":  int(p.get("retweets",  0) or 0),
    }

def _norm_macro(p: dict) -> dict:
    def _f(v):
        try: return float(v)
        except (TypeError, ValueError): return None

    value    = _f(p.get("value"))
    surprise = _f(p.get("surprise"))
    return {
        "series_id": p.get("series_id", p.get("event", "")),
        "event":     p.get("event",     ""),
        "country":   (p.get("country") or "").upper(),
        "value":     value,
        "surprise":  surprise,
        "impact":    p.get("impact",    "medium"),
        "title":     p.get("title",     ""),
        "units":     p.get("units",     ""),
    }

_NORMALIZERS = {
    "market":    _norm_market,
    "news":      _norm_news,
    "sentiment": _norm_sentiment,
    "macro":     _norm_macro,
}


# ── Online z-scaler ───────────────────────────────────────────────────────────

class RollingZScaler:
    def __init__(self, window: int = 200):
        self._window = window
        self._buf: List[float] = []

    def transform(self, value: float) -> float:
        self._buf.append(value)
        if len(self._buf) > self._window:
            self._buf.pop(0)
        arr    = np.array(self._buf, dtype=np.float32)
        mu, sd = arr.mean(), arr.std()
        return float((value - mu) / (sd + 1e-8))

    @property
    def mean(self) -> float: return float(np.mean(self._buf)) if self._buf else 0.0
    @property
    def std(self)  -> float: return float(np.std(self._buf))  if self._buf else 1.0


# ── Pipeline ──────────────────────────────────────────────────────────────────

class NormalizationPipeline:
    def __init__(self, raw_queue: asyncio.Queue, norm_queue: asyncio.Queue,
                 price_window: int = 200):
        self._raw     = raw_queue
        self._out     = norm_queue
        self._scaler  = RollingZScaler(price_window)
        self._stats:  Dict[str, int] = defaultdict(int)
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info("NormalizationPipeline started")
        while self._running:
            try:
                raw    = await asyncio.wait_for(self._raw.get(), timeout=1.0)
                normed = self._process(raw)
                if normed:
                    await self._out.put(normed)
                    self._stats[normed.type] += 1
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error("Normalization error: %s", exc, exc_info=True)

    def _process(self, raw) -> Optional[NormalizedEvent]:
        raw_type = raw.type.value if hasattr(raw.type, "value") else str(raw.type)
        norm_fn  = _NORMALIZERS.get(raw_type)
        if not norm_fn:
            logger.debug("Unknown event type: %s", raw_type)
            return None

        try:
            payload = norm_fn(raw.payload)
        except Exception as exc:
            logger.error("Normalization schema error [%s]: %s", raw_type, exc)
            return None

        if raw_type == "market":
            mid = payload.get("mid", 0)
            if mid > 0:
                payload["z_price"]     = self._scaler.transform(mid)
                payload["price_mean"]  = self._scaler.mean
                payload["price_std"]   = self._scaler.std

        lt = _clock.update(int(raw.timestamp * 1e3) % (2 ** 48))
        return NormalizedEvent(
            wall_time=raw.timestamp,
            logical_t=lt,
            type=raw_type,
            payload=payload,
            source=raw.source,
            confidence=raw.confidence,
        )

    async def stop(self) -> None:
        self._running = False

    @property
    def stats(self) -> dict:
        return dict(self._stats)
