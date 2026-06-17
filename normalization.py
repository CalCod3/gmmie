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
    """
    Hybrid online scaler:
      · O(1) per update using EWMA mean / variance (no full-buffer recompute)
      · Tracks rolling MAD (median absolute deviation) on a bounded deque so
        the scaler is robust to fat-tailed shocks (e.g. flash crashes,
        FOMC prints) which would inflate a Welford σ and depress z-scores.

    Returns the *robust* z-score by default: (x − median) / (1.4826·MAD).
    Falls back to EWMA-σ when the MAD buffer is cold (< 30 samples).
    """

    _MAD_NORM = 1.4826    # MAD → Normal-consistent σ
    _ROBUST_CLIP = 8.0    # clip extreme z (anti winner-takes-all)

    def __init__(self, window: int = 200, ewma_alpha: float = 0.02):
        from collections import deque
        self._window = window
        self._buf:    deque = deque(maxlen=window)
        self._alpha  = ewma_alpha
        self._mu_ewma: float = 0.0
        self._var_ewma: float = 1.0
        self._n: int = 0

    def transform(self, value: float) -> float:
        x = float(value)

        # EWMA mean / variance (O(1))
        if self._n == 0:
            self._mu_ewma = x
            self._var_ewma = 1.0
        else:
            d = x - self._mu_ewma
            self._mu_ewma  = self._mu_ewma + self._alpha * d
            self._var_ewma = (1 - self._alpha) * (self._var_ewma + self._alpha * d * d)
        self._n += 1
        self._buf.append(x)

        # Robust scaling once we have enough samples
        if len(self._buf) >= 30:
            arr = np.fromiter(self._buf, dtype=np.float64, count=len(self._buf))
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            sigma = self._MAD_NORM * mad if mad > 1e-12 else np.sqrt(self._var_ewma)
            z = (x - med) / (sigma + 1e-12)
        else:
            z = (x - self._mu_ewma) / (np.sqrt(self._var_ewma) + 1e-12)

        return float(np.clip(z, -self._ROBUST_CLIP, self._ROBUST_CLIP))

    @property
    def mean(self) -> float:
        if not self._buf:
            return 0.0
        return float(np.median(np.fromiter(self._buf, dtype=np.float64, count=len(self._buf))))

    @property
    def std(self) -> float:
        if len(self._buf) < 2:
            return 1.0
        arr = np.fromiter(self._buf, dtype=np.float64, count=len(self._buf))
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return float(self._MAD_NORM * mad) if mad > 1e-12 else float(np.sqrt(self._var_ewma))


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
