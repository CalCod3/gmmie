"""
Shared HTTP helpers for source ingesters: retry with exponential backoff,
respect-Retry-After-on-429, jittered sleep.

A single 429 from FRED used to abort the whole backfill in v1. This helper
turns transient failures into a 30–60s pause and a retry.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


def get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    max_attempts: int = 5,
    base_backoff_s: float = 1.5,
    max_backoff_s:  float = 60.0,
) -> httpx.Response:
    """GET with exponential backoff. Honors `Retry-After` on 429/503."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = client.get(url, params=params, headers=headers)
            if r.status_code in (429, 503):
                wait = _wait_from_response(r) or _backoff(attempt, base_backoff_s, max_backoff_s)
                logger.warning(
                    "HTTP %d on %s (attempt %d/%d) — sleeping %.1fs",
                    r.status_code, url, attempt, max_attempts, wait,
                )
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = _backoff(attempt, base_backoff_s, max_backoff_s)
                logger.warning("HTTP %d (5xx) — sleeping %.1fs", r.status_code, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except httpx.RequestError as exc:
            last_exc = exc
            wait = _backoff(attempt, base_backoff_s, max_backoff_s)
            logger.warning(
                "request error (%s) on %s attempt %d/%d — sleeping %.1fs",
                type(exc).__name__, url, attempt, max_attempts, wait,
            )
            time.sleep(wait)
    if last_exc:
        raise last_exc
    raise httpx.HTTPError(f"exhausted {max_attempts} attempts on {url}")


def _backoff(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with full jitter."""
    upper = min(cap, base * (2 ** (attempt - 1)))
    return random.uniform(0.0, upper)


def _wait_from_response(r: httpx.Response) -> Optional[float]:
    h = r.headers.get("Retry-After")
    if not h:
        return None
    try:
        return float(h)
    except (TypeError, ValueError):
        return None
