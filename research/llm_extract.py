"""
research.llm_extract — LLM-extracted structured news features.

This is the single biggest free-leverage move in the project. Commercial systems
pay RavenPack / Bloomberg ~$50k/yr for approximately the same output. We do it
with prompt caching + cheap models + a content-hash cache for under $20 to
backfill 5 years.

Two backends:
  · `claude` — Anthropic API; uses prompt caching on the long system prompt
    so per-call cost is dominated by the headline tokens (cents per 1k items).
  · `ollama` — local Llama-3.1-8B-Instruct via the Ollama HTTP API; free.

Output schema is identical across backends.

Caching:
  · Per-headline content hash; results stored in the lake.
  · System-prompt cache key is the prompt SHA so prompt changes invalidate.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from data_lake.db import LakeDB

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior gold-market analyst. For each news headline
you receive, return a single JSON object — nothing else — with these keys:

{
  "event_type":      one of [
                        "fed_policy", "cpi_release", "geopolitical",
                        "central_bank_buying", "etf_flow", "physical_demand",
                        "mining_supply", "dollar_move", "yields_move",
                        "risk_sentiment", "other"
                     ],
  "gold_impact":     float in [-1.0, 1.0],
  "confidence":      float in [0.0, 1.0],
  "horizon_minutes": integer >= 5,
  "novel":           boolean,
  "surprise_signed": float in [-1.0, 1.0],
  "rationale":       string <= 200 chars
}

Sign convention: gold_impact > 0 → gold likely to rise. Rate hike, dollar
strength, real-yield rise → impact < 0. Geopolitical escalation, dovish Fed,
weak USD, central-bank buying → impact > 0. Rehashes of older stories →
novel=false, low confidence.

Examples:

Headline: "Fed signals possible 50bp rate hike at next meeting amid sticky inflation"
{"event_type":"fed_policy","gold_impact":-0.55,"confidence":0.82,
 "horizon_minutes":2880,"novel":true,"surprise_signed":-0.4,
 "rationale":"Hawkish surprise; +rates → higher real yields → gold bearish"}

Headline: "PBoC adds 22 tonnes of gold to official reserves in November"
{"event_type":"central_bank_buying","gold_impact":0.45,"confidence":0.78,
 "horizon_minutes":7200,"novel":true,"surprise_signed":0.2,
 "rationale":"Structural central-bank bid; trend reinforces gold floor"}

Headline: "Spot gold trades flat as markets await Fed minutes"
{"event_type":"risk_sentiment","gold_impact":0.0,"confidence":0.3,
 "horizon_minutes":120,"novel":false,"surprise_signed":0.0,
 "rationale":"No new information; market in wait-mode"}

Headline: "Apple unveils new iPhone with bigger screen"
{"event_type":"other","gold_impact":0.0,"confidence":0.0,
 "horizon_minutes":5,"novel":false,"surprise_signed":0.0,
 "rationale":"Irrelevant to gold"}
"""

MODEL_VERSION = "v1-2026-06"   # bump when SYSTEM_PROMPT changes


@dataclass
class NewsExtraction:
    news_id:         str
    model_version:   str
    extracted_at:    dt.datetime
    event_type:      str
    gold_impact:     float
    confidence:      float
    horizon_minutes: int
    novel:           bool
    surprise_signed: float
    rationale:       str


# ── Backends ─────────────────────────────────────────────────────────────────

class _Backend:
    name: str = "base"
    def extract(self, text: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class ClaudeBackend(_Backend):
    """
    Anthropic SDK with prompt caching. The system prompt is cacheable so each
    headline costs only output + headline-input tokens (~50 + 100 tokens).
    Use claude-haiku-4-5 for cost — ~$1/1M output tokens.
    """
    name = "claude"

    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 api_key: Optional[str] = None):
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic SDK not installed") from exc
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model  = model

    def extract(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=400,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {"role": "user", "content": f"Headline: {text[:500]}"}
                ],
            )
            raw = msg.content[0].text.strip()
            return _safe_json(raw)
        except Exception as exc:
            logger.warning("Claude extract failed: %s", exc)
            return None


class OllamaBackend(_Backend):
    """Local Llama-3.1-8B via Ollama HTTP API. Free; ~50–200ms per headline."""
    name = "ollama"

    def __init__(self, model: str = "llama3.1:8b-instruct-q4_K_M",
                 base_url: str = "http://localhost:11434"):
        self._model = model
        self._base  = base_url

    def extract(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            r = httpx.post(
                f"{self._base}/api/chat",
                timeout=60.0,
                json={
                    "model": self._model,
                    "stream": False,
                    "format": "json",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f"Headline: {text[:500]}"},
                    ],
                    "options": {"temperature": 0.0, "num_predict": 400},
                },
            )
            r.raise_for_status()
            raw = r.json()["message"]["content"]
            return _safe_json(raw)
        except Exception as exc:
            logger.warning("Ollama extract failed: %s", exc)
            return None


def _safe_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    # Strip code fences if a model insisted on them
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find the first {...} substring
        i, j = raw.find("{"), raw.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(raw[i:j + 1])
            except Exception:
                return None
    return None


# ── Orchestrator ─────────────────────────────────────────────────────────────

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def load_pending(db: LakeDB, model_version: str = MODEL_VERSION,
                 limit: int = 1000) -> List[Tuple[str, str]]:
    """News rows that have not been extracted for this model version."""
    df = db.df(
        """
        SELECT n.id, COALESCE(n.title, '') || ' ' || COALESCE(n.text, '') AS body
        FROM news n
        LEFT JOIN news_features f
          ON f.news_id = n.id AND f.model_version = ?
        WHERE f.news_id IS NULL
        ORDER BY n.ts DESC
        LIMIT ?
        """,
        [model_version, limit],
    )
    return list(zip(df["id"], df["body"]))


def extract_and_store(db: LakeDB, backend: _Backend,
                      *, batch: int = 100, throttle_s: float = 0.0) -> int:
    pending = load_pending(db, MODEL_VERSION, limit=batch)
    if not pending:
        return 0
    rows = []
    for news_id, body in pending:
        out = backend.extract(body)
        if not out:
            continue
        rows.append((
            news_id, MODEL_VERSION, dt.datetime.utcnow(),
            str(out.get("event_type", "other"))[:32],
            _clip(out.get("gold_impact"), -1, 1),
            _clip(out.get("confidence"), 0, 1),
            int(max(5, min(7 * 24 * 60, out.get("horizon_minutes", 60) or 60))),
            bool(out.get("novel", False)),
            _clip(out.get("surprise_signed"), -1, 1),
            str(out.get("rationale", ""))[:240],
        ))
        if throttle_s:
            time.sleep(throttle_s)

    n = db.upsert(
        "news_features",
        ["news_id", "model_version", "extracted_at",
         "event_type", "gold_impact", "confidence",
         "horizon_minutes", "novel", "surprise_signed", "rationale"],
        rows,
        conflict_key=("news_id", "model_version"),
    )
    logger.info("LLM extract: %d/%d succeeded via %s", n, len(pending), backend.name)
    return n


def _clip(v, lo, hi):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return float(max(lo, min(hi, x)))


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    import argparse, sys
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    p = argparse.ArgumentParser(description="LLM-extract structured news features")
    p.add_argument("--backend", choices=["claude", "ollama"], default="ollama")
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--model",
                   help="model id (claude-haiku-4-5-... or llama3.1:8b-...)")
    p.add_argument("--loop", action="store_true",
                   help="keep extracting until no pending rows remain")
    args = p.parse_args(argv)

    backend: _Backend
    if args.backend == "claude":
        backend = ClaudeBackend(model=args.model or "claude-haiku-4-5-20251001")
    else:
        backend = OllamaBackend(model=args.model or "llama3.1:8b-instruct-q4_K_M")

    with LakeDB() as db:
        if args.loop:
            total = 0
            while True:
                n = extract_and_store(db, backend, batch=args.batch)
                total += n
                if n == 0:
                    break
            logger.info("Total extracted: %d", total)
        else:
            extract_and_store(db, backend, batch=args.batch)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
