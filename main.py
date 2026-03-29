"""
main.py — GMMIE Engine Orchestrator
=====================================
Wires all layers into a single async event loop.

BUGS FIXED:
  1. relative import `from .delivery` → `from delivery` (was killing every packet silently)
  2. feat_type "event" (returned by features.py macro path) never matched "macro" check —
     normalised to "macro" immediately after feature extraction.

Run:
    python main.py
    uvicorn delivery:app --host 0.0.0.0 --port 8001  # API only
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import numpy as np

from config import CONFIG
from ingestion import IngestionOrchestrator
from normalization import NormalizationPipeline
from features import FeaturePipeline
from world_state import MultiModalEmbeddingLayer, NSSMWorldStateModel
from engines import (
    OnlineGrangerCausal,
    TFTForecastEngine,
    MetaLearningEngine,
    PatternMemory,
)

# FIX 1: module-level import, NOT relative — avoids ImportError on every packet
from delivery import IntelligencePacket

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=CONFIG.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

_DEBUG_INTERVAL_S = 300   # 5 minutes


# ── Attention allocation ──────────────────────────────────────────────────────

class AttentionAllocationEngine:
    MODALITIES = ["market", "macro", "sentiment", "events"]

    def __init__(self):
        self._errors  = {m: 0.1 for m in self.MODALITIES}
        self._weights = {m: 1.0 / len(self.MODALITIES) for m in self.MODALITIES}

    def update(self, modality: str, error: float) -> None:
        if modality in self._errors:
            self._errors[modality] = 0.9 * self._errors[modality] + 0.1 * abs(error)

    def get_weights(self) -> dict:
        inv = np.array(
            [1.0 / (self._errors[m] + 1e-6) for m in self.MODALITIES], dtype=np.float32
        )
        exp = np.exp(inv - inv.max())
        w   = exp / exp.sum()
        self._weights = {m: float(v) for m, v in zip(self.MODALITIES, w)}
        return self._weights


# ── Regime classifier ─────────────────────────────────────────────────────────

def classify_regime(ws: dict) -> str:
    risk     = ws.get("risk_regime",      0.5)
    vol      = ws.get("volatility_state", 0.5)
    mom      = ws.get("momentum_short",   0.5)
    mom_long = ws.get("momentum_long",    0.5)
    liq      = ws.get("liquidity",        0.5)
    mr       = ws.get("mean_reversion",   0.5)

    if risk > 0.65 and vol < 0.4 and liq > 0.5: return "RISK-ON"
    if risk < 0.35 or vol > 0.7:                 return "RISK-OFF"
    if mom > 0.6 and mom_long > 0.55:            return "TRENDING"
    if mr > 0.6:                                  return "RANGING"
    return "RANGING"


# ── Main engine ───────────────────────────────────────────────────────────────

class GMMIEEngine:
    def __init__(self, config=None):
        self.cfg = config or CONFIG

        warnings = self.cfg.validate()
        for w in warnings:
            logger.warning("CONFIG WARNING: %s", w)

        self._raw_q:  asyncio.Queue = asyncio.Queue(maxsize=100_000)
        self._norm_q: asyncio.Queue = asyncio.Queue(maxsize=100_000)

        self.ingestion     = IngestionOrchestrator(self.cfg)
        self.normalization = NormalizationPipeline(self._raw_q, self._norm_q)
        self.features      = FeaturePipeline(self.cfg)
        self.embeddings    = MultiModalEmbeddingLayer(self.cfg)
        self.world_state   = NSSMWorldStateModel(
            input_dim=self.cfg.embeddings.fused_dim,
            latent_dim=self.cfg.world_state.latent_dim,
            hidden_dim=self.cfg.world_state.hidden_dim,
        )
        self.causal    = OnlineGrangerCausal()
        self.forecaster = TFTForecastEngine(horizons=self.cfg.forecast.horizons)
        self.meta       = MetaLearningEngine(self.cfg)
        self.memory     = PatternMemory(self.cfg)
        self.attention  = AttentionAllocationEngine()

        self._last_price:    float = 0.0
        self._last_ws:       dict  = {}
        self._last_forecast: dict  = {}
        self._last_causal:   dict  = {}
        self._tick_count:    int   = 0
        self._packets_sent:  int   = 0
        self._running:       bool  = False
        self._push_packet           = None

        # Per-type event counters for diagnostics
        self._ev_counts = {"market": 0, "text": 0, "macro": 0, "skipped": 0}
        self._last_debug_ts: float = 0.0

    def set_delivery_callback(self, fn) -> None:
        self._push_packet = fn

    async def run(self) -> None:
        self._running = True
        logger.info("GMMIE Engine starting…")
        await self.ingestion.start()

        norm_task   = asyncio.create_task(self.normalization.run(), name="normalization")
        bridge_task = asyncio.create_task(
            self._bridge_queues(self.ingestion.queue, self._raw_q), name="queue_bridge"
        )
        try:
            await self._process_loop()
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            await self.ingestion.stop()
            await self.normalization.stop()
            norm_task.cancel()
            bridge_task.cancel()
            self.memory.save()
            logger.info(
                "GMMIE Engine stopped. Ticks: %d | Packets sent: %d",
                self._tick_count, self._packets_sent,
            )

    async def _bridge_queues(self, src: asyncio.Queue, dst: asyncio.Queue) -> None:
        while self._running:
            try:
                item = await asyncio.wait_for(src.get(), timeout=1.0)
                await dst.put(item)
            except asyncio.TimeoutError:
                continue

    async def _process_loop(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._norm_q.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                self._maybe_emit_debug_log()
                continue
            except Exception as exc:
                logger.error("Processing error: %s", exc, exc_info=True)

    # ── 5-minute debug heartbeat ──────────────────────────────────────────────

    def _maybe_emit_debug_log(self) -> None:
        now = time.monotonic()
        if now - self._last_debug_ts < _DEBUG_INTERVAL_S:
            return
        self._last_debug_ts = now

        ws_clients = "?"
        try:
            from delivery import manager
            ws_clients = str(manager.n_clients)
        except Exception:
            pass

        logger.info(
            "[5-MIN BACKEND DEBUG] "
            "raw_q=%d norm_q=%d | events={market:%d text:%d macro:%d skip:%d} | "
            "market_ticks=%d packets_sent=%d | last_price=%.4f | ws_clients=%s | "
            "norm_stats=%s",
            self._raw_q.qsize(),
            self._norm_q.qsize(),
            self._ev_counts["market"],
            self._ev_counts["text"],
            self._ev_counts["macro"],
            self._ev_counts["skipped"],
            self._tick_count,
            self._packets_sent,
            self._last_price,
            ws_clients,
            self.normalization.stats,
        )

    # ── Core event processor ──────────────────────────────────────────────────

    async def _process_event(self, event) -> None:
        t0 = time.perf_counter()

        feat_result = self.features.process(event)
        if feat_result is None:
            self._ev_counts["skipped"] += 1
            return

        feat_type, feat_obj, feat_vec = feat_result

        # FIX 2: features.py macro path returns ("event", ...) but pipeline checks
        # for "macro" — normalise immediately so routing is consistent.
        if feat_type == "event":
            feat_type = "macro"

        if feat_type == "market":
            self._ev_counts["market"] += 1
            self.embeddings.update_market(feat_vec)
            price = feat_obj.price
            if price > 0:
                self._last_price = price
                self.causal.push("GOLD", price)
        elif feat_type == "text":
            self._ev_counts["text"] += 1
            self.embeddings.update_text(feat_vec)
            self.causal.push("SENTIMENT", feat_obj.sentiment_score)
            self.attention.update("sentiment", 1 - abs(feat_obj.sentiment_score))
        elif feat_type == "macro":
            self._ev_counts["macro"] += 1
            self.embeddings.update_macro(feat_vec)
            self.attention.update("macro", 1 - feat_obj.impact_score)
            if hasattr(feat_obj, "series_id"):
                sid = feat_obj.series_id
                val = feat_obj.value
                if val is not None:
                    if "DGS10" in sid or "DGS2" in sid:
                        self.causal.push("YIELDS", val)
                    elif "DTWEXBGS" in sid or "USD" in sid:
                        self.causal.push("DXY", val)
                    elif "VIXCLS" in sid:
                        self.causal.push("VIX", val)
                    elif "DCOILWTICO" in sid:
                        self.causal.push("OIL", val)
                    elif "FEDFUNDS" in sid:
                        self.causal.push("FED", val)
                    elif "CPIAUCSL" in sid or "T10YIE" in sid:
                        self.causal.push("INFLATION", val)

        # Full pipeline only on market ticks
        if feat_type != "market":
            return

        fused = self.embeddings.get_fused()
        if fused is None:
            return

        latent  = self.world_state.update(fused)
        ws_dict = self.world_state.get_world_state()
        self._last_ws = ws_dict

        self._last_causal = self.causal.update()
        regime = classify_regime(ws_dict)

        vol = feat_obj.vol_20 if hasattr(feat_obj, "vol_20") else 1.0
        self.forecaster.push(self._last_price, vol, latent)
        forecast = self.forecaster.predict()
        self._last_forecast = forecast

        confidence   = self.meta.get_confidence(latent, regime)
        attn_weights = self.attention.get_weights()
        similar      = self.memory.query(fused)

        if self._tick_count % 100 == 0 and self._tick_count > 0:
            self.memory.store(
                fused,
                label=f"tick_{self._tick_count}",
                outcome=0.0,
                regime=regime,
                timestamp=event.wall_time,
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        self._tick_count += 1

        if self._push_packet:
            try:
                packet = IntelligencePacket(
                    price=round(self._last_price, 5),
                    prediction={str(h): v for h, v in forecast.items()},
                    world_state={k: round(v, 4) for k, v in ws_dict.items()},
                    causal={k: round(v, 4) for k, v in self._last_causal.items()},
                    confidence=round(confidence, 4),
                    regime=regime,
                    memory=[
                        {
                            "label":      m.get("label", ""),
                            "similarity": round(m.get("similarity", 0), 4),
                            "outcome":    m.get("outcome", 0),
                            "regime":     m.get("regime", ""),
                        }
                        for m in similar[:3]
                    ],
                    attention={k: round(v, 4) for k, v in attn_weights.items()},
                    latency_ms=round(latency_ms, 3),
                    ts=event.wall_time,
                    normalization_stats=self.normalization.stats,
                )
                await self._push_packet(packet)
                self._packets_sent += 1
            except Exception as exc:
                logger.error("Delivery error: %s", exc, exc_info=True)

        if self._tick_count % 500 == 0:
            logger.info(
                "Tick %d | Price=%.4f | Regime=%s | Conf=%.2f | Lat=%.2fms | "
                "Sources=%s | Packets=%d",
                self._tick_count, self._last_price, regime, confidence, latency_ms,
                dict(self.normalization.stats), self._packets_sent,
            )

        # Trigger 5-min debug if enough time has passed (market-tick path)
        self._maybe_emit_debug_log()


# ── Entry point ───────────────────────────────────────────────────────────────

async def _run_server_safe(app, host: str, port: int) -> None:
    import uvicorn
    for attempt_port in range(port, port + 10):
        config = uvicorn.Config(app, host=host, port=attempt_port, log_level="warning")
        server = uvicorn.Server(config)
        try:
            if attempt_port != port:
                logger.warning("Port %d unavailable — trying %d", attempt_port - 1, attempt_port)
            logger.info(
                "API: http://%s:%d  |  WS: ws://%s:%d/ws/intelligence",
                host, attempt_port, host, attempt_port,
            )
            await server.serve()
            return
        except (SystemExit, OSError) as exc:
            code = exc.code if hasattr(exc, "code") else exc.errno
            logger.error("Server failed on port %d (%s).", attempt_port, code)
            continue
    logger.critical("Could not bind to any port in %d-%d.", port, port + 9)


async def main():
    from delivery import push_packet as _push, app

    engine = GMMIEEngine()
    engine.set_delivery_callback(_push)

    cfg = engine.cfg.delivery
    await asyncio.gather(
        engine.run(),
        _run_server_safe(app, cfg.host, cfg.port),
    )


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())