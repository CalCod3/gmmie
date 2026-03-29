"""
delivery.py — Layer 11: Real-Time Delivery
===========================================
FastAPI server:
  GET  /health              — liveness + diagnostics
  GET  /api/state           — last intelligence snapshot (JSON)
  GET  /api/candles         — recent OHLCV candle history
  GET  /api/causal          — current causal graph weights
  GET  /api/diagnostics     — data-flow health (queue sizes, packet counts)
  WS   /ws/intelligence     — streaming intelligence packets

5-minute debug log written to Python logger.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Set

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    _FASTAPI = True
except ImportError:
    logger.error("FastAPI not installed: pip install fastapi uvicorn[standard]")
    _FASTAPI = False

try:
    import redis.asyncio as aioredis
    _REDIS = True
except ImportError:
    _REDIS = False

_DEBUG_INTERVAL_S = 300   # 5 minutes


# ── Intelligence packet ───────────────────────────────────────────────────────

@dataclass
class Candle:
    t:     float
    open:  float
    high:  float
    low:   float
    close: float
    vol:   float

    def to_dict(self) -> dict:
        return {"t": self.t, "o": self.open, "h": self.high,
                "l": self.low, "c": self.close, "v": self.vol}


class CandleBuilder:
    def __init__(self, resolution_s: int = 60, max_candles: int = 500):
        self._res     = resolution_s
        self._candles: deque = deque(maxlen=max_candles)
        self._current: Optional[Candle] = None

    def push_tick(self, price: float, volume: float, ts: float) -> Optional[Candle]:
        bucket = (int(ts) // self._res) * self._res
        completed = None
        if self._current is None:
            self._current = Candle(bucket, price, price, price, price, volume)
        elif bucket > self._current.t:
            completed = self._current
            self._candles.append(completed)
            self._current = Candle(bucket, price, price, price, price, volume)
        else:
            c = self._current
            c.high  = max(c.high, price)
            c.low   = min(c.low,  price)
            c.close = price
            c.vol  += volume
        return completed

    def get_candles(self, n: int = 200) -> list:
        closed = list(self._candles)[-n:]
        result = [c.to_dict() for c in closed]
        if self._current:
            result.append(self._current.to_dict())
        return result


class IntelligencePacket:
    __slots__ = [
        "price", "prediction", "world_state", "causal",
        "confidence", "regime", "memory", "attention",
        "latency_ms", "ts", "normalization_stats",
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__slots__:
                setattr(self, k, v)

    def to_dict(self) -> dict:
        return {k: getattr(self, k, None) for k in self.__slots__}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ── Connection manager ────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self._active: Set[WebSocket] = set()
        self._total_connected: int   = 0
        self._total_messages:  int   = 0

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)
        self._total_connected += 1
        logger.info("WS client connected — active: %d (total ever: %d)",
                    len(self._active), self._total_connected)

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)
        logger.info("WS client disconnected — active: %d", len(self._active))

    async def broadcast(self, message: str) -> None:
        dead = set()
        for ws in list(self._active):
            try:
                await ws.send_text(message)
                self._total_messages += 1
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._active.discard(ws)

    @property
    def n_clients(self) -> int:
        return len(self._active)

    @property
    def total_messages(self) -> int:
        return self._total_messages


# ── Redis bridge ──────────────────────────────────────────────────────────────

class RedisBridge:
    def __init__(self, redis_url: Optional[str] = None, channel: str = "output"):
        self._url     = redis_url or os.getenv("REDIS_URL", "redis://redis:6379")
        self._channel = channel
        self._client  = None

    async def connect(self) -> None:
        if not _REDIS:
            logger.warning("redis-py not installed — Redis bridge disabled")
            return

        for i in range(5):
            try:
                self._client = await aioredis.from_url(self._url, decode_responses=True)
                await self._client.ping()
                logger.info("Redis connected: %s", self._url)
                return
            except Exception as exc:
                logger.warning("Redis retry %d failed: %s", i + 1, exc)
                await asyncio.sleep(1)

        logger.error("Redis unavailable — continuing without it")
        self._client = None

    async def publish(self, packet: IntelligencePacket) -> None:
        if not self._client:
            return
        try:
            await self._client.publish(self._channel, packet.to_json())
        except Exception as exc:
            logger.debug("Redis publish error: %s", exc)


# ── FastAPI app ───────────────────────────────────────────────────────────────

if _FASTAPI:
    app = FastAPI(title="GMMIE Intelligence API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"],
        allow_methods=["*"], allow_headers=["*"],
    )

    manager        = ConnectionManager()
    redis_bridge   = RedisBridge()
    candle_builder = CandleBuilder(resolution_s=60, max_candles=500)

    _latest_packet: dict = {}
    _latest_causal: dict = {}
    _packets_received: int = 0
    _last_debug_ts: float  = 0.0

    @app.on_event("startup")
    async def _startup():
        await redis_bridge.connect()
        asyncio.create_task(_debug_log_loop(), name="delivery_debug_log")

    async def _debug_log_loop() -> None:
        """5-minute delivery-layer debug log."""
        global _last_debug_ts
        while True:
            await asyncio.sleep(_DEBUG_INTERVAL_S)
            logger.info(
                "[5-MIN DELIVERY DEBUG] packets_rx=%d ws_clients=%d ws_msgs_sent=%d "
                "has_data=%s last_price=%s last_ts=%s",
                _packets_received,
                manager.n_clients,
                manager.total_messages,
                bool(_latest_packet),
                _latest_packet.get("price", "none"),
                _latest_packet.get("ts", "none"),
            )

    @app.get("/health")
    async def health():
        return {
            "status":          "ok",
            "ws_clients":      manager.n_clients,
            "packets_received": _packets_received,
            "has_data":        bool(_latest_packet),
            "ts":              time.time(),
        }

    @app.get("/api/state")
    async def get_state():
        if not _latest_packet:
            return {"error": "no data yet — waiting for first tick"}
        return _latest_packet

    @app.get("/api/candles")
    async def get_candles(limit: int = 200):
        return {"candles": candle_builder.get_candles(n=limit)}

    @app.get("/api/causal")
    async def get_causal():
        return {"causal": _latest_causal, "ts": time.time()}

    @app.get("/api/diagnostics")
    async def get_diagnostics():
        return {
            "packets_received": _packets_received,
            "ws_clients":       manager.n_clients,
            "ws_messages_sent": manager.total_messages,
            "has_state":        bool(_latest_packet),
            "last_price":       _latest_packet.get("price"),
            "last_ts":          _latest_packet.get("ts"),
            "causal_edges":     len(_latest_causal),
            "ts":               time.time(),
        }

    @app.websocket("/ws/intelligence")
    async def ws_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            # Send snapshot of latest data immediately on connect
            if _latest_packet:
                await websocket.send_text(
                    json.dumps({**_latest_packet, "type": "snapshot"})
                )
            while True:
                # Keep-alive ping every 25s; actual data pushed via broadcast()
                await asyncio.sleep(25)
                await websocket.send_text(json.dumps({"type": "ping", "ts": time.time()}))
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception:
            manager.disconnect(websocket)

    async def push_packet(packet: IntelligencePacket) -> None:
        global _latest_packet, _latest_causal, _packets_received

        data           = packet.to_dict()
        _latest_packet = data
        _packets_received += 1

        if data.get("causal"):
            _latest_causal = data["causal"]

        price = data.get("price", 0)
        ts    = data.get("ts", time.time())
        if price:
            candle_builder.push_tick(price, 0, ts)

        msg = json.dumps({**data, "type": "tick"})
        await manager.broadcast(msg)
        await redis_bridge.publish(packet)

        # Log first packet arrival
        if _packets_received == 1:
            logger.info(
                "FIRST PACKET received — price=%.4f regime=%s conf=%.3f ws_clients=%d",
                data.get("price", 0),
                data.get("regime", "?"),
                data.get("confidence", 0),
                manager.n_clients,
            )


def run_server(host: str = "0.0.0.0", port: int = 8001) -> None:
    if not _FASTAPI:
        raise RuntimeError("FastAPI not installed")
    uvicorn.run("delivery:app", host=host, port=port, reload=False, log_level="warning")