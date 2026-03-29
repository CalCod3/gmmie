"""
ingestion.py — Layer 1: Real-Time Data Ingestion
=================================================
All sources produce RawEvent and push to an asyncio.Queue.
No demo fallbacks. Sources that lack credentials log a warning and skip.

Sources:
  Market  — Twelve Data WebSocket (primary) | Alpha Vantage poll | OANDA stream
  News    — NewsAPI poll | GDELT 2.0 DOC API
  Social  — asyncpraw (Reddit) | Twitter API v2
  Macro   — FRED API (St. Louis Fed) — free, official
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    MARKET    = "market"
    NEWS      = "news"
    SENTIMENT = "sentiment"
    MACRO     = "macro"


@dataclass
class RawEvent:
    timestamp:  float
    type:       EventType
    payload:    Dict[str, Any]
    source:     str
    confidence: float = 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseIngester:
    def __init__(self, queue: asyncio.Queue, source_name: str):
        self.queue       = queue
        self.source_name = source_name
        self._running    = False

    async def start(self) -> None:
        self._running = True
        logger.info("Ingester started: %s", self.source_name)
        try:
            async for event in self._stream():
                await self.queue.put(event)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Ingester %s fatal error: %s", self.source_name, exc, exc_info=True)
        finally:
            self._running = False
            logger.info("Ingester stopped: %s", self.source_name)

    async def stop(self) -> None:
        self._running = False

    async def _stream(self) -> AsyncGenerator[RawEvent, None]:
        raise NotImplementedError
        yield


# ── 1a. Twelve Data WebSocket ─────────────────────────────────────────────────

class TwelveDataIngester(BaseIngester):
    """
    Real-time XAUUSD tick stream via Twelve Data WebSocket.
    Docs: https://twelvedata.com/docs#websocket-price
    Message format: {"event":"price","symbol":"XAU/USD","currency":"USD",
                      "exchange":"...","type":"...","timestamp":...,
                      "price":2345.12,"day_volume":...}
    """

    def __init__(self, queue: asyncio.Queue, api_key: str,
                 symbols: list = None, ws_url: str = "wss://ws.twelvedata.com/v1/quotes/price"):
        super().__init__(queue, "twelve_data")
        self._api_key = api_key
        self._symbols = symbols or ["XAU/USD"]
        self._ws_url  = ws_url

    async def _stream(self) -> AsyncGenerator[RawEvent, None]:
        if not self._api_key:
            logger.error("TWELVE_DATA_API_KEY not set — TwelveData ingester disabled")
            return

        import websockets

        url = f"{self._ws_url}?apikey={self._api_key}"
        backoff = 1.0

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    backoff = 1.0
                    # Subscribe
                    sub = json.dumps({"action": "subscribe", "params": {"symbols": ",".join(self._symbols)}})
                    await ws.send(sub)
                    logger.info("TwelveData: subscribed to %s", self._symbols)

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                        except json.JSONDecodeError:
                            continue

                        event_type = msg.get("event", "")
                        if event_type == "price":
                            yield RawEvent(
                                timestamp=float(msg.get("timestamp", time.time())),
                                type=EventType.MARKET,
                                payload={
                                    "symbol":    msg.get("symbol", "XAU/USD"),
                                    "mid":       float(msg.get("price", 0)),
                                    "bid":       float(msg.get("bid",   msg.get("price", 0))),
                                    "ask":       float(msg.get("ask",   msg.get("price", 0))),
                                    "spread":    float(msg.get("ask", 0)) - float(msg.get("bid", 0)),
                                    "volume":    float(msg.get("day_volume", 0) or 0),
                                    "exchange":  msg.get("exchange", ""),
                                },
                                source="twelve_data",
                            )
                        elif event_type == "heartbeat":
                            pass
                        elif event_type == "subscribe-status":
                            status = msg.get("status", "")
                            if status != "ok":
                                logger.warning("TwelveData subscribe status: %s | %s", status, msg.get("message", ""))

            except Exception as exc:
                logger.warning("TwelveData WS error: %s — reconnecting in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)


# ── 1b. Alpha Vantage polling (fallback) ──────────────────────────────────────

class AlphaVantageIngester(BaseIngester):
    """
    Polls Alpha Vantage CURRENCY_EXCHANGE_RATE for XAU/USD.
    Free tier: 25 requests/day. Use as fallback only.
    Docs: https://www.alphavantage.co/documentation/#currency-exchange
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, queue: asyncio.Queue, api_key: str, poll_interval: int = 60):
        super().__init__(queue, "alpha_vantage")
        self._api_key = api_key
        self._poll_s  = poll_interval

    async def _stream(self) -> AsyncGenerator[RawEvent, None]:
        if not self._api_key:
            logger.error("ALPHA_VANTAGE_API_KEY not set — AlphaVantage ingester disabled")
            return

        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    params = {
                        "function":    "CURRENCY_EXCHANGE_RATE",
                        "from_currency": "XAU",
                        "to_currency":   "USD",
                        "apikey":      self._api_key,
                    }
                    async with session.get(self.BASE_URL, params=params,
                                           timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        data = await resp.json(content_type=None)
                    rate = data.get("Realtime Currency Exchange Rate", {})
                    if not rate:
                        note = data.get("Note", data.get("Information", ""))
                        if note:
                            logger.warning("AlphaVantage rate-limit: %s", note[:120])
                    else:
                        price = float(rate.get("5. Exchange Rate", 0))
                        bid   = float(rate.get("8. Bid Price", price))
                        ask   = float(rate.get("9. Ask Price", price))
                        yield RawEvent(
                            timestamp=time.time(),
                            type=EventType.MARKET,
                            payload={
                                "symbol": "XAU/USD",
                                "mid":    price,
                                "bid":    bid,
                                "ask":    ask,
                                "spread": ask - bid,
                                "volume": 0.0,
                            },
                            source="alpha_vantage",
                            confidence=0.9,
                        )
                except Exception as exc:
                    logger.error("AlphaVantage error: %s", exc)

                await asyncio.sleep(self._poll_s)


# ── 1c. OANDA v20 Streaming ───────────────────────────────────────────────────

class OANDAIngester(BaseIngester):
    """
    Real-time XAUUSD streaming via OANDA v20 REST streaming endpoint.
    Docs: https://developer.oanda.com/rest-live-v20/pricing-df/
    """

    STREAM_URLS = {
        "practice": "https://stream-fxpractice.oanda.com",
        "live":     "https://stream-fxtrade.oanda.com",
    }

    def __init__(self, queue: asyncio.Queue, account_id: str,
                 access_token: str, environment: str = "practice"):
        super().__init__(queue, "oanda")
        self._account_id   = account_id
        self._access_token = access_token
        self._base_url     = self.STREAM_URLS.get(environment, self.STREAM_URLS["practice"])

    async def _stream(self) -> AsyncGenerator[RawEvent, None]:
        if not self._account_id or not self._access_token:
            logger.error("OANDA credentials not set — OANDA ingester disabled")
            return

        url = f"{self._base_url}/v3/accounts/{self._account_id}/pricing/stream"
        headers = {
            "Authorization":  f"Bearer {self._access_token}",
            "Accept-Datetime-Format": "UNIX",
        }
        params = {"instruments": "XAU_USD"}
        backoff = 1.0

        while self._running:
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(url, params=params,
                                           timeout=aiohttp.ClientTimeout(total=None)) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            logger.error("OANDA stream HTTP %d: %s", resp.status, body[:200])
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, 60)
                            continue

                        backoff = 1.0
                        async for line in resp.content:
                            if not self._running:
                                break
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                msg = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            if msg.get("type") == "PRICE":
                                bids = msg.get("bids", [{}])
                                asks = msg.get("asks", [{}])
                                bid  = float(bids[0].get("price", 0)) if bids else 0.0
                                ask  = float(asks[0].get("price", 0)) if asks else 0.0
                                mid  = (bid + ask) / 2
                                yield RawEvent(
                                    timestamp=float(msg.get("time", time.time())),
                                    type=EventType.MARKET,
                                    payload={
                                        "symbol": "XAU/USD",
                                        "bid":    bid,
                                        "ask":    ask,
                                        "mid":    mid,
                                        "spread": ask - bid,
                                        "volume": 0.0,
                                        "tradeable": msg.get("tradeable", True),
                                    },
                                    source="oanda",
                                )
            except Exception as exc:
                logger.warning("OANDA stream error: %s — reconnecting in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)


# ── 2a. NewsAPI ───────────────────────────────────────────────────────────────

class NewsAPIIngester(BaseIngester):
    """
    Polls NewsAPI /v2/everything for gold/macro headlines.
    Docs: https://newsapi.org/docs/endpoints/everything
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, queue: asyncio.Queue, api_key: str,
                 query: str = "", poll_interval: int = 60):
        super().__init__(queue, "newsapi")
        self._api_key = api_key
        self._query   = query or 'gold OR XAUUSD OR "Federal Reserve" OR CPI OR inflation'
        self._poll_s  = poll_interval
        self._seen:   set = set()

    async def _stream(self) -> AsyncGenerator[RawEvent, None]:
        if not self._api_key:
            logger.error("NEWSAPI_KEY not set — NewsAPI ingester disabled")
            return

        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    params = {
                        "q":        self._query,
                        "sortBy":   "publishedAt",
                        "pageSize": 50,
                        "language": "en",
                        "apiKey":   self._api_key,
                    }
                    async with session.get(self.BASE_URL, params=params,
                                           timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        data = await resp.json(content_type=None)

                    if data.get("status") != "ok":
                        logger.warning("NewsAPI error: %s", data.get("message", "unknown"))
                    else:
                        for art in data.get("articles", []):
                            art_id = art.get("url", "")
                            if art_id in self._seen:
                                continue
                            self._seen.add(art_id)
                            yield RawEvent(
                                timestamp=_parse_iso(art.get("publishedAt", "")),
                                type=EventType.NEWS,
                                payload={
                                    "title":       (art.get("title") or "").strip()[:512],
                                    "description": (art.get("description") or "").strip()[:1024],
                                    "text":        " ".join(filter(None, [
                                                       art.get("title", ""),
                                                       art.get("description", ""),
                                                   ])).strip(),
                                    "source":      (art.get("source") or {}).get("name", ""),
                                    "url":         art.get("url", ""),
                                    "author":      art.get("author", ""),
                                },
                                source="newsapi",
                                confidence=0.85,
                            )
                except Exception as exc:
                    logger.error("NewsAPI error: %s", exc)

                await asyncio.sleep(self._poll_s)


# ── 2b. GDELT 2.0 DOC API ────────────────────────────────────────────────────

class GDELTIngester(BaseIngester):
    """
    Polls GDELT 2.0 DOC API — free, no key required.
    Returns last 15 minutes of news articles matching the query.
    Docs: https://blog.gdeltproject.org/gdelt-2-0-our-global-database-of-society/
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, queue: asyncio.Queue, query: str = "", poll_interval: int = 300):
        super().__init__(queue, "gdelt")
        self._query  = query or "gold Federal Reserve inflation"
        self._poll_s = poll_interval
        self._seen:  set = set()

    async def _stream(self) -> AsyncGenerator[RawEvent, None]:
        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    params = {
                        "query":   self._query,
                        "mode":    "artlist",
                        "maxrecords": 50,
                        "timespan": "24h",
                        "format":  "json",
                        "sourcelang": "english",
                    }
                    async with session.get(self.BASE_URL, params=params,
                                           timeout=aiohttp.ClientTimeout(total=20)) as resp:
                        if resp.status != 200:
                            logger.warning("GDELT API returned status %d", resp.status)
                            await asyncio.sleep(self._poll_s)
                            continue
                        
                        # Check if response has content before parsing
                        text = await resp.text()
                        if not text or not text.strip():
                            logger.warning("GDELT API returned empty response")
                            await asyncio.sleep(self._poll_s)
                            continue
                        
                        try:
                            data = json.loads(text)
                        except json.JSONDecodeError as je:
                            logger.warning("GDELT JSON decode error: %s (response: %s...)", je, text[:100])
                            await asyncio.sleep(self._poll_s)
                            continue

                    for art in data.get("articles", []):
                        url = art.get("url", "")
                        if url in self._seen:
                            continue
                        self._seen.add(url)
                        seendate = art.get("seendate", "")
                        yield RawEvent(
                            timestamp=_parse_gdelt_date(seendate),
                            type=EventType.NEWS,
                            payload={
                                "title":  (art.get("title") or "").strip()[:512],
                                "text":   (art.get("title") or "").strip(),
                                "source": art.get("domain", ""),
                                "url":    url,
                                "language": art.get("language", ""),
                            },
                            source="gdelt",
                            confidence=0.70,
                        )
                except Exception as exc:
                    logger.error("GDELT error: %s", exc)

                await asyncio.sleep(self._poll_s)


# ── 3a. Reddit via asyncpraw ──────────────────────────────────────────────────

class RedditIngester(BaseIngester):
    """
    Streams new posts from finance/gold subreddits via asyncpraw.
    Docs: https://asyncpraw.readthedocs.io/
    """

    def __init__(self, queue: asyncio.Queue, client_id: str, client_secret: str,
                 username: str, password: str, user_agent: str,
                 subreddits: list = None, poll_interval: int = 90):
        super().__init__(queue, "reddit")
        self._client_id     = client_id
        self._client_secret = client_secret
        self._username      = username
        self._password      = password
        self._user_agent    = user_agent
        self._subreddits    = subreddits or ["Gold", "Economics", "investing", "Forex"]
        self._poll_s        = poll_interval
        self._seen:         set = set()

    async def _stream(self) -> AsyncGenerator[RawEvent, None]:
        if not self._client_id or not self._client_secret:
            logger.error("REDDIT_CLIENT_ID/SECRET not set — Reddit ingester disabled")
            return

        try:
            import asyncpraw
        except ImportError:
            logger.error("asyncpraw not installed: pip install asyncpraw")
            return

        reddit = asyncpraw.Reddit(
            client_id=self._client_id,
            client_secret=self._client_secret,
            username=self._username or None,
            password=self._password or None,
            user_agent=self._user_agent,
        )

        try:
            sub_str = "+".join(self._subreddits)
            subreddit = await reddit.subreddit(sub_str)

            while self._running:
                try:
                    async for post in subreddit.new(limit=25):
                        if post.id in self._seen:
                            continue
                        self._seen.add(post.id)
                        # Only posts newer than poll_s * 3 to avoid processing old items
                        if time.time() - post.created_utc > self._poll_s * 3:
                            continue
                        text = " ".join(filter(None, [post.title, post.selftext[:500]]))
                        yield RawEvent(
                            timestamp=float(post.created_utc),
                            type=EventType.SENTIMENT,
                            payload={
                                "title":     post.title[:512],
                                "text":      text,
                                "score":     post.score,
                                "subreddit": post.subreddit.display_name,
                                "url":       f"https://reddit.com{post.permalink}",
                                "num_comments": post.num_comments,
                            },
                            source=f"reddit/{post.subreddit.display_name}",
                            confidence=0.65,
                        )
                except Exception as exc:
                    logger.error("Reddit stream error: %s", exc)

                await asyncio.sleep(self._poll_s)
        finally:
            await reddit.close()


# ── 3b. Twitter / X API v2 ────────────────────────────────────────────────────

class TwitterIngester(BaseIngester):
    """
    Polls Twitter/X API v2 recent search for gold/macro tweets.
    Docs: https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-recent
    """

    BASE_URL = "https://api.twitter.com/2/tweets/search/recent"

    def __init__(self, queue: asyncio.Queue, bearer_token: str,
                 query: str = "", poll_interval: int = 30):
        super().__init__(queue, "twitter")
        self._token  = bearer_token
        self._query  = query or "(gold OR XAUUSD OR bullion) -is:retweet lang:en"
        self._poll_s = poll_interval
        self._since_id: Optional[str] = None

    async def _stream(self) -> AsyncGenerator[RawEvent, None]:
        if not self._token:
            logger.error("TWITTER_BEARER_TOKEN not set — Twitter ingester disabled")
            return

        headers = {"Authorization": f"Bearer {self._token}"}
        async with aiohttp.ClientSession(headers=headers) as session:
            while self._running:
                try:
                    params = {
                        "query":       self._query,
                        "max_results": 100,
                        "tweet.fields": "created_at,public_metrics,author_id",
                        "expansions":  "author_id",
                        "sort_order":  "recency",
                    }
                    if self._since_id:
                        params["since_id"] = self._since_id

                    async with session.get(self.BASE_URL, params=params,
                                           timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status == 429:
                            # Rate limited — back off
                            reset = int(resp.headers.get("x-rate-limit-reset", time.time() + 60))
                            wait  = max(10, reset - int(time.time()))
                            logger.warning("Twitter rate limited — waiting %ds", wait)
                            await asyncio.sleep(wait)
                            continue
                        if resp.status != 200:
                            logger.error("Twitter API HTTP %d", resp.status)
                            await asyncio.sleep(60)
                            continue
                        data = await resp.json()

                    tweets = data.get("data", [])
                    if tweets:
                        self._since_id = tweets[0]["id"]

                    for tweet in tweets:
                        m = tweet.get("public_metrics", {})
                        yield RawEvent(
                            timestamp=_parse_iso(tweet.get("created_at", "")),
                            type=EventType.SENTIMENT,
                            payload={
                                "text":          tweet.get("text", "")[:512],
                                "tweet_id":      tweet.get("id", ""),
                                "likes":         m.get("like_count", 0),
                                "retweets":      m.get("retweet_count", 0),
                                "replies":       m.get("reply_count", 0),
                                "impressions":   m.get("impression_count", 0),
                            },
                            source="twitter",
                            confidence=0.60,
                        )
                except Exception as exc:
                    logger.error("Twitter error: %s", exc)

                await asyncio.sleep(self._poll_s)


# ── 4. FRED API (macro data) ──────────────────────────────────────────────────

class FREDIngester(BaseIngester):
    """
    Fetches economic series from the Federal Reserve FRED API.
    Free, no rate limits for reasonable use.
    Docs: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
    """

    OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

    HIGH_IMPACT = {"FEDFUNDS", "CPIAUCSL", "UNRATE", "GDP", "T10YIE"}

    def __init__(self, queue: asyncio.Queue, api_key: str,
                 series: list = None, poll_interval: int = 3600):
        super().__init__(queue, "fred")
        self._api_key = api_key
        self._series  = series or ["FEDFUNDS", "CPIAUCSL", "DGS10", "VIXCLS", "DCOILWTICO"]
        self._poll_s  = poll_interval
        self._last_values: dict = {}

    async def _stream(self) -> AsyncGenerator[RawEvent, None]:
        if not self._api_key:
            logger.error("FRED_API_KEY not set — FRED ingester disabled")
            return

        async with aiohttp.ClientSession() as session:
            while self._running:
                for series_id in self._series:
                    try:
                        async for event in self._fetch_series(session, series_id):
                            yield event
                        # Small delay between series requests
                        await asyncio.sleep(1.0)
                    except Exception as exc:
                        logger.error("FRED [%s] error: %s", series_id, exc)

                await asyncio.sleep(self._poll_s)

    async def _fetch_series(self, session: aiohttp.ClientSession,
                            series_id: str) -> AsyncGenerator[RawEvent, None]:
        params = {
            "series_id":    series_id,
            "api_key":      self._api_key,
            "file_type":    "json",
            "sort_order":   "desc",
            "limit":        5,            # last 5 observations
            "observation_start": "2020-01-01",
        }
        async with session.get(self.OBS_URL, params=params,
                               timeout=aiohttp.ClientTimeout(total=15)) as resp:
            data = await resp.json(content_type=None)

        observations = data.get("observations", [])
        if not observations:
            return

        # Most recent valid observation
        for obs in observations:
            value_str = obs.get("value", ".")
            if value_str == ".":
                continue
            try:
                value = float(value_str)
            except ValueError:
                continue

            # Only emit if value changed
            if self._last_values.get(series_id) == value:
                break
            self._last_values[series_id] = value

            hi = series_id in self.HIGH_IMPACT
            yield RawEvent(
                timestamp=_parse_iso(obs.get("date", "") + "T12:00:00Z"),
                type=EventType.MACRO,
                payload={
                    "series_id": series_id,
                    "event":     series_id,
                    "value":     value,
                    "date":      obs.get("date", ""),
                    "units":     data.get("units", ""),
                    "title":     data.get("title", series_id),
                    "impact":    "high" if hi else "medium",
                    "country":   "US",
                },
                source="fred",
                confidence=1.0,
            )
            break  # emit only the latest changed value


# ── Orchestrator ──────────────────────────────────────────────────────────────

class IngestionOrchestrator:
    """
    Builds and starts all enabled ingesters. Exposes a single output queue.

    Usage:
        orch = IngestionOrchestrator(config)
        await orch.start()
        while True:
            event = await orch.queue.get()
            process(event)
    """

    def __init__(self, config):
        self.queue: asyncio.Queue[RawEvent] = asyncio.Queue(maxsize=100_000)
        self._tasks: list = []
        self._ingesters: list = []

        md  = config.market_data
        nws = config.news
        snt = config.sentiment
        mac = config.macro

        # ── Market ──
        if md.twelve_data_enabled and md.twelve_data_api_key:
            self._ingesters.append(TwelveDataIngester(
                self.queue, md.twelve_data_api_key, md.symbols, md.twelve_data_ws_url
            ))
        elif md.alpha_vantage_enabled and md.alpha_vantage_api_key:
            logger.warning("TwelveData unavailable — falling back to AlphaVantage polling")
            self._ingesters.append(AlphaVantageIngester(
                self.queue, md.alpha_vantage_api_key, md.alpha_vantage_poll_s
            ))

        if md.oanda_enabled and md.oanda_account_id and md.oanda_access_token:
            self._ingesters.append(OANDAIngester(
                self.queue, md.oanda_account_id, md.oanda_access_token, md.oanda_environment
            ))

        # ── News ──
        if nws.newsapi_enabled and nws.newsapi_key:
            self._ingesters.append(NewsAPIIngester(
                self.queue, nws.newsapi_key, nws.newsapi_query, nws.newsapi_poll_s
            ))

        if nws.gdelt_enabled:
            self._ingesters.append(GDELTIngester(
                self.queue, nws.gdelt_query, nws.gdelt_poll_s
            ))

        # ── Sentiment ──
        if snt.reddit_enabled and snt.reddit_client_id and snt.reddit_client_secret:
            self._ingesters.append(RedditIngester(
                self.queue,
                client_id=snt.reddit_client_id,
                client_secret=snt.reddit_client_secret,
                username=snt.reddit_username,
                password=snt.reddit_password,
                user_agent=snt.reddit_user_agent,
                subreddits=snt.reddit_subreddits,
                poll_interval=snt.reddit_poll_s,
            ))

        if snt.twitter_enabled and snt.twitter_bearer_token:
            self._ingesters.append(TwitterIngester(
                self.queue, snt.twitter_bearer_token, snt.twitter_query, snt.twitter_poll_s
            ))

        # ── Macro ──
        if mac.fred_enabled and mac.fred_api_key:
            self._ingesters.append(FREDIngester(
                self.queue, mac.fred_api_key, mac.fred_series, mac.fred_poll_s
            ))

        if not self._ingesters:
            raise RuntimeError(
                "No ingesters configured. Check your .env credentials and ENABLE_* flags."
            )

        logger.info("IngestionOrchestrator: %d sources configured", len(self._ingesters))

    async def start(self) -> None:
        for ing in self._ingesters:
            task = asyncio.create_task(ing.start(), name=ing.source_name)
            self._tasks.append(task)

    async def stop(self) -> None:
        for ing in self._ingesters:
            await ing.stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _parse_iso(s: str) -> float:
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return time.time()


def _parse_gdelt_date(s: str) -> float:
    """Parse GDELT date format: YYYYMMDDHHMMSS"""
    from datetime import datetime, timezone
    try:
        dt = datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return time.time()
