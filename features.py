"""
features.py — Layer 3: Feature & Signal Extraction
====================================================
Improvements over v1:
  · Market: added MACD (12/26/9), Bollinger Bands, OBV, Stochastic %K/%D,
    Williams %R, CMF (Chaikin Money Flow) — all pure numpy, no lookahead
  · NLP: FinBERT optional sentiment (falls back to VADER); improved
    emotion vector with TF-IDF inverse frequency weighting
  · Event: FRED surprise computation vs rolling average
  · feat_type for macro returns "macro" (aligned with main.py expectation)

IMPORTANT: The FeaturePipeline.process() method now returns ("macro", ...)
for macro events instead of ("event", ...) — the main.py fix normalises
this at the call site, but the type is corrected here too for clarity.
"""

from __future__ import annotations

import logging
import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

@dataclass
class MarketFeatures:
    timestamp:    float
    price:        float
    z_price:      float
    returns_1:    float
    returns_5:    float
    returns_20:   float
    vol_5:        float
    vol_20:       float
    rsi_14:       float
    atr_14:       float
    vwap_dev:     float
    spread_pct:   float
    momentum_5:   float
    momentum_20:  float
    frac_diff:    float
    macd:         float
    macd_signal:  float
    bb_pct:       float    # Bollinger %B: 0=lower band, 1=upper band
    stoch_k:      float    # Stochastic %K
    stoch_d:      float    # Stochastic %D (smoothed)
    williams_r:   float    # Williams %R
    obv_norm:     float    # OBV normalised to [-1, 1]
    cmf:          float    # Chaikin Money Flow
    raw: Dict = field(default_factory=dict, repr=False)


@dataclass
class TextFeatures:
    timestamp:       float
    source_type:     str
    sentiment_score: float       # [-1, +1]
    emotion_vector:  np.ndarray  # [fear, greed, uncertainty, calm] ∈ [0,1]
    topic_embedding: np.ndarray  # [embedding_dim]
    text_hash:       str = ""


@dataclass
class EventFeatures:
    timestamp:    float
    event_type:   str
    impact_score: float
    surprise:     Optional[float]
    country:      str
    series_id:    str = ""
    value:        Optional[float] = None


# ── Fractional Differencing ───────────────────────────────────────────────────

def _frac_diff_weights(d: float, size: int, tol: float = 1e-5) -> np.ndarray:
    w = np.zeros(size, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, size):
        w[k] = -w[k - 1] * (d - k + 1) / k
        if abs(w[k]) < tol:
            w = w[:k]
            break
    return w[::-1]


class FractionalDifferencer:
    def __init__(self, d: float = 0.4, window: int = 60):
        self.d        = d
        self._window  = window
        self._weights = _frac_diff_weights(d, window)
        self._buf: deque = deque(maxlen=window)

    def transform(self, price: float) -> float:
        self._buf.append(price)
        if len(self._buf) < len(self._weights):
            return 0.0
        buf = np.array(list(self._buf)[-len(self._weights):], dtype=np.float64)
        return float(np.dot(self._weights, buf))


# ── MACD helper ───────────────────────────────────────────────────────────────

class _EMATracker:
    """Incremental EMA computation."""
    def __init__(self, period: int):
        self._alpha = 2.0 / (period + 1)
        self._val: Optional[float] = None

    def update(self, x: float) -> float:
        if self._val is None:
            self._val = x
        else:
            self._val = self._alpha * x + (1 - self._alpha) * self._val
        return self._val


# ── Market feature extractor ──────────────────────────────────────────────────

class MarketFeatureExtractor:
    def __init__(self, windows: List[int] = None, frac_diff_d: float = 0.4):
        self._w        = windows or [5, 20, 60, 200]
        self._prices:   deque = deque(maxlen=max(self._w) + 1)
        self._highs:    deque = deque(maxlen=max(self._w))
        self._lows:     deque = deque(maxlen=max(self._w))
        self._spreads:  deque = deque(maxlen=20)
        self._volumes:  deque = deque(maxlen=max(self._w))
        self._pv_sum   = 0.0
        self._v_sum    = 0.0
        self._fd       = FractionalDifferencer(d=frac_diff_d)
        self._obv:     float = 0.0

        # MACD EMAs
        self._ema12 = _EMATracker(12)
        self._ema26 = _EMATracker(26)
        self._macd_signal = _EMATracker(9)

        # Stochastic
        self._stoch_k_hist: deque = deque(maxlen=3)

    def extract(self, event) -> Optional[MarketFeatures]:
        p      = event.payload
        price  = p.get("mid", 0.0)
        spread = p.get("spread", 0.0)
        vol    = p.get("volume", 0.0)
        z      = p.get("z_price", 0.0)

        if price <= 0:
            return None

        # Update price buffers
        self._prices.append(price)
        self._spreads.append(spread)
        vol_f = float(vol or 0.0)
        self._volumes.append(vol_f)

        # Approximate high/low from spread
        self._highs.append(price + spread / 2)
        self._lows.append(price - spread / 2)

        # VWAP
        self._pv_sum += price * max(vol_f, 1e-10)
        self._v_sum  += max(vol_f, 1e-10)

        # OBV
        prices = list(self._prices)
        if len(prices) >= 2:
            self._obv += vol_f if price >= prices[-2] else -vol_f

        n = len(prices)
        if n < 2:
            return None

        def ret(lag: int) -> float:
            if n <= lag: return 0.0
            d = prices[-(lag + 1)]
            return (prices[-1] - d) / d if d != 0 else 0.0

        def rolling_std(lag: int) -> float:
            if n < lag: return 0.0
            return float(np.std(prices[-lag:]))

        def momentum(lag: int) -> float:
            return prices[-1] - prices[-lag] if n > lag else 0.0

        vwap     = self._pv_sum / self._v_sum
        vwap_dev = (price - vwap) / (vwap + 1e-8)

        # MACD
        e12   = self._ema12.update(price)
        e26   = self._ema26.update(price)
        macd  = e12 - e26
        sig   = self._macd_signal.update(macd)

        # Bollinger Bands (20-period)
        bb_pct = 0.5
        if n >= 20:
            arr = np.array(prices[-20:])
            mu  = arr.mean()
            sd  = arr.std() + 1e-8
            bb_pct = float((price - (mu - 2 * sd)) / (4 * sd))
            bb_pct = float(np.clip(bb_pct, 0.0, 1.0))

        # Stochastic %K (14-period)
        stoch_k, stoch_d = 50.0, 50.0
        lk = 14
        if n >= lk:
            lo = min(self._lows) if len(self._lows) >= lk else price
            hi = max(self._highs) if len(self._highs) >= lk else price
            rng = hi - lo
            stoch_k = float((price - lo) / rng * 100) if rng > 1e-8 else 50.0
            self._stoch_k_hist.append(stoch_k)
            stoch_d = float(np.mean(list(self._stoch_k_hist)))

        # Williams %R (14-period)
        williams_r = -50.0
        if n >= lk:
            hi = max(list(self._highs)[-lk:]) if len(self._highs) >= lk else price
            lo = min(list(self._lows)[-lk:]) if len(self._lows) >= lk else price
            rng = hi - lo
            williams_r = float(((hi - price) / rng) * -100) if rng > 1e-8 else -50.0

        # CMF — Chaikin Money Flow (14-period)
        cmf = 0.0
        if n >= 14 and len(self._volumes) >= 14:
            vs  = np.array(list(self._volumes)[-14:], dtype=np.float64)
            ps  = np.array(prices[-14:], dtype=np.float64)
            his = np.array(list(self._highs)[-14:], dtype=np.float64)
            los = np.array(list(self._lows)[-14:], dtype=np.float64)
            rngs = his - los + 1e-8
            mfv  = ((2 * ps - his - los) / rngs) * vs
            denom = vs.sum()
            cmf  = float(mfv.sum() / denom) if denom > 1e-8 else 0.0

        # OBV normalised to recent range
        obv_norm = 0.0
        obv_abs = abs(self._obv)
        if obv_abs > 0:
            obv_norm = float(np.clip(self._obv / (obv_abs + 1e-8), -1.0, 1.0))

        return MarketFeatures(
            timestamp=event.wall_time,
            price=price,
            z_price=z,
            returns_1=ret(1),
            returns_5=ret(5),
            returns_20=ret(20),
            vol_5=rolling_std(5),
            vol_20=rolling_std(20),
            rsi_14=_rsi(prices, 14),
            atr_14=_atr(prices, 14),
            vwap_dev=vwap_dev,
            spread_pct=spread / (price + 1e-8),
            momentum_5=momentum(5),
            momentum_20=momentum(20),
            frac_diff=self._fd.transform(price),
            macd=float(macd),
            macd_signal=float(sig),
            bb_pct=bb_pct,
            stoch_k=stoch_k / 100.0,
            stoch_d=stoch_d / 100.0,
            williams_r=(williams_r + 100) / 100.0,   # normalise to [0, 1]
            obv_norm=obv_norm,
            cmf=float(np.clip(cmf, -1.0, 1.0)),
            raw=p,
        )

    def to_vector(self, feat: MarketFeatures) -> np.ndarray:
        """25-dimensional market feature vector."""
        return np.array([
            feat.z_price,
            feat.returns_1,    feat.returns_5,    feat.returns_20,
            feat.vol_5,        feat.vol_20,
            feat.rsi_14 / 100.0,
            feat.atr_14,
            feat.vwap_dev,     feat.spread_pct,
            feat.momentum_5,   feat.momentum_20,
            feat.frac_diff,
            feat.macd,         feat.macd_signal,
            feat.bb_pct,
            feat.stoch_k,      feat.stoch_d,
            feat.williams_r,
            feat.obv_norm,     feat.cmf,
        ], dtype=np.float32)


def _rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0).mean()
    losses = np.where(deltas < 0, -deltas, 0.0).mean()
    if losses < 1e-10:
        return 100.0
    return float(100 - 100 / (1 + gains / losses))


def _atr(prices: list, period: int = 14) -> float:
    if len(prices) < 2:
        return 0.0
    trs = [abs(prices[i] - prices[i - 1]) for i in range(-min(period, len(prices) - 1), 0)]
    return float(np.mean(trs)) if trs else 0.0


# ── NLP feature extractor ─────────────────────────────────────────────────────

class NLPFeatureExtractor:
    """
    Real semantic text embeddings via sentence-transformers.
    Sentiment: FinBERT (if available) → VADER → keyword fallback.
    Emotion:   Multi-label with inverse-document-frequency weighting.
    """

    EMOTION_VOCAB = {
        "fear": [
            "fear", "panic", "crash", "collapse", "crisis", "warning", "danger",
            "plunge", "tumble", "spiral", "contagion", "default", "recession",
            "depression", "shock", "turmoil", "rout", "meltdown", "carnage",
            "selloff", "bloodbath", "free fall", "systemic",
        ],
        "greed": [
            "rally", "surge", "soar", "bullish", "boom", "gain", "profit",
            "record", "high", "upside", "breakout", "strong", "buy",
            "inflow", "demand", "optimism", "growth", "momentum", "outperform",
            "euphoria", "frenzy", "fomo", "all-time high",
        ],
        "uncertainty": [
            "uncertain", "unclear", "mixed", "wait", "pending", "ambiguous",
            "conflicting", "volatile", "choppy", "cautious", "hesitant",
            "undecided", "opaque", "murky", "crosscurrent", "diverge",
            "unpredictable", "unstable", "unknown", "risk",
        ],
        "calm": [
            "stable", "steady", "hold", "flat", "unchanged", "range",
            "consolidate", "stabilize", "pause", "contained", "anchored",
            "orderly", "balanced", "tempered", "moderate", "gradual",
        ],
    }
    # IDF-like weights per emotion (rarer = higher weight)
    _IDF = {"fear": 1.4, "greed": 1.2, "uncertainty": 1.0, "calm": 0.9}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self._model        = None
        self._finbert      = None
        self._vader        = None
        self._model_name   = model_name
        self._device       = device
        self._embedding_dim = 384
        self._load_models()

    def _load_models(self) -> None:
        # Sentence Transformers
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformer: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name, device=self._device)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info("sentence-transformer loaded — dim=%d", self._embedding_dim)
        except ImportError:
            logger.error("sentence-transformers not installed: pip install sentence-transformers")
        except Exception as exc:
            logger.error("Failed to load sentence-transformer: %s", exc)

        # FinBERT (financial domain sentiment — higher accuracy than VADER for finance)
        try:
            from transformers import pipeline
            self._finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0 if self._device == "cuda" else -1,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded for financial sentiment")
        except Exception:
            logger.info("FinBERT unavailable — falling back to VADER")

        # VADER fallback
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyser loaded")
        except ImportError:
            logger.warning("vaderSentiment not installed: pip install vaderSentiment")

    def extract(self, event) -> Optional[TextFeatures]:
        text = (event.payload.get("text") or
                event.payload.get("title") or "").strip()
        if not text or len(text) < 10:
            return None

        sentiment  = self._sentiment(text)
        emotion    = self._emotion_vector(text)
        topic_emb  = self._embed(text)

        if topic_emb is None:
            return None

        return TextFeatures(
            timestamp=event.wall_time,
            source_type=event.type,
            sentiment_score=sentiment,
            emotion_vector=emotion,
            topic_embedding=topic_emb,
            text_hash=hashlib.md5(text.encode()).hexdigest()[:12],
        )

    def _sentiment(self, text: str) -> float:
        # FinBERT: trained on financial texts, better calibrated for gold/macro
        if self._finbert is not None:
            try:
                result = self._finbert(text[:512])[0]
                label  = result["label"].lower()
                score  = result["score"]
                if label == "positive":
                    return float(score)
                elif label == "negative":
                    return float(-score)
                else:
                    return 0.0
            except Exception:
                pass

        # VADER fallback
        if self._vader:
            return float(self._vader.polarity_scores(text)["compound"])

        # Keyword fallback
        words = text.lower().split()
        pos = sum(1 for w in words if w in {"gain", "rise", "rally", "up", "bull", "strong"})
        neg = sum(1 for w in words if w in {"fall", "drop", "crash", "down", "bear", "weak"})
        return float((pos - neg) / (pos + neg + 1e-8))

    def _emotion_vector(self, text: str) -> np.ndarray:
        words = text.lower().split()
        n     = len(words) + 1e-8
        scores = []
        for emo, vocab in self.EMOTION_VOCAB.items():
            raw = sum(1 for w in words if any(v in w for v in vocab)) / n
            scores.append(raw * self._IDF.get(emo, 1.0))
        arr   = np.array(scores, dtype=np.float32)
        total = arr.sum()
        return arr / (total + 1e-8)

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        try:
            emb = self._model.encode(
                text[:512], normalize_embeddings=True, show_progress_bar=False
            )
            return emb.astype(np.float32)
        except Exception as exc:
            logger.error("Embedding error: %s", exc)
            return None

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def to_vector(self, feat: TextFeatures) -> np.ndarray:
        return np.concatenate([
            [feat.sentiment_score],
            feat.emotion_vector,       # 4-d
            feat.topic_embedding,      # 384-d
        ]).astype(np.float32)


# ── Event encoder ─────────────────────────────────────────────────────────────

class EventEncoder:
    """
    Maps FRED series IDs to structured feature vectors.
    Computes surprise as deviation from rolling mean (online).
    """

    SERIES_IMPACT = {
        "FEDFUNDS":          ("FED_RATE",       1.00),
        "CPIAUCSL":          ("CPI",            0.95),
        "UNRATE":            ("UNEMPLOYMENT",   0.80),
        "DGS10":             ("YIELD_10Y",      0.90),
        "DGS2":              ("YIELD_2Y",       0.80),
        "DTWEXBGS":          ("USD_INDEX",      0.85),
        "VIXCLS":            ("VIX",            0.85),
        "DCOILWTICO":        ("OIL_WTI",        0.70),
        "GOLDAMGBD228NLBM":  ("GOLD_FIX",       0.75),
        "GDP":               ("GDP",            0.85),
        "T10YIE":            ("BREAKEVEN_INFL", 0.90),
    }

    CATEGORIES = [
        "FED_RATE", "CPI", "UNEMPLOYMENT", "YIELD_10Y", "YIELD_2Y",
        "USD_INDEX", "VIX", "OIL_WTI", "GOLD_FIX", "GDP",
        "BREAKEVEN_INFL", "OTHER",
    ]

    def __init__(self):
        self._rolling: Dict[str, deque] = {}   # rolling history for surprise

    def encode(self, event) -> Optional[EventFeatures]:
        # Accept both "macro" and "event" type strings
        if event.type not in ("macro", "event"):
            return None
        p          = event.payload
        series_id  = p.get("series_id", p.get("event", "OTHER"))
        event_info = self.SERIES_IMPACT.get(series_id, ("OTHER", 0.40))
        event_type, base_impact = event_info

        value    = p.get("value")
        surprise = p.get("surprise")

        # Compute surprise from rolling mean if not provided
        if surprise is None and value is not None:
            val = float(value)
            if series_id not in self._rolling:
                self._rolling[series_id] = deque(maxlen=12)
            hist = self._rolling[series_id]
            if len(hist) >= 2:
                mu  = float(np.mean(hist))
                sd  = float(np.std(hist)) + 1e-8
                surprise = (val - mu) / sd
            hist.append(val)

        return EventFeatures(
            timestamp=event.wall_time,
            event_type=event_type,
            impact_score=float(min(1.0, base_impact)),
            surprise=float(surprise) if surprise is not None else None,
            country=p.get("country", "US"),
            series_id=series_id,
            value=float(value) if value is not None else None,
        )

    def to_vector(self, feat: EventFeatures) -> np.ndarray:
        onehot = np.zeros(len(self.CATEGORIES), dtype=np.float32)
        if feat.event_type in self.CATEGORIES:
            onehot[self.CATEGORIES.index(feat.event_type)] = 1.0
        return np.concatenate([
            onehot,
            [feat.impact_score, feat.surprise or 0.0],
        ])


# ── Unified pipeline ──────────────────────────────────────────────────────────

class FeaturePipeline:
    def __init__(self, config=None):
        cfg = config.features if config else None
        self.market = MarketFeatureExtractor(
            windows=[5, 20, 60, 200],
            frac_diff_d=cfg.frac_diff_d if cfg else 0.4,
        )
        self.nlp = NLPFeatureExtractor(
            model_name=cfg.nlp_model  if cfg else "all-MiniLM-L6-v2",
            device=cfg.nlp_device     if cfg else "cpu",
        )
        self.events = EventEncoder()

    def process(self, event) -> Optional[Tuple]:
        t = event.type
        if t == "market":
            feat = self.market.extract(event)
            if feat:
                return ("market", feat, self.market.to_vector(feat))
        elif t in {"news", "sentiment"}:
            feat = self.nlp.extract(event)
            if feat:
                return ("text", feat, self.nlp.to_vector(feat))
        elif t in {"macro", "event"}:
            feat = self.events.encode(event)
            if feat:
                # Return "macro" (not "event") for consistent routing in main.py
                return ("macro", feat, self.events.to_vector(feat))
        return None

    @property
    def text_embedding_dim(self) -> int:
        return self.nlp.embedding_dim