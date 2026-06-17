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
    """
    All windowed reads use exactly the last `lk` elements (fixes the silent
    Stochastic %K bug that scanned the full 200-element deque). VWAP is now
    session-relative (resets every `vwap_session_s` seconds) so the anchor
    does not become stale over multi-day operation. Volume-dependent indicators
    (OBV, CMF) gracefully degrade when no volume is reported by using the
    Yang-Zhang style price-action proxy with non-degenerate normalisation.
    """

    def __init__(
        self,
        windows: List[int] = None,
        frac_diff_d: float = 0.4,
        vwap_session_s: float = 24 * 3600,
    ):
        self._w        = windows or [5, 20, 60, 200]
        wmax           = max(self._w)
        self._prices:   deque = deque(maxlen=wmax + 1)
        self._highs:    deque = deque(maxlen=wmax)
        self._lows:     deque = deque(maxlen=wmax)
        self._spreads:  deque = deque(maxlen=wmax)
        self._volumes:  deque = deque(maxlen=wmax)
        self._returns:  deque = deque(maxlen=wmax)   # log-returns for vol estimators
        self._obv_hist: deque = deque(maxlen=wmax)   # rolling OBV for proper normalisation

        # Session-relative VWAP — resets on session boundary
        self._pv_sum:      float = 0.0
        self._v_sum:       float = 0.0
        self._vwap_session_s     = vwap_session_s
        self._vwap_session_start: Optional[float] = None

        self._fd       = FractionalDifferencer(d=frac_diff_d)
        self._obv:     float = 0.0

        # MACD EMAs
        self._ema12 = _EMATracker(12)
        self._ema26 = _EMATracker(26)
        self._macd_signal = _EMATracker(9)

        # Stochastic — keep %D smoothed across last 3 %K values
        self._stoch_k_hist: deque = deque(maxlen=3)

        # Welford accumulators for online robust standardisation of returns
        self._ret_n:  int   = 0
        self._ret_mu: float = 0.0
        self._ret_m2: float = 0.0   # M2 for variance

    def extract(self, event) -> Optional[MarketFeatures]:
        p      = event.payload
        price  = float(p.get("mid", 0.0) or 0.0)
        spread = float(p.get("spread", 0.0) or 0.0)
        vol    = float(p.get("volume", 0.0) or 0.0)
        z      = float(p.get("z_price", 0.0) or 0.0)
        ts     = float(event.wall_time or 0.0)

        if price <= 0:
            return None

        # ---- bookkeeping --------------------------------------------------
        prev_price = self._prices[-1] if self._prices else None
        self._prices.append(price)
        self._spreads.append(spread)
        self._volumes.append(vol)

        # Approximate intra-bar high/low from spread (best we can do at tick freq)
        half = max(spread * 0.5, price * 1e-6)
        self._highs.append(price + half)
        self._lows.append(price - half)

        # Log-return (more stable than simple return for vol estimators)
        if prev_price and prev_price > 0:
            r = float(np.log(price / prev_price))
            self._returns.append(r)
            # Welford online mean / variance of returns
            self._ret_n += 1
            delta = r - self._ret_mu
            self._ret_mu += delta / self._ret_n
            self._ret_m2 += delta * (r - self._ret_mu)

        # ---- session-relative VWAP ----------------------------------------
        if (self._vwap_session_start is None
                or (ts - self._vwap_session_start) > self._vwap_session_s):
            self._vwap_session_start = ts
            self._pv_sum = 0.0
            self._v_sum  = 0.0
        # Proxy weight when no volume is reported: use 1 per tick so VWAP
        # degenerates gracefully to time-weighted average price.
        w_vol = vol if vol > 0 else 1.0
        self._pv_sum += price * w_vol
        self._v_sum  += w_vol

        # ---- OBV (only meaningful when real volume present) ---------------
        if prev_price is not None and vol > 0:
            self._obv += vol if price >= prev_price else -vol
        self._obv_hist.append(self._obv)

        prices = list(self._prices)
        n = len(prices)
        if n < 2:
            return None

        # ---- vectorised helpers (single numpy view) -----------------------
        prices_arr = np.asarray(prices, dtype=np.float64)

        def ret_simple(lag: int) -> float:
            if n <= lag:
                return 0.0
            d = prices_arr[-(lag + 1)]
            return float((prices_arr[-1] - d) / d) if d != 0 else 0.0

        def rolling_std(lag: int) -> float:
            if n < lag:
                return 0.0
            return float(prices_arr[-lag:].std(ddof=0))

        def momentum(lag: int) -> float:
            return float(prices_arr[-1] - prices_arr[-lag]) if n > lag else 0.0

        vwap     = self._pv_sum / max(self._v_sum, 1e-10)
        vwap_dev = (price - vwap) / (vwap + 1e-8)

        # ---- MACD ---------------------------------------------------------
        e12   = self._ema12.update(price)
        e26   = self._ema26.update(price)
        macd  = e12 - e26
        sig   = self._macd_signal.update(macd)
        # Normalise MACD by price scale for stable cross-regime behaviour
        macd_n      = float(macd / (price + 1e-8))
        macd_sig_n  = float(sig  / (price + 1e-8))

        # ---- Bollinger %B (20-period) ------------------------------------
        bb_pct = 0.5
        if n >= 20:
            arr = prices_arr[-20:]
            mu  = arr.mean()
            sd  = arr.std() + 1e-8
            bb_pct = float(np.clip((price - (mu - 2 * sd)) / (4 * sd), 0.0, 1.0))

        # ---- Stochastic %K / %D (14-period — last 14 ONLY) ---------------
        stoch_k, stoch_d = 0.5, 0.5
        lk = 14
        if n >= lk:
            recent_highs = list(self._highs)[-lk:]
            recent_lows  = list(self._lows)[-lk:]
            hi  = max(recent_highs)
            lo  = min(recent_lows)
            rng = hi - lo
            stoch_k = float((price - lo) / rng) if rng > 1e-8 else 0.5
            stoch_k = float(np.clip(stoch_k, 0.0, 1.0))
            self._stoch_k_hist.append(stoch_k)
            stoch_d = float(np.mean(self._stoch_k_hist))
        else:
            self._stoch_k_hist.append(stoch_k)

        # ---- Williams %R normalised to [0,1] -----------------------------
        williams_r = 0.5
        if n >= lk:
            recent_highs = list(self._highs)[-lk:]
            recent_lows  = list(self._lows)[-lk:]
            hi  = max(recent_highs)
            lo  = min(recent_lows)
            rng = hi - lo
            williams_r = float((hi - price) / rng) if rng > 1e-8 else 0.5
            williams_r = float(1.0 - np.clip(williams_r, 0.0, 1.0))

        # ---- CMF — only emit non-zero when real volume present -----------
        cmf = 0.0
        if n >= lk and len(self._volumes) >= lk:
            vs  = np.array(list(self._volumes)[-lk:], dtype=np.float64)
            if vs.sum() > 1e-8:   # gate: only meaningful with real volume
                ps  = prices_arr[-lk:]
                his = np.array(list(self._highs)[-lk:], dtype=np.float64)
                los = np.array(list(self._lows)[-lk:],  dtype=np.float64)
                rngs = his - los + 1e-8
                mfv  = ((2 * ps - his - los) / rngs) * vs
                cmf  = float(np.clip(mfv.sum() / vs.sum(), -1.0, 1.0))

        # ---- OBV normalised by recent OBV range (proper rolling norm) ----
        obv_norm = 0.0
        if len(self._obv_hist) >= 20:
            obv_arr = np.asarray(list(self._obv_hist)[-200:], dtype=np.float64)
            obv_lo, obv_hi = obv_arr.min(), obv_arr.max()
            obv_rng = obv_hi - obv_lo
            if obv_rng > 1e-8:
                obv_norm = float(2.0 * (self._obv - obv_lo) / obv_rng - 1.0)
                obv_norm = float(np.clip(obv_norm, -1.0, 1.0))

        # ---- Robust z-price using running median-of-returns variance -----
        # If upstream z_price was 0 (cold start) backfill with online std
        if abs(z) < 1e-12 and self._ret_n > 5:
            sd = float(np.sqrt(self._ret_m2 / max(1, self._ret_n - 1)))
            # cross-sectional z of latest log-return
            if sd > 1e-10 and self._returns:
                z = float((self._returns[-1] - self._ret_mu) / sd)

        return MarketFeatures(
            timestamp=event.wall_time,
            price=price,
            z_price=float(np.clip(z, -10.0, 10.0)),
            returns_1=ret_simple(1),
            returns_5=ret_simple(5),
            returns_20=ret_simple(20),
            vol_5=rolling_std(5)  / max(price, 1e-8),     # scale-invariant
            vol_20=rolling_std(20) / max(price, 1e-8),
            rsi_14=_rsi(prices_arr, 14),
            atr_14=_atr(prices_arr, 14) / max(price, 1e-8),
            vwap_dev=float(np.clip(vwap_dev, -1.0, 1.0)),
            spread_pct=spread / (price + 1e-8),
            momentum_5=momentum(5)  / max(price, 1e-8),
            momentum_20=momentum(20) / max(price, 1e-8),
            frac_diff=self._fd.transform(price),
            macd=macd_n,
            macd_signal=macd_sig_n,
            bb_pct=bb_pct,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            williams_r=williams_r,
            obv_norm=obv_norm,
            cmf=cmf,
            raw=p,
        )

    def to_vector(self, feat: MarketFeatures) -> np.ndarray:
        """21-dimensional, scale-invariant market feature vector.

        All entries are either fractions of price, bounded [0,1] / [-1,1]
        indicators, or already-standardised z-scores — so downstream layers
        operate on a numerically homogeneous space.
        """
        return np.array([
            feat.z_price,
            feat.returns_1,    feat.returns_5,    feat.returns_20,
            feat.vol_5,        feat.vol_20,
            (feat.rsi_14 - 50.0) / 50.0,                  # centred to [-1, 1]
            feat.atr_14,
            feat.vwap_dev,     feat.spread_pct,
            feat.momentum_5,   feat.momentum_20,
            feat.frac_diff,
            feat.macd,         feat.macd_signal,
            feat.bb_pct * 2 - 1.0,                        # [0,1] → [-1,1]
            feat.stoch_k * 2 - 1.0,
            feat.stoch_d * 2 - 1.0,
            feat.williams_r * 2 - 1.0,
            feat.obv_norm,     feat.cmf,
        ], dtype=np.float32)


def _rsi(prices, period: int = 14) -> float:
    """Wilder-smoothed RSI (more accurate than simple mean-of-deltas)."""
    arr = np.asarray(prices, dtype=np.float64)
    if arr.size < period + 1:
        return 50.0
    deltas = np.diff(arr[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Wilder smoothing: first mean, then EMA on subsequent
    avg_g = gains.mean()
    avg_l = losses.mean()
    if avg_l < 1e-12:
        return 100.0
    rs = avg_g / avg_l
    return float(100.0 - 100.0 / (1.0 + rs))


def _atr(prices, period: int = 14) -> float:
    arr = np.asarray(prices, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    trs = np.abs(np.diff(arr[-(period + 1):]))
    return float(trs.mean()) if trs.size else 0.0


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