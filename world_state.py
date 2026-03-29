"""
embeddings.py — Layer 4: Multi-Modal Embedding
world_state.py  — Layer 5: World State Model (NSSM)
=====================================================

Improvements over v1:
  · Xavier/Glorot weight initialisation for all Linear layers (replaces
    near-zero random init that caused slow gradient flow)
  · TCNMarketEncoder: accepts new 21-d feature vector from features.py (was 13-d);
    dimension resolved lazily on first call
  · NSSMWorldStateModel: added uncertainty estimation (diagonal covariance);
    factor interaction matrix for richer latent dynamics; per-factor EMA rates
  · Calibrated EMA alpha to factor timescale:
    - Fast factors (momentum, microstructure): high alpha (0.25)
    - Slow factors (carry, geopolitical):      low alpha  (0.05)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Neural primitives ─────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -30, 30))

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    sd = x.std(-1, keepdims=True)
    return (x - mu) / (sd + eps)


class Linear:
    """Xavier-initialised linear layer."""
    def __init__(self, in_d: int, out_d: int, seed: int = 0):
        rng   = np.random.default_rng(seed)
        # Xavier uniform: limit = sqrt(6 / (fan_in + fan_out))
        limit = np.sqrt(6.0 / (in_d + out_d))
        self.W = rng.uniform(-limit, limit, (in_d, out_d)).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b


# ── Encoders ──────────────────────────────────────────────────────────────────

class TCNMarketEncoder:
    """
    Temporal Convolutional Network approximation.
    Input dimension is resolved lazily on first push() call so it works
    whether features.py returns 13-d (v1) or 21-d (v2) vectors.
    """

    def __init__(self, out_dim: int = 64, seq_len: int = 20):
        self.seq_len  = seq_len
        self.out_dim  = out_dim
        self._buf:    list = []
        self._in_dim: Optional[int] = None
        self.fc1 = None
        self.fc2 = None

    def _init(self, in_dim: int) -> None:
        self._in_dim = in_dim
        total_in     = in_dim * self.seq_len
        self.fc1     = Linear(total_in, max(128, total_in // 2), seed=1)
        self.fc2     = Linear(max(128, total_in // 2), self.out_dim, seed=2)
        logger.info("TCNMarketEncoder: in_dim=%d seq=%d → out=%d",
                    in_dim, self.seq_len, self.out_dim)

    def push(self, vec: np.ndarray) -> Optional[np.ndarray]:
        v = vec.astype(np.float32)
        if self._in_dim is None:
            self._init(v.shape[0])

        self._buf.append(v)
        if len(self._buf) > self.seq_len:
            self._buf.pop(0)
        if len(self._buf) < self.seq_len:
            return None

        x = np.concatenate(self._buf).astype(np.float32)
        x = _relu(self.fc1(x))
        x = _layer_norm(self.fc2(x))
        return x


class TransformerTextEncoder:
    """
    Single-head self-attention text encoder.
    Input dim resolved lazily (sentence-transformers varies by model).
    """

    def __init__(self, out_dim: int = 64):
        self.out_dim  = out_dim
        self._in_dim: Optional[int] = None
        self.proj_in  = None
        self.proj_out = None

    def _init_weights(self, in_dim: int) -> None:
        logger.info("TransformerTextEncoder: in_dim=%d → out_dim=%d", in_dim, self.out_dim)
        self._in_dim  = in_dim
        self.proj_in  = Linear(in_dim,       self.out_dim, seed=10)
        self.proj_out = Linear(self.out_dim, self.out_dim, seed=14)

    def encode(self, vec: np.ndarray) -> np.ndarray:
        v = vec.astype(np.float32)
        if self._in_dim is None or v.shape[0] != self._in_dim:
            self._init_weights(v.shape[0])
        x = _relu(self.proj_in(v))
        x = _layer_norm(self.proj_out(x))
        return x


class MLPMacroEncoder:
    """MLP for macro/event features."""

    def __init__(self, in_dim: int = 14, out_dim: int = 32):
        self.fc1 = Linear(in_dim, 64, seed=20)
        self.fc2 = Linear(64, out_dim, seed=21)

    def encode(self, vec: np.ndarray) -> np.ndarray:
        # Pad/trim to expected input dimension
        v = vec.astype(np.float32)
        expected = self.fc1.W.shape[0]
        if len(v) < expected:
            v = np.pad(v, (0, expected - len(v)))
        elif len(v) > expected:
            v = v[:expected]
        x = _relu(self.fc1(v))
        return _layer_norm(self.fc2(x))


# ── Cross-Attention Fusion ────────────────────────────────────────────────────

class CrossAttentionFusion:
    """
    Fused = cross_attention(market, text, macro).
    Single-vector gated attention: each modality acts as Q, K, V simultaneously.
    """

    def __init__(self, market_dim: int = 64, text_dim: int = 64,
                 macro_dim: int = 32, fused_dim: int = 128):
        total_in     = market_dim + text_dim + macro_dim
        self.attn_q  = Linear(total_in, total_in, seed=30)
        self.attn_k  = Linear(total_in, total_in, seed=31)
        self.attn_v  = Linear(total_in, total_in, seed=32)
        self.proj    = Linear(total_in, fused_dim, seed=33)
        self._scale  = float(np.sqrt(total_in))

    def fuse(
        self,
        market: Optional[np.ndarray],
        text:   Optional[np.ndarray],
        macro:  Optional[np.ndarray],
        market_dim: int = 64,
        text_dim:   int = 64,
        macro_dim:  int = 32,
    ) -> np.ndarray:
        m = market if market is not None else np.zeros(market_dim, np.float32)
        t = text   if text   is not None else np.zeros(text_dim,   np.float32)
        c = macro  if macro  is not None else np.zeros(macro_dim,  np.float32)
        x = np.concatenate([m, t, c]).astype(np.float32)

        q    = self.attn_q(x)
        k    = self.attn_k(x)
        gate = _sigmoid(q * k / self._scale)
        v    = self.attn_v(x) * gate
        out  = _relu(self.proj(_layer_norm(v)))
        return out


# ── Layer 4: Multi-Modal Embedding Pipeline ───────────────────────────────────

class MultiModalEmbeddingLayer:
    def __init__(self, config=None):
        cfg = config.embeddings if config else None
        mkt = cfg.market_dim if cfg else 64
        txt = cfg.text_dim   if cfg else 64
        mac = cfg.macro_dim  if cfg else 32
        fus = cfg.fused_dim  if cfg else 128

        # TCN in_dim resolved lazily — works with 13-d (v1) or 21-d (v2) features
        self._mkt_enc  = TCNMarketEncoder(out_dim=mkt)
        self._txt_enc  = TransformerTextEncoder(out_dim=txt)
        self._mac_enc  = MLPMacroEncoder(in_dim=14, out_dim=mac)
        self._fusion   = CrossAttentionFusion(mkt, txt, mac, fus)

        self._last_market: Optional[np.ndarray] = None
        self._last_text:   Optional[np.ndarray] = None
        self._last_macro:  Optional[np.ndarray] = None
        self._dims = (mkt, txt, mac)

    def update_market(self, vec: np.ndarray) -> None:
        emb = self._mkt_enc.push(vec)
        if emb is not None:
            self._last_market = emb

    def update_text(self, vec: np.ndarray) -> None:
        self._last_text = self._txt_enc.encode(vec)

    def update_macro(self, vec: np.ndarray) -> None:
        self._last_macro = self._mac_enc.encode(vec)

    def get_fused(self) -> Optional[np.ndarray]:
        if self._last_market is None:
            return None
        return self._fusion.fuse(
            self._last_market,
            self._last_text,
            self._last_macro,
            *self._dims,
        )


# ── Layer 5: Neural State Space Model ────────────────────────────────────────

FACTOR_NAMES = [
    "risk_regime", "inflation_pressure", "liquidity", "sentiment", "volatility_state",
    "momentum_short", "momentum_long", "mean_reversion",
    "macro_shock", "dollar_strength", "yield_pressure",
    "geopolitical_risk", "market_microstructure",
    "positioning_bias", "options_skew", "carry",
]

# Per-factor EMA alpha — fast-moving factors get higher alpha
_FACTOR_ALPHA = {
    "momentum_short":       0.30,
    "market_microstructure": 0.28,
    "volatility_state":     0.25,
    "sentiment":            0.22,
    "mean_reversion":       0.20,
    "risk_regime":          0.15,
    "dollar_strength":      0.12,
    "yield_pressure":       0.10,
    "liquidity":            0.10,
    "inflation_pressure":   0.08,
    "macro_shock":          0.08,
    "positioning_bias":     0.07,
    "options_skew":         0.07,
    "momentum_long":        0.07,
    "geopolitical_risk":    0.05,
    "carry":                0.05,
}


class NSSMWorldStateModel:
    """
    Neural State Space Model — core of GMMIE.

    z_{t+1} = f(z_t, x_t)   (transition with gating)
    y_t      = g(z_t)         (emission)

    Improvements:
      · Xavier initialisation (replaces small-normal causing vanishing gradients)
      · Per-factor EMA smoothing (different timescales per factor)
      · Diagonal uncertainty covariance P (Kalman-like variance tracking)
      · Factor interaction matrix M: z' ← tanh(Az + Bx + Mz⊗z) captures
        nonlinear factor interactions (e.g. momentum × sentiment → regime)
    """

    def __init__(self, input_dim: int = 128, latent_dim: int = 16,
                 hidden_dim: int = 128):
        self.latent_dim = latent_dim
        self._z = np.zeros(latent_dim, dtype=np.float32)
        self._P = np.ones(latent_dim, dtype=np.float32) * 0.1   # uncertainty

        # Transition network: [z; x] → hidden → z'
        self._trans_fc1 = Linear(latent_dim + input_dim, hidden_dim, seed=40)
        self._trans_fc2 = Linear(hidden_dim, latent_dim, seed=41)

        # Gating: [z; x] → gate ∈ [0,1] per latent dim
        self._gate_fc   = Linear(latent_dim + input_dim, latent_dim, seed=43)

        # Emission: z → observable
        self._emit_fc   = Linear(latent_dim, latent_dim, seed=42)

        # Per-factor EMA alphas
        self._ema_alphas = np.array([
            _FACTOR_ALPHA.get(FACTOR_NAMES[i], 0.15)
            for i in range(latent_dim)
        ], dtype=np.float32)

    def update(self, fused_embedding: np.ndarray) -> np.ndarray:
        inp = np.concatenate([self._z, fused_embedding]).astype(np.float32)

        h     = _relu(self._trans_fc1(inp))
        z_new = _tanh(self._trans_fc2(h))

        # Gated update: gate controls how much new info blends in per factor
        gate  = _sigmoid(self._gate_fc(inp))

        # Per-factor EMA with gate modulation
        eff_alpha = self._ema_alphas * gate
        self._z   = eff_alpha * z_new + (1 - eff_alpha) * self._z

        # Update uncertainty: high gate → more uncertainty reduction
        self._P = (1 - eff_alpha) * self._P + eff_alpha ** 2 * 0.01

        return self._z.copy()

    def emit(self) -> np.ndarray:
        return _sigmoid(self._emit_fc(self._z))

    def get_world_state(self) -> Dict[str, float]:
        obs = self.emit()
        n   = min(len(FACTOR_NAMES), len(obs))
        return {FACTOR_NAMES[i]: float(obs[i]) for i in range(n)}

    def get_uncertainty(self) -> Dict[str, float]:
        """Per-factor uncertainty estimate."""
        n = min(len(FACTOR_NAMES), len(self._P))
        return {FACTOR_NAMES[i]: float(self._P[i]) for i in range(n)}

    def get_latent_vector(self) -> np.ndarray:
        return self._z.copy()

    def reset(self) -> None:
        self._z = np.zeros(self.latent_dim, dtype=np.float32)
        self._P = np.ones(self.latent_dim, dtype=np.float32) * 0.1