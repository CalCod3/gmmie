"""
config.py — Central configuration, sourced entirely from environment variables.
Never hardcode credentials. Call load_dotenv() before importing CONFIG.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()  # reads .env if present


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()

def _env_bool(key: str, default: bool = True) -> bool:
    return _env(key, str(default)).lower() in ("1", "true", "yes")

def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default

def _env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default

def _env_list(key: str, default: str = "") -> List[str]:
    raw = _env(key, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass
class MarketDataConfig:
    twelve_data_api_key: str  = field(default_factory=lambda: _env("TWELVE_DATA_API_KEY"))
    twelve_data_ws_url:  str  = "wss://ws.twelvedata.com/v1/quotes/price"
    twelve_data_enabled: bool = field(default_factory=lambda: _env_bool("ENABLE_TWELVE_DATA"))

    alpha_vantage_api_key: str  = field(default_factory=lambda: _env("ALPHA_VANTAGE_API_KEY"))
    alpha_vantage_enabled: bool = field(default_factory=lambda: _env_bool("ENABLE_ALPHA_VANTAGE"))
    alpha_vantage_poll_s:  int  = 60

    polygon_api_key: str  = field(default_factory=lambda: _env("POLYGON_API_KEY"))
    polygon_ws_url:  str  = "wss://socket.polygon.io/forex"

    oanda_account_id:   str  = field(default_factory=lambda: _env("OANDA_ACCOUNT_ID"))
    oanda_access_token: str  = field(default_factory=lambda: _env("OANDA_ACCESS_TOKEN"))
    oanda_environment:  str  = field(default_factory=lambda: _env("OANDA_ENVIRONMENT", "practice"))
    oanda_enabled:      bool = field(default_factory=lambda: _env_bool("ENABLE_OANDA", False))

    symbols: List[str] = field(default_factory=lambda: ["XAU/USD"])


@dataclass
class NewsConfig:
    newsapi_key:     str  = field(default_factory=lambda: _env("NEWSAPI_KEY"))
    newsapi_enabled: bool = field(default_factory=lambda: _env_bool("ENABLE_NEWSAPI"))
    newsapi_query:   str  = 'gold OR XAUUSD OR "Federal Reserve" OR CPI OR inflation OR bullion'
    newsapi_poll_s:  int  = 60

    gdelt_enabled:  bool = field(default_factory=lambda: _env_bool("ENABLE_GDELT"))
    gdelt_query:    str  = "gold price Federal Reserve inflation"
    gdelt_poll_s:   int  = 300


@dataclass
class SentimentConfig:
    twitter_bearer_token: str  = field(default_factory=lambda: _env("TWITTER_BEARER_TOKEN"))
    twitter_enabled:      bool = field(default_factory=lambda: _env_bool("ENABLE_TWITTER", False))
    twitter_query:        str  = "(gold OR XAUUSD OR bullion OR #Gold) -is:retweet lang:en"
    twitter_poll_s:       int  = 30

    reddit_client_id:     str  = field(default_factory=lambda: _env("REDDIT_CLIENT_ID"))
    reddit_client_secret: str  = field(default_factory=lambda: _env("REDDIT_CLIENT_SECRET"))
    reddit_username:      str  = field(default_factory=lambda: _env("REDDIT_USERNAME"))
    reddit_password:      str  = field(default_factory=lambda: _env("REDDIT_PASSWORD"))
    reddit_user_agent:    str  = field(default_factory=lambda: _env("REDDIT_USER_AGENT", "GMMIE/1.0"))
    reddit_enabled:       bool = field(default_factory=lambda: _env_bool("ENABLE_REDDIT"))
    reddit_subreddits:    List[str] = field(default_factory=lambda: [
        "Gold", "Economics", "investing", "wallstreetbets", "Forex", "economy"
    ])
    reddit_poll_s: int = 90


@dataclass
class MacroConfig:
    fred_api_key:  str  = field(default_factory=lambda: _env("FRED_API_KEY"))
    fred_enabled:  bool = field(default_factory=lambda: _env_bool("ENABLE_FRED"))
    fred_base_url: str  = "https://api.stlouisfed.org/fred"
    fred_poll_s:   int  = 3600

    fred_series: List[str] = field(default_factory=lambda: [
        "FEDFUNDS", "CPIAUCSL", "UNRATE", "DGS10", "DGS2",
        "DTWEXBGS", "VIXCLS", "DCOILWTICO", "GOLDAMGBD228NLBM",
        "GDP", "T10YIE",
    ])

    trading_economics_key:     str  = field(default_factory=lambda: _env("TRADING_ECONOMICS_API_KEY"))
    trading_economics_enabled: bool = field(default_factory=lambda: _env_bool("ENABLE_TRADING_ECONOMICS", False))


@dataclass
class FeatureConfig:
    windows:        List[int] = field(default_factory=lambda: [5, 20, 60, 200])
    frac_diff_d:    float     = 0.4
    nlp_model:      str       = field(default_factory=lambda: _env("NLP_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    nlp_device:     str       = field(default_factory=lambda: _env("DEVICE", "cpu"))
    nlp_batch_size: int       = 16
    nlp_max_length: int       = 256


@dataclass
class EmbeddingConfig:
    market_dim: int   = 64
    text_dim:   int   = 64
    macro_dim:  int   = 32
    fused_dim:  int   = 128
    n_heads:    int   = 4
    dropout:    float = 0.1


@dataclass
class WorldStateConfig:
    latent_dim:  int = 16
    hidden_dim:  int = 128
    n_layers:    int = 2
    factor_names: List[str] = field(default_factory=lambda: [
        "risk_regime", "inflation_pressure", "liquidity",
        "sentiment", "volatility_state",
        "momentum_short", "momentum_long", "mean_reversion",
        "macro_shock", "dollar_strength", "yield_pressure",
        "geopolitical_risk", "market_microstructure",
        "positioning_bias", "options_skew", "carry",
    ])


@dataclass
class ForecastConfig:
    d_model:    int        = 128
    n_heads:    int        = 4
    n_layers:   int        = 2
    dropout:    float      = 0.1
    max_lag:    int        = 60
    horizons:   List[int]  = field(default_factory=lambda: [5, 30, 300])
    quantiles:  List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])


@dataclass
class MetaLearningConfig:
    replay_buffer_size:  int   = 10_000
    min_buffer_to_train: int   = 500
    retrain_every:       int   = 100
    xgb_n_estimators:    int   = 100
    xgb_max_depth:       int   = 4
    xgb_learning_rate:   float = 0.05


@dataclass
class MemoryConfig:
    faiss_index_type: str = "IVFFlat"
    embedding_dim:    int = 128
    top_k:            int = 5
    db_path:          str = "data/memory.faiss"
    metadata_path:    str = "data/memory_meta.pkl"


@dataclass
class DeliveryConfig:
    host:               str       = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port:               int       = field(default_factory=lambda: _env_int("PORT", 8001))
    redis_url:          str       = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379"))
    redis_channel:      str       = "output"
    ws_path:            str       = "/ws/intelligence"
    cors_origins:       List[str] = field(default_factory=lambda: ["*"])
    candle_buffer_size: int       = 500


@dataclass
class LakeConfig:
    """Offline data lake + research pipeline."""
    lake_path:           str = field(default_factory=lambda: _env(
        "GMMIE_LAKE_PATH", "data/lake/gmmie.duckdb"))
    cache_dir:           str = field(default_factory=lambda: _env(
        "GMMIE_CACHE_DIR", "data/cache"))
    backtest_train_days: int = 252 * 8
    backtest_test_days:  int = 63
    embargo_days:        int = 5


@dataclass
class LLMConfig:
    """Strategist + extractor."""
    backend:        str = field(default_factory=lambda: _env("LLM_BACKEND", "ollama"))
    claude_model:   str = field(default_factory=lambda: _env(
        "CLAUDE_MODEL", "claude-haiku-4-5-20251001"))
    ollama_model:   str = field(default_factory=lambda: _env(
        "OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M"))
    ollama_url:     str = field(default_factory=lambda: _env(
        "OLLAMA_URL", "http://localhost:11434"))
    anthropic_key:  str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))


@dataclass
class GMMIEConfig:
    market_data: MarketDataConfig   = field(default_factory=MarketDataConfig)
    news:        NewsConfig         = field(default_factory=NewsConfig)
    sentiment:   SentimentConfig    = field(default_factory=SentimentConfig)
    macro:       MacroConfig        = field(default_factory=MacroConfig)
    features:    FeatureConfig      = field(default_factory=FeatureConfig)
    embeddings:  EmbeddingConfig    = field(default_factory=EmbeddingConfig)
    world_state: WorldStateConfig   = field(default_factory=WorldStateConfig)
    forecast:    ForecastConfig     = field(default_factory=ForecastConfig)
    meta:        MetaLearningConfig = field(default_factory=MetaLearningConfig)
    memory:      MemoryConfig       = field(default_factory=MemoryConfig)
    delivery:    DeliveryConfig     = field(default_factory=DeliveryConfig)
    lake:        LakeConfig         = field(default_factory=LakeConfig)
    llm:         LLMConfig          = field(default_factory=LLMConfig)

    device:           str  = field(default_factory=lambda: _env("DEVICE", "cpu"))
    log_level:        str  = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    enable_profiling: bool = False

    def validate(self) -> List[str]:
        warnings = []
        md = self.market_data
        if not md.twelve_data_api_key and not md.alpha_vantage_api_key and not md.oanda_access_token:
            warnings.append("No market data source configured. Set TWELVE_DATA_API_KEY, ALPHA_VANTAGE_API_KEY, or OANDA_ACCESS_TOKEN.")
        if not self.news.newsapi_key and not self.news.gdelt_enabled:
            warnings.append("No news source configured.")
        if not self.macro.fred_api_key:
            warnings.append("FRED_API_KEY not set — macro data unavailable.")
        return warnings


CONFIG = GMMIEConfig()
