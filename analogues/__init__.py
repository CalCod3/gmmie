"""Historical-analogue retrieval (RAG) for forecasting + strategist dossier."""
from .index   import build_index, load_index, AnalogueIndex
from .retrieve import retrieve, AnalogueHit

__all__ = ["build_index", "load_index", "AnalogueIndex",
           "retrieve", "AnalogueHit"]
