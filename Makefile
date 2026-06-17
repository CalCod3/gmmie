# GMMIE — research & runtime orchestration.
# Convention: every target is idempotent. `make` (no args) prints help.

SHELL := /bin/bash
PY    := python3

HORIZON     ?= 20d
EPOCHS      ?= 200
LLM_BACKEND ?= ollama
LLM_BATCH   ?= 500

# ── meta ──────────────────────────────────────────────────────────────────────

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "GMMIE — supremacy build"
	@echo
	@echo "Setup"
	@echo "  install        — pip install -r requirements.txt"
	@echo "  check          — syntax-check every Python module"
	@echo
	@echo "Data lake"
	@echo "  lake           — backfill every historical source"
	@echo "  lake-fred      — FRED macro only"
	@echo "  lake-yahoo     — Yahoo OHLCV only"
	@echo "  lake-cot       — CFTC COT only"
	@echo "  lake-gld       — SPDR GLD holdings only"
	@echo "  lake-fomc      — FOMC statements only"
	@echo "  lake-info      — table row counts"
	@echo
	@echo "Research"
	@echo "  llm-extract    — LLM news feature extraction"
	@echo "  train          — train hybrid forecaster (HORIZON=5d|20d|60d)"
	@echo "  backtest       — train + walk-forward backtest"
	@echo "  predict        — emit a prediction for today (loads active model)"
	@echo "  settle         — settle outcomes for any elapsed predictions"
	@echo "  skill          — print recent forecaster skill stats"
	@echo "  thesis         — generate today's strategist thesis"
	@echo "  registry       — show active-model registry"
	@echo "  diagnostics    — show importance + drift for active model"
	@echo "  analogues      — top-K historical analogues for today"
	@echo
	@echo "Runtime"
	@echo "  run            — start the live engine + API (legacy real-time)"
	@echo
	@echo "Hygiene"
	@echo "  clean-cache    — wipe __pycache__"
	@echo "  clean-lake     — DANGEROUS: delete data/lake/"
	@echo "  clean-models   — DANGEROUS: delete data/models/"

# ── setup ────────────────────────────────────────────────────────────────────

.PHONY: install check test
install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -v --tb=short

check:
	@$(PY) - <<-'PY'
	import ast, pathlib, sys
	root = pathlib.Path('.')
	files = sorted(p for p in root.rglob('*.py')
	               if '.venv' not in p.parts and '__pycache__' not in p.parts)
	bad = 0
	for f in files:
	    try:
	        ast.parse(f.read_text())
	    except SyntaxError as e:
	        print(f"ERR  {f}: {e}")
	        bad += 1
	print(f"{len(files)-bad}/{len(files)} OK")
	sys.exit(1 if bad else 0)
	PY

# ── data lake ────────────────────────────────────────────────────────────────

.PHONY: lake lake-fred lake-yahoo lake-cot lake-gld lake-fomc lake-info

lake:
	$(PY) -m data_lake.build_lake --all --since 2010-01-01

lake-fred:
	$(PY) -m data_lake.build_lake --fred --since 2000-01-01

lake-yahoo:
	$(PY) -m data_lake.build_lake --yahoo --since 1990-01-01

lake-cot:
	$(PY) -m data_lake.build_lake --cot --cot-start-year 2010

lake-gld:
	$(PY) -m data_lake.build_lake --gld

lake-fomc:
	$(PY) -m data_lake.build_lake --fomc --since 2008-01-01

lake-info:
	@$(PY) - <<-'PY'
	from data_lake.db import LakeDB
	with LakeDB(read_only=True) as db:
	    for tbl in ("prices_d","macro_d","cot_disagg","etf_flows","news",
	                "news_features","fomc_statements","predictions","outcomes","theses"):
	        try:
	            n = db.scalar(f"SELECT COUNT(*) FROM {tbl}") or 0
	            print(f"  {tbl:18s} {n:>10,}")
	        except Exception as e:
	            print(f"  {tbl:18s}  (n/a: {e})")
	PY

# ── research ─────────────────────────────────────────────────────────────────

.PHONY: llm-extract train backtest predict settle skill thesis registry

llm-extract:
	$(PY) -m research.llm_extract --backend $(LLM_BACKEND) --batch $(LLM_BATCH) --loop

train:
	$(PY) -m research.train --horizon $(HORIZON) --epochs $(EPOCHS)

backtest:
	$(PY) -m research.train --horizon $(HORIZON) --epochs $(EPOCHS) --backtest

predict:
	$(PY) -m research.predict --horizon $(HORIZON) --predict --settle

settle:
	$(PY) -m research.predict --horizon $(HORIZON) --settle

skill:
	$(PY) -m research.predict --horizon $(HORIZON) --skill

thesis:
	$(PY) -m research.strategist --horizon $(HORIZON)

registry:
	@$(PY) -c "import json; from research.registry import show; print(json.dumps(show(), indent=2))"

diagnostics:
	@$(PY) - <<-'PY'
	import json, pathlib
	from research.registry import active_path
	p = active_path('$(HORIZON)')
	if not p:
	    print("no active model for horizon $(HORIZON)"); raise SystemExit(1)
	d = p / "diagnostics.json"
	if d.exists():
	    print(d.read_text())
	else:
	    print("no diagnostics.json in", p)
	PY

analogues:
	@$(PY) - <<-'PY'
	import datetime as dt, pandas as pd
	from analogues import load_index, retrieve, analogue_summary
	from data_lake.db import LakeDB
	from research.features import build_panel
	idx = load_index('$(HORIZON)')
	if not idx:
	    print("no analogue index for $(HORIZON) — run `make train HORIZON=$(HORIZON)`")
	    raise SystemExit(1)
	with LakeDB(read_only=True) as db:
	    panel = build_panel(db)
	row = panel.tail(1).iloc[0]
	hits = retrieve(idx, row, k=8)
	for h in hits:
	    print(f"  {h.date}  sim={h.similarity:.3f}  realised={h.realised_logret:+.4f}  VIX={h.vix}")
	import json
	print(json.dumps(analogue_summary(hits), indent=2))
	PY

# ── runtime ──────────────────────────────────────────────────────────────────

.PHONY: run
run:
	$(PY) main.py

# ── hygiene ──────────────────────────────────────────────────────────────────

.PHONY: clean-cache clean-lake clean-models
clean-cache:
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "__pycache__ wiped"

clean-lake:
	@read -p "Delete data/lake/ ? [y/N] " ans && [ "$$ans" = "y" ] && rm -rf data/lake/ && echo "lake deleted" || echo "aborted"

clean-models:
	@read -p "Delete data/models/ ? [y/N] " ans && [ "$$ans" = "y" ] && rm -rf data/models/ && echo "models deleted" || echo "aborted"
