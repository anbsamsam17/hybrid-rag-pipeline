# Hybrid RAG pipeline — common tasks.
# On Windows, run these in Git Bash (with `make`), or run the underlying command directly.
.PHONY: help install up down ingest serve eval eval-attribution eval-corrective eval-ragas test lint fmt typecheck clean

help:
	@echo "install    - editable install with dev extras"
	@echo "up/down    - start/stop Qdrant (docker compose)"
	@echo "ingest     - build the dense+sparse index from the corpus (+ meta.json)"
	@echo "serve      - run the FastAPI service"
	@echo "eval       - run the evaluation harness (comparison table + CI95)"
	@echo "eval-attribution - aggregate the measured attribution_rate over the golden set (needs LLM)"
	@echo "eval-corrective  - paired corrective-vs-baseline eval: activation rate + secondary deltas (needs LLM)"
	@echo "eval-ragas       - RAGAS-style generation quality: faithfulness + answer_relevancy (needs LLM)"
	@echo "test/lint/fmt/typecheck - quality gates"

install:
	pip install -e ".[dev]"

up:
	docker compose up -d qdrant

down:
	docker compose down

ingest:
	python -m rag.indexing.build

serve:
	uvicorn rag.api.app:app --reload --port 8000

eval:
	python -m rag.eval.harness

# Separate from `eval` on purpose: `make eval` stays LLM-free / key-free; this target calls the
# real LLM to aggregate the measured attribution_rate over the golden set (no CI in v1).
eval-attribution:
	python -m rag.eval.attribution

# Also separate from `eval` (LLM-required, mirrors eval-attribution): the paired corrective-vs-
# baseline comparison. PRIMARY = trace-only activation rate (judge-free); SECONDARY = correctness
# (real LLM judge), attribution regression guard, recall, cost. single_run, no CI.
eval-corrective:
	python -m rag.eval.corrective

# Also separate from `eval` (LLM-required, mirrors eval-attribution): the RAGAS-STYLE generation
# quality metrics (faithfulness + answer_relevancy) reimplemented over the Anthropic SDK. RAGAS is
# credited as the spec, NOT run as a library. single_run, no CI; publishable only when fully real.
eval-ragas:
	python -m rag.eval.generation_quality

test:
	pytest

lint:
	ruff check src tests && black --check src tests

fmt:
	ruff check --fix src tests && black src tests

typecheck:
	mypy src

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
