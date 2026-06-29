# Architecture

> Status: scaffolding. This document tracks the system design; it is kept in sync with
> `src/rag/` and cross-links the Architecture Decision Records in `docs/decisions/`.

## Pipeline stages

```
Corpus ─▶ ingestion/    loaders (PDF/DOCX/MD/HTML) → chunking (fixed | recursive | semantic)
       ─▶ indexing/     dense embeddings → Qdrant │ sparse → BM25   (+ reproducible meta.json)
       ─▶ retrieval/    dense.py · sparse.py · fusion.py (RRF, k=60) · rerank.py (cross-encoder)
       ─▶ generation/   citation-enforced prompt → Pydantic Answer/Citation (structured output)
       ─▶ verification/ attribution checker: each claim ↔ its source span (measured rate)
       ─▶ agentic/      corrective_rag.py — LangGraph StateGraph (optional self-correction)
       ─▶ api/          FastAPI: /ingest /query /eval, SSE streaming, observability
       ─▶ eval/         metrics.py (recall@k · nDCG@k · MRR) + bootstrap.py (paired CI95)
                        [harness + golden set wiring + RAGAS + attribution_rate forthcoming]
```

Configuration is centralized in `src/rag/config.py` (`Settings`). Each `src/rag/<module>`
owns exactly one stage; the agentic layer composes the retrieval + verification modules
rather than duplicating them.

## Decisions

Architecture Decision Records live in [`decisions/`](decisions/) (use `/adr-new`):

- ADR-0001 — hybrid retrieval (dense + sparse) over dense-only _(to be written)_
- ADR-0002 — Reciprocal Rank Fusion over weighted score fusion _(to be written)_
- ADR-0003 — chunking strategy choice, justified by retrieval metrics _(to be written)_
- [ADR-0004](decisions/ADR-0004-eval-metrics-and-paired-bootstrap.md) — retrieval metric
  definitions (recall@k / nDCG@k / MRR) and paired percentile bootstrap CI95

## Evaluation-first

No metric reaches the README unless it is reproducible from `make eval` over the committed
golden set. See [`../data/README.md`](../data/README.md) for the corpus/eval data layout.
