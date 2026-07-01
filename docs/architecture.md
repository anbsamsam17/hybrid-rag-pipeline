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
       ─▶ eval/         Retrieval evaluation harness (increment 2, complete):
                          · metrics.py — recall@k, nDCG@k, reciprocal_rank, MRR (pure, backend-agnostic)
                          · bootstrap.py — paired percentile CI95 (seed-stable, bit-exact reproducible)
                          · models.py — EvalProvenance, EvalReport, RetrievalMetrics, ConfigComparison
                          · golden.py — load the labeled golden set (never indexed)
                          · harness.py — run_eval() orchestrates hermetic eval-scoped index build,
                            scores 4 configs (dense-only / sparse-only / hybrid / hybrid+rerank) on
                            the committed golden set, computes per-config metrics + paired bootstrap
                            CI95 comparisons, enforces anti-leakage guards (golden path, coverage,
                            public corpus), emits byte-diffable JSON artifact + console report.
                          · k_values = (1, 3, 5, 10); headline metric = nDCG@10; secondary = recall@5.
                          · Runs via `make eval` → `python -m rag.eval.harness`.
                        [RAGAS faithfulness/answer-relevance + attribution_rate aggregation
                         forthcoming — increment 3, requires LLM judge]
```

Configuration is centralized in `src/rag/config.py` (`Settings`). Each `src/rag/<module>`
owns exactly one stage; the agentic layer composes the retrieval + verification modules
rather than duplicating them.

## Decisions

Architecture Decision Records live in [`decisions/`](decisions/) and guide cross-cutting
design. Use the `/adr-new` slash command to create one (assigned in sequence; `rag-architect`
authors, `docs-historian` formats and cross-links).

### Evaluation (increment 2, complete)

- [ADR-0004](decisions/ADR-0004-eval-metrics-and-paired-bootstrap.md) — **Retrieval metric
  definitions** (recall@k / nDCG@k / reciprocal_rank / MRR) as pure, backend-agnostic functions;
  paired percentile bootstrap CI95 (seed-stable 12345, B=10000 default); edge-case handling
  (dedup-first, honor k, finite validation). Consumed by the harness.
- [ADR-0005](decisions/ADR-0005-retrieval-eval-harness.md) — **Retrieval evaluation harness**
  (hermetic eval-scoped index build on public `sample_dir`, never touching production or private
  corpus; single `K_RETRIEVE=10` retrieval per config per query; anti-leakage guards that raise
  not warn: golden-path, eval-corpus-is-public, golden-coverage; paired bootstrap CI95 on
  headline nDCG@10 + secondary recall@5; pre-registered PRIMARY endpoint for multiplicity
  mitigation; byte-diffable JSON artifact + console report with `publishable` flag).

### Retrieval & indexing (increments 1–2)

- ADR-0001 — hybrid retrieval (dense + sparse) over dense-only _(to be written)_
- ADR-0002 — Reciprocal Rank Fusion (k=60) over weighted score fusion _(to be written)_
- ADR-0003 — chunking strategy choice (recursive 512/64), justified by retrieval metrics _(to be written)_

## Evaluation-first

No metric reaches the README unless it is reproducible from `make eval` over the committed
golden set. See [`../README.md#evaluation-results`](../README.md#evaluation-results) for the
published comparison table (4 retrieval configurations, paired bootstrap CI95 on recall@k /
nDCG@k / MRR). See [`../data/README.md`](../data/README.md) for the corpus/eval data layout.
