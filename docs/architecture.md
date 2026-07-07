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
       ─▶ agentic/      corrective_rag.py — LangGraph StateGraph (optional self-correction, ADR-0007):
                          analyze → retrieve → grade docs → (low relevance) rewrite & retry →
                          generate → verify → (ungrounded) regenerate. Pure composition over the
                          existing HybridRetriever.retrieve / generate_answer / verify_answer;
                          TypedDict channel state, Pydantic boundary (CorrectiveRAGRequest/Result);
                          bounded rewrite + regenerate budgets + derived recursion_limit (provably
                          terminates); honest abstention accepted (0-citation never regenerated);
                          degrades to best-effort answer + measured VerificationReport, never raises
                          or loops. Offline-fake tested; opt-in (agentic_enabled=False by default).
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
                          · attribution.py — attribution_rate aggregation (increment 3a, ADR-0006):
                            reuses verify_answer + generate_answer over hybrid+rerank @ top_k_rerank,
                            shares prepare_hermetic_eval with the harness; micro (pooled) headline +
                            macro + macro-over-answered + n_abstained; NO CI (LLM not bit-exact);
                            publishable only when LLM+embedder+reranker are all real. Offline-fake
                            tested (byte-stable). Runs via `make eval-attribution` →
                            `python -m rag.eval.attribution` (LLM-required; NOT part of `make eval`).
                        [RAGAS faithfulness/answer-relevance forthcoming — later increment,
                         requires LLM judge]
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
- [ADR-0006](decisions/ADR-0006-attribution-rate-aggregation.md) — **Attribution-rate
  aggregation** (offline-fake-first): a separate `make eval-attribution` entry point that reuses
  `verify_answer` + `generate_answer` over the real answering config (hybrid+rerank @
  `top_k_rerank`), sharing the hermetic build + anti-leakage guards via the extracted
  `prepare_hermetic_eval`; micro (pooled) headline immune to the 0-citation convention, macro +
  macro-over-answered + `n_abstained` secondary; no CI in v1 (LLM not bit-exact reproducible);
  `publishable` only when LLM + embedder + reranker are all real; measures grounding, not
  correctness. `make eval` stays LLM-free.

### Agentic (self-corrective RAG, increment 4)

- [ADR-0007](decisions/ADR-0007-self-corrective-rag-stategraph.md) — **Self-corrective RAG layer**
  as a bounded LangGraph `StateGraph` that **composes on top of** (never duplicates) the existing
  `HybridRetriever.retrieve` / `generate_answer` / `verify_answer`: nodes `analyze → retrieve →
  grade docs → (low relevance) rewrite & retry → generate → verify → (ungrounded) regenerate`.
  TypedDict channel state with a Pydantic boundary (`CorrectiveRAGRequest` / frozen
  `CorrectiveRAGResult`); a new agentic-owned `CorrectiveLLM` Protocol (batch structured doc-grading
  + query-rewrite via `messages.parse`, SDK rules honored) leaving the generation contract untouched.
  Two bounded counters (`agentic_max_query_rewrites=2`, `agentic_max_regenerations=1`) + a derived
  `recursion_limit` backstop **provably terminate** the graph; honest abstention (0 citations) is
  accepted and never regenerated; on budget exhaustion it degrades to a best-effort `Answer` + its
  **measured** `VerificationReport` + a trace, never raising or looping. `langgraph>=0.2` is already a
  runtime dep (no pyproject change). Offline-fake tested; `agentic_enabled=False` by default; the
  API surface and a corrective-vs-baseline eval are deferred.
- [ADR-0008](decisions/ADR-0008-corrective-vs-baseline-eval.md) — **Corrective-vs-baseline eval**
  (offline-fake-first): a separate `make eval-corrective` entry point that runs the SAME golden set
  through two arms differing only in agentic off (single pass) vs on (`run_corrective_rag`), over
  ONE shared hermetic index / backends / generation LLM (reusing `prepare_hermetic_eval` +
  `recall_at_k`/`ndcg_at_k` + the measured `attribution_rate`). **Primary metric = activation rate**
  (does the loop fire? read trace-only from the frozen `CorrectiveRAGResult`: `n_rewrites` /
  `n_regenerations` / `final_query != original_query`; judge-free, byte-stable, expected ~0/50 on
  this saturated corpus). Secondary (all expected ≈ 0 delta, directional only): answer correctness
  vs `reference_answer` (blind LLM-judge + deterministic lexical F1), attribution regression guard,
  recall/nDCG on final contexts, and a trace-derived cost delta (`2*n_rewrites + n_regenerations +
  1` extra LLM calls/query — +1 grade call even at zero activation). No CI, `single_run`,
  `publishable` only when generation LLM + corrective LLM + judge + embedder + reranker are all
  real. Adds no instrumentation to the agentic layer; `make eval` stays LLM-free.

### Retrieval & indexing (increments 1–2)

- [ADR-0001](decisions/ADR-0001-hybrid-retrieval.md) — **Hybrid retrieval (dense + sparse)**
  over dense-only: robust coverage across lexical+semantic queries; four-way comparison
  (dense / sparse / hybrid / hybrid+rerank) with paired bootstrap CI95; honest caveats
  (n=50 underpowered, RRF alone does not beat BM25 on this corpus, BM25 strength is
  corpus-specific).
- [ADR-0002](decisions/ADR-0002-reciprocal-rank-fusion.md) — **Reciprocal Rank Fusion (k=60),
  hand-written**, over weighted score fusion: score-scale-independent; one literature-default
  hyperparameter; deterministic and tie-stable; pure and dependency-free; no head-to-head
  weighted-fusion ablation (decision rests on reproducibility/overfitting, not a measured win).
- [ADR-0003](decisions/ADR-0003-chunking-strategy.md) — **Chunking strategy: recursive
  character splitter, 512 / 64 (chars)**, over fixed or semantic: boundary quality (para/line/sentence);
  determinism (ids span-derived); the golden set is frozen to this config, so any change
  invalidates it and requires re-minting; no eval sweep vs alternatives (corpus-specific
  selection, not measured).

## Evaluation-first

No metric reaches the README unless it is reproducible from `make eval` over the committed
golden set. See [`../README.md#evaluation-results`](../README.md#evaluation-results) for the
published comparison table (4 retrieval configurations, paired bootstrap CI95 on recall@k /
nDCG@k / MRR). See [`../data/README.md`](../data/README.md) for the corpus/eval data layout.
