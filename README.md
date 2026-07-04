# Hybrid RAG Pipeline

> Production-grade Retrieval-Augmented Generation with **hybrid retrieval** (dense + sparse), **Reciprocal Rank Fusion**, **verified citations**, and a **rigorous retrieval evaluation harness**.

[![CI](https://img.shields.io/badge/CI-pending-lightgrey)](#)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-in%20development-orange)](#roadmap)

> 🚧 **Status: in active development.** This README describes the target design. Sections marked _(planned)_ are not implemented yet; the [roadmap](#roadmap) tracks progress. No benchmark numbers are published until they are reproducible via `make eval`.

---

## Why this project

Most RAG demos stop at `embeddings → top-k → LLM` over a single PDF. That proves nothing about retrieval quality. This project is built around the questions that actually matter in production:

- **Is the retrieval any good?** Measured rigorously with `recall@k`, `nDCG@k`, `MRR` on a labeled evaluation set, comparing **dense-only vs. sparse-only vs. hybrid vs. hybrid+rerank**. The comparison table includes paired bootstrap **CI95** to distinguish real wins from noise. The harness itself is hermetic (eval-scoped index, never touches production), anti-leakage (golden set validation, corpus freshness guards), reproducible (bit-exact, byte-diffable JSON artifact), and fully offline-testable.
- **Are the answers grounded?** Every generated claim is checked against its cited source span; the pipeline reports a measured **citation attribution rate** (not assumed). _(Coming: RAGAS faithfulness/answer-relevance metrics.)_
- **Does it hold up as a system?** Async API, containerized vector store, observability (latency p95, cost/request), CI, and architecture decision records capturing the key tradeoffs.

The differentiator is not the stack — it is the **evaluation rigor** and the **verified citations**.

## Evaluation results

**Retrieval quality: hybrid+rerank vs. baselines** (n=50 golden queries, paired bootstrap CI95, B=10000)

| Configuration | R@1 | R@5 | R@10 | nDCG@5 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|
| dense-only | 0.870 | 0.960 | 0.980 | 0.928 | 0.935 | 0.920 |
| sparse-only | 0.930 | 1.000 | 1.000 | 0.971 | 0.971 | 0.964 |
| hybrid | 0.910 | 0.980 | 1.000 | 0.958 | 0.965 | 0.953 |
| hybrid+rerank | 0.950 | 1.000 | 1.000 | 0.983 | 0.983 | 0.977 |

### Key observations (sample-specific)

**All differences are directional, not statistically significant at n=50.** Bootstrap CI95 comparisons on nDCG@10 (headline metric):

- **hybrid+rerank vs. dense-only** (pre-registered primary): diff=+0.048, CI95=[−0.007, +0.113] — **not significant**
- hybrid vs. dense-only: diff=+0.031, CI95=[−0.003, +0.069] — **not significant**
- hybrid+rerank vs. hybrid: diff=+0.018, CI95=[−0.025, +0.063] — **not significant**
- hybrid vs. sparse-only: diff=−0.006, CI95=[−0.048, +0.030] — **not significant**

**Caveats:**
- **Recall@10 saturates near 1.0** because the sample corpus has only one relevant chunk per query; recall@k is not a discriminating signal here. Use **nDCG@5 and nDCG@10** as the primary signal.
- **BM25 (sparse-only) is a surprisingly strong baseline** on this factual corpus (nDCG@10 = 0.971), outperforming dense-only (0.935). Dense embedding quality is data-dependent, not free.
- **Sample size (n=50)** means wider confidence intervals. Larger evaluation sets (e.g., n≥200) would tighten bounds and may reveal true wins or losses.

### Attribution rate (measured)

**Configuration:** hybrid+rerank, top_k_rerank=5

**Result:** micro (pooled) attribution_rate = **1.000** (57/57 grounded citations across all 50 answered queries)

**Secondary metrics:**
- Macro attribution_rate: 1.000
- Macro over answered queries: 1.000 (n_answered=50)
- Abstentions: 0/50

**Mandatory caveats:**

Measures grounding, not correctness: every supporting quote is a real span of its cited chunk; says nothing about factual correctness vs. reference answers, completeness, or whether every claim is cited.

Single run, no CI — LLM generation is not bit-exact reproducible; point estimate only.

n=50 in a small easy-corpus regime (12 docs / 39 chunks, single-fact verbatim-quotable queries, top-5 contexts); a 1.000 here will not generalize to larger/noisier corpora or multi-hop queries.

Verification is lexical (normalized substring / hardened token-overlap vs. the cited chunk only), not semantic entailment.

**Provenance (distinct from retrieval table above):**
- git commit: 57659b52e465
- corpus SHA-256: beb2701a7daea638
- Embedder: BAAI/bge-small-en-v1.5 (SentenceTransformer)
- Reranker: BAAI/bge-reranker-base (top_k=5)
- LLM: Claude Sonnet 4.6 (verification)
- Reproducible via: `make eval-attribution` (requires `ANTHROPIC_API_KEY`; not part of `make eval`)

### Reproducibility

**Provenance:**
- Embedder: BAAI/bge-small-en-v1.5 (SentenceTransformer)
- Reranker: BAAI/bge-reranker-base (cross-encoder)
- Corpus: data/sample (12 documents, 39 chunks)
- Golden set: 50 labeled queries (lexical + semantic)
- git commit: 9284155 (re-measured after a history rewrite changed commit SHAs; all
  metrics reproduced bit-identical to the original 50-query run — trees unchanged)
- corpus SHA-256: beb2701a
- Bootstrap: seed=12345, B=10000, paired percentile CI95
- numpy: 2.1.3

To reproduce these numbers:
```bash
make eval
```
or
```bash
python -m rag.eval.harness
```

The harness builds an evaluation-scoped index (never touches production), runs the four retrieval configurations, computes metrics, applies paired bootstrap, and outputs a reproducible JSON artifact + this table.

---

## Architecture (target)

```
Corpus ─▶ Ingestion (loaders, chunking strategies)
       ─▶ Indexing  (dense embeddings → Qdrant │ sparse → BM25)
       ─▶ Retrieval (dense ┐
                      sparse ┼─▶ RRF fusion ─▶ cross-encoder rerank)
       ─▶ Generation (citation-enforced prompt → structured output)
       ─▶ Verification (each citation ↔ source span)
       ─▶ (Optional) Self-corrective RAG: grade docs → rewrite query → retry retrieval;
                     verify answer → regenerate on low grounding
       ─▶ Eval harness (recall@k · nDCG@k · MRR · faithfulness · attribution)
```

### Self-corrective RAG (optional, opt-in)

An optional **self-corrective RAG** layer ([ADR-0007](docs/decisions/ADR-0007-self-corrective-rag-stategraph.md)) composes a bounded LangGraph `StateGraph` on top of the existing retrieval, generation, and verification pipeline. It implements two feedback loops:

1. **Grade → rewrite → retry retrieval.** After retrieval, an LLM grades each retrieved chunk's relevance. If too few docs are relevant (default: `< 1`), the query is rewritten and retrieval retried, up to `agentic_max_query_rewrites` times (default: 2, so ≤ 3 total retrievals).
2. **Verify → regenerate → retry generation.** After generation, the attribution checker scores grounding. If the answer has unsupported citations, it is regenerated, up to `agentic_max_regenerations` times (default: 1, so ≤ 2 total generations). Honest 0-citation abstentions are accepted and never retried.

**Constraints:**
- **Opt-in only** (`agentic_enabled=False` by default). The base single-pass pipeline (retrieve → generate → verify) is the unchanged default.
- **Provably bounded.** Two strictly-monotonic counters plus a derived `recursion_limit = (R+1)*3 + (G+1)*3 + 5` backstop guarantee termination.
- **Degrades gracefully.** On budget exhaustion, returns the best-effort answer with its **measured** `VerificationReport` and a trace; never raises or loops.
- **Pure composition.** Reuses existing `HybridRetriever.retrieve()`, `generate_answer()`, and `verify_answer()` verbatim; no fusion/rerank/attribution logic is duplicated.

**Evaluation status:** _(intentionally not yet measured)_ — no corrective-vs-baseline numbers are published. The impact of query rewriting on retrieval recall and of regeneration on grounding rate are the key signals. When measured, they will be reported via a paired bootstrap comparison over the golden set alongside n_rewrites and n_regenerations costs, with a regression guard (must not reduce attribution below baseline).

## Tech stack

| Layer | Choice |
|-------|--------|
| Orchestration | LangChain (+ LangGraph for the agentic layer) |
| Embeddings | OpenAI `text-embedding-3-small` / `BAAI/bge-small-en-v1.5` (local) |
| Vector store | Qdrant |
| Sparse retrieval | BM25 (`rank_bm25`) with hand-implemented RRF |
| Reranker | `BAAI/bge-reranker-base` (cross-encoder) |
| Generation | GPT-4o / Claude Sonnet, structured output via Pydantic |
| API | FastAPI (async, SSE streaming) |
| Evaluation | custom retrieval-metrics harness + RAGAS |
| Infra | Docker Compose, GitHub Actions CI |

## Roadmap

- [x] Ingestion + 3 chunking strategies (fixed / recursive / semantic)
- [x] Indexing (dense + sparse) with reproducible `meta.json`
- [x] Retrieval: dense, sparse, RRF fusion (k=60), cross-encoder rerank
- [x] Generation with enforced, structured citations
- [x] Citation verification (lexical + LLM-judge)
- [x] **Evaluation harness** (increment 2) — retrieval metrics (recall@k · nDCG@k · MRR) + paired bootstrap CI95, 4-config comparison table, anti-leakage guards, reproducible via `make eval`
- [x] FastAPI service with streaming + observability
- [x] Docker Compose + GitHub Actions CI (ruff · black · mypy · pytest) + tests
- [x] _(increment 3a)_ **attribution_rate aggregation** over the golden set (`make eval-attribution`) — measured micro (pooled) headline + macro + abstention split, on the hybrid+rerank answering config; published in [`README.md § Attribution rate (measured)`](#attribution-rate-measured)
- [ ] _(increment 3b)_ RAGAS faithfulness + answer-relevance
- [x] _(increment 4)_ **Self-corrective RAG** (LangGraph `StateGraph` over the existing pipeline; opt-in via `agentic_enabled=False`; two bounded feedback loops: grade+rewrite retrieval, verify+regenerate generation; provably terminates via recursion limit; offline-testable with deterministic fakes; evaluation deferred)

## Development

This repository is developed with a multi-agent orchestration setup (see [`.claude/`](.claude) and [`CLAUDE.md`](CLAUDE.md)) — specialized agents handle architecture, implementation, adversarial review, and documentation.

## Related work

- [`machine-learning-traffic-redressement-platform`](https://github.com/anbsamsam17/machine-learning-traffic-redressement-platform) — production ML platform with rigorous statistical evaluation (bootstrap CI95, paired McNemar, drift analysis). This project applies the same evaluation rigor to LLM retrieval.

## License

MIT — see [LICENSE](LICENSE).
