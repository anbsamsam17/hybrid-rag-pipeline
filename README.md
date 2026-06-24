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

- **Is the retrieval any good?** Measured with `recall@k`, `nDCG@k`, `MRR` on a labeled evaluation set — comparing dense-only vs. sparse-only vs. hybrid vs. hybrid+rerank.
- **Are the answers grounded?** Every generated claim is checked against its cited source span; the pipeline reports a measured **citation attribution rate** rather than assuming faithfulness.
- **Does it hold up as a system?** Async API, containerized vector store, observability (latency p95, cost/request), CI, and architecture decision records.

The differentiator is not the stack — it is the **evaluation** and the **citation verification**.

## Architecture (target)

```
Corpus ─▶ Ingestion (loaders, chunking strategies)
       ─▶ Indexing  (dense embeddings → Qdrant │ sparse → BM25)
       ─▶ Retrieval (dense ┐
                      sparse ┼─▶ RRF fusion ─▶ cross-encoder rerank)
       ─▶ Generation (citation-enforced prompt → structured output)
       ─▶ Verification (each citation ↔ source span)
       ─▶ Eval harness (recall@k · nDCG@k · MRR · faithfulness · attribution)
```

An optional **self-corrective RAG** layer (LangGraph) grades retrieved documents and rewrites the query on low relevance before generating.

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

- [ ] Ingestion + 3 chunking strategies (fixed / recursive / semantic)
- [ ] Indexing (dense + sparse) with reproducible `meta.json`
- [ ] Retrieval: dense, sparse, RRF fusion, cross-encoder rerank
- [ ] Generation with enforced, structured citations
- [ ] Citation verification + attribution rate
- [ ] **Evaluation harness** + labeled golden set + comparison report
- [ ] FastAPI service with streaming + observability
- [ ] Docker Compose + CI + tests
- [ ] _(optional)_ Self-corrective RAG (LangGraph)

## Development

This repository is developed with a multi-agent orchestration setup (see [`.claude/`](.claude) and [`CLAUDE.md`](CLAUDE.md)) — specialized agents handle architecture, implementation, adversarial review, and documentation.

## Related work

- [`machine-learning-traffic-redressement-platform`](https://github.com/anbsamsam17/machine-learning-traffic-redressement-platform) — production ML platform with rigorous statistical evaluation (bootstrap CI95, paired McNemar, drift analysis). This project applies the same evaluation rigor to LLM retrieval.

## License

MIT — see [LICENSE](LICENSE).
