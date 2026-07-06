# ADR-0001 — Hybrid retrieval (dense + sparse) over dense-only or sparse-only

- Status: Accepted (retrospective — recorded 2026-07-02)
- Date: 2026-07-02 (decision made and implemented earlier; recorded here retrospectively)
- Deciders: rag-architect (implemented by retrieval-engineer)
- Scope: `src/rag/retrieval/{dense.py,sparse.py,hybrid.py}` and the two indices they read
  (`src/rag/indexing/{embeddings.py → Qdrant,sparse.py → BM25}`). The fusion primitive itself
  is decided separately in [ADR-0002](ADR-0002-reciprocal-rank-fusion.md); this ADR only
  decides *that* we combine dense and sparse rather than committing to a single retriever.

## Context

The whole premise of the repo is that a real retrieval system should not stop at
`embed → top-k`. Two families of retriever have complementary failure modes:

- **Dense** (`DenseRetriever`, `src/rag/retrieval/dense.py`) embeds the query with the same
  bi-encoder used at index time (`BAAI/bge-small-en-v1.5`) and does ANN search in Qdrant
  (cosine). It captures paraphrase / semantic similarity but can miss exact terms, rare
  tokens, IDs, and numbers.
- **Sparse** (`SparseRetriever` over `BM25Index`, `src/rag/retrieval/sparse.py`,
  `src/rag/indexing/sparse.py`) scores lexical overlap with BM25 Okapi (k1=1.5, b=0.75). It
  nails exact-term and keyword queries but is blind to synonyms and rephrasings.

The decision is whether to ship one of them or both. Because the corpus is a factual
driving-rules corpus (12 documents / 39 chunks) and the golden set deliberately mixes
**lexical** and **semantic** query phrasings, neither failure mode is hypothetical here.

## Decision drivers

- **Robustness across query types** — the golden set is a lexical+semantic mix; a single
  retriever is strong on one half and weak on the other.
- **No overclaiming** — at n=50 the eval cannot *prove* a hybrid win, so the decision must
  stand on robustness and query-type coverage, not on a claimed significant metric gap.
- **Reproducibility** — both indices build deterministically and their provenance is captured
  in `meta.json` (corpus SHA-256, embedding model+dim, BM25 params), so the comparison is
  auditable.
- **Clean seam** — dense and sparse expose the *same* `list[tuple[chunk_id, score]]` contract,
  so a fusion layer can consume them without knowing which scorer produced what.

## Options considered

**(a) Dense-only.** Simplest; one index, one score scale. Cons: on this corpus it is the
**weakest** config measured (nDCG@10 = 0.935, MRR = 0.920) — embedding quality is
data-dependent and this factual corpus does not play to bge-small's strengths. Misses
exact-term queries.

**(b) Sparse-only (BM25).** No model, no GPU, cheap, and — surprisingly — the **strongest
single** retriever here (nDCG@10 = 0.971, R@5 = 1.000). Cons: brittle on paraphrase/semantic
queries; a strong BM25 on a clean keyword-rich corpus does not generalize to noisier or more
conversational corpora. Betting the system on it overfits to this corpus's lexical character.

**(c) Hybrid (dense + sparse, chosen).** Run both, fuse by rank (RRF, ADR-0002), then rerank.
Pros: covers both query families; `hybrid+rerank` leads on **every** tabled metric. Cons:
two indices to build and keep in sync; more moving parts; and — honestly — the *plain* RRF
hybrid (no rerank) does **not** beat sparse-only on this corpus (nDCG@10 0.965 vs 0.971).
The cross-encoder rerank is what carries hybrid past the BM25 baseline.

## Decision

Ship **hybrid**: `HybridRetriever` (`src/rag/retrieval/hybrid.py`) runs dense (`top_k_dense=20`)
and sparse (`top_k_sparse=20`), fuses the two ranked id lists with hand-written RRF (k=60),
hydrates the fused pool from Qdrant payloads (including sparse-only ids the dense search never
returned), and optionally reranks the fused top-N with the `BAAI/bge-reranker-base`
cross-encoder down to `top_k_rerank=5`. Dense-only and sparse-only remain first-class,
independently testable retrievers and are kept as **eval baselines** — they are not dead code,
they are the control arms of the headline comparison.

The decision rests on **robustness across query types**, not on a proven metric win. At n=50
no paired comparison is significant, so the honest justification is coverage: hybrid is the
only config that is not systematically weak on either the lexical or the semantic half of the
query distribution.

## Consequences

Positive:
- One retrieval path handles both lexical and semantic queries; `hybrid+rerank` is the
  best-measured config on every metric and is the default answering config downstream.
- Dense/sparse survive as clean baselines, so the eval can always re-audit whether hybrid is
  still earning its complexity on a future corpus.
- Both index builds are reproducible and provenance-stamped (`meta.json`), so the four-way
  comparison is auditable end to end.

Negative / accepted:
- **Two indices, more surface.** Qdrant + BM25 must be built from the *same* chunk set or the
  fusion joins garbage; the build pipeline enforces a single chunking pass feeding both.
- **The win is directional, not proven at n=50.** Every paired CI95 straddles zero. We do not
  round this up to a win; a larger golden set is the only real fix.
- **Plain RRF hybrid underperforms sparse-only here.** The value of the hybrid path on this
  corpus is realized only *after* the rerank; on a corpus where BM25 is weaker the plain
  hybrid margin should widen. This is a corpus-specific caveat, stated, not hidden.
- **BM25's strength is corpus-specific.** Do not generalize "sparse ≈ hybrid" beyond this
  clean, keyword-rich factual corpus.

## Evidence

From the committed reproducible `make eval` run (README "Evaluation results"; n=50 golden
queries, paired percentile bootstrap, B=10000, seed=12345; corpus `data/sample`, 12 docs /
39 chunks; embedder `BAAI/bge-small-en-v1.5`, reranker `BAAI/bge-reranker-base`; git 7e8ccb3,
corpus SHA-256 `beb2701a`, numpy 2.1.3):

| Configuration | R@1 | R@5 | R@10 | nDCG@5 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|
| dense-only | 0.870 | 0.960 | 0.980 | 0.928 | 0.935 | 0.920 |
| sparse-only | 0.930 | 1.000 | 1.000 | 0.971 | 0.971 | 0.964 |
| hybrid | 0.910 | 0.980 | 1.000 | 0.958 | 0.965 | 0.953 |
| hybrid+rerank | 0.950 | 1.000 | 1.000 | 0.983 | 0.983 | 0.977 |

Paired bootstrap CI95 on nDCG@10 (headline metric): hybrid+rerank vs dense-only
(pre-registered primary) diff=+0.048, CI95=[−0.007, +0.113] — **not significant**; hybrid vs
dense-only +0.031 [−0.003, +0.069] — not significant; hybrid+rerank vs hybrid +0.018
[−0.025, +0.063] — not significant; hybrid vs sparse-only −0.006 [−0.048, +0.030] — not
significant (hybrid slightly *behind* sparse before rerank). Recall@10 saturates at ~1.0
(one relevant chunk per query) and is not a discriminating signal; nDCG is the axis of
separation. See [ADR-0005](ADR-0005-retrieval-eval-harness.md) for the harness that produced
these numbers.

## Cross-links

Fusion primitive: [ADR-0002](ADR-0002-reciprocal-rank-fusion.md). Chunking that both indices
consume: [ADR-0003](ADR-0003-chunking-strategy.md). Eval harness/metrics that measured this:
[ADR-0004](ADR-0004-eval-metrics-and-paired-bootstrap.md) /
[ADR-0005](ADR-0005-retrieval-eval-harness.md). Linked from `docs/architecture.md` (Decisions).
