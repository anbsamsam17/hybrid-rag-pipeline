# ADR-0002 — Reciprocal Rank Fusion (RRF, k=60), hand-written, over weighted score fusion

- Status: Accepted (retrospective — recorded 2026-07-02)
- Date: 2026-07-02 (decision made and implemented earlier; recorded here retrospectively)
- Deciders: rag-architect (implemented by retrieval-engineer)
- Scope: `src/rag/retrieval/fusion.py` (the `reciprocal_rank_fusion` primitive) and its call
  site in `src/rag/retrieval/hybrid.py`. Decides *how* the dense and sparse ranked lists from
  [ADR-0001](ADR-0001-hybrid-retrieval.md) are merged before rerank. The `rrf_k=60` constant
  is pinned in `src/rag/config.py`.

## Context

Hybrid retrieval (ADR-0001) produces two ranked lists whose scores are on **incomparable
scales**: Qdrant returns cosine similarities (bounded ~[−1, 1]) while BM25 returns unbounded
Okapi scores whose magnitude depends on corpus statistics and query length. Any fusion that
adds or interpolates the *scores* directly must first normalize them onto a common scale, and
the choice of normalization (min-max, z-score, softmax) plus any per-retriever weight becomes
a set of hyperparameters. On a 50-query golden set, tuning those hyperparameters is a fast
route to overfitting the eval. The fusion step is also on the critical path for
reproducibility: it must be deterministic and tie-stable so a rebuild yields byte-identical
rankings.

## Decision drivers

- **Score-scale independence** — cosine and BM25 are not comparable; a fusion that never reads
  raw scores sidesteps the normalization problem entirely.
- **Few, robust hyperparameters** — a fusion with one literature-default knob beats one whose
  weights must be tuned against a small eval set.
- **Determinism / tie-stability** — equal fused scores must break deterministically so
  `/repro-audit` sees identical orderings across rebuilds; no set/dict iteration order may leak.
- **Transparency** — the primitive is a repo signature ("hand-written RRF"); it must be pure,
  dependency-free, and line-by-line defensible.

## Options considered

**(a) Weighted score fusion (normalized score interpolation).** `w·norm(dense) +
(1−w)·norm(sparse)`. Pros: can in principle exploit score *magnitude*, not just rank. Cons:
requires a normalization scheme and a weight `w`; both are hyperparameters with no principled
default; tuning `w` on n=50 overfits and would make the headline table a function of a fitted
constant; normalization is itself brittle (a single outlier BM25 score skews min-max). Every
added knob is a reproducibility and overclaiming liability.

**(b) Rerank-only (skip fusion; feed a single retriever's top-N to the cross-encoder).**
Pros: fewer stages. Cons: throws away the second retriever's *recall* — a relevant chunk that
only BM25 surfaced never enters the rerank pool, so the cross-encoder can never recover it.
Fusion-before-rerank exists precisely to build a high-recall candidate pool from both lists;
`HybridRetriever` hydrates sparse-only ids the dense search never returned for exactly this
reason.

**(c) Reciprocal Rank Fusion, k=60, hand-written (chosen).** Score = Σ over lists of
`1/(k + rank)`, rank 1-based, k=60. Pros: reads **rank only**, so no score normalization and
no per-retriever weight; one hyperparameter with a strong literature default (Cormack, Clarke
& Buettcher, 2009); pure and deterministic. Cons: discards score magnitude (a dense hit at
0.99 and one at 0.71 both count as "rank 1"); k is a single global constant, not per-corpus
tuned. Both are accepted — magnitude is exactly the incomparable quantity we want to ignore.

## Decision

Fuse with the hand-written `reciprocal_rank_fusion(rankings, k=60)` in
`src/rag/retrieval/fusion.py`. Load-bearing implementation details, each tested:

1. **Rank is 1-based**: the top item of a list contributes `1/(k+1)`, never `1/(k+0)`. An
   off-by-one here silently inflates every top item's weight.
2. **Duplicates within one list collapse to that id's best (first) rank** in that list, so a
   repeated id cannot be double-counted inside one retriever's contribution.
3. **Tie-stable ordering by first-appearance.** Equal fused scores break by the order in which
   an id was first seen while scanning `rankings` left-to-right, top-to-bottom (a `first_seen`
   counter), then the final sort key is `(-fused_score, first_seen_index)`. This never leaks
   `set`/`dict` iteration order. Ties are **real** here: an id that appears only in the dense
   list at rank r and a different id that appears only in the sparse list at rank r both score
   `1/(k+r)`, so the tie-break decides their relative order.
4. **`k <= 0` raises** (`ValueError`) — the denominator must stay strictly positive.

`k=60` and the fixed call order `reciprocal_rank_fusion([dense_ids, sparse_ids], ...)` in
`hybrid.py` are pinned; the constant lives in `Settings.rrf_k` and is recorded in `meta.json`.

## Consequences

Positive:
- No score normalization and no tuned weight → nothing about fusion can be overfit to the
  n=50 golden set; the fused ranking is a pure function of the two input orderings and k.
- Deterministic and tie-stable → a rebuild reproduces byte-identical fused orderings, which is
  what makes the eval bit-exact and `/repro-audit` meaningful.
- Pure, dependency-free, hand-written → defensible line by line and unit-tested including the
  1-based-rank and tie-stability invariants.

Negative / accepted:
- **Magnitude is discarded.** RRF cannot distinguish a barely-relevant rank-1 hit from a
  strong one; the cross-encoder rerank downstream is where score-sensitive reordering happens.
- **k is a single global constant**, not per-corpus tuned. This is deliberate — a tuned k
  would reintroduce the overfitting risk we chose RRF to avoid — but it means k=60 is a
  literature prior, not an eval-optimized value on this corpus.
- **Tie-break is first-appearance, not by `chunk_id`.** It is fully deterministic *because*
  both input lists are themselves tie-stable (BM25 breaks ties by corpus index; dense comes
  from a deterministic ANN order) **and** the list order passed to RRF is fixed
  (`[dense, sparse]`). A `chunk_id`-based tie-break would be robust to input-order changes too;
  first-appearance is not, so the fixed call order in `hybrid.py` is part of the contract and
  must not be reordered casually.

## Evidence

There is **no committed head-to-head ablation** of RRF vs weighted fusion in the eval
artifacts — the decision rests on the reproducibility/overfitting argument above, not on a
measured RRF-vs-weighted delta, and this ADR does not claim one. What *is* measured: the RRF
hybrid path is exercised as the `hybrid` and `hybrid+rerank` rows of the committed n=50 table
(README "Evaluation results"), which are reproducible via `make eval` (git 7e8ccb3, seed
12345, B=10000). The fusion primitive's correctness (1-based rank, duplicate collapse,
tie-stability) is covered by unit tests, not by benchmark numbers. If a weighted-fusion
baseline is ever added, it should enter the harness as a fifth config so the comparison is
paired and reproducible rather than asserted.

## Cross-links

Consumes the two ranked lists from [ADR-0001](ADR-0001-hybrid-retrieval.md); the fused pool it
produces is reranked and then scored by the harness in
[ADR-0005](ADR-0005-retrieval-eval-harness.md) using the metrics fixed in
[ADR-0004](ADR-0004-eval-metrics-and-paired-bootstrap.md) (whose rank convention — 1-based,
best-first — matches this primitive by design). Linked from `docs/architecture.md` (Decisions).
