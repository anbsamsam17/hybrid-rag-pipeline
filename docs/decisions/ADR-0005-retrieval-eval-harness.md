# ADR-0005 — Retrieval evaluation harness: hermetic index, single-list scoring, and headline comparisons

- Status: Accepted
- Date: 2026-06-30
- Deciders: rag-architect (implemented by eval-scientist)
- Scope: `src/rag/eval/{harness.py,golden.py}` plus the `EvalProvenance` / `EvalReport`
  models in `src/rag/eval/models.py`. This increment is **retrieval-only**: it scores
  dense-only / sparse-only / hybrid / hybrid+rerank on `recall@k` / `nDCG@k` / `MRR` with a
  paired bootstrap CI95. The RAGAS faithfulness/answer-relevance metrics and the custom
  `attribution_rate` aggregation are **explicitly out of scope** here — they require an LLM
  judge and land in increment 3 — so this harness never instantiates an LLM and runs fully
  offline in tests. Consumes the metric + bootstrap contracts fixed in
  [ADR-0004](ADR-0004-eval-metrics-and-paired-bootstrap.md).

## Context

ADR-0004 fixed the pure metric definitions (recall@k / nDCG@k / RR / MRR) and the paired
percentile bootstrap. What was missing was the harness that turns the committed golden set
into the repo's headline deliverable: a per-config comparison table with confidence intervals.
This is the most credibility-sensitive code in the repo — a silent join bug, a leaked golden
label, or a number typed into a doc by hand would destroy the signal more thoroughly than a
missing feature. The harness therefore has to be reproducible, hermetic (it must not touch the
production index), offline-testable, and aggressively defended against leakage and
overclaiming.

The golden set has **exactly one relevant chunk per query** (`|relevant| == 1`), which has a
direct consequence the report must state: recall@k is binary per query and recall@10 saturates
near 1.0, so it is not a discriminating signal. nDCG@10 and MRR are the signals that separate
the configs; recall is reported for completeness but never headlined.

## Decision drivers

- **Reproducibility** — same corpus + same seed ⇒ bit-identical metrics and a byte-diffable
  artifact (mirrors the `meta.json` discipline; the bootstrap seed/B are recorded).
- **No metric leakage** — the golden labels must never be indexed, and the eval must score
  against the same corpus the golden set was minted from (no test-on-train).
- **No overclaiming** — `n` is reported next to every interval; a CI that straddles zero is
  reported as "not distinguishable", never rounded up to a win; fake-backend runs are marked
  non-publishable.
- **Clean seam / single source of truth** — reuse `metrics.py` / `bootstrap.py` / `models.py`
  and the existing retrievers via dependency injection; duplicate no fusion/rerank/metric math.
- **Offline testability** — the whole path runs with a hashing embedder, an in-memory Qdrant,
  and a deterministic fake reranker, with no torch, no network, and no API key.

## Options considered

**(a) Eval index: hermetic eval-scoped build vs reuse the production collection.**
Reusing the prod collection couples eval to whatever happens to be indexed (risking a
stale/leaked corpus) and can mutate prod state. Chosen: a **hermetic eval-scoped index** built
on the public `sample_dir` into a `<collection>_eval` collection under `storage_dir/eval`, so a
`make eval` never reads or writes the production index.

**(b) Scoring: one `K_RETRIEVE = max(k)` list per (config, query) vs re-retrieving per cutoff.**
Re-retrieving per cutoff (a separate call for k=1,3,5,10) is wasteful and can subtly disagree
across cutoffs. Chosen: retrieve **once** at `K_RETRIEVE = 10`, project to a best-first
`list[chunk_id]`, and evaluate every cutoff off that one list. Crucially, **no re-sort and no
extra tie-break** is applied on top of the retrievers — they are already tie-stable, and
re-sorting would leak dict/set iteration order into the ranks. A documented consequence: MRR is
`RR@K_RETRIEVE`, i.e. a relevant chunk ranked beyond 10 counts as a miss (0).

**(c) Headline metric: nDCG@10 + recall@5 vs recall-only.**
With `|relevant| == 1`, recall@10 saturates and recall-only would imply the configs are
indistinguishable. Chosen: **headline nDCG@10, secondary recall@5**, both bootstrapped; recall
is tabled for completeness but never headlined, with the saturation caveat printed.

**(d) Golden coverage: hard-fail vs warn.**
A golden relevant id missing from the index silently scores ~0 recall for that query and
quietly drags every aggregate down. Chosen: **hard-fail** — `run_eval` raises
`GoldenCoverageError` listing the missing ids, with the message "golden id not in index —
chunking config likely drifted from the config that minted golden.jsonl, or wrong corpus
indexed." A warning would let a corrupted run produce authoritative-looking numbers.

## Decision

1. **Adaptation retrieval → ranked_ids (single list per (config, query)).** Each config is run
   once at `K_RETRIEVE = max(k_values) = 10`, `k` passed explicitly (never relying on
   `top_k_rerank`): dense-only / sparse-only project their `(chunk_id, score)` hits; the two
   hybrids are the same `HybridRetriever` with `settings.model_copy(update={"use_reranker":
   False})` and `{"use_reranker": True}` and the same injected reranker. No re-sort, no
   tie-break on top.

2. **`k_values = (1, 3, 5, 10)`**, recall and nDCG computed at the same four cutoffs (the
   `RetrievalMetrics` validator requires `set(recall) == set(ndcg) == set(k_values)`), MRR once.

3. **Hermetic eval index + DI.** `eval_settings = settings.model_copy(update={corpus_dir:
   sample_dir, qdrant_collection: <coll>+"_eval", storage_dir: storage_dir/"eval"})`. The BM25
   index and the `meta.json` provenance live under `storage_dir/eval`; the dense vectors are
   isolated by the eval-scoped collection name, and **in Qdrant on-disk mode the `qdrant_path`
   is additionally redirected to `storage_dir/eval/qdrant`** so the eval vectors are truly
   isolated on disk (and a prod-locked on-disk store can't deadlock `make eval`). In server
   mode (`qdrant_url`) the eval-scoped collection name is sufficient isolation. Real `make eval`
   resolves backends lazily inside `run_eval` (`get_embedder(settings)` → bge-small,
   `QdrantVectorStore.from_settings(eval_settings)`, `get_reranker(settings)`); a missing
   `sentence-transformers`/`torch`/`qdrant` is a loud, actionable failure, never a fake
   fallback. Tests inject `HashingEmbedder` + in-memory Qdrant + `LexicalOverlapReranker`.

4. **Anti-leakage guards that raise (never warn):** (1) the golden path must not be under the
   eval corpus; (2) `load_corpus` does not accept `.jsonl`, so the golden file is never ingested
   (a structural property, not a runtime check); (3) the eval corpus (`sample_dir`) must **not**
   resolve to the private prod corpus (`corpus_dir`) — otherwise the harness would index
   proprietary docs and could emit `publishable` numbers over a non-public corpus; (4) **every**
   golden relevant id must be in the built index, else hard-fail listing the missing ids.
   Guards (1) and (3) are path-only and run **before** the build, so a leaky/private corpus is
   never indexed at all; guard (4) runs after the build (it needs the index). _(Guard (3)
   replaces an earlier tautological check that compared the eval corpus to `sample_dir` — i.e.
   to itself — and so could never fire; the meaningful comparison is against `corpus_dir`.)_

5. **Paired bootstrap** on per-query vectors built in the **frozen golden order** (index `i` =
   the same query in both arms). Four comparisons (dense→hybrid, sparse→hybrid,
   hybrid→hybrid+rerank, dense→hybrid+rerank) on both the headline (nDCG@10) and secondary
   (recall@5) metric, `B = 10000`, `seed = 12345`, `n = 16` (illustrative — `n` is derived from
   the committed golden set at run time, not fixed; the golden set has since grown). `n` is
   printed next to every
   interval; a non-significant diff is never presented as a win. Because 8 comparisons (4 pairs
   × 2 metrics) are reported, a **single pre-registered PRIMARY endpoint** — nDCG@10, dense-only
   → hybrid+rerank — carries confirmatory weight; the other 7 are secondary/exploratory and the
   console prints a multiplicity caveat (≈34% chance of ≥1 spurious "significant" under the null
   across 8 tests).

6. **Outputs.** The **console** is the only publishable surface: a provenance header, the
   per-config table (`config | R@1 | R@5 | R@10 | nDCG@5 | nDCG@10 | MRR`, metric means at 3
   decimals), the honesty + multiplicity caveats, and one line per comparison tagged `[PRIMARY]`
   / `[exploratory]` with `diff` and CI bounds at **4 decimals** (so a genuinely significant CI
   whose lower bound is a positive epsilon does not misprint as `[+0.000, …]`). A reproducible
   JSON artifact (`storage_dir/eval/eval_results.json`, sorted keys, byte-diffable) dumps the
   full `EvalReport`; `corpus_sha256` and library versions come from the eval `meta.json` the
   build wrote. `publishable` is `False` whenever a fake backend produced the rankings.

## Consequences

Positive:
- A `make eval` run is reproducible and the artifact is byte-diffable, so a `/repro-audit` can
  diff metrics against a prior committed run.
- The eval never touches the production index (hermetic, eval-scoped collection; on-disk
  vectors redirected under `storage_dir/eval/qdrant`).
- Leakage and join bugs fail loudly (three guards that raise — golden-under-corpus,
  eval-corpus-is-private, golden-coverage — plus the structural `.jsonl`-not-ingested property),
  and the harness is fully offline-testable.

Negative / accepted:
- **CIs are wide at small n (illustratively n = 16).** This is accepted and must be *signalled*,
  not hidden: every interval prints the actual `n` (illustratively `n=16`), and even a
  "significant" interval is wide at this size. A larger golden set is the only real fix; it is the
  direction of travel (the set has since grown) and `n` is always read from the golden set, never
  hard-coded.
- **Multiplicity across 8 comparisons.** Reporting 4 pairs × 2 metrics inflates the chance of a
  spurious "significant" (~34% under the null). Accepted and mitigated by *pre-registering one
  PRIMARY endpoint* (nDCG@10, dense-only → hybrid+rerank) as the only confirmatory result and
  printing a multiplicity caveat; the other 7 are explicitly exploratory. A formal correction
  (e.g. Holm) is deferred — at small n (illustratively n = 16) the honest framing is
  "directional", not a corrected p.
- **golden.jsonl is coupled to the chunking config.** The `relevant_chunk_ids` are span-derived
  (`sha256(rel_path:start:end)[:16]`), valid only for the chunk spans the current chunking
  config produces (minted under recursive / 512 / 64). The harness and `scripts/validate_golden.py`
  both read the same `Settings` so they agree by construction, but a deliberate chunking change
  requires **re-minting** the golden set — and the coverage guard (4) hard-fails until it is.
- **Fake-embedder numbers are non-publishable.** The `HashingEmbedder` path is a bag-of-tokens
  fake whose rankings are uninterpretable; such runs are marked `publishable=false` and exist
  only to exercise the harness. Only a real bge-small `make eval` produces publishable numbers.
- MRR is `RR@10` (relevant beyond rank 10 = miss), stated in the report.

## Measured numbers

**Intentionally left blank.** Per the headline-metric rule, no comparison numbers are recorded
in this ADR (or anywhere in `docs/`/README) until the **first reproducible `make eval` run with
the real bge-small embedder** has produced them; they are then published by `docs-historian`
from that run, never invented or hand-edited. The offline test path asserts structure,
invariants, determinism, and that the leakage guards raise — never metric values.

## Cross-links

Builds on [ADR-0004](ADR-0004-eval-metrics-and-paired-bootstrap.md) (metric definitions + paired
percentile bootstrap), which is the contract this harness consumes. The hermetic build + the
anti-leakage guards decided here are extracted into a shared `prepare_hermetic_eval` helper and
reused by the attribution harness in
[ADR-0006](ADR-0006-attribution-rate-aggregation.md). Linked from `docs/architecture.md`
(Decisions).
