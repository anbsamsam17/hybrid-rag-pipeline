# ADR-0004 — Retrieval metric definitions and paired percentile bootstrap

- Status: Accepted
- Date: 2026-06-27
- Deciders: rag-architect (implemented by eval-scientist)
- Scope: `src/rag/eval/{metrics.py,bootstrap.py,models.py}` — the metrics + statistics
  core only. The harness (`rag.eval.harness`), the golden set wiring, RAGAS, and the
  `attribution_rate` aggregation are out of scope here and land in the next increment.

## Context

The headline credibility of this repo is its evaluation rigor: a comparison of
dense-only vs sparse-only vs hybrid vs hybrid+rerank on `recall@k`, `nDCG@k`, `MRR`
with bootstrap CI95. A metric whose definition is ambiguous, or that silently inflates
(e.g. recall capped at 1.0 when it should report 1/3), or a bootstrap that is unpaired
or time-seeded, destroys that signal more thoroughly than a missing feature. These
definitions are therefore load-bearing and are fixed here so every published number is
reproducible and defensible after the fact.

The metric functions must be **pure** (no I/O, no global state, no randomness) and
**backend-agnostic**: they operate on `ranked_ids: Sequence[str]` and
`relevant: Collection[str]`, never on `RetrievalResult` or any index object. This keeps
the `eval` module boundary clean (`eval` imports `retrieval`/`verification`; the metrics
core imports neither) and makes the math unit-testable against hand-computed fixtures.

## Decision drivers

- IR-standard, textbook-defensible definitions (Manning et al., *Introduction to
  Information Retrieval*) over bespoke variants.
- Determinism / bit-exact reproducibility (mirrors the project's `meta.json` discipline).
- Adversarial safety: each definition must make the "wrong-but-plausible" inflation a
  *test failure*, not a silent pass.
- Simplicity and transparency of the statistics over marginal statistical sophistication.

## Decision

### 1. Rank convention

`ranked_ids` is ordered best-first; rank is 1-based (`rank = index + 1`). This matches
`RetrievalResult.rank` (`ge=1`) and the hand-written RRF (`1/(k+rank)`, rank 1-based).

### 2. Duplicate handling (shared pre-step)

Every metric first **deduplicates `ranked_ids` keeping first occurrence**, then applies
the `k` cutoff. A retriever that returns the same `chunk_id` twice must not inflate any
metric or waste a top-k slot. `relevant` is coerced to a `set` internally for O(1)
membership; the golden contract (below) already forbids duplicate relevant ids.

Worked example: `ranked=["a","a","b"]`, `relevant={"b"}`, `k=2`. Dedup → `["a","b"]`,
top-2 contains `b` → `recall@2 = 1.0`. (Cut-then-dedup would give `["a"]` → `0.0`, which
is wrong: the duplicate must not consume the second slot.)

### 3. recall@k — proportion of relevant retrieved (standard IR)

`recall@k = |relevant ∩ dedup(ranked)[:k]| / |relevant|`.

Denominator is `|relevant|` — **not** `min(|relevant|, k)`. This is the standard IR
definition and is intentionally honest: when `|relevant| > k`, recall@k is capped below
1.0 because not all relevant docs can fit in k slots. Using `min(|relevant|, k)` would
silently inflate recall to 1.0 and is forbidden.

Worked example: `relevant={r1,r2,r3}`, one relevant at rank 1, `k=1` →
`recall@1 = 1/3 ≈ 0.333`, **not** 1.0.

### 4. nDCG@k — binary gains, log2 discount, ideal cap at min(|relevant|, k)

Binary relevance (the golden set is a *set* of relevant ids; no graded labels):

```
DCG@k  = Σ_{i=1..k}  rel_i / log2(i + 1)          # rel_i ∈ {0,1}, i 1-based
IDCG@k = Σ_{i=1..min(|relevant|, k)}  1 / log2(i + 1)
nDCG@k = DCG@k / IDCG@k
```

With binary gains the "traditional" (`rel/log2(i+1)`) and "exponential"
(`(2^rel - 1)/log2(i+1)`) formulations coincide, so there is no ambiguity. IDCG is
capped at `min(|relevant|, k)` so a perfect ranking scores exactly 1.0 even when
`|relevant| > k`.

Worked example: `relevant={r1..r5}`, top-3 all relevant, `k=3` → `nDCG@3 = 1.0`
(IDCG capped at 3) while `recall@3 = 3/5 = 0.6`. These two together prove the two
conventions are distinct and both correct.

### 5. reciprocal_rank and MRR

`reciprocal_rank(ranked, relevant) = 1 / rank_of_first_relevant`, 0.0 if none. It takes
no `k`; for RR@k the caller passes `dedup(ranked)[:k]`. `MRR` is the arithmetic mean of
per-query reciprocal ranks; `mrr([]) == 0.0`.

### 6. Edge cases (normative, total functions — raise only on non-positive k or non-finite MRR input)

| Case | recall@k | nDCG@k | reciprocal_rank |
|------|----------|--------|-----------------|
| `k > len(ranked)` | score what's present, no padding | same | n/a (no k) |
| `k <= 0` | **raise ValueError** | **raise ValueError** | n/a |
| `relevant` empty | 0.0 (0/0 → 0.0 by convention) | 0.0 (IDCG=0 → 0.0) | 0.0 |
| no relevant in top-k | 0.0 | 0.0 | 0.0 (if none anywhere) |
| duplicates in `ranked` | dedup-first-occurrence (§2) | same | same |
| `ranked` empty | 0.0 | 0.0 | 0.0 |

`k <= 0` raises (mirrors `reciprocal_rank_fusion`'s `k<=0` guard); a non-positive cutoff
is a programming error, and a silent 0.0 would mask a harness bug. Empty `relevant`
returns 0.0 rather than NaN/raise so the functions are total; the `GoldenItem` contract
nonetheless forbids empty relevant upstream, so this branch is pure defense.
**No metric ever returns NaN or inf.**

`mrr` additionally **raises ValueError on any non-finite reciprocal rank** (NaN/inf): a NaN
silently poisons the arithmetic mean (`mean` of anything containing NaN is NaN), which would
contradict the "never NaN" guarantee, so it is rejected at the boundary rather than propagated.
The input contract is `Sequence[float]`; misses must be encoded as `0.0` by the caller.

### 7. Golden contract (`GoldenItem`)

Frozen Pydantic. `relevant_chunk_ids` stored as `tuple[str, ...]` (not `set`/`frozenset`)
so the model dump is **deterministic and diffable**, with a validator enforcing
non-empty and no duplicates. The harness converts to a `set` at the metric boundary.
`query_id` and `query` carry `min_length=1`: an empty `query_id` is a silent golden↔results
join trap (it would match nothing, or everything, depending on the join), so it fails at load.

#### Model value bounds (defensive typing)

A retrieval metric score is a fraction in `[0, 1]`; a value outside that range signals a
harness bug (e.g. recall divided by the wrong denominator) and must fail validation rather than
be stored as an authoritative-looking number. The frozen result models therefore constrain:

- `QueryMetrics.reciprocal_rank`, `RetrievalMetrics.mrr`, and the **values** of the
  `recall` / `ndcg` dicts in both models to `[0.0, 1.0]` (`Field(ge=0, le=1)`).
- `RetrievalMetrics` adds a `model_validator(mode="after")` requiring
  `set(recall) == set(ndcg) == set(k_values)` — the reported cutoffs must be mutually
  consistent, so a number can never be mislabeled against the wrong `k`.
- Metric dicts are inserted with `k` keys sorted ascending (and `k_values` sorted) for a
  byte-deterministic, diffable dump.

### 8. Paired percentile bootstrap (CI95)

We compare two configs on a per-query metric. The configs are scored on the **same**
queries, so we use a **paired** bootstrap: resample query *indices* with replacement and
apply the *same* indices to both configs' per-query vectors. Pairing cancels per-query
difficulty variance and yields a correct, tighter CI on the difference than an unpaired
bootstrap would.

- Statistic: `diff = mean(treatment) - mean(baseline)` (positive ⇒ treatment better).
- Point estimate: the **observed** diff on the full sample (not the bootstrap mean).
- CI: percentile method — `[p2.5, p97.5]` of the bootstrap diff distribution.
- `B (n_resamples)` default **10000** (stable 2.5/97.5 percentiles).
- Seed: fixed default **12345** via `numpy.random.default_rng(seed)`. Never time-seeded,
  never the legacy global `numpy.random`. Vectorized:
  `idx = rng.integers(0, n, size=(B, n))` then index both vectors with `idx`.
- `significant = not (ci_low <= 0 <= ci_high)` (0 outside the interval).
- Validation (raise ValueError): unequal lengths, empty inputs, **`n < 2`**, **`B < 1000`**,
  `ci` not in (0,1), any NaN/inf in inputs.

#### Hard floors on the bootstrap (degenerate-CI guards)

Two floors are enforced so a degenerate interval can never be silently flagged significant:

- **`n >= 2` paired observations.** With a single pair every resample draws the only index,
  so the bootstrap diff distribution is a constant: the CI collapses to width 0 and *any*
  non-zero observed diff is misread as significant. A paired bootstrap has no resampling
  variance below 2 pairs, so `n < 2` raises rather than emitting a meaningless interval. (This
  is a statistics floor, separate from the per-query-count honesty caveat below: even at the
  minimum `n = 2`, a CI is not "impressive" and the report must state `n`.)
- **`B >= 1000` resamples.** A tiny `B` (e.g. `B = 1`) makes the "CI" a single resample that
  can contradict the observed estimate and flip the significance flag run-to-run; 1000 is the
  documented minimum for usable 2.5/97.5 percentiles. The default 10000 stays well above it.

#### Percentile vs BCa — the contested choice

| Method | Pros | Cons |
|--------|------|------|
| **Percentile (chosen)** | Simple, transparent, fully reproducible from (data, seed, B); no jackknife; trivially unit-testable; standard for this kind of paired comparison | Slight under-coverage when the diff distribution is biased/skewed |
| BCa (bias-corrected accelerated) | Better coverage under skew/bias | Adds bias-correction `z0` + acceleration via jackknife; more code, more failure surface, harder to test bit-exactly; over-engineered for a golden set of this size |

We choose **percentile** for v1: transparency and reproducibility dominate at this golden-set
size, and the method is easy to defend line-by-line. BCa is recorded as a deferred option
should coverage diagnostics later show material skew.

## Consequences

Positive:
- Every headline number has a single, citable definition with worked examples; an
  inflated recall or a mis-discounted nDCG is a failing unit test, not a shipped lie.
- Metrics are pure and backend-agnostic → the `eval` boundary stays clean and the math is
  testable without an index, a corpus, or any LLM.
- Bootstrap is paired, seeded, and percentile → CI95 is bit-exact reproducible and
  recorded alongside `seed`/`B`/`n_queries` for defense.

Negative / accepted:
- recall@k can read "low" when `|relevant| > k`; this is correct and must be explained in
  the report, not patched away.
- Percentile CIs may slightly under-cover under heavy skew (accepted; BCa deferred).
- Binary-gain nDCG ignores graded relevance (the golden set has none); revisit only if
  graded labels are introduced.

## Reproducibility notes

Metric outputs are deterministic given inputs. Bootstrap outputs are deterministic given
(data, `seed`, `B`). The `numpy` version (which fixes the pairwise-summation order behind
`mean`/`percentile`) must be captured in the run's `meta.json` so CI bounds reproduce
bit-exactly across environments.

## Cross-links

Linked from `docs/architecture.md` (Decisions). The harness increment that consumes these
contracts (and the actual measured comparison numbers + CI95) is
[ADR-0005](ADR-0005-retrieval-eval-harness.md).
