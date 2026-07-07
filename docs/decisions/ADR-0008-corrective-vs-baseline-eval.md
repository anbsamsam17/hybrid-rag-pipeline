# ADR-0008 — Corrective-vs-baseline evaluation: activation rate is the primary metric (offline-fake-first)

- Status: Proposed
- Date: 2026-07-07
- Deciders: rag-architect (implemented by eval-scientist)
- Scope: `src/rag/eval/corrective.py` (new) + a small `AnswerCorrectnessJudge` abstraction
  (Protocol + real `AnthropicAnswerCorrectnessJudge` + deterministic `FakeAnswerCorrectnessJudge`, in
  `src/rag/eval/`) + the `CorrectiveEvalProvenance` / `CorrectiveQueryRecord` /
  `CorrectiveEvalReport` models in `src/rag/eval/models.py` + a `make eval-corrective` target.
  It **reads** the frozen boundary output of
  [ADR-0007](ADR-0007-self-corrective-rag-stategraph.md)'s
  `rag.agentic.corrective_rag.run_corrective_rag`; it adds **no** instrumentation to the agentic
  layer. It **reuses** the hermetic build + anti-leakage guards
  ([ADR-0005](ADR-0005-retrieval-eval-harness.md)'s `prepare_hermetic_eval`), the metric core
  ([ADR-0004](ADR-0004-eval-metrics-and-paired-bootstrap.md)'s `recall_at_k` / `ndcg_at_k`), and
  the measured attribution rate ([ADR-0006](ADR-0006-attribution-rate-aggregation.md)'s
  `verify_answer`). **RAGAS, the API surface, and any change to `corrective_rag.py` are out of
  scope.**

## Context

ADR-0007 shipped the self-corrective layer and deliberately deferred the confirming metric: a
paired corrective-vs-baseline comparison over the committed golden set. This ADR designs that
comparison. It has to answer one question honestly: **does the corrective loop actually improve
over the single-pass baseline on this corpus, and at what cost — or is it a no-op we are paying
for?**

Three structural facts, all grounded in the current code, force the design:

- **The baseline already saturates the obvious quality metric.** The published attribution run
  scores `attribution_rate = 1.000` (micro, pooled) on this easy 12-doc driving-theory corpus.
  Attribution therefore **cannot discriminate** the two arms: the corrective layer only helps
  where the baseline already fails, and here it almost never fails. Choosing attribution as the
  headline would guarantee a null result dressed up as a measurement.
- **The corrective loop only fires on failure conditions the baseline rarely hits.** In
  `corrective_rag.py`, `route_after_grade` routes to `rewrite_query` **only** when
  `len(relevant) < agentic_min_relevant_docs` (default 1, i.e. *zero* graded-relevant docs), and
  `route_after_verify` routes to `regenerate` **only** when `n_citations > 0` **and**
  `attribution_rate < agentic_min_attribution_rate` (default 1.0). On a corpus where hybrid+rerank
  reliably returns an on-topic chunk and the fake/real generator grounds at 1.0, both triggers are
  starved. The honest expectation is therefore **near-zero activation**.
- **The corrective result already exposes its own control-flow trace.** `CorrectiveRAGResult` is
  frozen and carries `n_rewrites`, `n_regenerations`, `original_query` / `final_query`,
  `terminated_reason ∈ {grounded, abstained, regenerate_budget_exhausted, recursion_limit}`,
  `rewrite_budget_exhausted`, the **measured** `report: VerificationReport`, the final `answer`,
  and the final `contexts`. Whether a rewrite or regeneration was triggered is **already
  measurable** from these fields — no new hooks in the agentic module.

Given all three, the first thing to measure is not answer quality at all — it is **whether the
corrective RETRY LOOP fires**. Activation (`n_rewrites > 0` or `n_regenerations > 0`) is trace-only
and judge-free, so we can measure it almost for free. But activation alone does **not** prove a
no-op: the graph *also* grades and **filters** the context set before generation
(`_generate_node` generates over the graded-relevant subset), so with the retry loop idle the
corrective arm can still generate over fewer contexts than the baseline's full retrieved set. The
harness therefore reports a second judge-free number, `contexts_identical_rate` (both arms' final
contexts identical). A **true no-op** requires `activation == 0` **and** `contexts_identical_rate
== 1`; otherwise the honest headline is "the retry loop never fired, but grading still filtered
contexts on N/50 queries — see the recall/attribution/correctness deltas for the effect."

## Decision drivers

- **Pick a metric that can actually discriminate.** Attribution is saturated; it must be a
  regression guard, not the headline.
- **Cheapest honest signal first.** Activation rate is derivable from the trace with **zero** LLM
  judge calls and is byte-stable under fakes; run it first.
- **A fair comparison changes exactly one thing.** Same golden queries, same hermetic index, same
  backends, same generation LLM; the *only* difference is agentic off (single pass) vs on
  (`run_corrective_rag`). Same anti-leakage guards.
- **`make eval` stays LLM-free / key-free.** This is a *separate* opt-in entry point
  (`make eval-corrective`) that needs an LLM + `langgraph`, exactly as `make eval-attribution` is
  separate (ADR-0006).
- **Honesty about non-determinism.** Generation *and* the correctness judge are non-reproducible;
  no bootstrap CI, `single_run=True`, publishable only from a fully-real run (ADR-0006 discipline).
- **No fishing.** Pre-register a single primary endpoint (activation rate) so a spurious
  correctness "win" mined out of generator+judge noise can never be reported as a result.
- **Offline-fake-first.** The whole eval runs on fakes (fake generation LLM, fake corrective LLM,
  fake judge, fake backends); exactly **one** real run at the very end produces publishable
  numbers.

## Options considered

**(a) What is the primary metric?**
- **(a1) attribution_rate delta.** Rejected as *primary*: saturated at 1.000 on this corpus, so the
  delta is structurally ~0 regardless of whether the layer helps. Kept as a **secondary regression
  guard** (the corrective arm must not *reduce* attribution — ADR-0007's stated guard).
- **(a2) answer-correctness delta vs `reference_answer`.** This is where a real lift *would* show,
  and all 50 golden rows carry a `reference_answer`. But at ~0 retry activation the corrective
  final answer is a second draw from the *same* generator (a regeneration, when it fires, is a
  pure re-call over identical contexts — ADR-0007 4b) over contexts that are at most a graded
  *subset* of the baseline's retrieved set — never a genuinely different retrieval. So the delta
  is dominated by generator+judge noise (plus whatever the grade-filter dropped), not a validated
  corrective lift. Kept as a **secondary** endpoint, explicitly *not* confirmatory.
- **(a3) retry-loop activation rate.** Over the 50 queries, the fraction where the corrective
  RETRY LOOP fired (`n_rewrites > 0` **or** `n_regenerations > 0`; and separately
  `final_query != original_query`). Cheap, judge-free, byte-stable under fakes. **Chosen as the
  pre-registered PRIMARY.** It answers "did the retry loop fire?", NOT the broader "did the layer
  do anything?" — the grade-and-filter step can change the context set with the loop idle, so this
  is paired with the judge-free `contexts_identical_rate` and only both together license a no-op
  claim. On this corpus the honest expectation is ~0/50 retry activation — the headline finding,
  qualified by whether grading also left the contexts unchanged.

**(b) How is answer correctness judged?**
- **(b1) LLM judge only.** Non-reproducible → no CI, publishable-gated. Correctness-sensitive, so
  it must run on `claude-opus-4-8` / `claude-sonnet-4-6` with adaptive thinking + high effort and
  never the banned params (mirrors the verification LLM-judge discipline).
- **(b2) Lexical only** (token-F1 of `reference_answer` content tokens present in the candidate).
  Deterministic and byte-stable; informative here because reference answers are short factual spans
  ("Dipped headlights.").
- **Chosen: both.** LLM judge is the headline correctness signal (publishable-gated, single_run);
  the lexical F1 is the **deterministic floor** reported alongside so a fake/offline run still
  exercises the correctness path byte-stably and a reader can sanity-check the judge.

**(c) Cost axis: wall-clock latency vs derived LLM-call count.**
- **(c1) Wall-clock p50/p95.** Machine-dependent, non-reproducible; fine as a soft note, useless as
  a defensible number.
- **(c2) LLM-call count derived from the trace.** Deterministic given the trace:
  `grade_calls = n_rewrites + 1`, `rewrite_calls = n_rewrites`, `generate_calls = n_regenerations
  + 1`; `extra_llm_calls_vs_baseline = 2*n_rewrites + n_regenerations + 1`. **Chosen** as the
  primary cost axis. At the expected `n_rewrites = n_regenerations = 0`, this is **+1 grade call per
  query** — the honest "you pay for the grading even when it does nothing" number. (Extra query
  embeddings on rewrite passes are local/bge-small → $0 API cost, noted as latency only.)

**(d) Separate module vs a flag on the attribution/retrieval harness.** A flag would couple an
LLM+langgraph path into `make eval`. **Chosen: a separate `rag.eval.corrective` module + entry
point**, mirroring ADR-0006, so `make eval` stays offline and this run is opt-in.

**(e) Reuse `AttributionReport` vs a dedicated `CorrectiveEvalReport`.** Attribution's shape has no
place for activation, per-arm correctness, recall deltas, or the trace/cost fields. **Chosen: a
dedicated `CorrectiveEvalReport`** carrying both arms and the trace, mirroring the ADR-0006
precedent of a purpose-built provenance/report rather than an ill-fitting reuse.

## Decision

1. **Separate module + entry point.** `rag.eval.corrective.run_corrective_eval(settings, *,
   llm=None, corrective=None, judge=None, embedder=None, store=None, reranker=None) ->
   CorrectiveEvalReport`, run via `python -m rag.eval.corrective` / `make eval-corrective`.
   `make eval` and `make eval-attribution` are unchanged; `make eval` stays LLM-free.

2. **One hermetic build, two arms, one differing knob.** Call `prepare_hermetic_eval(settings,
   embedder, store)` **once** (its anti-leakage + golden-coverage guards run before any number is
   trusted). Build **one** `HybridRetriever(use_reranker=True)`, one generation `LLMClient`, one
   `CorrectiveLLM`. Iterate the golden set in the **frozen file order**. For each query, at
   `k = eval_settings.top_k_rerank` (=5) so both arms retrieve identically on the first pass:
   - **Baseline arm (agentic OFF):** `retrieve(query, k) → generate_answer(query, contexts) →
     verify_answer(answer, contexts)` — the *same* single pass as the ADR-0006 attribution harness
     and `RagService.answer_query`. To avoid a third copy of that sequence, factor a shared
     `answer_once(...)` helper (seam-cleanliness note handed to eval-scientist / citation boundary).
   - **Corrective arm (agentic ON):** `run_corrective_rag(CorrectiveRAGRequest(query=query,
     k=eval_settings.top_k_rerank), retriever=<same>, llm=<same>, corrective=<CorrectiveLLM>,
     settings=eval_settings)`.
   The two arms share the identical retriever/llm/settings; only the presence of the
   grade/rewrite/regenerate graph differs. **Baseline is NOT approximated by `run_corrective_rag`
   with budgets set to 0** — that still runs a grade call and filters contexts by grading, so it is
   not a true single pass.

3. **PRIMARY metric (pre-registered, confirmatory): retry-loop activation rate (+ the judge-free
   contexts-identical rate).** From each `CorrectiveRAGResult`, per query:
   `activated := (n_rewrites > 0) or (n_regenerations > 0)`;
   `final_query_changed := (final_query != original_query)`;
   `contexts_identical := (baseline final chunk_ids == corrective final chunk_ids)`.
   Report `activation_rate = n_activated / n_queries`, the split
   `n_rewrite_activated` / `n_regenerate_activated` / `n_final_query_changed`, the
   `terminated_reason` histogram, **and** `contexts_identical_rate` with per-arm context counts.
   Both numbers are judge-free and byte-stable under fakes. **Run this first** — a **true no-op**
   is asserted only when `activation == 0` AND `contexts_identical_rate == 1`; if activation is
   0/50 but grading filtered contexts on some queries, the honest finding is "the retry loop never
   fired, but the grade node still changed the context set on N/50 queries," and the secondary
   deltas below carry the effect.

4. **SECONDARY metrics (all expected ≈ 0 delta; reported with honesty banners, none confirmatory):**
   - **Answer correctness vs `reference_answer`** (both arms): `AnswerCorrectnessJudge.judge(query,
     reference_answer, candidate_answer) -> CorrectnessVerdict{correct: bool, score: float, reason:
     str}`, judged **blind** to which arm produced the answer. Report `baseline_correctness_rate` /
     `corrective_correctness_rate` (LLM judge) and the deterministic
     `baseline_lexical_f1_mean` / `corrective_lexical_f1_mean`. At ~0 activation this measures
     generator+judge noise, not corrective lift — stated in the report.
   - **Attribution regression guard:** aggregate `report.attribution_rate` for both arms (micro,
     reusing the ADR-0006 pooled definition). The corrective arm must **not** score lower than the
     baseline; a reduction is a regression, not a wash.
   - **Retrieval recall/nDCG delta on final contexts:** `recall_at_k` / `ndcg_at_k` (ADR-0004) of
     the baseline first-pass contexts vs the corrective **final** `contexts`, against
     `relevant_chunk_ids`. Non-zero only on queries where a rewrite actually changed retrieval
     (expected: none here).
   - **Cost:** mean `extra_llm_calls_vs_baseline = 2*n_rewrites + n_regenerations + 1` over the
     golden set (derived from the trace, deterministic). Wall-clock is a non-reproducible soft note.

5. **No CI; single_run; publishable only when fully real.** No bootstrap (both the generator and
   the judge are non-reproducible — a CI would be false precision, ADR-0006). `single_run=True`,
   point estimates + the per-query `CorrectiveQueryRecord` distribution.
   `publishable = True` **iff** the generation LLM is `AnthropicLLMClient` **and** the corrective
   LLM is `AnthropicCorrectiveLLM` **and** the judge is `AnthropicAnswerCorrectnessJudge` **and** the
   embedder is `SentenceTransformerEmbedder` **and** the reranker is `CrossEncoderReranker`. Any
   fake flips it to `False`. The activation/cost/recall/lexical-F1 numbers are deterministic and
   byte-stable under fakes; only the LLM-judge correctness rate is non-reproducible.

6. **Pre-registration + multiplicity honesty.** PRIMARY = activation rate (expected ~0/50). All
   secondary endpoints are directional; the correctness delta is explicitly **not** a win/loss
   claim at ~0 activation. This is recorded in the report header so a reader cannot mistake a
   noise-driven correctness difference for a corrective effect.

7. **Anti-leakage, inherited and extended.** `reference_answer` lives in the protected
   `golden.jsonl` and is **never indexed** (the existing golden-not-under-corpus guard covers it).
   The judge is shown **only** `(query, reference_answer, candidate_answer)` — never the corpus,
   never `relevant_chunk_ids`, and **never the arm label** (blind judging), so it cannot be biased
   toward the corrective arm. Queries with an empty `reference_answer` are excluded from the
   correctness metric and are **not** counted in `n_judged` (only queries that carry a reference
   increment it); all 50 committed golden rows carry a reference, so here `n_judged = 50`.

8. **Distinct, byte-diffable artifact.** Dumped (sorted keys) to
   `storage_dir/eval/corrective_results.json` — a name distinct from `eval_results.json` and
   `attribution_results.json`. Byte-stable under the deterministic fakes; a real run is
   `single_run` and not bit-exact.

9. **Judge abstraction mirrors the corrective-LLM pattern.** `AnswerCorrectnessJudge` Protocol in
   `eval/`; `AnthropicAnswerCorrectnessJudge` lazy-imports `anthropic`, uses `messages.parse`,
   adaptive thinking + high effort, and **never** `temperature` / `top_p` / `top_k` /
   `budget_tokens` / prefill. It judges on a model DIFFERENT from the generator by default
   (`default_judge_model`: if generation is sonnet, judge on opus, else on sonnet — both
   correctness-sensitive 4.x) to blunt self-preference bias in the ABSOLUTE correctness rate; the
   resolved model is recorded as `judge_model` in provenance. This does not affect the
   corrective-vs-baseline DELTA (both arms share the judge). `FakeAnswerCorrectnessJudge` scores
   lexical token-F1 vs the reference and thresholds it, deterministic and key-free. Living in
   `eval/` respects the dependency rule (eval depends on generation/verification/retrieval/agentic;
   nothing depends on eval).

## Consequences

Positive:
- The **headline is honest and cheap**: activation rate is trace-only, judge-free, byte-stable
  under fakes, and on this corpus it is expected to say plainly "the corrective layer is a no-op
  here." That is a stronger, more defensible statement than a manufactured quality delta.
- **One differing knob**: same golden set, same hermetic index, same backends, same generation LLM,
  same anti-leakage guards — the comparison is fair by construction, and the corrective arm reads
  the *existing* frozen `CorrectiveRAGResult` (no new instrumentation, no agentic churn).
- **Fully offline-testable**: fake generation LLM, fake corrective LLM, fake judge, fake backends
  → byte-stable activation/cost/recall/lexical-F1; tests assert structure/invariants/determinism,
  never a real value. Exactly one real run publishes.
- **Regression-safe**: attribution is retained as a guard (corrective must not reduce it), and the
  cost axis makes the "+1 grade call/query for nothing" price explicit.
- **Future-proof**: on a harder corpus the *same* harness surfaces real activation, a real recall
  recovery from rewrite, and a real correctness lift — the secondary endpoints become live without
  a redesign.

Negative / accepted:
- **The primary result is expected to be null** (activation ~0/50). Accepted — a truthful null with
  a measured cost is the honest outcome, and the harness is built to detect a real effect the
  moment the corpus warrants one.
- **Correctness and the LLM-judge are non-reproducible** → no CI, `single_run`, publishable only
  when fully real; a real run is not bit-exact (ADR-0006 caveat carries over).
- **At ~0 activation the correctness delta is generator+judge noise**, so it is reported as
  directional only and can never be headlined as a win/loss.
- **`llm_model` / judge-model aliases are recorded verbatim** (ADR-0006 alias-drift caveat applies).
- **The real run adds LLM cost**: an extra grade call per query plus the judge calls (estimate
  below), the price of measuring a layer that is off by default.

## Measured numbers

**Intentionally left blank.** Per the headline-metric rule, no activation / correctness /
attribution / cost numbers are recorded here or in `docs/` / README until the **first reproducible
`make eval-corrective` run** with the real Anthropic generation client + real `AnthropicCorrectiveLLM`
+ real `AnthropicAnswerCorrectnessJudge` + bge-small embedder + cross-encoder reranker has produced them;
`docs-historian` then publishes them from that run, never invented. The pre-registered
**prediction** (not a published metric) is activation ≈ 0/50 and every secondary delta ≈ 0 on this
corpus. The offline test path asserts structure, invariants, determinism, and the activation/cost
accounting — never a real LLM correctness value.

**Estimated real-run API cost (planning only, not a benchmark):** baseline 50 generation calls;
corrective arm ≈ 50 grade + 50 generate = ~100 calls at the expected zero activation (hard upper
bound per query `(R+1)` grade + `R` rewrite + `(G+1)` generate = 3+2+2 = 7 → ≤ 350 with `R=2, G=1`);
correctness judge 2 arms × 50 = 100 calls. **Total ≈ 250 LLM calls expected, ≤ ~500 worst case**,
small structured outputs each — on the order of a few US dollars depending on model and thinking
depth. Exact figures are not published.

## Cross-links

Reads the frozen `CorrectiveRAGResult` boundary of
[ADR-0007](ADR-0007-self-corrective-rag-stategraph.md) (the confirming metric that ADR deferred);
reuses [ADR-0005](ADR-0005-retrieval-eval-harness.md)'s `prepare_hermetic_eval` (hermetic build +
anti-leakage guards), [ADR-0004](ADR-0004-eval-metrics-and-paired-bootstrap.md)'s `recall_at_k` /
`ndcg_at_k`, and [ADR-0006](ADR-0006-attribution-rate-aggregation.md)'s measured `attribution_rate`
+ the offline-fake-first / no-CI / publishable-only-when-real discipline. Linked from
`docs/architecture.md` (Decisions).
