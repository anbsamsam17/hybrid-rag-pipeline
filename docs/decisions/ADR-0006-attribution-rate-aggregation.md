# ADR-0006 — Attribution-rate aggregation over the golden set (offline-fake-first)

- Status: Proposed
- Date: 2026-07-01
- Deciders: rag-architect (implemented by eval-scientist)
- Scope: `src/rag/eval/attribution.py` (new) + the `AttributionQueryRecord` /
  `AttributionProvenance` / `AttributionReport` models in `src/rag/eval/models.py` + the
  hermetic-eval helper `prepare_hermetic_eval` extracted into `src/rag/eval/harness.py` +
  the `make eval-attribution` target. **RAGAS faithfulness / answer-relevance are explicitly
  out of scope** and land in a later increment. Consumes the measured per-answer
  `attribution_rate` from `rag.verification.citations.verify_answer` and the hermetic build +
  anti-leakage guards fixed in [ADR-0005](ADR-0005-retrieval-eval-harness.md).

## Context

[ADR-0005](ADR-0005-retrieval-eval-harness.md) built the retrieval harness but **deferred**
attribution and RAGAS to "increment 3" because they need an LLM. The verification stage already
produces a *measured* per-answer `attribution_rate` (grounded citations / total citations, by a
deterministic lexical span check — never declared). What is still missing is a **defensible
golden-set aggregate** of that rate for the real answering configuration.

That aggregate is credibility-sensitive in the same way the retrieval numbers are: a single
authoritative-looking headline number that silently averages abstentions and grounding failures
together, or that is computed on a re-retrieval rather than on the contexts the model actually
saw, or that gets typed into a doc without a real run behind it, would destroy the signal. So the
aggregation is **offline-fake-first** (fully exercised by deterministic fakes, byte-stable under
them) and **publishable only from a real run**.

Two structural facts shape the design:

- **`verify_answer`'s 0-citation rule is `attribution_rate = 0.0`.** A query where the model
  abstains (cites nothing) therefore looks identical, at the per-answer rate, to a query where the
  model cited a span that failed to ground. Any aggregate must not blur those two.
- **LLM generation is not bit-exact reproducible** even with `temperature` removed on the 4.x
  models. So a per-run bootstrap CI over queries would report a precision the generator does not
  have.

## Decision drivers

- **`make eval` stays LLM-free / key-free.** The retrieval harness must remain fully offline; the
  attribution run is a *separate* entry point (`make eval-attribution`) that may call the LLM.
- **Single source of the anti-leakage guards.** The hermetic build + the three leakage guards
  must not be copy-pasted into a second harness where they could drift; they get extracted once.
- **Measure the REAL answering config.** Attribution must reflect what the pipeline actually
  answers with: hybrid + rerank at `top_k_rerank` (=5), verified against the SAME contexts handed
  to generation — never `K_RETRIEVE=10` and never a re-retrieval.
- **Honesty about micro vs macro vs abstention.** The headline must be immune to the 0-citation
  convention, and the abstention effect must always be visible.
- **Honesty about non-determinism.** No CI in v1; point estimates + the per-query distribution.
- **No published number without a real run.** The measured numbers stay blank until the first
  real `make eval-attribution`.

## Options considered

**(a) A flag on `run_eval` vs a separate module.** Adding `--attribution` to `run_eval` would
couple the LLM-free retrieval harness to an LLM path and risk `make eval` acquiring a key
dependency. Chosen: **a separate `rag.eval.attribution` module** with its own entry point, so
`make eval` stays offline and the attribution run is opt-in.

**(b) Macro-only vs micro-primary + macro + abstentions.** A macro (mean of per-query rates)
alone conflates "cited but ungrounded" (0.0 *with* citations) and "abstained" (0.0 by the
0-citation rule), and is dragged down by abstentions in a way that hides *why*. Chosen:
**micro (pooled `total_grounded / total_citations`) as the headline** — immune to the 0-citation
convention because an abstaining query contributes nothing to either pool — **plus macro, macro
over answered-only, and `n_abstained`** as secondary so the abstention effect is explicit. Macro
is never reported alone.

**(c) Intra-run bootstrap CI vs point estimate + distribution.** A bootstrap over queries within
one LLM run would manufacture a confidence interval the non-reproducible generator does not
support. Chosen: **point estimates + the full per-query distribution + `single_run=True`**; the
paired bootstrap (`rag.eval.bootstrap`) is for the single-config retrieval comparison and does
not apply here.

**(d) Reuse `EvalProvenance` vs a dedicated `AttributionProvenance`.** `EvalProvenance` carries
`seed` / `n_resamples` / headline/secondary *retrieval* metric names and no LLM identity — wrong
shape for a single-config LLM measurement. Chosen: **a dedicated `AttributionProvenance`** that
records `llm_class` / `llm_model` (the identity a judged number turns on) and drops the bootstrap
fields.

## Decision

1. **Separate module + entry point.** `rag.eval.attribution.run_attribution_eval(settings, *,
   llm=None, embedder=None, store=None, reranker=None) -> AttributionReport`, run via
   `python -m rag.eval.attribution` / `make eval-attribution`. `make eval` is unchanged and
   remains LLM-free.

2. **One hermetic helper.** The hermetic eval-scoped build + the anti-leakage guards
   (golden-not-under-corpus, eval-corpus-is-public, golden-coverage) are extracted from `run_eval`
   into `prepare_hermetic_eval(settings, *, embedder=None, store=None) -> HermeticEval`, used by
   **both** `run_eval` and `run_attribution_eval`. `run_eval`'s behavior is unchanged (the harness
   tests stay green); the guards exist in exactly one place.

3. **Real answering config, same contexts.** Per golden query, in the frozen file order:
   `HybridRetriever(use_reranker=True).retrieve(query, k=settings.top_k_rerank)` →
   `generate_answer(query, contexts, llm=llm, settings=settings)` →
   `verify_answer(answer, contexts)` against the **same** `contexts` objects (no re-retrieval).
   `k = top_k_rerank` (=5), not `K_RETRIEVE`.

4. **Micro is the headline.** `micro_attribution_rate = total_grounded / total_citations` (0.0
   when there are no citations at all). `macro_attribution_rate` = mean of per-query rates
   (abstentions contribute 0.0); `macro_attribution_rate_answered` = macro over `abstained ==
   False` queries; `n_abstained` = count of `n_citations == 0`. `abstained := (n_citations == 0)`.
   Macro is never reported alone.

5. **No CI in v1.** Point estimates + the per-query `AttributionQueryRecord` distribution;
   `single_run=True`. LLM non-reproducibility is stated in the report and docstrings.

6. **Publishable only when fully real.** `publishable = (llm_class == "AnthropicLLMClient" AND
   embedder_class == "SentenceTransformerEmbedder" AND reranker_class == "CrossEncoderReranker")`.
   Any fake → `publishable = False`. Provenance records `llm_class` / `llm_model` (from
   `settings.llm_model`) plus the corpus SHA-256 / git SHA / library versions read from the eval
   `meta.json`.

7. **Distinct, byte-diffable artifact.** The report is dumped (sorted keys) to
   `storage_dir/eval/attribution_results.json` — a name distinct from the retrieval harness's
   `eval_results.json`. Under the deterministic fakes the artifact is byte-stable; a real LLM run
   is not bit-exact and is marked `single_run`. `attribution_rate` measures **grounding** (are
   cited spans supported), NOT correctness or completeness — stated in the report and docstrings.

## Consequences

Positive:
- Fully offline-testable and **byte-stable under the deterministic fakes**, so the structure /
  invariants / abstention accounting are covered without an API key.
- The anti-leakage guards and the hermetic build live in **one** helper shared by both harnesses,
  so they cannot drift.
- Reporting is honest: a micro headline immune to the 0-citation convention, macro + macro-over-
  answered + `n_abstained` to expose abstentions, and an explicit grounding-not-correctness scope.

Negative / accepted:
- A real run is **not bit-exact reproducible** (LLM generation), so the artifact is byte-stable
  only under the fakes; a real run is a `single_run` point estimate.
- **No confidence interval in v1** — accepted; a CI would be false precision for a non-
  reproducible single-config measurement.
- The measured headline number is **left blank** until the first real `make eval-attribution`.
- **`llm_model` alias drift is known**: provenance records `settings.llm_model` (e.g.
  `claude-sonnet-4-6`), which is an alias that may resolve to different underlying weights over
  time; the alias is recorded verbatim, not pinned to a content hash.

## Measured numbers

**Intentionally left blank.** Per the headline-metric rule, no attribution numbers are recorded
here (or anywhere in `docs/`/README) until the **first reproducible `make eval-attribution` run
with the real Anthropic client + bge-small embedder + cross-encoder reranker** has produced them;
they are then published by `docs-historian` from that run, never invented or hand-edited. The
offline test path asserts structure, invariants, determinism, and the abstention accounting —
never a real LLM attribution value.

## Cross-links

Builds on [ADR-0005](ADR-0005-retrieval-eval-harness.md) (the hermetic build + anti-leakage
guards this ADR extracts into `prepare_hermetic_eval` and reuses) and
[ADR-0004](ADR-0004-eval-metrics-and-paired-bootstrap.md) (the metric/bootstrap core). Linked
from `docs/architecture.md` (Decisions).
