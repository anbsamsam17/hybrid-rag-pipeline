# ADR-0009 — RAGAS-style generation metrics (faithfulness + answer-relevancy) reimplemented over the Anthropic SDK (offline-fake-first)

- Status: Accepted
- Date: 2026-07-08
- Deciders: rag-architect (to be implemented by eval-scientist; scorer prompts reviewed by citation-verifier)
- Scope: `src/rag/eval/generation_quality.py` (new orchestrator) + a scorer abstraction
  (`FaithfulnessScorer` / `AnswerRelevancyScorer` Protocols + real
  `AnthropicFaithfulnessScorer` / `AnthropicAnswerRelevancyScorer` + deterministic
  `FakeFaithfulnessScorer` / `FakeAnswerRelevancyScorer`, in `src/rag/eval/generation_scorers.py`)
  + the `GenerationQualityQueryRecord` / `GenerationQualityProvenance` / `GenerationQualityReport`
  models in `src/rag/eval/models.py` + a `make eval-ragas` target + one new `Settings` field
  (`ragas_answer_relevancy_n_questions`). It **reuses** the hermetic build + anti-leakage guards
  ([ADR-0005](ADR-0005-retrieval-eval-harness.md)'s `prepare_hermetic_eval`), the DI-clean
  `generate_answer` ([ADR-0006](ADR-0006-attribution-rate-aggregation.md)), the local bge-small
  `SentenceTransformerEmbedder`, and the judge-model / offline-fake-first / no-CI / publishable-only-
  when-real discipline of [ADR-0006](ADR-0006-attribution-rate-aggregation.md) and
  [ADR-0008](ADR-0008-corrective-vs-baseline-eval.md). It **does not** add the `ragas` library to the
  runtime path, and it **does not** change generation, verification, retrieval, or the agentic layer.

## Context

`CLAUDE.md` §1 and the README both promise **RAGAS faithfulness** and **RAGAS answer-relevance** as
part of the "rigorous evaluation" differentiator, but neither metric is implemented. This ADR
designs that increment as the third generation-quality signal, alongside the measured
`attribution_rate` (ADR-0006). Like the attribution and corrective harnesses, the numbers here are
credibility-sensitive: a faithfulness score that silently sits at `1.000` because a broken verifier
always answers "supported", an answer-relevancy that leaks the golden reference, or a headline that
implies we ran the canonical RAGAS library when we did not, would each destroy the very rigor signal
the metric is supposed to demonstrate.

Two structural facts force the central decision:

- **The `ragas` library conflicts head-on with this repo's binding Anthropic SDK rules.**
  `CLAUDE.md` §2 makes non-negotiable that *every* LLM call goes through the official `anthropic`
  SDK with `thinking={"type": "adaptive"}` + `output_config={"effort": ...}`, and **never** passes
  `temperature` / `top_p` / `top_k` / `budget_tokens` (all 400 on `claude-opus-4-8` /
  `claude-sonnet-4-6`) and **never** uses assistant-prefill. `ragas>=0.2` drives its LLM steps
  through `LangchainLLMWrapper` around `ChatAnthropic`, which sets a default `temperature` and has
  no adaptive-thinking / effort surface. Running RAGAS as shipped would therefore either 400 on
  every call (temperature removed on the 4.x models) or force a second, rule-violating LLM path
  into the codebase. The whole repo already routes LLM calls through one idiom —
  `AnthropicLLMClient`, `AnthropicCorrectiveLLM`, `AnthropicAnswerCorrectnessJudge` — all lazy-SDK,
  `messages.parse`, adaptive thinking, high effort, fail-closed. A LangChain-wrapped RAGAS breaks
  that invariant.

- **The corpus is easy and near-ceiling; the metric is instrumentation, not a trophy.** The
  published attribution run scores `attribution_rate = 1.000` (micro) on this 12-doc driving-theory
  corpus. Faithfulness — grounding of *all* answer claims against the *full* retrieved context — is
  very likely also near-ceiling here (the generator is grounded and the corpus is small and
  on-topic). So faithfulness must be pre-registered as **directional and regime-bound**: a near-`1.0`
  is a property of an easy corpus plus a grounded generator, **not** a differentiator "win", and a
  saturated metric is dangerously indistinguishable from one that is stuck at `1.0` because of a bug.

The design therefore inherits the ADR-0006 posture wholesale — separate opt-in entry point,
offline-fake-first, no CI, publishable only from a fully real run — and adds the RAGAS-specific
honesty guards the metric needs.

## Decision drivers

- **Keep the binding SDK rules intact.** No LangChain LLM wrapper, no `temperature`, no banned
  params, no prefill; every LLM call is the official `anthropic` SDK with adaptive thinking + effort,
  exactly like `judge.py` / `llm.py` / the corrective LLM.
- **Offline-fake-first + reproducible.** The whole eval must run byte-stably in CI with no API key
  and no GPU (deterministic fakes), and a real run must record full provenance. This is the same
  discipline that lets `attribution.py` and `corrective.py` be tested without a key.
- **`make eval` stays LLM-free / key-free.** Generation metrics need an LLM, so they get a *separate*
  `make eval-ragas` target, exactly as `make eval-attribution` (ADR-0006) and `make eval-corrective`
  (ADR-0008) are separate.
- **No overclaiming — credit the spec, own the engine.** If we do not literally run the RAGAS
  library, we must never label the numbers as if we did. They are a **faithful reimplementation of
  the published RAGAS algorithms**, credited as such everywhere they surface.
- **Position the three grounding/quality metrics so nothing is double-counted.** attribution
  (cited-span grounding), faithfulness (all-claim grounding vs full context), answer-relevancy
  (does the answer address the question) measure three distinct things and must be reported as such.
- **Honesty about non-determinism and saturation.** LLM statement-decomposition / NLI /
  question-generation are non-reproducible → point estimates + per-query distribution, no intra-run
  bootstrap. A negative fixture must force each score off its ceiling, or the metric is unfalsifiable.

## Options considered

**(a) The engine: RAGAS library vs reimplement vs hybrid.**

- **(a1) Use the `ragas` library as-is.** *Pros:* canonical, well-known name; zero algorithm work.
  *Cons:* it drives LLMs through `LangchainLLMWrapper`/`ChatAnthropic`, which sets `temperature` and
  offers no adaptive-thinking/effort — a direct violation of the binding SDK rules that would 400 on
  the 4.x models this repo targets, or force a rule-breaking second LLM path. It defaults
  answer-relevancy embeddings to OpenAI (new provider, API cost, non-determinism). It is not
  offline-fake-first (needs a key to run at all), so the CI/test path could not exercise it
  byte-stably. Its internal prompts change across minor versions, so a published number would be
  pinned to a moving prompt we do not control. **Rejected**: it breaks the one invariant the repo's
  LLM discipline rests on and cannot be made offline-deterministic.
- **(a2) Reimplement faithfulness + answer_relevancy over the Anthropic SDK, faithful to the
  published RAGAS algorithms, crediting RAGAS as the spec.** *Pros:* keeps the binding SDK rules
  intact (lazy `anthropic`, `messages.parse`, adaptive thinking, high effort, fail-closed); follows
  the *exact* established pattern (`AnswerCorrectnessJudge` Protocol + real Anthropic + deterministic
  Fake); offline-fake-first and byte-stable in CI; full reproducibility provenance; reuses the local
  bge-small embedder at $0 API cost; the algorithm is transparent and auditable in-repo. *Cons:* we
  own the decomposition/NLI/question-gen prompts, so our numbers are "RAGAS-style", not
  library-identical — we must be explicit about that. **Chosen.**
- **(a3) Hybrid — RAGAS prompts as source, our SDK as engine.** *Pros:* prompt fidelity to RAGAS.
  *Cons:* importing RAGAS purely for prompt strings couples us to its prompt versioning and pulls a
  heavyweight dep for a few templates; the LLM step still must be reimplemented on our SDK anyway.
  **Rejected as a dependency** but **partially adopted in spirit**: (a2)'s scorer docstrings cite the
  RAGAS algorithm and mirror its prompt intent without importing the library.

**(b) The `ragas` dependency in `pyproject.toml`.** Carrying `ragas>=0.2` as a runtime dep while not
using it is a small but real integrity smell (it implies usage). **Chosen: remove `ragas` from the
runtime dependencies.** If a one-off external cross-check against canonical RAGAS is ever wanted, it
belongs in an optional `[ragas-crosscheck]` extra used in a throwaway validation script, never in the
published path.

**(c) Faithfulness aggregation: macro-only vs micro-primary (mirror ADR-0006).** A macro mean of
per-answer faithfulness conflates a 0-statement abstention (`0.0` by convention) with a
"claimed-but-unsupported" `0.0`, exactly the failure ADR-0006 rejected for attribution.
**Chosen: micro (pooled) `total_supported / total_statements` as the headline** — immune to the
0-statement convention (an abstaining answer contributes nothing to either pool) — **plus macro,
macro-over-answered, and `n_faith_abstained`** as secondary. `faith_abstained := (n_statements == 0)`.

**(d) Answer-relevancy aggregation and the noncommittal rule.** Answer-relevancy is a per-query
similarity in roughly `[0, 1]`, not a count ratio, so there is no meaningful "pool". **Chosen:
macro mean over all queries as the headline**, with a principled asymmetry vs faithfulness:
a **noncommittal / refusal answer is included as `0`** (it genuinely fails to address the question —
that is what low relevancy *means*), whereas a faithfulness abstention is *excluded* (there is
nothing to ground). `n_noncommittal` and a committal-only mean are reported alongside so the effect
is always visible. This asymmetry is deliberate and pre-registered.

**(e) Answer-relevancy embeddings: local bge-small vs OpenAI default.** RAGAS defaults to OpenAI
embeddings. **Chosen: reuse the repo's `SentenceTransformerEmbedder` (bge-small, already a real
backend, $0 API cost)** — the same embedder the index uses. It keeps the run offline-capable,
deterministic at the embedding step (bge-small on CPU is a pure function of its inputs given a fixed
model revision), and adds no new provider. The generated questions are still LLM-produced (so the
*end-to-end* answer-relevancy number is non-reproducible), but the fake path uses `HashingEmbedder`
for byte-stability, and publishable gates on the real bge-small.

**(f) Score range for answer-relevancy: clamp to `[0, 1]` vs keep signed cosine.** Cosine of
normalized embeddings is theoretically `[-1, 1]`. Clamping negatives to `0` would hide a genuinely
divergent answer. **Chosen: store answer-relevancy as a `float` in a documented `[-1, 1]` range,
unclamped** (the noncommittal gate still forces exact `0`), while faithfulness stays the `[0, 1]`
`Score` type. Honest signal beats a tidy range.

**(g) One combined report vs two.** Both metrics score the **same** generated answers over the same
hybrid+rerank pass. **Chosen: one `GenerationQualityReport`** (faithfulness block + answer-relevancy
block + one provenance + one per-query record), computed from a single generation pass, mirroring the
ADR-0006/0008 precedent of a purpose-built report. Distinct artifact
`storage_dir/eval/generation_quality_results.json`.

## Decision

1. **Reimplement, do not import.** Faithfulness and answer_relevancy are reimplemented over the
   official `anthropic` SDK, faithful to the published RAGAS algorithms and **credited as
   "RAGAS-style (reimplementation of the published algorithm); RAGAS credited as the spec"** in the
   report header, docstrings, README, and `architecture.md`. `ragas` is removed from the runtime
   dependencies. The module is named `generation_quality.py` (not `ragas.py`) to avoid shadowing the
   installed package name and to avoid implying we vendored RAGAS.

2. **Scorer abstraction mirrors the judge pattern (ADR-0008 §9).** In `src/rag/eval/generation_scorers.py`:
   - `FaithfulnessScorer` Protocol: `score(question, answer, contexts) -> FaithfulnessResult`
     (`statements: tuple[StatementVerdict, ...]`, `n_statements`, `n_supported`).
     `AnthropicFaithfulnessScorer` performs the RAGAS two-step: (i) decompose the answer into atomic
     statements, (ii) verify each statement is inferable from the **full retrieved context**, both via
     `client.messages.parse(...)` with adaptive thinking + high effort, never the banned params, never
     prefill. `FakeFaithfulnessScorer` is deterministic: sentence-split the answer, ground each
     statement against the pooled context text by a public token-overlap floor (no reach into
     `verification` internals), so a fabricated claim scores unsupported.
   - `AnswerRelevancyScorer` Protocol: `score(question, answer, contexts) -> AnswerRelevancyResult`
     (`generated_questions`, `noncommittal`, `mean_similarity`, `score`).
     `AnthropicAnswerRelevancyScorer` generates `N` candidate questions (+ a `noncommittal` flag) from
     the answer via one `messages.parse` call, embeds the original and generated questions with the
     injected `SentenceTransformerEmbedder`, and returns `mean_i cos(q, q_i) * (1 - noncommittal)`.
     `FakeAnswerRelevancyScorer` templates deterministic questions from answer tokens, uses a lexical
     refusal heuristic for `noncommittal`, and embeds with `HashingEmbedder`.
   - Both scorers default to `default_judge_model(settings.llm_model)` (reused from `judge.py`) so the
     scorer model **differs from the generator** by default, blunting self-preference on the
     faithfulness verdicts; the resolved model is recorded in provenance and is constructor-overridable.

3. **Blind by construction (anti-leakage).** Neither scorer signature exposes `reference_answer` or
   `relevant_chunk_ids`. Faithfulness sees only `(question, answer, retrieved contexts)`;
   answer-relevancy sees only `(question, answer, contexts)` and structurally needs **no ground
   truth**. The golden set is never indexed (inherited guard). Neither metric can be gamed by, or
   leak, the golden labels — a property to be asserted in tests.

4. **One generation pass, real answering config.** `run_generation_quality_eval(settings, *,
   llm=None, faithfulness=None, answer_relevancy=None, embedder=None, store=None, reranker=None) ->
   GenerationQualityReport`, run via `python -m rag.eval.generation_quality` / `make eval-ragas`.
   Per golden query, in frozen file order: `HybridRetriever(use_reranker=True).retrieve(query,
   k=top_k_rerank)` → `generate_answer(query, contexts, llm=llm, settings=eval_settings)` → score the
   **same** answer with **both** scorers over the **same** contexts (no re-retrieval, no re-generation).
   `prepare_hermetic_eval` supplies the shared hermetic build + anti-leakage + golden-coverage guards.

5. **Faithfulness headline = micro (pooled).** `micro_faithfulness = total_supported /
   total_statements` (over answered queries). `macro_faithfulness` (mean of per-query faithfulness,
   abstentions counted `0.0`), `macro_faithfulness_answered`, and `n_faith_abstained`
   (`n_statements == 0`) are secondary and always reported so abstention is visible. Macro is never
   reported alone. Fail-closed: a statement whose verdict cannot be parsed counts as **not supported**.

6. **Answer-relevancy headline = macro mean over all queries.** Noncommittal answers are **included
   as `0`**; `n_noncommittal` and a committal-only mean are reported alongside. Stored as a `float`
   in a documented `[-1, 1]` range (unclamped). Fail-closed: an unparseable question-generation
   response yields `score = 0.0` for that answer (never silently high).

7. **No CI; single_run; publishable only when fully real.** No bootstrap (decomposition / NLI /
   question-gen are non-reproducible → an intra-run CI would be false precision, ADR-0006).
   `single_run = True`; point estimates + the per-query `GenerationQualityQueryRecord` distribution
   (including `n_statements` per query so a suspicious `n=1` is visible). `publishable = True` **iff**
   the generation LLM is `AnthropicLLMClient` **and** both scorers are the `Anthropic*` classes **and**
   the embedder is `SentenceTransformerEmbedder` **and** the reranker is `CrossEncoderReranker`. Any
   fake flips it `False`. The fake path (fake generator + fake scorers + `HashingEmbedder`) is fully
   deterministic and byte-stable.

8. **Provenance + config.** `GenerationQualityProvenance` records `generator_llm_class` / `llm_model`,
   `faithfulness_scorer_class` / `answer_relevancy_scorer_class` / `scorer_model`, `embedder_class` /
   `embedding_model`, `reranker_class`, `top_k_rerank`, `answer_relevancy_n_questions`, plus the
   corpus SHA-256 / git SHA / library versions read from the eval `meta.json`, `n_queries`,
   `single_run`, and `publishable`. One new `Settings` field:
   `ragas_answer_relevancy_n_questions: int = Field(default=3, gt=0)` (RAGAS default N); the scorer
   model reuses `default_judge_model` rather than adding a settings field. `make eval` and the other
   eval targets are unchanged; `make eval-ragas` is opt-in and LLM-required.

9. **Distinct, byte-diffable artifact.** Dumped (sorted keys) to
   `storage_dir/eval/generation_quality_results.json` — a name distinct from `eval_results.json`,
   `attribution_results.json`, and `corrective_results.json`. Byte-stable under the deterministic
   fakes; a real run is `single_run` and not bit-exact.

10. **Three-metric positioning (no double-counting).** The report and docs state crisply:
    - **attribution_rate** (ADR-0006) — grounding of the citations the model *made*; denominator =
      citations; deterministic lexical check. "Are the model's own citations honest?"
    - **faithfulness** (this ADR) — grounding of **all** answer statements against the **full**
      retrieved context, cited or not; denominator = statements; LLM NLI. "Are all the answer's claims
      supported by the evidence, whether or not the model cited them?" Broader scope than attribution;
      the two use different denominators and different checkers and are **never summed** or presented
      as one improving the other.
    - **answer_relevancy** (this ADR) — does the answer address the **question**; needs **no** ground
      truth; embedding-cosine over LLM-generated questions. An orthogonal axis: an answer can be
      faithful but off-topic, or on-topic but hallucinated.

## Consequences

Positive:
- **The binding SDK rules stay intact.** Every LLM call is the official `anthropic` SDK with adaptive
  thinking + high effort, fail-closed, `messages.parse` — no LangChain wrapper, no `temperature`, no
  banned params, no prefill. The reimplementation is the *stronger* engineering statement: it demonstrates we
  understand the RAGAS algorithm well enough to own it, rather than importing a black box that would
  break our own rules.
- **Fully offline-testable and byte-stable under fakes**, so structure / invariants / abstention +
  noncommittal accounting / the negative fixtures are covered in CI without a key or GPU. Exactly one
  real `make eval-ragas` publishes.
- **Honest, non-overlapping reporting** of attribution vs faithfulness vs answer-relevancy, with
  faithfulness pre-registered as directional/regime-bound (near-ceiling on this corpus is not a win).
- **$0 API cost for embeddings** (reused local bge-small) and no new provider.
- **Future-proof**: on a harder corpus the same harness surfaces a real faithfulness gap and a real
  relevancy signal without a redesign.

Negative / accepted:
- **The numbers are "RAGAS-style", not RAGAS-library-identical.** Accepted and stated explicitly
  everywhere; we never claim to have run canonical RAGAS. A one-off external cross-check remains
  possible via an optional extra but is never the published path.
- **Faithfulness is decomposition-sensitive.** `supported / statements` depends on how the answer is
  split; a one-giant-statement decomposition can mask unsupported sub-claims and an over-split can
  dilute the denominator. Mitigated by recording `n_statements` per query, pinning the decomposition
  prompt, failing closed, and a mandatory negative test fixture (a fabricated claim **must** drop
  faithfulness below `1.0`), so a saturated `1.000` cannot hide an always-"supported" bug.
- **Non-reproducible → no CI, `single_run`, publishable only when fully real**; a real run is not
  bit-exact (ADR-0006 caveat carries over). `scorer_model` / `llm_model` aliases are recorded
  verbatim (ADR-0006 alias-drift caveat); the bge-small model revision is likewise an alias recorded
  via `library_versions`.
- **Self-preference risk on faithfulness** if the scorer model equals the generator model; mitigated
  by defaulting the scorer to a different model (`default_judge_model`) and printing a self-preference
  banner when `scorer_model == llm_model`.
- **The real run adds LLM cost**: per query, ~2 faithfulness calls (decompose + verify) + 1
  answer-relevancy call, over the 50 golden queries — small structured outputs, on the order of a few
  US dollars; exact figures are not published.

## Measured numbers

First reproducible `make eval-ragas` run (`publishable=true`), published in
[`README.md § Generation quality — RAGAS-style (measured)`](../../README.md#generation-quality--ragas-style-measured).
All backends real: `AnthropicLLMClient` generator (`claude-sonnet-4-6`) + real
`AnthropicFaithfulnessScorer` / `AnthropicAnswerRelevancyScorer` (`claude-opus-4-8`, distinct from
the generator) + bge-small embedder + cross-encoder reranker; hybrid+rerank, top_k_rerank=5, N=3,
n=50, git_sha `3e9b399`, corpus_sha256 `beb2701a`, `single_run=true` (no CI).

- **faithfulness — micro (headline) = 0.981** (210/214 pooled supported statements); macro 0.986;
  macro over answered 0.986 (n_answered=50); 0/50 abstentions.
- **answer-relevancy — macro (headline) = 0.828** (mean over all queries); committal-only 0.828;
  0/50 noncommittal; per-query range 0.661–0.958.

The pre-registered prediction held: faithfulness near-ceiling (regime-bound, **directional — not a
differentiator win**) and answer-relevancy high but below `1.0`. Crucially, faithfulness did **not**
saturate at `1.000`: the NLI flagged 4 unsupported statements across 2 queries (q08 5/7, q48 3/5),
live evidence the scorer discriminates rather than always answering "supported" — the exact failure
the mandatory fabricated-claim fixture guards against. Numbers are **RAGAS-style** (reimplementation),
never presented as the canonical `ragas` library's output, and are never summed with — nor claimed to
improve — the measured `attribution_rate`. Republish only from a fresh reproducible run (`single_run`
point estimates drift run-to-run, exactly the no-CI caveat), never hand-edited.

## Cross-links

Inherits the offline-fake-first / no-CI / publishable-only-when-real / micro-vs-macro-vs-abstention
discipline of [ADR-0006](ADR-0006-attribution-rate-aggregation.md) (attribution aggregation) and the
Protocol + real-Anthropic + deterministic-Fake + blind-judging + `default_judge_model` pattern of
[ADR-0008](ADR-0008-corrective-vs-baseline-eval.md); reuses
[ADR-0005](ADR-0005-retrieval-eval-harness.md)'s `prepare_hermetic_eval` (hermetic build +
anti-leakage guards) and [ADR-0004](ADR-0004-eval-metrics-and-paired-bootstrap.md)'s eval-model
discipline. Complements — and is explicitly **not** double-counted with — the measured
`attribution_rate`. Linked from `docs/architecture.md` (Decisions).
