# ADR-0007 — Self-corrective RAG layer: a bounded LangGraph StateGraph over the existing pipeline

- Status: Proposed
- Date: 2026-07-02
- Deciders: rag-architect (implemented by agentic-graph-engineer)
- Scope: `src/rag/agentic/corrective_rag.py` (new — currently an empty stub) plus the
  agentic-owned Pydantic contracts (`CorrectiveRAGRequest`, `CorrectiveRAGResult`,
  `DocRelevanceGrade`, `GradeResponse`, `RewrittenQuery`) and the corrective-LLM abstraction
  (`CorrectiveLLM` Protocol + `AnthropicCorrectiveLLM` + deterministic fakes) it introduces,
  plus the five `agentic_*` fields added to `src/rag/config.py`. **The API surface, streaming,
  and a corrective-vs-baseline eval are explicitly out of scope** and land in later increments.
  The layer **composes on top of** — and must not duplicate — the hybrid retrieval
  ([ADR-0001](ADR-0001-hybrid-retrieval.md)/[ADR-0002](ADR-0002-reciprocal-rank-fusion.md)),
  the citation-enforced generation, and the measured attribution check
  ([ADR-0006](ADR-0006-attribution-rate-aggregation.md)).

## Context

The pipeline answers a query in one straight shot: `HybridRetriever.retrieve` → `generate_answer`
→ `verify_answer` (this is exactly `RagService.answer_query`, `src/rag/api/service.py:112`). That
single pass has two silent failure modes the rest of the repo already *measures* but never *acts*
on:

1. **Bad retrieval.** A lexically-mismatched or under-specified query returns contexts that do not
   contain the answer. Generation then correctly abstains (0 citations) or, worse, reaches for a
   weakly-related span. Nothing tries a reformulation.
2. **Ungrounded generation.** The model cites a chunk whose quote does not lexically ground;
   `verify_answer` flags it (`report.unsupported` non-empty, `attribution_rate < 1.0`) but the
   pass-through returns it as-is.

The optional self-corrective layer (in `CLAUDE.md` §3 and the target architecture) adds two bounded
feedback loops on top of the existing modules: **grade the retrieved docs and, on low relevance,
rewrite the query and retry**; **verify the answer and, on grounding failure, regenerate**. The
credibility-sensitive part is not the loops — it is proving they *always terminate*, that they
never mask an honest abstention as a failure to be retried forever, and that the layer reuses the
already-measured retrieval / attribution logic rather than growing a second, drifting copy of it.

Three structural facts shape the design:

- **The generation `LLMClient` Protocol only exposes `generate_answer`** (`src/rag/generation/llm.py:52`).
  Doc-grading and query-rewrite are new LLM capabilities with no home in the generation contract.
- **`verify_answer`'s 0-citation rule is `attribution_rate = 0.0`** (`src/rag/verification/citations.py:261`).
  An honest abstention ("the context does not answer this") is *indistinguishable at the rate* from
  a cited-but-ungrounded answer. A regenerate loop that triggers on `attribution_rate < threshold`
  would loop on every abstention — burning the budget trying to force citations onto an
  unanswerable query. This is the single most important correctness trap in this ADR.
- **LLM calls are not bit-exact reproducible** (established in ADR-0006). The graph's *structure*
  is deterministic; its LLM-backed nodes are not. Tests therefore drive it entirely with
  deterministic fakes (no API key at build time), exactly as generation/verification/eval already do.

## Decision drivers

- **Composition, not duplication.** The graph nodes must call the *existing* `HybridRetriever.retrieve`,
  `generate_answer`, and `verify_answer` verbatim. No fusion, rerank, or attribution logic is
  re-implemented in `agentic/`.
- **Guaranteed termination.** Two independent, strictly-monotonic, finitely-bounded counters plus a
  derived `recursion_limit` backstop. The adversarial reviewer will attack an unbounded loop first;
  the design must make an infinite loop impossible, not merely unlikely.
- **Honest abstention.** An answer with 0 citations is *accepted*, never regenerated — abstention is
  a correct outcome, not a grounding failure.
- **Graceful degradation, no exception-swallowing.** When a budget exhausts, the caller gets the
  best-effort `Answer` **with its measured `VerificationReport`** and a trace of why it stopped —
  never an exception, never an infinite loop, and never a fabricated "success". Genuine infra
  errors (SDK/network) still propagate; only the RAG-quality control loops degrade.
- **Offline-testable with fakes, no key at build time.** Retriever, embedder, generation LLM, and
  the new corrective LLM are all dependency-injected; the whole graph runs on fakes.
- **No eval-signal leakage.** The layer rewrites the *query text* only; it never reads golden
  labels, never special-cases eval queries, and touches no index build (so no `meta.json` impact).

## Options considered

**(1) Where do grade + rewrite LLM calls live?**
- **(1a) Extend the generation `LLMClient` Protocol** with `grade_documents` / `rewrite_query`.
  Con: pollutes the generation contract with agentic-only concerns; forces `FakeLLMClient` /
  `AnthropicLLMClient` (owned by `citation-verifier`) to grow methods generation never uses; blurs
  the module boundary — generation would "know" about grading.
- **(1b) Agentic-owned `CorrectiveLLM` Protocol** (new, in `agentic/`) with `grade_documents` and
  `rewrite_query`, a real `AnthropicCorrectiveLLM` that reuses the *exact* SDK path the generation
  client uses (`client.messages.parse(output_format=...)`, adaptive thinking + effort, no banned
  params), and deterministic fakes. The graph is injected with the existing `LLMClient` (for the
  `generate` node, passed straight to `generate_answer`) **and** a `CorrectiveLLM`. In production a
  single Anthropic-backed object may satisfy both Protocols. **Chosen** — generation stays untouched,
  the graph is fully DI/testable, and the SDK rules are honored in one new place.

**(2) State schema: frozen Pydantic vs mutable Pydantic vs TypedDict?**
- Frozen Pydantic (the repo default for `Answer`/`RetrievalResult`) cannot be a LangGraph channel —
  nodes return partial updates merged into mutable state. A mutable Pydantic state model works on
  langgraph ≥ 0.2 but adds validation friction on every partial merge.
- **Chosen: a `TypedDict` (`CorrectiveRAGState`, `total=False`) for the graph-internal channel state,
  Pydantic at the module boundary** (`CorrectiveRAGRequest` in, frozen `CorrectiveRAGResult` out).
  This honors "inter-module data crosses as Pydantic models" while using the LangGraph-idiomatic,
  reducer-friendly channel type internally. All fields are last-write-wins (no custom reducers, no
  accumulator sets) — one less nondeterminism source.

**(3) How is "low relevance" decided?**
- A per-doc LLM grade (`relevant: bool`) is the structured, defensible signal; a bare cosine/RRF
  score threshold would just re-encode first-stage retrieval and add nothing.
- **Chosen: batch-grade all retrieved docs in one structured `messages.parse` call** (`GradeResponse.grades`),
  key grades by `chunk_id`, **fail-open on omissions** (a context with no returned grade defaults to
  `relevant=True`, and grades for unknown ids are ignored) so a flaky grader can never silently drop
  a retrieved doc. **Low relevance ⇔ `len(relevant) < agentic_min_relevant_docs`** (default 1: "we
  found at least one relevant doc"). The graded-relevant subset (order preserved) becomes the
  generation context, tightening grounding precision; on an empty subset the graph falls back to the
  full retrieved set so the model can abstain honestly rather than see nothing.

**(4) How does regenerate actually correct anything without a generation-contract change?**
- **(4a) Additive `feedback` param** to `build_grounding_prompt`/`generate_answer` telling the model
  which citations failed. Genuinely corrective, but expands the generation contract and pushes work
  into `citation-verifier`'s module.
- **(4b) Bounded pure re-call** of the *same* `generate_answer(original_query, contexts, llm, settings)`.
  Zero blast radius into generation; correction comes from the real client's run-to-run variation
  (ADR-0006 established generation is not bit-exact). **Chosen for v1** — strictest composition, no
  cross-module churn for the implementing agent. Named weakness: against a *deterministic* fake a
  re-call cannot change the outcome, so the "regenerate-then-passes" test uses a **scripted fake**
  (fabricated quote first call, grounded quote second). (4a) is the ADR-approved upgrade path once a
  corrective eval shows regeneration's lift justifies the contract change.

**(5) Regenerate trigger vs the 0-citation rule.**
- **Chosen: accept iff `n_citations == 0` (honest abstention) OR `attribution_rate >=
  agentic_min_attribution_rate` (default 1.0 ⇔ no unsupported citations).** Regenerate *only* when
  there is at least one unsupported citation and budget remains. This is the load-bearing guard that
  stops the regenerate loop from chasing citations on an unanswerable query.

## Decision

1. **Module = pure composition.** `src/rag/agentic/corrective_rag.py` builds and runs a LangGraph
   `StateGraph`. Its nodes call the existing contracts verbatim (exact signatures below); it
   re-implements no retrieval, fusion, rerank, or attribution logic.

   | Node | Calls (existing / new) |
   |------|------------------------|
   | `analyze` | LLM-free init: `current_query = query`, counters = 0 (no API call; deterministic) |
   | `retrieve` | `HybridRetriever.retrieve(current_query, k=settings.top_k_rerank)` → `list[RetrievalResult]` |
   | `grade_documents` | `CorrectiveLLM.grade_documents(current_query, retrieved)` → `list[DocRelevanceGrade]` |
   | `rewrite_query` | `CorrectiveLLM.rewrite_query(query, current_query, retrieved)` → `RewrittenQuery` |
   | `generate` | `generate_answer(query, contexts, llm=llm, settings=settings)` → `Answer` *(original query; contexts = relevant subset, else retrieved)* |
   | `verify` | `verify_answer(answer, contexts)` → `VerificationReport` *(same contexts the generator saw — ADR-0006 principle)* |

   Exact existing signatures the nodes bind to:
   - `HybridRetriever.retrieve(self, query: str, *, k: int | None = None) -> list[RetrievalResult]`
   - `generate_answer(query: str, contexts: list[RetrievalResult], *, llm: LLMClient, settings: Settings) -> Answer`
   - `verify_answer(answer: Answer, contexts: list[RetrievalResult]) -> VerificationReport`
   - `LLMClient.generate_answer(self, query: str, contexts: list[RetrievalResult]) -> Answer`

2. **Edges.** `START → analyze → retrieve → grade_documents`; conditional `route_after_grade`:
   `generate` if `len(relevant) >= agentic_min_relevant_docs` **or** `n_rewrites >=
   agentic_max_query_rewrites`, else `rewrite_query`; `rewrite_query → retrieve`; `generate →
   verify`; conditional `route_after_verify`: `END` if accepted (abstention or grounded), `generate`
   (regenerate) if `report.unsupported` and `n_regenerations < agentic_max_regenerations`, else `END`
   (best-effort). Counters are incremented in `rewrite_query` / on the regenerate branch, before
   re-entering the loop.

3. **Agentic-owned contracts** (new Pydantic models, frozen where they are records):
   - `DocRelevanceGrade{ chunk_id: str, relevant: bool, reason: str }`
   - `GradeResponse{ grades: list[DocRelevanceGrade] }` — the `messages.parse(output_format=...)` schema
   - `RewrittenQuery{ query: str, rationale: str }`
   - `CorrectiveLLM` Protocol: `grade_documents(query, contexts) -> list[DocRelevanceGrade]`,
     `rewrite_query(original_query, current_query, contexts) -> RewrittenQuery`
   - `CorrectiveRAGRequest{ query: str, k: int | None = None }` (boundary input)
   - `CorrectiveRAGResult{ answer: Answer, report: VerificationReport, contexts: list[RetrievalResult],
     original_query: str, final_query: str, n_rewrites: int, n_regenerations: int,
     rewrite_budget_exhausted: bool, terminated_reason: TerminatedReason }` (frozen boundary output;
     `rewrite_budget_exhausted` and the `TerminatedReason` literal are explained in decision 6)
   - `CorrectiveRAGState(TypedDict, total=False)` — graph-internal only, never crosses the boundary.

4. **Corrective LLM follows the SDK rules exactly.** `AnthropicCorrectiveLLM` lazy-imports `anthropic`,
   calls `client.messages.parse(output_format=GradeResponse / RewrittenQuery)` with
   `thinking={"type": "adaptive"}`, `output_config={"effort": ...}`, model `settings.llm_model`
   (`claude-sonnet-4-6` default — grade/rewrite are the cheap, higher-volume path), and **never**
   passes `temperature`/`top_p`/`top_k`/`budget_tokens` or uses prefill. Grade/rewrite prompts are
   kept simple and are pure functions of `(query, contexts)` (no timestamps, no set iteration).

5. **Termination guarantee (proof).** Let `R = agentic_max_query_rewrites`, `G =
   agentic_max_regenerations`. `n_rewrites` and `n_regenerations` are initialized to 0, are strictly
   incremented before re-entering their respective loops, are never reset or decremented, and each
   conditional routes to a terminal path once its counter reaches its fixed budget. Hence: retrievals
   ≤ `R + 1`, generations ≤ `G + 1`; total node executions are finite. Each retrieval-loop cycle
   (`retrieve` -> `grade_documents` -> `rewrite_query`) and each regenerate-loop cycle (`generate` ->
   `verify` -> `regenerate`) costs exactly 3 node executions, so the true worst case is
   `3*R + 3*G + 6` (the `+6` covers `analyze`/`START`, the final `retrieve`/`grade_documents` pass
   after the rewrite budget is spent, the final `generate`/`verify` pass after the regenerate budget
   is spent, and a terminal marker node). As a belt-and-suspenders backstop the graph is invoked with
   `recursion_limit = (R + 1) * 3 + (G + 1) * 3 + 5` — a derived bound (not a drifting config) chosen
   to stay comfortably above `3*R + 3*G + 6` for every `R, G >= 0` (an earlier `(G + 1) * 2` term
   under-counted the regenerate cycle by one node and crossed below the true worst case at `G >= 4`,
   which the adversarial review caught: it fired the recursion backstop on a legitimate
   always-fabricating run instead of the intended `regenerate_budget_exhausted` outcome — fixed by
   giving the regenerate cycle the same `* 3` multiplier as the retrieval cycle). If LangGraph ever
   raises `GraphRecursionError` the layer catches it, logs at error level, and returns the last
   best-effort state with `terminated_reason = "recursion_limit"`. Two independent bounds + one hard
   cap ⇒ the graph provably halts.

6. **Failure semantics.** The graph **always** returns a `CorrectiveRAGResult` carrying the
   best-effort `Answer`, its **measured** `VerificationReport`, the contexts, and the trace
   (`n_rewrites`, `n_regenerations`, `rewrite_budget_exhausted`,
   `terminated_reason ∈ {"grounded", "abstained", "regenerate_budget_exhausted",
   "recursion_limit"}`). It never raises for a RAG-quality failure and never swallows a genuine
   collaborator exception into a fake answer — SDK/network errors propagate; only the control loops
   degrade gracefully. Note `"rewrite_budget_exhausted"` is deliberately NOT a member of the
   `terminated_reason` set: `route_after_verify`'s own grounded/abstained/regenerate-exhausted
   determination always runs after `generate` and always has the final say, so a value only
   `grade_documents` could set would never actually survive to the boundary as the FINAL reason. That
   fact — caught in adversarial review — is instead tracked honestly as its own
   `rewrite_budget_exhausted: bool` field: True iff the last doc-grading pass still had too few
   relevant docs after the rewrite budget ran out (i.e. `generate` proceeded on a best-effort context
   set), independent of whatever `verify` subsequently decides.

7. **Config additions** (`src/rag/config.py`, one `--- Agentic (self-corrective RAG) ---` block):
   `agentic_enabled: bool = False`, `agentic_max_query_rewrites: int = Field(default=2, ge=0)`,
   `agentic_max_regenerations: int = Field(default=1, ge=0)`, `agentic_min_relevant_docs: int =
   Field(default=1, ge=1)`, `agentic_min_attribution_rate: float = Field(default=1.0, ge=0.0, le=1.0)`.
   Budget rationale: **2 rewrites** (≤ 3 retrievals) — a single reformulation often fails to fix a
   lexical miss, two bounded attempts trade recall recovery against latency/cost, beyond that is
   diminishing returns; **1 regeneration** (≤ 2 generations) — a bounded single retry catches a
   transient ungrounded citation without thrashing on a non-deterministic generator.

8. **`langgraph` is already a runtime dependency** (`pyproject.toml:23`, `langgraph>=0.2`) and is
   installed in CI via `pip install -e ".[dev]"` — **no pyproject change is needed**. The graph
   builder defers the `langgraph` import (mirroring the repo's lazy-import discipline) so
   `import rag.agentic` stays cheap and the Pydantic contracts import even in a stripped environment.

## Consequences

Positive:
- The self-corrective layer is a **pure composition**: retrieval, fusion, rerank, generation, and the
  measured attribution check all live in exactly one place and cannot drift into an agentic copy.
- **Provably terminating** — two bounded counters plus a derived `recursion_limit` backstop; graceful
  degradation returns a best-effort answer *with its measured report*, never a hang or a swallowed error.
- **Honest abstention is preserved**: the 0-citation case is accepted, so the regenerate loop cannot
  chase citations onto an unanswerable query — the ADR-0006 discipline carries into the graph.
- **Fully offline-testable** with deterministic fakes (retriever, generation LLM, corrective LLM); no
  API key at build time, consistent with generation/verification/eval.
- Generation and verification contracts are **untouched** — zero blast radius into other agents' modules.

Negative / accepted:
- **Regeneration is a bounded pure re-call** (Option 4b): against a *deterministic* client it cannot
  change the outcome, so its practical lift comes only from the real client's run-to-run variation.
  The stronger additive-`feedback` mechanism (4a) is deferred to a measured upgrade.
- Grading/rewrite add **LLM latency and cost** (up to `R` extra grade+rewrite round-trips and `G`
  extra generations); the layer is `agentic_enabled=False` by default and opt-in.
- A real corrective run is **not bit-exact reproducible** (LLM-backed nodes), exactly as ADR-0006;
  determinism and byte-stability hold only under the fakes.
- `settings.llm_model` is an **alias** recorded verbatim (ADR-0006's known alias-drift caveat applies
  identically here).

## Measured numbers

**Intentionally left blank.** Per the headline-metric rule, no corrective-vs-baseline numbers are
recorded here until a reproducible run produces them. The **confirming metric** (a later increment,
out of scope for this ADR) is a paired corrective-vs-baseline comparison over the committed golden
set: (a) **final-retrieval `recall@k` / `nDCG@k`** after rewrite vs the one-shot hybrid baseline
(does query rewrite recover the relevant chunk on lexical-mismatch queries?), and (b) the **micro
`attribution_rate`** and **`n_abstained`** from the ADR-0006 aggregation, with the mean
`n_rewrites` / `n_regenerations` as the cost axis. A regression guard applies: the corrective layer
must not *reduce* attribution vs the baseline. For v1 the confirmation is the deterministic test
suite (happy-path, rewrite-path, rewrite-budget-exhaustion, regenerate-path,
regenerate-budget-exhaustion, and byte-level determinism under fakes).

## Cross-links

Composes on top of [ADR-0001](ADR-0001-hybrid-retrieval.md) /
[ADR-0002](ADR-0002-reciprocal-rank-fusion.md) (the hybrid+RRF retrieval the `retrieve` node calls),
the citation-enforced generation, and [ADR-0006](ADR-0006-attribution-rate-aggregation.md) (the
measured `attribution_rate` and the 0-citation/abstention discipline the `verify` node inherits).
Linked from `docs/architecture.md` (Decisions).
