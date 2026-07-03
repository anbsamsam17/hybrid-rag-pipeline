"""Self-corrective RAG: a bounded LangGraph ``StateGraph`` over the existing pipeline.

Implements ADR-0007 (``docs/decisions/ADR-0007-self-corrective-rag-stategraph.md``). This
module is a **pure controller**: it composes the existing, already-tested modules —
:class:`~rag.retrieval.hybrid.HybridRetriever`, :func:`~rag.generation.generate.generate_answer`,
:func:`~rag.verification.citations.verify_answer` — into two bounded feedback loops:

1. **Grade -> rewrite -> retry.** After retrieval, an LLM (or deterministic fake) judge grades
   each retrieved chunk's relevance. If too few are relevant, the query is rewritten and
   retrieval retried, up to ``settings.agentic_max_query_rewrites`` times.
2. **Verify -> regenerate -> retry.** After generation, the existing attribution checker scores
   the answer. If it has unsupported citations, generation is retried, up to
   ``settings.agentic_max_regenerations`` times. An honest 0-citation abstention is *accepted*,
   never retried (the load-bearing anti-thrash guard from ADR-0007 decision driver #5).

Both loops are bounded by strictly-monotonic counters that are checked *before* the routing
decision and incremented *as the retry action is taken*, so the graph provably halts (see
:func:`_derive_recursion_limit` for the belt-and-suspenders backstop). Nothing here
re-implements fusion, rerank, or attribution logic — the graph's only original logic is
grading, rewriting, and deciding when to retry.

The graph channel state (:class:`CorrectiveRAGState`) is a plain ``TypedDict`` (LangGraph's
idiomatic, reducer-free channel type — see ADR-0007 decision (2)); Pydantic models are used at
the module boundary (:class:`CorrectiveRAGRequest` in, frozen :class:`CorrectiveRAGResult` out)
and for the new agentic-owned records (:class:`DocRelevanceGrade`, :class:`GradeResponse`,
:class:`RewrittenQuery`).

``langgraph`` is deferred-imported inside :func:`build_corrective_rag` so importing this module
(and constructing the Pydantic contracts / fakes) never requires it, mirroring the lazy-import
discipline used for ``anthropic`` in :mod:`rag.generation.llm`.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from rag.config import Settings
from rag.generation.generate import generate_answer
from rag.generation.llm import LLMClient
from rag.generation.models import Answer
from rag.indexing.sparse import tokenize
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.models import RetrievalResult
from rag.verification.citations import verify_answer
from rag.verification.models import VerificationReport

logger = logging.getLogger(__name__)

# Effort for the grade/rewrite structured-output calls. Kept explicit (not left to a client
# default) per CLAUDE.md's "never silently pass-mark" discipline for correctness-sensitive
# LLM-judge calls.
_EFFORT = "high"
# Grading/rewriting are small structured JSON objects; comfortably bounded.
_GRADE_MAX_TOKENS = 2048
_REWRITE_MAX_TOKENS = 512


# =====================================================================================
# Agentic-owned Pydantic contracts (ADR-0007 decision 3)
# =====================================================================================


class DocRelevanceGrade(BaseModel):
    """One retrieved chunk's relevance verdict, keyed by ``chunk_id``."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="The graded chunk's id (must match a RetrievalResult).")
    relevant: bool = Field(description="True iff this chunk helps answer the query.")
    reason: str = Field(description="One short sentence explaining the verdict.")


class GradeResponse(BaseModel):
    """The ``messages.parse(output_format=...)`` schema for a batch doc-grading call."""

    model_config = ConfigDict(frozen=True)

    grades: list[DocRelevanceGrade] = Field(
        default_factory=list, description="One grade per graded chunk, any order."
    )


class RewrittenQuery(BaseModel):
    """A reformulated query plus the rationale for the reformulation."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="The reformulated query to retrieve with next.")
    rationale: str = Field(description="Why this reformulation should retrieve better docs.")


#: The actually-reachable set of final `terminated_reason` values. `"rewrite_budget_exhausted"`
#: is deliberately NOT a member: `verify`'s own terminal routing (accept_grounded /
#: accept_abstained / exhaust_regenerate) always runs after `generate` and always has the final
#: say, so a value only `grade_documents` could set would never survive to the boundary — see
#: `rewrite_budget_exhausted: bool` on `CorrectiveRAGResult` for the honest, separately-tracked
#: signal that the rewrite loop specifically ran out of budget.
TerminatedReason = Literal[
    "grounded", "abstained", "regenerate_budget_exhausted", "recursion_limit"
]


class CorrectiveRAGRequest(BaseModel):
    """Boundary input: what the caller asks the corrective graph to answer."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="The user's question.")
    k: int | None = Field(
        default=None, description="Final retrieval count; None defers to settings.top_k_rerank."
    )


class CorrectiveRAGResult(BaseModel):
    """Boundary output: always returned, never raised for a RAG-quality failure.

    Carries the best-effort :class:`~rag.generation.models.Answer`, its **measured**
    :class:`~rag.verification.models.VerificationReport`, the exact contexts generation and
    verification saw, and a trace of the control flow (rewrite/regenerate counts and why the
    run stopped).
    """

    model_config = ConfigDict(frozen=True)

    answer: Answer
    report: VerificationReport
    contexts: list[RetrievalResult] = Field(
        description="The SAME contexts generate() and verify() saw for the final answer."
    )
    original_query: str
    final_query: str
    n_rewrites: int = Field(ge=0)
    n_regenerations: int = Field(ge=0)
    rewrite_budget_exhausted: bool = Field(
        default=False,
        description="True iff the LAST doc-grading pass still had too few relevant docs after "
        "the rewrite budget ran out (generation then proceeded on a best-effort context set). "
        "Tracked separately from terminated_reason because verify's own grounded/abstained/"
        "regenerate-exhausted determination always has the final say once it runs.",
    )
    terminated_reason: TerminatedReason = Field(
        description='One of: "grounded" | "abstained" | "regenerate_budget_exhausted" | '
        '"recursion_limit" — see rewrite_budget_exhausted for the separate rewrite-loop signal.'
    )


# =====================================================================================
# Graph-internal state (ADR-0007 decision 2): TypedDict, last-write-wins, no reducers.
# =====================================================================================


class CorrectiveRAGState(TypedDict, total=False):
    """LangGraph channel state; never crosses the module boundary (see ``CorrectiveRAGResult``).

    Beyond the fields the ADR calls out at minimum, two internal-plumbing fields are carried
    through the channel because they must survive repeated node re-entries within one run:

    * ``k`` — the request's requested result count. ``build_corrective_rag`` does not accept a
      per-request ``k`` (its signature is retriever/llm/corrective/settings only), so the value
      is threaded through the initial invoke state instead of being bound at build time.
    * ``contexts`` — the exact context list ``generate`` used, stashed so ``verify`` scores the
      IDENTICAL list (never re-derives it), per ADR-0006's "same contexts" principle.

    ``rewrite_budget_exhausted`` is set (both True and False) by every ``grade_documents`` pass,
    so last-write-wins naturally reflects only the FINAL grading pass — the one that actually
    decided to fall through to ``generate``.
    """

    query: str
    current_query: str
    k: int | None
    retrieved: list[RetrievalResult]
    relevant: list[RetrievalResult]
    grades: list[DocRelevanceGrade]
    contexts: list[RetrievalResult]
    answer: Answer
    report: VerificationReport
    n_rewrites: int
    n_regenerations: int
    rewrite_budget_exhausted: bool
    terminated_reason: TerminatedReason


# =====================================================================================
# CorrectiveLLM: agentic-owned Protocol (do NOT extend the generation LLMClient Protocol)
# =====================================================================================


@runtime_checkable
class CorrectiveLLM(Protocol):
    """Grade retrieved docs and rewrite a query. Owned by ``agentic/`` — new capabilities the
    generation ``LLMClient`` Protocol has no business exposing (ADR-0007 decision 1b)."""

    def grade_documents(
        self, query: str, contexts: list[RetrievalResult]
    ) -> list[DocRelevanceGrade]:
        """Return one relevance grade per context (fail-open handled by the calling node)."""
        ...

    def rewrite_query(
        self, original_query: str, current_query: str, contexts: list[RetrievalResult]
    ) -> RewrittenQuery:
        """Return a reformulated query, given the docs the current query failed to satisfy."""
        ...


def _build_grade_prompt(query: str, contexts: list[RetrievalResult]) -> str:
    """Pure prompt builder for doc-grading: a function of (query, contexts) only."""
    lines = [
        "Judge whether each retrieved chunk below is relevant enough to help answer the "
        "query. A chunk is relevant if it contains information that could support an answer.",
        f"Query: {query}",
        "",
        "Chunks:",
    ]
    for ctx in contexts:
        lines.append(f"- chunk_id={ctx.chunk_id}: {ctx.text}")
    lines.append("")
    lines.append("Return one grade per chunk_id listed above.")
    return "\n".join(lines)


def _build_rewrite_prompt(
    original_query: str, current_query: str, contexts: list[RetrievalResult]
) -> str:
    """Pure prompt builder for query rewriting: a function of the three args only."""
    lines = [
        "The current query retrieved chunks that do not sufficiently answer the original "
        "question. Reformulate the query (expand, disambiguate, or add synonyms) so a "
        "retrieval system is more likely to find the relevant chunks.",
        f"Original question: {original_query}",
        f"Current query: {current_query}",
        "",
        "Chunks retrieved by the current query (for context on what's mismatched):",
    ]
    for ctx in contexts:
        lines.append(f"- chunk_id={ctx.chunk_id}: {ctx.text}")
    lines.append("")
    lines.append("Return a meaningfully different query and a short rationale.")
    return "\n".join(lines)


class AnthropicCorrectiveLLM:
    """Real ``CorrectiveLLM``: lazy-imports ``anthropic``, structured ``messages.parse`` calls.

    Mirrors :class:`rag.generation.llm.AnthropicLLMClient._ensure_client` exactly: constructing
    this object never imports the SDK or touches the network (no key required at build time);
    the import happens on first :meth:`grade_documents` / :meth:`rewrite_query` call. Both
    methods use adaptive thinking + high effort and NEVER pass ``temperature`` / ``top_p`` /
    ``top_k`` / ``budget_tokens``, and NEVER use assistant-prefill (all four 400 on the SDK's
    4.x models this repo targets).
    """

    def __init__(self, settings: Settings) -> None:
        """Store config; defer SDK import and client construction until first use."""
        self._settings = settings
        self._client: object | None = None

    def _ensure_client(self) -> object:
        """Lazily import ``anthropic`` and build the SDK client once."""
        if self._client is None:
            import anthropic  # lazy: package imports fine without the SDK installed

            self._client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        return self._client

    def grade_documents(
        self, query: str, contexts: list[RetrievalResult]
    ) -> list[DocRelevanceGrade]:
        """Batch-grade every context's relevance to ``query`` in one structured call."""
        if not contexts:
            return []
        client = self._ensure_client()
        prompt = _build_grade_prompt(query, contexts)
        logger.info("grading %d docs: model=%s", len(contexts), self._settings.llm_model)
        response = client.messages.parse(  # type: ignore[attr-defined]
            model=self._settings.llm_model,
            max_tokens=_GRADE_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": _EFFORT},
            output_format=GradeResponse,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = response.parsed_output
        if parsed is None:
            # Fail-open: a grader that returns nothing parseable must not silently drop
            # retrieved docs. Treat every context as relevant rather than starving generation.
            logger.warning("grade_documents: parsed_output is None; failing open (all relevant)")
            return [
                DocRelevanceGrade(chunk_id=ctx.chunk_id, relevant=True, reason="parse failed")
                for ctx in contexts
            ]
        return parsed.grades

    def rewrite_query(
        self, original_query: str, current_query: str, contexts: list[RetrievalResult]
    ) -> RewrittenQuery:
        """Reformulate ``current_query`` given the docs it failed to satisfy."""
        client = self._ensure_client()
        prompt = _build_rewrite_prompt(original_query, current_query, contexts)
        logger.info("rewriting query: model=%s", self._settings.llm_model)
        response = client.messages.parse(  # type: ignore[attr-defined]
            model=self._settings.llm_model,
            max_tokens=_REWRITE_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": _EFFORT},
            output_format=RewrittenQuery,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = response.parsed_output
        if parsed is None:
            logger.warning("rewrite_query: parsed_output is None; kept the current query")
            return RewrittenQuery(query=current_query, rationale="parse failed; kept query")
        return parsed


class FakeCorrectiveLLM:
    """Deterministic, dependency-free, key-free ``CorrectiveLLM`` for tests and offline runs.

    Grades by lexical overlap between the query's tokens and each chunk's text (the same
    :func:`~rag.indexing.sparse.tokenize` the fake reranker uses): a chunk is ``relevant`` iff
    at least one query token appears in it. Rewrites by appending the most-frequent content
    token found across the contexts that is NOT already in the current query — a genuine,
    deterministic attempt to change what retrieval will find (never a no-op echo when there is
    any new token available), so a rewrite is never gratuitously wasted.
    """

    def grade_documents(
        self, query: str, contexts: list[RetrievalResult]
    ) -> list[DocRelevanceGrade]:
        """Grade each context ``relevant`` iff it shares >= 1 token with ``query``."""
        query_tokens = set(tokenize(query))
        grades: list[DocRelevanceGrade] = []
        for ctx in contexts:
            overlap = len(query_tokens & set(tokenize(ctx.text)))
            grades.append(
                DocRelevanceGrade(
                    chunk_id=ctx.chunk_id,
                    relevant=overlap >= 1,
                    reason=f"lexical overlap with query = {overlap}",
                )
            )
        return grades

    def rewrite_query(
        self, original_query: str, current_query: str, contexts: list[RetrievalResult]
    ) -> RewrittenQuery:
        """Append the most-frequent new content token found in ``contexts`` to the query."""
        current_tokens = set(tokenize(current_query))
        counts: dict[str, int] = {}
        for ctx in contexts:
            for token in tokenize(ctx.text):
                if token not in current_tokens:
                    counts[token] = counts.get(token, 0) + 1
        if not counts:
            # Nothing new to add across all retrieved contexts (pathological/degenerate case):
            # echo the query. The caller's rewrite_query node still counts this as a used
            # attempt so the retry budget remains a hard bound regardless.
            return RewrittenQuery(query=current_query, rationale="no new terms found; kept query")
        # Deterministic tie-break: highest count, then by reversed-character order (NOT plain
        # alphabetical — this compares each token's characters back-to-front) so the pick is
        # stable and reproducible regardless of dict/set iteration order.
        best_token = max(counts, key=lambda tok: (counts[tok], tuple(reversed(tok))))
        new_query = f"{current_query} {best_token}"
        return RewrittenQuery(
            query=new_query, rationale=f"expanded query with content token '{best_token}'"
        )


# =====================================================================================
# Nodes (each is a thin wrapper over an existing contract; DI'd via functools.partial)
# =====================================================================================


def _analyze_node(state: CorrectiveRAGState) -> dict[str, Any]:
    """LLM-free init: seed ``current_query`` and zero both retry counters."""
    return {
        "current_query": state["query"],
        "n_rewrites": 0,
        "n_regenerations": 0,
    }


def _retrieve_node(state: CorrectiveRAGState, *, retriever: HybridRetriever) -> dict[str, Any]:
    """``HybridRetriever.retrieve`` verbatim over the current (possibly rewritten) query."""
    current_query = state.get("current_query", state["query"])
    retrieved = retriever.retrieve(current_query, k=state.get("k"))
    return {"retrieved": retrieved}


def _grade_documents_node(
    state: CorrectiveRAGState, *, corrective: CorrectiveLLM, settings: Settings
) -> dict[str, Any]:
    """Grade retrieved docs; fail-open on ungraded/unknown ids (ADR-0007 decision 3)."""
    current_query = state.get("current_query", state["query"])
    retrieved = state.get("retrieved") or []
    grades = corrective.grade_documents(current_query, retrieved)
    grade_by_id = {grade.chunk_id: grade for grade in grades}

    relevant: list[RetrievalResult] = []
    for ctx in retrieved:
        grade = grade_by_id.get(ctx.chunk_id)  # grades for unknown ids are simply never looked up
        if grade is None or grade.relevant:  # fail-open: no grade -> keep it
            relevant.append(ctx)

    # `rewrite_budget_exhausted` is a separate boolean trace field (NOT part of
    # `terminated_reason` — see the `TerminatedReason` docstring for why): it records whether
    # THIS pass fell through to `generate` because the rewrite budget ran out rather than
    # because docs turned out sufficient. Set on every pass (both True and False) so
    # last-write-wins naturally reflects only the final grading pass.
    n_rewrites = state.get("n_rewrites", 0)
    exhausted = (
        len(relevant) < settings.agentic_min_relevant_docs
        and n_rewrites >= settings.agentic_max_query_rewrites
    )
    return {"grades": grades, "relevant": relevant, "rewrite_budget_exhausted": exhausted}


def _rewrite_query_node(state: CorrectiveRAGState, *, corrective: CorrectiveLLM) -> dict[str, Any]:
    """Reformulate the query and strictly increment ``n_rewrites`` before retrying retrieval."""
    query = state["query"]
    current_query = state.get("current_query", query)
    retrieved = state.get("retrieved") or []
    rewritten = corrective.rewrite_query(query, current_query, retrieved)
    if rewritten.query == current_query:
        logger.info("rewrite_query: reformulation did not change the query text")
    return {
        "current_query": rewritten.query,
        # Always increment, even on an unchanged query: the retry budget is a hard bound on
        # NODE EXECUTIONS, not on "meaningfully different" attempts — see FakeCorrectiveLLM's
        # docstring for why a genuine implementation should almost never hit this case anyway.
        "n_rewrites": state.get("n_rewrites", 0) + 1,
    }


def _generate_node(
    state: CorrectiveRAGState, *, llm: LLMClient, settings: Settings
) -> dict[str, Any]:
    """``generate_answer`` over the graded-relevant subset (falling back to all retrieved).

    Answers the ORIGINAL query (never ``current_query`` — the user asked the original
    question; only retrieval uses the rewritten form). Stashes the exact ``contexts`` used so
    ``verify`` scores the identical list.
    """
    query = state["query"]
    relevant = state.get("relevant") or []
    retrieved = state.get("retrieved") or []
    contexts = relevant if relevant else retrieved
    answer = generate_answer(query, contexts, llm=llm, settings=settings)
    return {"answer": answer, "contexts": contexts}


def _verify_node(state: CorrectiveRAGState) -> dict[str, Any]:
    """``verify_answer`` over the SAME contexts ``generate`` saw (ADR-0006 principle)."""
    answer = state["answer"]
    contexts = state.get("contexts") or []
    report = verify_answer(answer, contexts)
    return {"report": report}


def _accept_grounded_node(state: CorrectiveRAGState) -> dict[str, Any]:
    """Terminal marker: the answer is grounded enough to accept."""
    return {"terminated_reason": "grounded"}


def _accept_abstained_node(state: CorrectiveRAGState) -> dict[str, Any]:
    """Terminal marker: an honest 0-citation abstention — accepted, never regenerated."""
    return {"terminated_reason": "abstained"}


def _regenerate_node(state: CorrectiveRAGState) -> dict[str, Any]:
    """Strictly increment ``n_regenerations`` before re-entering ``generate``."""
    return {"n_regenerations": state.get("n_regenerations", 0) + 1}


def _exhaust_regenerate_node(state: CorrectiveRAGState) -> dict[str, Any]:
    """Terminal marker: regenerate budget exhausted; return the last (ungrounded) report."""
    return {"terminated_reason": "regenerate_budget_exhausted"}


# =====================================================================================
# Conditional routing (ADR-0007 hard rule: pure, deterministic, unit-testable in isolation)
# =====================================================================================

_RouteAfterGrade = Literal["generate", "rewrite_query"]
_RouteAfterVerify = Literal[
    "accept_grounded", "accept_abstained", "regenerate", "exhaust_regenerate"
]


def route_after_grade(state: CorrectiveRAGState, *, settings: Settings) -> _RouteAfterGrade:
    """Route to ``generate`` if docs are relevant-enough OR the rewrite budget is exhausted;
    otherwise ``rewrite_query``. A pure function of ``state``/``settings`` — no side effects."""
    relevant = state.get("relevant") or []
    if len(relevant) >= settings.agentic_min_relevant_docs:
        return "generate"
    if state.get("n_rewrites", 0) >= settings.agentic_max_query_rewrites:
        return "generate"
    return "rewrite_query"


def route_after_verify(state: CorrectiveRAGState, *, settings: Settings) -> _RouteAfterVerify:
    """Route on the verification report + regenerate budget. Pure, no side effects.

    Order matters (ADR-0007 decision 5): an honest 0-citation abstention is accepted BEFORE the
    attribution-rate check even runs, so an unanswerable query can never be mistaken for an
    ungrounded one and sent into the regenerate loop.

    NOTE (inherited limitation, ADR-0006): "abstained" here means exactly ``n_citations == 0`` —
    ``verify_answer`` scores CITATIONS, not the answer text, so a confident but uncited answer
    (one that asserts something without citing any chunk) also lands here as "accepted", not as
    a verified refusal. This module does not change that verification-layer behavior.
    """
    report = state["report"]
    if report.n_citations == 0:
        return "accept_abstained"
    if report.attribution_rate >= settings.agentic_min_attribution_rate:
        return "accept_grounded"
    if state.get("n_regenerations", 0) < settings.agentic_max_regenerations:
        return "regenerate"
    return "exhaust_regenerate"


# =====================================================================================
# Graph assembly + public entry point
# =====================================================================================


def _derive_recursion_limit(settings: Settings) -> int:
    """Derive the belt-and-suspenders ``recursion_limit`` from the two retry budgets.

    ``(R + 1) * 3 + (G + 1) * 3 + 5`` per ADR-0007 decision 5 — NOT a separate config knob, so
    it can never drift independently of the budgets it backstops. Each retrieval pass costs up
    to 3 node steps (retrieve, grade_documents, +rewrite_query); each FULL regenerate cycle is
    also 3 node executions (generate, verify, +regenerate bookkeeping) before looping back into
    generate, so it gets the same per-cycle multiplier as the retrieval loop (an earlier ``* 2``
    here under-counted the regenerate cycle by one node and could cross below the true worst
    case ``3R + 3G + 6`` at G >= 4, firing the recursion backstop on a legitimate run instead of
    the intended ``regenerate_budget_exhausted`` outcome); +5 covers analyze/START and the
    terminal marker nodes, giving this formula comfortable margin over the true worst case.
    """
    r = settings.agentic_max_query_rewrites
    g = settings.agentic_max_regenerations
    return (r + 1) * 3 + (g + 1) * 3 + 5


def build_corrective_rag(
    *,
    retriever: HybridRetriever,
    llm: LLMClient,
    corrective: CorrectiveLLM,
    settings: Settings,
) -> Any:
    """Build and compile the corrective-RAG ``StateGraph``.

    ``langgraph`` is imported here (not at module scope) so ``import rag.agentic`` stays cheap
    and works even without the package installed. Every node/edge below composes an EXISTING
    contract (retriever/llm/corrective) or is pure orchestration bookkeeping (grade routing,
    counters, terminal markers) — no fusion, rerank, or attribution logic lives here.
    """
    from langgraph.graph import END, START, StateGraph

    graph: StateGraph[CorrectiveRAGState] = StateGraph(CorrectiveRAGState)

    graph.add_node("analyze", _analyze_node)
    graph.add_node("retrieve", functools.partial(_retrieve_node, retriever=retriever))
    graph.add_node(
        "grade_documents",
        functools.partial(_grade_documents_node, corrective=corrective, settings=settings),
    )
    graph.add_node("rewrite_query", functools.partial(_rewrite_query_node, corrective=corrective))
    graph.add_node("generate", functools.partial(_generate_node, llm=llm, settings=settings))
    graph.add_node("verify", _verify_node)
    graph.add_node("accept_grounded", _accept_grounded_node)
    graph.add_node("accept_abstained", _accept_abstained_node)
    graph.add_node("regenerate", _regenerate_node)
    graph.add_node("exhaust_regenerate", _exhaust_regenerate_node)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        functools.partial(route_after_grade, settings=settings),
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        functools.partial(route_after_verify, settings=settings),
        {
            "accept_grounded": "accept_grounded",
            "accept_abstained": "accept_abstained",
            "regenerate": "regenerate",
            "exhaust_regenerate": "exhaust_regenerate",
        },
    )
    graph.add_edge("accept_grounded", END)
    graph.add_edge("accept_abstained", END)
    graph.add_edge("exhaust_regenerate", END)
    graph.add_edge("regenerate", "generate")

    return graph.compile()


def _fallback_answer() -> Answer:
    """A degenerate but valid ``Answer`` for the (structurally unreachable in normal budgets)
    case where the run is interrupted before ``generate`` ever executes."""
    return Answer(text="The corrective RAG run terminated before an answer was generated.")


def _fallback_report() -> VerificationReport:
    """A degenerate but valid ``VerificationReport`` matching :func:`_fallback_answer`."""
    return VerificationReport(attribution_rate=0.0, checks=[], unsupported=[], n_citations=0)


def _state_to_result(
    state: CorrectiveRAGState,
    *,
    request: CorrectiveRAGRequest,
    override_reason: TerminatedReason | None,
) -> CorrectiveRAGResult:
    """Map the final (or last-best-effort) graph state to the boundary ``CorrectiveRAGResult``."""
    answer = state.get("answer") or _fallback_answer()
    report = state.get("report") or _fallback_report()
    contexts = state.get("contexts") or state.get("retrieved") or []
    terminated_reason: TerminatedReason = (
        override_reason or state.get("terminated_reason") or "recursion_limit"
    )
    return CorrectiveRAGResult(
        answer=answer,
        report=report,
        contexts=contexts,
        original_query=request.query,
        final_query=state.get("current_query", request.query),
        n_rewrites=state.get("n_rewrites", 0),
        n_regenerations=state.get("n_regenerations", 0),
        rewrite_budget_exhausted=bool(state.get("rewrite_budget_exhausted", False)),
        terminated_reason=terminated_reason,
    )


def run_corrective_rag(
    request: CorrectiveRAGRequest,
    *,
    retriever: HybridRetriever,
    llm: LLMClient,
    corrective: CorrectiveLLM,
    settings: Settings,
    graph: Any | None = None,
) -> CorrectiveRAGResult:
    """Run the corrective-RAG graph for ``request`` and always return a result.

    Builds (or reuses a prebuilt) compiled graph and invokes it with the derived
    ``recursion_limit`` backstop (:func:`_derive_recursion_limit`). The graph's own two bounded
    counters are the primary termination guarantee; the recursion limit is belt-and-suspenders.
    On ``GraphRecursionError`` (which should not occur given the counters, but is caught rather
    than trusted blindly), logs at ERROR and returns the last observed state as a best-effort
    result with ``terminated_reason = "recursion_limit"``. Never raises for a RAG-quality
    failure; genuine collaborator exceptions (SDK/network/retriever errors) propagate.
    """
    from langgraph.errors import GraphRecursionError

    compiled = (
        graph
        if graph is not None
        else build_corrective_rag(
            retriever=retriever, llm=llm, corrective=corrective, settings=settings
        )
    )
    recursion_limit = _derive_recursion_limit(settings)

    initial_state: CorrectiveRAGState = {
        "query": request.query,
        "current_query": request.query,
        "k": request.k,
        "n_rewrites": 0,
        "n_regenerations": 0,
    }

    last_state: CorrectiveRAGState = dict(initial_state)  # type: ignore[assignment]
    override_reason: TerminatedReason | None = None
    try:
        for snapshot in compiled.stream(
            initial_state, config={"recursion_limit": recursion_limit}, stream_mode="values"
        ):
            last_state = snapshot
    except GraphRecursionError:
        logger.error(
            "corrective RAG hit the recursion_limit backstop (%d); returning best-effort result",
            recursion_limit,
        )
        override_reason = "recursion_limit"

    return _state_to_result(last_state, request=request, override_reason=override_reason)
