"""Tests for the self-corrective RAG graph (``rag.agentic.corrective_rag``), ADR-0007.

Fully offline: routing functions are unit-tested directly on synthetic state (no LLM, no
graph). Full-graph tests use a deterministic, dependency-free STUB retriever (a fixed doc list,
no qdrant/bm25/langgraph-model dependency beyond ``langgraph`` itself) plus
:class:`~rag.agentic.corrective_rag.FakeCorrectiveLLM` / small scripted subclasses and
:class:`~rag.generation.llm.FakeLLMClient` / a scripted generation client — mirroring how
``test_hybrid.py`` / ``test_generation.py`` mock their collaborators, per the "mock the LLM /
retrieval calls so graph-logic tests are fast and deterministic" discipline. One smaller
end-to-end smoke test at the bottom wires the REAL :class:`HybridRetriever` (offline backends)
to prove the composition also works over genuine retrieval.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from rag.agentic.corrective_rag import (  # noqa: E402
    AnthropicCorrectiveLLM,
    CorrectiveLLM,
    CorrectiveRAGRequest,
    DocRelevanceGrade,
    FakeCorrectiveLLM,
    RewrittenQuery,
    _derive_recursion_limit,
    build_corrective_rag,
    route_after_grade,
    route_after_verify,
    run_corrective_rag,
)
from rag.config import Settings  # noqa: E402
from rag.generation.llm import FakeLLMClient  # noqa: E402
from rag.generation.models import Answer, Citation  # noqa: E402
from rag.retrieval.models import RetrievalResult  # noqa: E402
from rag.verification.models import VerificationReport  # noqa: E402

# --- Shared fixtures / fakes ----------------------------------------------------------------

DOC_TEXT = "elephants roam the savanna grasslands searching for water at dusk"


def _ctx(chunk_id: str = "doc1", text: str = DOC_TEXT, rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=1.0,
        rank=rank,
        text=text,
        rel_path=f"{chunk_id}.md",
        heading_path=[],
        metadata={},
        sources=["dense"],
    )


class _StubRetriever:
    """Deterministic, dependency-free retriever stub: always returns the same fixed docs.

    Satisfies the informal ``retrieve(query, *, k=None) -> list[RetrievalResult]`` interface
    without touching qdrant/bm25, so graph-logic tests stay fast and offline. Records every
    query it was called with (for tests that want to assert the rewrite actually re-ran
    retrieval).
    """

    def __init__(self, docs: list[RetrievalResult] | None = None) -> None:
        self._docs = docs if docs is not None else [_ctx()]
        self.queries: list[str] = []

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievalResult]:
        self.queries.append(query)
        return list(self._docs)


class _EmptyRetriever:
    """Always returns no results (for the abstention test: no docs -> 0 citations)."""

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievalResult]:
        return []


class _AlwaysIrrelevantCorrectiveLLM(FakeCorrectiveLLM):
    """Scripted grader: every doc is graded irrelevant, no matter the query."""

    def grade_documents(
        self, query: str, contexts: list[RetrievalResult]
    ) -> list[DocRelevanceGrade]:
        return [
            DocRelevanceGrade(chunk_id=ctx.chunk_id, relevant=False, reason="scripted: irrelevant")
            for ctx in contexts
        ]


class _ScriptedLLMClient:
    """Scripted generation ``LLMClient``: cites the top context with a per-call quote mode.

    ``modes`` is a sequence of ``"fabricate"`` / ``"ground"``; the last entry repeats for any
    call beyond the scripted length. This is the "regenerate needs a scripted fake" mechanism
    ADR-0007 calls for (Option 4b): a deterministic fake can't change its own answer on a bare
    retry, so the test drives the retry explicitly.
    """

    _FABRICATED_QUOTE = "zzqxv totally fabricated unsupported claim wzzqxv"

    def __init__(self, modes: list[str]) -> None:
        self._modes = modes
        self.calls = 0

    def generate_answer(self, query: str, contexts: list[RetrievalResult]) -> Answer:
        mode = self._modes[min(self.calls, len(self._modes) - 1)]
        self.calls += 1
        if not contexts:
            return Answer(text="no contexts", citations=[])
        top = contexts[0]
        quote = self._FABRICATED_QUOTE if mode == "fabricate" else top.text[:60]
        return Answer(
            text=f"[1] ({top.chunk_id}): {quote}",
            citations=[
                Citation(chunk_id=top.chunk_id, rel_path=top.rel_path, supporting_quote=quote)
            ],
        )


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "agentic_max_query_rewrites": 2,
        "agentic_max_regenerations": 1,
        "agentic_min_relevant_docs": 1,
        "agentic_min_attribution_rate": 1.0,
    }
    base.update(overrides)
    return Settings(**base)


# --- route_after_grade: pure, unit-testable in isolation -------------------------------------


def test_route_after_grade_generates_when_enough_relevant() -> None:
    settings = _settings()
    state = {"relevant": [_ctx()], "n_rewrites": 0}
    assert route_after_grade(state, settings=settings) == "generate"


def test_route_after_grade_rewrites_when_insufficient_and_budget_remains() -> None:
    settings = _settings()
    state = {"relevant": [], "n_rewrites": 0}
    assert route_after_grade(state, settings=settings) == "rewrite_query"


def test_route_after_grade_falls_through_at_budget_exhaustion() -> None:
    settings = _settings(agentic_max_query_rewrites=2)
    state = {"relevant": [], "n_rewrites": 2}
    assert route_after_grade(state, settings=settings) == "generate"


def test_route_after_grade_never_exceeds_budget_even_over_the_limit() -> None:
    """A pathological n_rewrites value above the budget must still fall through, not loop."""
    settings = _settings(agentic_max_query_rewrites=2)
    state = {"relevant": [], "n_rewrites": 999}
    assert route_after_grade(state, settings=settings) == "generate"


# --- route_after_verify: pure, unit-testable in isolation ------------------------------------


def test_route_after_verify_abstains_on_zero_citations_even_with_no_regen_budget() -> None:
    """The 0-citation guard fires BEFORE the regen-budget check (anti-thrash guard)."""
    settings = _settings(agentic_max_regenerations=0)
    report = VerificationReport(attribution_rate=0.0, checks=[], unsupported=[], n_citations=0)
    state = {"report": report, "n_regenerations": 0}
    assert route_after_verify(state, settings=settings) == "accept_abstained"


def test_route_after_verify_accepts_grounded() -> None:
    settings = _settings()
    report = VerificationReport(attribution_rate=1.0, checks=[], unsupported=[], n_citations=1)
    state = {"report": report, "n_regenerations": 0}
    assert route_after_verify(state, settings=settings) == "accept_grounded"


def test_route_after_verify_regenerates_when_budget_remains() -> None:
    settings = _settings(agentic_max_regenerations=1)
    report = VerificationReport(attribution_rate=0.0, checks=[], unsupported=["c1"], n_citations=1)
    state = {"report": report, "n_regenerations": 0}
    assert route_after_verify(state, settings=settings) == "regenerate"


def test_route_after_verify_exhausts_when_budget_spent() -> None:
    settings = _settings(agentic_max_regenerations=1)
    report = VerificationReport(attribution_rate=0.0, checks=[], unsupported=["c1"], n_citations=1)
    state = {"report": report, "n_regenerations": 1}
    assert route_after_verify(state, settings=settings) == "exhaust_regenerate"


# --- _derive_recursion_limit ------------------------------------------------------------------


def test_derive_recursion_limit_matches_formula() -> None:
    settings = _settings(agentic_max_query_rewrites=2, agentic_max_regenerations=1)
    # (R+1)*3 + (G+1)*3 + 5 = 3*3 + 2*3 + 5 = 9 + 6 + 5 = 20
    assert _derive_recursion_limit(settings) == 20


def test_derive_recursion_limit_has_margin_over_true_worst_case() -> None:
    """The derived limit must never cross below the true worst case `3R + 3G + 6`.

    Regression guard for the MAJOR finding: an earlier formula used `(G + 1) * 2`, under-
    counting a full regenerate cycle (generate -> verify -> regenerate is 3 node executions,
    not 2), which crossed below the true worst case at G >= 4 and fired the recursion backstop
    on a legitimate run instead of `regenerate_budget_exhausted`.
    """
    for r in range(4):
        for g in range(6):
            settings = _settings(agentic_max_query_rewrites=r, agentic_max_regenerations=g)
            true_worst_case = 3 * r + 3 * g + 6
            assert _derive_recursion_limit(settings) >= true_worst_case


# --- Full-graph tests: happy path -------------------------------------------------------------


def test_happy_path_no_retries_grounded() -> None:
    settings = _settings()
    retriever = _StubRetriever([_ctx()])
    request = CorrectiveRAGRequest(query="Where do elephants roam?")

    result = run_corrective_rag(
        request,
        retriever=retriever,
        llm=FakeLLMClient(),
        corrective=FakeCorrectiveLLM(),
        settings=settings,
    )

    assert result.terminated_reason == "grounded"
    assert result.n_rewrites == 0
    assert result.n_regenerations == 0
    assert result.rewrite_budget_exhausted is False
    assert result.report.attribution_rate == 1.0
    assert result.original_query == "Where do elephants roam?"
    assert result.final_query == "Where do elephants roam?"
    assert len(retriever.queries) == 1


# --- Full-graph tests: rewrite path ------------------------------------------------------------


def test_rewrite_path_recovers_then_grounds() -> None:
    settings = _settings(agentic_max_query_rewrites=2)
    retriever = _StubRetriever([_ctx()])
    # Zero lexical overlap with DOC_TEXT -> first grade marks it irrelevant.
    request = CorrectiveRAGRequest(query="banana smoothie recipe")

    result = run_corrective_rag(
        request,
        retriever=retriever,
        llm=FakeLLMClient(),
        corrective=FakeCorrectiveLLM(),
        settings=settings,
    )

    assert result.n_rewrites == 1  # one rewrite recovered relevance; no need to burn the 2nd
    assert result.terminated_reason == "grounded"
    assert result.rewrite_budget_exhausted is False  # recovered before the budget ran out
    assert result.final_query != result.original_query
    # Retrieval re-ran with the rewritten query (2 calls: original + 1 retry).
    assert len(retriever.queries) == 2
    assert retriever.queries[0] == "banana smoothie recipe"
    assert retriever.queries[1] == result.final_query


# --- Full-graph tests: rewrite-budget exhaustion ------------------------------------------------


def test_rewrite_budget_exhaustion_terminates_with_best_effort_result() -> None:
    """Grader always irrelevant: exactly `agentic_max_query_rewrites` rewrites, no hang.

    `terminated_reason` is NOT asserted to be "rewrite_budget_exhausted" here: that value is
    not part of the reachable `TerminatedReason` set (`verify`'s own grounded/abstained/
    regenerate-exhausted determination always has the final say once it runs — see
    `route_after_verify`'s docstring and `CorrectiveRAGResult.rewrite_budget_exhausted`'s
    description for why). Instead, the honest, separately-tracked `rewrite_budget_exhausted`
    boolean is what proves the rewrite loop actually ran out of budget (rather than found
    docs sufficient) on its way to `generate`.
    """
    settings = _settings(agentic_max_query_rewrites=2)
    retriever = _StubRetriever([_ctx()])
    request = CorrectiveRAGRequest(query="banana smoothie recipe")

    result = run_corrective_rag(
        request,
        retriever=retriever,
        llm=FakeLLMClient(),
        corrective=_AlwaysIrrelevantCorrectiveLLM(),
        settings=settings,
    )

    assert result.n_rewrites == 2  # exactly the budget, never more
    assert result.rewrite_budget_exhausted is True
    # Fallback: generate ran over the full retrieved set (relevant was always empty).
    assert [c.chunk_id for c in result.contexts] == ["doc1"]
    assert result.terminated_reason in {"grounded", "abstained"}  # never raised, never hung
    # Retrieval ran exactly budget+1 times (initial + 2 rewrites), never more.
    assert len(retriever.queries) == 3


# --- Full-graph tests: regenerate path ----------------------------------------------------------


def test_regenerate_path_recovers_then_grounds() -> None:
    settings = _settings(agentic_max_regenerations=1)
    retriever = _StubRetriever([_ctx()])
    scripted_llm = _ScriptedLLMClient(modes=["fabricate", "ground"])
    request = CorrectiveRAGRequest(query="Where do elephants roam?")  # grades relevant on pass 1

    result = run_corrective_rag(
        request,
        retriever=retriever,
        llm=scripted_llm,
        corrective=FakeCorrectiveLLM(),
        settings=settings,
    )

    assert result.n_regenerations == 1
    assert result.terminated_reason == "grounded"
    assert result.report.attribution_rate == 1.0
    assert scripted_llm.calls == 2


# --- Full-graph tests: regenerate-budget exhaustion ---------------------------------------------


def test_regenerate_budget_exhaustion_returns_ungrounded_report_intact() -> None:
    settings = _settings(agentic_max_regenerations=1)
    retriever = _StubRetriever([_ctx()])
    scripted_llm = _ScriptedLLMClient(modes=["fabricate", "fabricate"])
    request = CorrectiveRAGRequest(query="Where do elephants roam?")

    result = run_corrective_rag(
        request,
        retriever=retriever,
        llm=scripted_llm,
        corrective=FakeCorrectiveLLM(),
        settings=settings,
    )

    assert result.n_regenerations == 1  # exactly the budget, never more
    assert result.terminated_reason == "regenerate_budget_exhausted"
    assert result.report.attribution_rate == 0.0
    assert result.report.unsupported == ["doc1"]
    assert scripted_llm.calls == 2  # generate ran exactly budget+1 times


# --- Full-graph tests: abstention is accepted, never regenerated --------------------------------


def test_abstention_is_accepted_not_regenerated() -> None:
    # max_query_rewrites=0 -> falls straight through to generate on the first (empty) pass.
    settings = _settings(agentic_max_query_rewrites=0, agentic_max_regenerations=1)
    request = CorrectiveRAGRequest(query="Where do elephants roam?")

    result = run_corrective_rag(
        request,
        retriever=_EmptyRetriever(),
        llm=FakeLLMClient(),
        corrective=FakeCorrectiveLLM(),
        settings=settings,
    )

    assert result.report.n_citations == 0
    assert result.terminated_reason == "abstained"
    assert result.n_regenerations == 0  # the abstention trap: never spend the regen budget
    assert result.contexts == []


# --- Determinism ---------------------------------------------------------------------------


def test_determinism_same_inputs_same_trace() -> None:
    settings = _settings(agentic_max_query_rewrites=2)
    request = CorrectiveRAGRequest(query="banana smoothie recipe")

    result_a = run_corrective_rag(
        request,
        retriever=_StubRetriever([_ctx()]),
        llm=FakeLLMClient(),
        corrective=FakeCorrectiveLLM(),
        settings=settings,
    )
    result_b = run_corrective_rag(
        request,
        retriever=_StubRetriever([_ctx()]),
        llm=FakeLLMClient(),
        corrective=FakeCorrectiveLLM(),
        settings=settings,
    )

    assert result_a.model_dump_json() == result_b.model_dump_json()


# --- Termination under always-worst-case fakes (both loops pathological) ------------------------


def test_terminates_under_always_worst_case_fakes() -> None:
    """Grader never relevant AND generator never grounds: both loops must still halt."""
    settings = _settings(agentic_max_query_rewrites=2, agentic_max_regenerations=1)
    retriever = _StubRetriever([_ctx()])
    scripted_llm = _ScriptedLLMClient(modes=["fabricate"])  # every call fabricates
    request = CorrectiveRAGRequest(query="banana smoothie recipe")

    result = run_corrective_rag(
        request,
        retriever=retriever,
        llm=scripted_llm,
        corrective=_AlwaysIrrelevantCorrectiveLLM(),
        settings=settings,
    )

    # Both budgets fully consumed, never exceeded.
    assert result.n_rewrites == 2
    assert result.rewrite_budget_exhausted is True
    assert result.n_regenerations == 1
    assert result.terminated_reason == "regenerate_budget_exhausted"
    assert result.report.attribution_rate == 0.0
    # Bounded node executions: retrieve <= R+1 = 3, generate <= G+1 = 2.
    assert len(retriever.queries) == 3
    assert scripted_llm.calls == 2


def test_terminates_within_derived_recursion_limit_structurally() -> None:
    """Rebuilding the graph fresh and invoking it directly never needs the recursion backstop."""
    settings = _settings(agentic_max_query_rewrites=2, agentic_max_regenerations=1)
    graph = build_corrective_rag(
        retriever=_StubRetriever([_ctx()]),
        llm=_ScriptedLLMClient(modes=["fabricate"]),
        corrective=_AlwaysIrrelevantCorrectiveLLM(),
        settings=settings,
    )
    recursion_limit = _derive_recursion_limit(settings)
    # invoke (not stream) exercises the same recursion_limit path run_corrective_rag uses;
    # it must complete without raising GraphRecursionError.
    final_state = graph.invoke(
        {
            "query": "banana smoothie recipe",
            "current_query": "banana smoothie recipe",
            "k": None,
            "n_rewrites": 0,
            "n_regenerations": 0,
        },
        config={"recursion_limit": recursion_limit},
    )
    assert final_state["terminated_reason"] == "regenerate_budget_exhausted"


def test_recursion_limit_has_margin_at_high_regenerate_budget() -> None:
    """Regression guard for the MAJOR finding: G >= 4 must NOT fire the recursion backstop.

    An earlier `_derive_recursion_limit` formula counted only 2 node steps per regenerate
    cycle (generate, verify) when a full cycle is actually 3 (generate, verify, regenerate
    bookkeeping). That under-count crossed below the true worst case `3R + 3G + 6` at G >= 4,
    so a legitimate always-fabricating run hit `GraphRecursionError` and reported
    `terminated_reason == "recursion_limit"` instead of the correct
    `"regenerate_budget_exhausted"`. This must not happen with the corrected formula.
    """
    settings = _settings(agentic_max_query_rewrites=0, agentic_max_regenerations=4)
    retriever = _StubRetriever([_ctx()])
    scripted_llm = _ScriptedLLMClient(modes=["fabricate"])  # every call fabricates
    request = CorrectiveRAGRequest(query="Where do elephants roam?")  # grades relevant immediately

    result = run_corrective_rag(
        request,
        retriever=retriever,
        llm=scripted_llm,
        corrective=FakeCorrectiveLLM(),
        settings=settings,
    )

    assert result.n_regenerations == 4
    assert result.terminated_reason == "regenerate_budget_exhausted"
    assert scripted_llm.calls == 5  # G+1 generations, never truncated by the recursion backstop


# --- Protocol conformance --------------------------------------------------------------------


def test_fake_corrective_llm_satisfies_protocol() -> None:
    assert isinstance(FakeCorrectiveLLM(), CorrectiveLLM)


def test_anthropic_corrective_llm_satisfies_protocol_and_is_lazy() -> None:
    client = AnthropicCorrectiveLLM(Settings())
    assert isinstance(client, CorrectiveLLM)
    assert client._client is None  # SDK client not built at construction time


def test_fake_corrective_llm_rewrite_is_deterministic_and_changes_query() -> None:
    fake = FakeCorrectiveLLM()
    contexts = [_ctx()]
    rw1 = fake.rewrite_query("q", "banana smoothie recipe", contexts)
    rw2 = fake.rewrite_query("q", "banana smoothie recipe", contexts)
    assert rw1 == rw2
    assert isinstance(rw1, RewrittenQuery)
    assert rw1.query != "banana smoothie recipe"


# --- Smoke test over REAL retrieval components (separate from the fast graph-logic tests) ------


def test_smoke_real_hybrid_retriever(tmp_path: Path) -> None:
    """A single, smaller end-to-end run over the REAL HybridRetriever (offline backends)."""
    pytest.importorskip("qdrant_client")
    pytest.importorskip("rank_bm25")

    from rag.indexing.build import build_index
    from rag.indexing.embeddings import HashingEmbedder
    from rag.indexing.sparse import BM25Index
    from rag.indexing.vector_store import QdrantVectorStore
    from rag.retrieval.hybrid import HybridRetriever
    from rag.retrieval.rerank import LexicalOverlapReranker

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "elephants.md").write_text(f"# Elephants\n\n{DOC_TEXT}\n", encoding="utf-8")

    settings = _settings(
        corpus_dir=corpus,
        sample_dir=corpus,
        storage_dir=tmp_path / "storage",
        chunk_strategy="recursive",
        chunk_size=1024,
        chunk_overlap=0,
        qdrant_collection="corrective_rag_smoke",
    )
    store = QdrantVectorStore.in_memory(settings.qdrant_collection)
    build_index(settings, embedder=HashingEmbedder(), store=store)
    bm25 = BM25Index.load(settings.storage_dir)
    retriever = HybridRetriever(
        embedder=HashingEmbedder(),
        store=store,
        bm25=bm25,
        reranker=LexicalOverlapReranker(),
        settings=settings,
    )

    request = CorrectiveRAGRequest(query="Where do elephants roam?")
    result = run_corrective_rag(
        request,
        retriever=retriever,
        llm=FakeLLMClient(),
        corrective=FakeCorrectiveLLM(),
        settings=settings,
    )

    assert result.terminated_reason == "grounded"
    assert result.report.attribution_rate == 1.0
    assert result.contexts  # self-contained, hydrated real RetrievalResults
