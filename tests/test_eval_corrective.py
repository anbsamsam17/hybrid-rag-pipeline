"""Tests for the corrective-vs-baseline eval harness (ADR-0008), run fully OFFLINE.

Mirrors ``tests/test_eval_attribution.py``: the full-run tests exercise the genuine hermetic
build -> baseline single pass + corrective graph -> aggregate path with the proven offline
backends (:class:`HashingEmbedder` + in-memory Qdrant + :class:`LexicalOverlapReranker`) and the
DETERMINISTIC fakes (:class:`FakeLLMClient`, :class:`FakeCorrectiveLLM`,
:class:`FakeAnswerCorrectnessJudge`) — no torch, no network, no API key, no real LLM value ever
asserted. The judge-free PRIMARY (activation, cost, recall, lexical-F1) is byte-stable, so the
whole artifact is asserted byte-identical across runs.

They assert STRUCTURE, INVARIANTS, DETERMINISM, and the load-bearing design guards:
* the baseline is a TRUE single pass (it does NOT grade/filter — corrective may),
* the judge is BLIND (sees only ``(query, reference, candidate)``, never the arm),
* ``publishable`` flips ``False`` on ANY fake (including the judge),
* the trace-derived activation + ``extra_llm_calls`` accounting is exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# The corrective build path needs the offline index backends AND langgraph (the corrective arm
# compiles a StateGraph); skip the whole module cleanly (not a silent pass) if any is unavailable.
pytest.importorskip("qdrant_client")
pytest.importorskip("rank_bm25")
pytest.importorskip("langgraph")

from rag.agentic.corrective_rag import (  # noqa: E402
    CorrectiveRAGRequest,
    CorrectiveRAGResult,
    DocRelevanceGrade,
    FakeCorrectiveLLM,
    run_corrective_rag,
)
from rag.config import PROJECT_ROOT, Settings  # noqa: E402
from rag.eval.corrective import (  # noqa: E402
    CONFIG,
    CORRECTIVE_RESULTS_FILENAME,
    K_VALUES,
    _is_publishable,
    _trace_fields,
    answer_once,
    compute_extra_llm_calls,
    render_corrective_report,
    run_corrective_eval,
)
from rag.eval.golden import load_golden  # noqa: E402
from rag.eval.judge import (  # noqa: E402
    AnswerCorrectnessJudge,
    AnthropicAnswerCorrectnessJudge,
    CorrectnessVerdict,
    FakeAnswerCorrectnessJudge,
    default_judge_model,
    lexical_f1,
)
from rag.eval.models import CorrectiveEvalReport  # noqa: E402
from rag.generation.llm import FakeLLMClient  # noqa: E402
from rag.generation.models import Answer  # noqa: E402
from rag.indexing.build import build_index  # noqa: E402
from rag.indexing.embeddings import HashingEmbedder  # noqa: E402
from rag.indexing.sparse import BM25Index  # noqa: E402
from rag.indexing.vector_store import QdrantVectorStore  # noqa: E402
from rag.retrieval.models import RetrievalResult  # noqa: E402
from rag.retrieval.rerank import LexicalOverlapReranker  # noqa: E402
from rag.verification.models import VerificationReport  # noqa: E402

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "golden.jsonl"

# Derive n from the committed golden set — NEVER hard-code it.
GOLDEN = load_golden(GOLDEN_PATH)
N_GOLDEN = len(GOLDEN)


# --- Small deterministic collaborators ------------------------------------------------------


def _ctx(chunk_id: str, text: str, rank: int = 1) -> RetrievalResult:
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


class _TwoDocStubRetriever:
    """Returns two fixed docs regardless of query/k: doc1 overlaps the query, doc2 is disjoint.

    Duck-types ``retrieve(query, *, k=None)`` for both ``answer_once`` and ``run_corrective_rag``,
    so a graph-logic comparison stays offline. ``FakeCorrectiveLLM`` will grade doc1 relevant and
    doc2 irrelevant, which lets the corrective arm FILTER to a strict subset while the single-pass
    baseline keeps BOTH — the crux of the "baseline is a true single pass" guard.
    """

    def __init__(self) -> None:
        self._docs = [
            _ctx("doc1", "elephants roam the savanna grasslands", rank=1),
            _ctx("doc2", "banana smoothie blender recipe", rank=2),
        ]

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievalResult]:
        return list(self._docs)


class _RecordingJudge:
    """A judge that RECORDS every ``(query, reference, candidate)`` it is called with.

    Its whole purpose is the blindness proof: its signature exposes only the three blind strings,
    so a test can assert the harness never smuggles an arm label / corpus / relevant ids into the
    judge, and that both arms of a query are judged against the SAME reference.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def judge(self, query: str, reference_answer: str, candidate_answer: str) -> CorrectnessVerdict:
        self.calls.append((query, reference_answer, candidate_answer))
        return CorrectnessVerdict(correct=True, score=1.0, reason="recording stub")


def _settings(storage: Path, **overrides: object) -> Settings:
    """Eval settings pinned to the committed sample corpus + golden set (config that minted it)."""
    base: dict[str, object] = {
        "corpus_dir": PROJECT_ROOT / "data" / "corpus",  # unused by eval (it uses sample_dir)
        "sample_dir": SAMPLE_DIR,
        "golden_path": GOLDEN_PATH,
        "storage_dir": storage,
        "chunk_strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "qdrant_collection": "eval_corrective_test",
    }
    base.update(overrides)
    return Settings(**base)


def _run(storage: Path, *, judge: object | None = None) -> CorrectiveEvalReport:
    """Run the corrective eval fully offline (fresh in-memory store each call)."""
    return run_corrective_eval(
        _settings(storage),
        llm=FakeLLMClient(),
        corrective=FakeCorrectiveLLM(),
        judge=judge if judge is not None else FakeAnswerCorrectnessJudge(),  # type: ignore[arg-type]
        embedder=HashingEmbedder(),
        store=QdrantVectorStore.in_memory("eval_corrective_test"),
        reranker=LexicalOverlapReranker(),
    )


# --- PRIMARY: activation-rate math on synthetic traces (pure, no LLM, no graph) ----------------


def _synthetic_result(
    *, n_rewrites: int, n_regenerations: int, original: str, final: str
) -> CorrectiveRAGResult:
    """A minimal, valid ``CorrectiveRAGResult`` carrying just the trace fields under test."""
    return CorrectiveRAGResult(
        answer=Answer(text="answer", citations=[]),
        report=VerificationReport(attribution_rate=1.0, checks=[], unsupported=[], n_citations=0),
        contexts=[],
        original_query=original,
        final_query=final,
        n_rewrites=n_rewrites,
        n_regenerations=n_regenerations,
        rewrite_budget_exhausted=False,
        terminated_reason="grounded",
    )


def test_trace_fields_zero_activation() -> None:
    tf = _trace_fields(_synthetic_result(n_rewrites=0, n_regenerations=0, original="q", final="q"))
    assert tf.activated is False
    assert tf.final_query_changed is False
    assert tf.extra_llm_calls == 1  # the +1 grade call, paid even for nothing


def test_trace_fields_rewrite_activation_is_effective_when_query_changed() -> None:
    tf = _trace_fields(
        _synthetic_result(n_rewrites=1, n_regenerations=0, original="q", final="q expanded")
    )
    assert tf.activated is True
    assert tf.final_query_changed is True
    assert tf.extra_llm_calls == 3  # 2*1 + 0 + 1


def test_trace_fields_regenerate_activation_without_query_change() -> None:
    tf = _trace_fields(_synthetic_result(n_rewrites=0, n_regenerations=1, original="q", final="q"))
    assert tf.activated is True  # control-flow activation
    assert tf.final_query_changed is False  # a regeneration never changes the query text
    assert tf.extra_llm_calls == 2  # 2*0 + 1 + 1


def test_trace_fields_control_flow_activation_can_be_a_false_positive() -> None:
    """n_rewrites can increment on an UNCHANGED query -> activated True but effective False.

    This is the exact overclaim the separate ``final_query_changed`` signal guards against.
    """
    tf = _trace_fields(_synthetic_result(n_rewrites=1, n_regenerations=0, original="q", final="q"))
    assert tf.activated is True
    assert tf.final_query_changed is False


@pytest.mark.parametrize(
    "r,g,expected",
    [(0, 0, 1), (1, 0, 3), (0, 1, 2), (2, 1, 6), (3, 2, 9)],
)
def test_extra_llm_calls_formula(r: int, g: int, expected: int) -> None:
    assert compute_extra_llm_calls(r, g) == expected


# --- baseline is a TRUE single pass (does NOT grade/filter; corrective may) ---------------------


def test_answer_once_returns_all_retrieved_contexts_no_grading() -> None:
    retriever = _TwoDocStubRetriever()
    settings = _settings_bare()
    _, _, contexts = answer_once(
        "elephants roam", k=5, retriever=retriever, llm=FakeLLMClient(), settings=settings
    )
    # The single-pass baseline keeps BOTH retrieved docs (no grade, no filter), including the
    # disjoint doc2 a grader would drop.
    assert [c.chunk_id for c in contexts] == ["doc1", "doc2"]


def test_baseline_keeps_all_while_corrective_filters_same_retriever() -> None:
    """Same retriever + same query: baseline keeps both docs; corrective grade-filters to doc1."""
    settings = _settings_bare()
    retriever = _TwoDocStubRetriever()

    _, _, baseline_contexts = answer_once(
        "elephants roam", k=5, retriever=retriever, llm=FakeLLMClient(), settings=settings
    )
    corrective = run_corrective_rag(
        CorrectiveRAGRequest(query="elephants roam", k=5),
        retriever=retriever,
        llm=FakeLLMClient(),
        corrective=FakeCorrectiveLLM(),  # grades doc1 relevant (overlap), doc2 irrelevant
        settings=settings,
    )

    assert [c.chunk_id for c in baseline_contexts] == ["doc1", "doc2"]  # NOT graded/filtered
    assert [c.chunk_id for c in corrective.contexts] == ["doc1"]  # graded + filtered
    # The differing knob is real: the two arms produced different final context sets.
    assert baseline_contexts != corrective.contexts


def _settings_bare(**overrides: object) -> Settings:
    """Budget-pinned settings for the graph-logic comparison (no corpus/index needed)."""
    base: dict[str, object] = {
        "agentic_max_query_rewrites": 2,
        "agentic_max_regenerations": 1,
        "agentic_min_relevant_docs": 1,
        "agentic_min_attribution_rate": 1.0,
    }
    base.update(overrides)
    return Settings(**base)


# --- M1/M2: a FULL run where grading filters -> pins the baseline wiring + the honesty fix -----


def _two_doc_corpus_settings(tmp_path: Path) -> tuple[Settings, str]:
    """Write a 2-doc corpus (on-topic + off-topic) + a 1-row golden set; return (settings, id).

    Chunk size is large so each file is a single chunk. The relevant id is discovered by a scratch
    build (chunk ids are content hashes, deterministic across builds with identical chunking), so
    the eval's coverage guard passes and the golden row points at the on-topic chunk.
    """
    corpus = tmp_path / "sample"
    corpus.mkdir()
    (corpus / "ontopic.md").write_text(
        "# Elephants\n\nelephants roam the savanna grasslands searching for water at dusk\n",
        encoding="utf-8",
    )
    (corpus / "offtopic.md").write_text(
        "# Smoothies\n\nbanana smoothie blender recipe with yoghurt honey and ice\n",
        encoding="utf-8",
    )
    chunking: dict[str, object] = {"chunk_size": 1024, "chunk_overlap": 0}

    # Scratch build to discover the on-topic chunk id (tokens contain "elephants"). build_index
    # indexes settings.corpus_dir, so the scratch build points corpus_dir at the corpus; the eval
    # run instead indexes it via sample_dir (with a distinct corpus_dir for the leak guard). Both
    # index the same files with the same chunking, so the chunk ids match.
    scratch = _settings(
        tmp_path / "scratch",
        sample_dir=corpus,
        corpus_dir=corpus,
        qdrant_collection="corrective_filter_discover",
        **chunking,
    )
    build_index(scratch, embedder=HashingEmbedder(), store=QdrantVectorStore.in_memory("disc"))
    bm25 = BM25Index.load(scratch.storage_dir)
    ontopic_id = next(
        cid
        for cid, toks in zip(bm25.chunk_ids, bm25.corpus_tokens, strict=True)
        if "elephants" in toks
    )

    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    golden_path = golden_dir / "golden.jsonl"
    golden_path.write_text(
        json.dumps(
            {
                "query_id": "f1",
                "query": "elephants roam the savanna grasslands",
                "relevant_chunk_ids": [ontopic_id],
                "reference_answer": "Elephants roam the savanna grasslands.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    settings = _settings(
        tmp_path / "eval_storage",
        sample_dir=corpus,
        corpus_dir=tmp_path / "private",
        golden_path=golden_path,
        qdrant_collection="corrective_filter_eval",
        **chunking,
    )
    return settings, ontopic_id


def test_baseline_wiring_full_run_grading_filters_arms_diverge(tmp_path: Path) -> None:
    """End-to-end: with a query whose top-k holds an off-topic doc, grading FILTERS the corrective
    arm to 1 context while the baseline (answer_once) keeps BOTH. This pins that
    ``run_corrective_eval`` wires the baseline to a TRUE single pass: swapping it to
    ``run_corrective_rag(R=G=0)`` would grade+filter the baseline too, collapsing
    ``baseline_n_contexts`` to 1 and flipping ``contexts_identical`` to True — failing here.
    It is also the M1 honesty case: activation is 0 yet the arms did NOT see the same contexts.
    """
    settings, _ontopic_id = _two_doc_corpus_settings(tmp_path)
    report = run_corrective_eval(
        settings,
        llm=FakeLLMClient(),
        corrective=FakeCorrectiveLLM(),
        judge=FakeAnswerCorrectnessJudge(),
        embedder=HashingEmbedder(),
        store=QdrantVectorStore.in_memory("corrective_filter_eval"),
        reranker=LexicalOverlapReranker(),
    )

    assert report.n_queries == 1
    (record,) = report.per_query
    # Baseline kept BOTH retrieved docs (true single pass, no grade/filter).
    assert record.baseline_n_contexts == 2
    # Corrective grade-filtered the off-topic doc -> 1 context.
    assert record.corrective_n_contexts == 1
    assert record.contexts_identical is False

    # M1 honesty: the retry loop never fired, yet the arms are NOT equivalent (grading changed the
    # context set) -> this must NOT be reported as a "no-op".
    assert record.activated is False
    assert report.n_activated == 0
    assert report.n_contexts_identical == 0
    assert report.contexts_identical_rate == 0.0
    text = render_corrective_report(report)
    assert "NOT a no-op" in text
    assert "PROVABLE NO-OP" not in text
    assert "TRUE NO-OP" not in text


def test_render_true_no_op_only_when_activation_and_contexts_agree(tmp_path: Path) -> None:
    """On the committed golden set with offline fakes, grading keeps all contexts -> TRUE no-op."""
    report = _run(tmp_path / "storage")
    # Offline the fake grader keeps every retrieved context (each shares a token with its query).
    assert report.contexts_identical_rate == 1.0
    assert report.n_activated == 0
    text = render_corrective_report(report)
    assert "TRUE NO-OP" in text
    assert "NOT a no-op" not in text


# --- judge blindness ---------------------------------------------------------------------------


def test_judge_is_blind_sees_only_query_reference_candidate(tmp_path: Path) -> None:
    recorder = _RecordingJudge()
    _run(tmp_path / "storage", judge=recorder)

    ref_by_query = {item.query: (item.reference_answer or "").strip() for item in GOLDEN}
    all_relevant_ids = {cid for item in GOLDEN for cid in item.relevant_chunk_ids}

    # Both arms judged for every golden row (all 50 carry a reference) -> 2 calls per query.
    assert len(recorder.calls) == 2 * N_GOLDEN
    for query, reference, _candidate in recorder.calls:
        # The judge ONLY ever receives the golden query + its golden reference (plus a candidate,
        # which is the model's own answer and out of the harness's control).
        assert query in ref_by_query
        assert reference == ref_by_query[query]  # the true golden reference, nothing else
        # Blindness of the HARNESS-controlled inputs: neither the query nor the reference smuggles
        # an arm label or a relevant-chunk-id into the judge (that is what would bias it).
        for field in (query, reference):
            assert "baseline" not in field.lower()
            assert "corrective" not in field.lower()
            assert not any(cid in field for cid in all_relevant_ids)

    # For each query the SAME (query, reference) pair is judged twice (once per arm), differing
    # only by the candidate — the pairing that makes the correctness comparison fair.
    pairs = [(q, r) for q, r, _ in recorder.calls]
    for query in ref_by_query:
        assert pairs.count((query, ref_by_query[query])) == 2


# --- publishable flips False on ANY fake (per-dimension gate incl. the judge) -------------------


def _named(name: str) -> object:
    """A dummy whose class ``__name__`` is ``name``, to unit-test the publishable gate per axis."""
    return type(name, (), {})()


def test_publishable_true_only_when_every_backend_is_real() -> None:
    real = {
        "llm": _named("AnthropicLLMClient"),
        "corrective": _named("AnthropicCorrectiveLLM"),
        "judge": _named("AnthropicAnswerCorrectnessJudge"),
        "embedder": _named("SentenceTransformerEmbedder"),
        "reranker": _named("CrossEncoderReranker"),
    }
    assert _is_publishable(**real) is True  # type: ignore[arg-type]

    # Flipping ANY single axis to a fake class name flips publishable off — including the judge,
    # which a judged correctness number turns on.
    for axis, fake_name in [
        ("llm", "FakeLLMClient"),
        ("corrective", "FakeCorrectiveLLM"),
        ("judge", "FakeAnswerCorrectnessJudge"),
        ("embedder", "HashingEmbedder"),
        ("reranker", "LexicalOverlapReranker"),
    ]:
        broken = dict(real)
        broken[axis] = _named(fake_name)
        assert _is_publishable(**broken) is False, axis  # type: ignore[arg-type]


def test_full_fake_run_is_not_publishable(tmp_path: Path) -> None:
    prov = _run(tmp_path / "storage").provenance
    assert prov.publishable is False
    assert prov.baseline_llm_class == "FakeLLMClient"
    assert prov.corrective_llm_class == "FakeCorrectiveLLM"
    assert prov.judge_class == "FakeAnswerCorrectnessJudge"
    assert prov.embedder_class == "HashingEmbedder"
    assert prov.reranker_class == "LexicalOverlapReranker"


# --- provenance coherence ----------------------------------------------------------------------


def test_provenance_is_coherent_with_settings(tmp_path: Path) -> None:
    report = _run(tmp_path / "storage")
    p = report.provenance
    assert report.config == CONFIG
    assert p.n_queries == N_GOLDEN
    assert p.n_judged == N_GOLDEN  # every committed golden row carries a reference_answer
    assert p.top_k_rerank == 5  # the real answering config, both arms retrieve at k=5
    assert p.agentic_max_query_rewrites == 2
    assert p.agentic_max_regenerations == 1
    assert p.agentic_min_relevant_docs == 1
    assert p.agentic_min_attribution_rate == 1.0
    assert p.single_run is True
    # The judge records its OWN model (a distinct model from the generator by default); the fake
    # judge has no real model, so it records the explicit fake sentinel, never the generator model.
    assert p.judge_model == "fake-lexical-f1"
    assert p.judge_model != p.llm_model
    assert p.corpus_sha256  # a real corpus fingerprint was recorded


# --- structural invariants of the aggregate ----------------------------------------------------


def test_aggregate_invariants_hold(tmp_path: Path) -> None:
    report = _run(tmp_path / "storage")

    assert report.n_queries == N_GOLDEN
    assert len(report.per_query) == N_GOLDEN
    # activation rate is a fraction and equals the counted activations / n.
    assert 0.0 <= report.activation_rate <= 1.0
    assert report.activation_rate == report.n_activated / report.n_queries
    assert report.n_activated == sum(1 for r in report.per_query if r.activated)
    # contexts-identical rate is judge-free and equals the counted identical arms / n.
    assert 0.0 <= report.contexts_identical_rate <= 1.0
    assert report.contexts_identical_rate == report.n_contexts_identical / report.n_queries
    assert report.n_contexts_identical == sum(1 for r in report.per_query if r.contexts_identical)
    # terminated_reason histogram partitions exactly the queries.
    assert sum(report.terminated_reason_counts.values()) == N_GOLDEN
    # recall dicts cover exactly the declared cutoffs.
    assert set(report.baseline_recall_mean) == set(K_VALUES)
    assert set(report.corrective_recall_mean) == set(K_VALUES)

    for record in report.per_query:
        # The per-query cost is EXACTLY the trace formula (2*R + G + 1).
        assert record.extra_llm_calls == compute_extra_llm_calls(
            record.n_rewrites, record.n_regenerations
        )
        # activated iff a rewrite or a regeneration happened.
        assert record.activated == (record.n_rewrites > 0 or record.n_regenerations > 0)
        # The corrective arm never sees MORE contexts than the baseline (it can only grade-filter
        # the baseline's retrieved set), and identical contexts require equal counts (necessary
        # condition — same list implies same length).
        assert record.corrective_n_contexts <= record.baseline_n_contexts
        if record.contexts_identical:
            assert record.baseline_n_contexts == record.corrective_n_contexts
        if record.baseline_n_contexts != record.corrective_n_contexts:
            assert record.contexts_identical is False
        assert set(record.baseline_recall) == set(K_VALUES)
        assert set(record.corrective_recall) == set(K_VALUES)
        # every judged row has a bool verdict per arm; unjudged rows are None (none here).
        assert record.baseline_correct is not None
        assert record.corrective_correct is not None


def test_attribution_regression_guard_not_below_baseline(tmp_path: Path) -> None:
    # Under the grounded fake both arms ground at 1.0, so the corrective arm never REDUCES
    # attribution — the ADR-0007 regression guard holds (delta >= 0).
    report = _run(tmp_path / "storage")
    assert report.corrective_micro_attr >= report.baseline_micro_attr
    assert report.attr_delta == pytest.approx(
        report.corrective_micro_attr - report.baseline_micro_attr
    )


# --- determinism of the judge-free primary metric ----------------------------------------------


def test_primary_metric_is_deterministic(tmp_path: Path) -> None:
    report_a = _run(tmp_path / "a")
    report_b = _run(tmp_path / "b")

    # PRIMARY (trace-only) fields are identical across runs.
    assert report_a.activation_rate == report_b.activation_rate
    assert report_a.n_activated == report_b.n_activated
    assert report_a.n_rewrite_activated == report_b.n_rewrite_activated
    assert report_a.n_regenerate_activated == report_b.n_regenerate_activated
    assert report_a.n_final_query_changed == report_b.n_final_query_changed
    assert report_a.terminated_reason_counts == report_b.terminated_reason_counts
    assert report_a.mean_extra_llm_calls == report_b.mean_extra_llm_calls
    # And the full report round-trips identically (deterministic fakes end to end).
    assert report_a.model_dump() == report_b.model_dump()


def test_writes_byte_stable_distinct_artifact(tmp_path: Path) -> None:
    storage_a = tmp_path / "a"
    storage_b = tmp_path / "b"
    _run(storage_a)
    _run(storage_b)

    assert CORRECTIVE_RESULTS_FILENAME == "corrective_results.json"
    artifact_a = (storage_a / "eval" / CORRECTIVE_RESULTS_FILENAME).read_bytes()
    artifact_b = (storage_b / "eval" / CORRECTIVE_RESULTS_FILENAME).read_bytes()
    # Two fully-deterministic runs serialize byte-identically (sorted keys, no timestamp).
    assert artifact_a == artifact_b
    # And the artifact round-trips back into a valid report.
    CorrectiveEvalReport.model_validate_json(artifact_a.decode("utf-8"))


# --- console report ----------------------------------------------------------------------------


def test_render_report_headlines_activation_and_flags_non_publishable(tmp_path: Path) -> None:
    text = render_corrective_report(_run(tmp_path / "storage"))
    assert "NOT PUBLISHABLE" in text
    assert "activation_rate" in text  # the pre-registered PRIMARY headline
    assert "PRIMARY" in text
    assert "SECONDARY" in text
    assert "NOISE" in text  # the honesty caveat about the correctness delta at ~0 activation
    assert "REGRESSION GUARD" in text
    assert f"n={N_GOLDEN}" in text


# --- lexical-F1 floor + judge protocol conformance ---------------------------------------------


def test_lexical_f1_basic_properties() -> None:
    assert lexical_f1("Dipped headlights.", "Dipped headlights.") == 1.0
    assert lexical_f1("Dipped headlights.", "banana smoothie") == 0.0
    assert lexical_f1("", "anything") == 0.0
    assert lexical_f1("anything", "") == 0.0
    partial = lexical_f1("dipped headlights", "use dipped headlights when foggy")
    assert 0.0 < partial < 1.0


def test_fake_judge_is_deterministic_and_blind_to_query() -> None:
    fake = FakeAnswerCorrectnessJudge()
    v1 = fake.judge("q1", "Dipped headlights.", "Use dipped headlights.")
    v2 = fake.judge("DIFFERENT QUERY", "Dipped headlights.", "Use dipped headlights.")
    # The fake ignores the query (correctness = reference-vs-candidate) -> identical, deterministic.
    assert v1 == v2
    assert isinstance(v1, CorrectnessVerdict)


def test_judge_classes_satisfy_protocol_and_real_is_lazy() -> None:
    assert isinstance(FakeAnswerCorrectnessJudge(), AnswerCorrectnessJudge)
    judge = AnthropicAnswerCorrectnessJudge(Settings())
    assert isinstance(judge, AnswerCorrectnessJudge)
    assert judge._client is None  # SDK client not built at construction time (no key needed)


def test_default_judge_model_differs_from_generator() -> None:
    # N1 anti self-preference: the judge defaults to a DIFFERENT model than the generator.
    assert default_judge_model("claude-sonnet-4-6") == "claude-opus-4-8"
    assert default_judge_model("claude-opus-4-8") == "claude-sonnet-4-6"
    assert default_judge_model("claude-sonnet-4-6") != "claude-sonnet-4-6"


def test_real_judge_resolves_distinct_model_and_records_it() -> None:
    # With the default sonnet generator, the real judge runs on opus (distinct) and exposes it on
    # .model so provenance can record exactly what judged — no self-preference by default.
    settings = Settings(llm_model="claude-sonnet-4-6")
    judge = AnthropicAnswerCorrectnessJudge(settings)
    assert judge.model == "claude-opus-4-8"
    assert judge.model != settings.llm_model
    # An explicit override is honored (e.g. cheaper bulk judging).
    override = AnthropicAnswerCorrectnessJudge(settings, model="claude-sonnet-4-6")
    assert override.model == "claude-sonnet-4-6"


def test_grade_documents_partition_used_by_filter() -> None:
    """Sanity: FakeCorrectiveLLM grades the disjoint doc irrelevant (the filter test needs this)."""
    grades = FakeCorrectiveLLM().grade_documents(
        "elephants roam",
        [
            _ctx("doc1", "elephants roam the savanna grasslands"),
            _ctx("doc2", "banana smoothie blender recipe"),
        ],
    )
    by_id = {g.chunk_id: g for g in grades}
    assert isinstance(by_id["doc1"], DocRelevanceGrade)
    assert by_id["doc1"].relevant is True
    assert by_id["doc2"].relevant is False
