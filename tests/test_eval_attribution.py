"""Tests for the attribution-rate aggregation harness (ADR-0006), run fully OFFLINE.

These exercise the genuine build -> retrieve (hybrid+rerank) -> generate -> verify -> aggregate
path with the proven offline backends (:class:`HashingEmbedder` + in-memory Qdrant +
:class:`LexicalOverlapReranker`) and the DETERMINISTIC fake LLMs
(:class:`FakeLLMClient` / :class:`FabricatingFakeLLMClient`) — no torch, no network, no API key.

They assert **structure, invariants, determinism, and the abstention accounting** — never a real
LLM attribution value (the fakes are engineered so the grounding verdict is fully determined:
FakeLLMClient quotes a real substring of the top context -> grounded; FabricatingFakeLLMClient
quotes a sentinel -> ungrounded). No config ordering or retrieval ranking is asserted, because
the bag-of-tokens embedder produces uninterpretable rankings (marked non-publishable).
"""

from __future__ import annotations

from pathlib import Path

import pytest

# The attribution build path needs the lightweight index backends; skip the whole module cleanly
# (not a silent pass) if they are unavailable.
pytest.importorskip("qdrant_client")
pytest.importorskip("rank_bm25")

from rag.config import PROJECT_ROOT, Settings  # noqa: E402
from rag.eval.attribution import (  # noqa: E402
    ATTRIBUTION_RESULTS_FILENAME,
    CONFIG,
    render_attribution_report,
    run_attribution_eval,
)
from rag.eval.golden import load_golden  # noqa: E402
from rag.eval.models import AttributionReport  # noqa: E402
from rag.generation.llm import FabricatingFakeLLMClient, FakeLLMClient  # noqa: E402
from rag.generation.models import Answer, Citation  # noqa: E402
from rag.indexing.embeddings import HashingEmbedder  # noqa: E402
from rag.indexing.vector_store import QdrantVectorStore  # noqa: E402
from rag.retrieval.models import RetrievalResult  # noqa: E402
from rag.retrieval.rerank import LexicalOverlapReranker  # noqa: E402

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "golden.jsonl"

# Derive n from the committed golden set — NEVER hard-code it (the offline tests bind to the real
# data/eval/golden.jsonl, so growing the golden set must keep these assertions correct).
N_GOLDEN = len(load_golden(GOLDEN_PATH))

# A sentinel quote engineered to share no meaningful content tokens with any driving-content
# chunk, so verify_answer marks it ungrounded (mirrors FabricatingFakeLLMClient's own sentinel).
_FABRICATED_QUOTE = "zzqxv totally fabricated unsupported claim wzzqxv"
_GROUNDED_QUOTE_CHARS = 60  # leading slice length that is a real substring of the cited context


class HeterogeneousFakeLLMClient:
    """Test-only :class:`~rag.generation.llm.LLMClient` whose grounding OUTCOME varies by query.

    Structurally satisfies the ``LLMClient`` Protocol (``generate_answer(query, contexts) ->
    Answer``). Its whole purpose is to force ``micro != macro != macro_over_answered`` and a
    non-zero abstention, which the homogeneous fakes (always 1 grounded citation, never abstain)
    cannot exercise:

    * for ``mixed_query`` it emits **two** citations to a real chunk — one grounded (a real
      leading substring of the top context) and one fabricated (a sentinel quote) — so that
      query's per-answer rate is exactly 0.5 (2 citations, 1 grounded);
    * for ``abstain_query`` it emits **zero** citations (an explicit "no answer in context"), so
      that query abstains (rate 0.0, ``n_citations == 0``);
    * for every other query it emits **one** grounded citation (rate 1.0, like
      :class:`~rag.generation.llm.FakeLLMClient`).
    """

    def __init__(self, *, mixed_query: str, abstain_query: str) -> None:
        self._mixed_query = mixed_query
        self._abstain_query = abstain_query

    def generate_answer(self, query: str, contexts: list[RetrievalResult]) -> Answer:
        if not contexts or query == self._abstain_query:
            return Answer(text="The provided context does not answer this question.", citations=[])
        top = contexts[0]
        if query == self._mixed_query:
            # 1 grounded (real substring) + 1 fabricated (sentinel) citation -> rate 0.5.
            return Answer(
                text=f"Based on [{top.chunk_id}] (partly): mixed grounding.",
                citations=[
                    Citation(
                        chunk_id=top.chunk_id,
                        rel_path=top.rel_path,
                        supporting_quote=top.text[:_GROUNDED_QUOTE_CHARS],
                    ),
                    Citation(
                        chunk_id=top.chunk_id,
                        rel_path=top.rel_path,
                        supporting_quote=_FABRICATED_QUOTE,
                    ),
                ],
            )
        # Default: exactly one grounded citation (real substring of the top context) -> rate 1.0.
        return Answer(
            text=f"Based on [{top.chunk_id}]: grounded.",
            citations=[
                Citation(
                    chunk_id=top.chunk_id,
                    rel_path=top.rel_path,
                    supporting_quote=top.text[:_GROUNDED_QUOTE_CHARS],
                )
            ],
        )


class AbstainingFakeLLMClient:
    """Test-only ``LLMClient`` that ABSTAINS (zero citations) on every query.

    Drives ``total_citations == 0`` so the micro denominator is zero — exercising the
    division-by-zero guard (``micro_attribution_rate == 0.0`` with no ``ZeroDivisionError``).
    """

    def generate_answer(self, query: str, contexts: list[RetrievalResult]) -> Answer:
        return Answer(text="The provided context does not answer this question.", citations=[])


def _settings(storage: Path, **overrides: object) -> Settings:
    """Eval settings pinned to the committed sample corpus + golden set.

    Chunking is pinned to the config that minted golden.jsonl (recursive / 512 / 64) so the
    golden-coverage guard passes; storage is a temp dir so nothing leaks into the repo's real
    ``storage/``.
    """
    base: dict[str, object] = {
        "corpus_dir": PROJECT_ROOT / "data" / "corpus",  # unused by eval (it uses sample_dir)
        "sample_dir": SAMPLE_DIR,
        "golden_path": GOLDEN_PATH,
        "storage_dir": storage,
        "chunk_strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "qdrant_collection": "eval_attribution_test",
    }
    base.update(overrides)
    return Settings(**base)


def _run(storage: Path, llm: object) -> AttributionReport:
    """Run the attribution harness with the offline fakes (fresh in-memory store each call)."""
    return run_attribution_eval(
        _settings(storage),
        llm=llm,  # type: ignore[arg-type]
        embedder=HashingEmbedder(),
        store=QdrantVectorStore.in_memory("eval_attribution_test"),
        reranker=LexicalOverlapReranker(),
    )


def test_grounded_fake_gives_full_attribution(tmp_path: Path) -> None:
    report = _run(tmp_path / "storage", FakeLLMClient())

    assert isinstance(report, AttributionReport)
    assert report.config == CONFIG
    assert report.n_queries == N_GOLDEN
    assert len(report.per_query) == N_GOLDEN

    # The grounded fake cites a REAL substring of the top context for every query -> fully
    # grounded: micro == macro == 1.0, nothing abstains, and the pools agree.
    assert report.micro_attribution_rate == 1.0
    assert report.macro_attribution_rate == 1.0
    assert report.macro_attribution_rate_answered == 1.0
    assert report.n_abstained == 0
    assert report.n_answered == N_GOLDEN
    assert report.total_citations > 0
    assert report.total_grounded == report.total_citations

    for record in report.per_query:
        assert record.n_citations > 0
        assert record.n_grounded == record.n_citations
        assert record.attribution_rate == 1.0
        assert record.abstained is False


def test_grounded_fake_is_not_publishable(tmp_path: Path) -> None:
    prov = _run(tmp_path / "storage", FakeLLMClient()).provenance
    # Any fake in the stack flips publishable off; provenance records the LLM identity.
    assert prov.publishable is False
    assert prov.llm_class == "FakeLLMClient"
    assert prov.embedder_class == "HashingEmbedder"
    assert prov.reranker_class == "LexicalOverlapReranker"
    assert prov.single_run is True
    assert prov.top_k_rerank == 5  # the real answering config, not K_RETRIEVE=10


def test_fabricating_fake_fails_grounding_without_abstaining(tmp_path: Path) -> None:
    report = _run(tmp_path / "storage", FabricatingFakeLLMClient())

    # The fabricating fake cites a real chunk id but an UNSUPPORTED quote for every query:
    # a grounding failure (rate 0.0) that still emits citations -> NOT an abstention.
    assert report.micro_attribution_rate == 0.0
    assert report.macro_attribution_rate == 0.0
    assert report.macro_attribution_rate_answered == 0.0
    assert report.total_grounded == 0
    assert report.total_citations > 0
    assert report.n_abstained == 0
    assert report.n_answered == N_GOLDEN

    for record in report.per_query:
        assert record.n_citations > 0  # cited, but ungrounded
        assert record.n_grounded == 0
        assert record.attribution_rate == 0.0
        assert record.abstained is False


def test_run_is_deterministic(tmp_path: Path) -> None:
    # Deterministic fakes + deterministic retrieval/verification -> identical report objects.
    report_a = _run(tmp_path / "a", FakeLLMClient())
    report_b = _run(tmp_path / "b", FakeLLMClient())
    assert report_a.model_dump() == report_b.model_dump()


def test_writes_byte_stable_distinct_artifact(tmp_path: Path) -> None:
    storage_a = tmp_path / "a"
    storage_b = tmp_path / "b"
    _run(storage_a, FakeLLMClient())
    _run(storage_b, FakeLLMClient())

    # Distinct filename from the retrieval harness's eval_results.json.
    assert ATTRIBUTION_RESULTS_FILENAME == "attribution_results.json"
    artifact_a = (storage_a / "eval" / ATTRIBUTION_RESULTS_FILENAME).read_bytes()
    artifact_b = (storage_b / "eval" / ATTRIBUTION_RESULTS_FILENAME).read_bytes()
    # Two deterministic runs serialize byte-identically (sorted keys, no timestamp in the report).
    assert artifact_a == artifact_b
    # And the artifact round-trips back into a valid AttributionReport.
    AttributionReport.model_validate_json(artifact_a.decode("utf-8"))


def test_render_report_states_headline_scope_and_flags_non_publishable(tmp_path: Path) -> None:
    text = render_attribution_report(_run(tmp_path / "storage", FakeLLMClient()))
    assert "NOT PUBLISHABLE" in text
    assert "micro attribution_rate" in text  # headline is the micro rate
    assert "macro" in text  # macro reported alongside, never alone
    assert "abstention" in text.lower()
    assert "GROUNDING" in text  # states what the rate does NOT mean (correctness/completeness)
    assert f"n={N_GOLDEN}" in text


def test_micro_macro_abstention_are_distinct_with_exact_values(tmp_path: Path) -> None:
    # THE honesty test: the homogeneous fakes always leave micro==macro and n_abstained==0, so
    # a bug replacing micro=grounded/citations with micro=macro would pass every other test. A
    # heterogeneous fake breaks that symmetry with hand-computable exact values.
    golden = load_golden(GOLDEN_PATH)
    queries = [item.query for item in golden]
    mixed_query = queries[0]
    abstain_query = queries[-1]
    # Guard the design: the two special queries must be distinct and each unique in the golden set
    # so the fake's per-query branching hits exactly one query each (no accidental double-match).
    assert mixed_query != abstain_query
    assert queries.count(mixed_query) == 1
    assert queries.count(abstain_query) == 1

    report = _run(
        tmp_path / "storage",
        HeterogeneousFakeLLMClient(mixed_query=mixed_query, abstain_query=abstain_query),
    )

    # Per-query design (all hand-computed): n_default queries -> 1/1 grounded (rate 1.0); the
    # mixed query -> 1/2 grounded (rate 0.5); the abstain query -> 0 citations (rate 0.0).
    n_default = N_GOLDEN - 2
    expected_total_citations = n_default * 1 + 2 + 0
    expected_total_grounded = n_default * 1 + 1 + 0
    expected_micro = expected_total_grounded / expected_total_citations
    expected_macro = (n_default * 1.0 + 0.5 + 0.0) / N_GOLDEN
    expected_macro_answered = (n_default * 1.0 + 0.5) / (N_GOLDEN - 1)
    # Concretely at n=50: total_citations=50, total_grounded=49, micro=49/50=0.98,
    # macro=48.5/50=0.97, macro_answered=48.5/49~=0.98980 -- three FRANKLY distinct numbers.

    assert report.n_queries == N_GOLDEN
    assert report.total_citations == expected_total_citations
    assert report.total_grounded == expected_total_grounded
    assert report.n_abstained == 1
    assert report.n_answered == N_GOLDEN - 1

    assert report.micro_attribution_rate == pytest.approx(expected_micro)
    assert report.macro_attribution_rate == pytest.approx(expected_macro)
    assert report.macro_attribution_rate_answered == pytest.approx(expected_macro_answered)

    # Load-bearing: micro, macro, and macro-over-answered are all DISTINCT here. A regression that
    # computed micro AS the macro (mean of per-query rates) would flip micro to expected_macro and
    # trip the first two assertions below; the abstention split makes macro != macro_answered.
    assert report.micro_attribution_rate != pytest.approx(expected_macro)
    assert report.micro_attribution_rate != pytest.approx(report.macro_attribution_rate)
    assert report.macro_attribution_rate != pytest.approx(report.macro_attribution_rate_answered)
    # Micro (pooled) is HIGHER than macro here because the low-rate query (0.5) carries 2 citations
    # in the pool but only 1/50 of the macro mean -- exactly the distinction ADR-0006 protects.
    assert report.micro_attribution_rate > report.macro_attribution_rate

    # Per-query records reflect the exact design, keyed by the ORIGINAL golden query_id.
    by_id = {record.query_id: record for record in report.per_query}
    mixed = by_id[golden[0].query_id]
    abstained = by_id[golden[-1].query_id]
    assert mixed.n_citations == 2
    assert mixed.n_grounded == 1
    assert mixed.attribution_rate == pytest.approx(0.5)
    assert mixed.abstained is False
    assert abstained.n_citations == 0
    assert abstained.n_grounded == 0
    assert abstained.attribution_rate == 0.0
    assert abstained.abstained is True


def test_all_abstain_micro_is_zero_without_zero_division(tmp_path: Path) -> None:
    # Division-by-zero guard: when EVERY query abstains, total_citations == 0, so micro's
    # denominator is zero. The harness must report 0.0 (documented convention), never raise.
    report = _run(tmp_path / "storage", AbstainingFakeLLMClient())

    assert report.total_citations == 0
    assert report.total_grounded == 0
    assert report.micro_attribution_rate == 0.0  # no ZeroDivisionError
    assert report.macro_attribution_rate == 0.0
    assert report.macro_attribution_rate_answered == 0.0  # no answered queries -> 0.0 by convention
    assert report.n_abstained == N_GOLDEN
    assert report.n_answered == 0
    for record in report.per_query:
        assert record.abstained is True
        assert record.n_citations == 0
        assert record.attribution_rate == 0.0
