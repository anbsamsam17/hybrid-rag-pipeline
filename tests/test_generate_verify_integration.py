"""Integration: generate (fake) -> verify, fully offline.

Proves the two stages agree end-to-end: the grounded fake's own output verifies to
``attribution_rate == 1.0`` (it cites a real quote), and the fabricating fake's output
verifies to ``attribution_rate < 1.0`` (the unsupported path). No API key, no network.
"""

from __future__ import annotations

from rag.config import Settings
from rag.generation.generate import generate_answer
from rag.generation.llm import FabricatingFakeLLMClient, FakeLLMClient
from rag.retrieval.models import RetrievalResult
from rag.verification.citations import verify_answer


def _contexts() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="c1",
            score=1.0,
            rank=1,
            text="Reciprocal rank fusion combines ranked lists using the constant k=60.",
            rel_path="rrf.md",
            heading_path=["Fusion"],
            metadata={},
            sources=["dense", "sparse"],
        ),
        RetrievalResult(
            chunk_id="c2",
            score=0.5,
            rank=2,
            text="BM25 is a sparse lexical ranking function.",
            rel_path="bm25.md",
            heading_path=["Sparse"],
            metadata={},
            sources=["sparse"],
        ),
    ]


def test_fake_answer_verifies_to_full_attribution() -> None:
    settings = Settings()
    contexts = _contexts()
    answer = generate_answer(
        "what constant does RRF use?",
        contexts,
        llm=FakeLLMClient(),
        settings=settings,
    )
    report = verify_answer(answer, contexts)
    # The fake quotes a real substring of the top chunk -> fully grounded.
    assert report.attribution_rate == 1.0
    assert report.n_citations == 1
    assert report.unsupported == []


def test_fabricating_fake_answer_verifies_below_full_attribution() -> None:
    settings = Settings()
    contexts = _contexts()
    answer = generate_answer(
        "what constant does RRF use?",
        contexts,
        llm=FabricatingFakeLLMClient(),
        settings=settings,
    )
    report = verify_answer(answer, contexts)
    # The fabricating fake cites a real chunk id but an unsupported quote -> not grounded.
    assert report.attribution_rate < 1.0
    assert report.attribution_rate == 0.0
    assert report.unsupported == ["c1"]


def test_pipeline_is_deterministic_end_to_end() -> None:
    settings = Settings()
    contexts = _contexts()

    def run() -> object:
        answer = generate_answer("q", contexts, llm=FakeLLMClient(), settings=settings)
        return verify_answer(answer, contexts)

    assert run() == run()
