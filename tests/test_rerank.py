"""Tests for the rerank abstraction and the deterministic fake reranker.

No model download: only the :class:`LexicalOverlapReranker` / :class:`IdentityReranker` fakes
are exercised. The real :class:`CrossEncoderReranker` is verified only for lazy construction
(it must not import ``sentence_transformers`` at build time).
"""

from __future__ import annotations

from rag.config import Settings
from rag.retrieval.models import RetrievalResult
from rag.retrieval.rerank import (
    CrossEncoderReranker,
    IdentityReranker,
    LexicalOverlapReranker,
    get_reranker,
)


def _result(chunk_id: str, text: str, rank: int, score: float = 0.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        rank=rank,
        text=text,
        rel_path="doc.md",
        heading_path=[],
        metadata={},
        sources=["dense"],
    )


def _candidates() -> list[RetrievalResult]:
    # Incoming order is intentionally NOT the overlap order, so a real reorder is observable.
    return [
        _result("c1", "the cat sat on the mat", rank=1),
        _result("c2", "quantum entanglement of distant particles", rank=2),
        _result("c3", "a dog and a cat in the yard", rank=3),
    ]


def test_fake_reranker_reorders_by_overlap() -> None:
    reranker = LexicalOverlapReranker()
    out = reranker.rerank("quantum entanglement particles", _candidates(), top_k=3)
    # c2 has the most query-token overlap -> must be promoted to rank 1.
    assert out[0].chunk_id == "c2"
    assert out[0].rank == 1
    # Ranks are a clean 1..n sequence.
    assert [r.rank for r in out] == [1, 2, 3]
    # The reranker stamped its overlap score onto the result.
    assert out[0].score == 3.0  # quantum, entanglement, particles all present


def test_fake_reranker_respects_top_k() -> None:
    reranker = LexicalOverlapReranker()
    out = reranker.rerank("cat", _candidates(), top_k=2)
    assert len(out) == 2
    assert [r.rank for r in out] == [1, 2]


def test_fake_reranker_deterministic() -> None:
    reranker = LexicalOverlapReranker()
    a = reranker.rerank("cat mat", _candidates(), top_k=3)
    b = reranker.rerank("cat mat", _candidates(), top_k=3)
    assert [(r.chunk_id, r.rank, r.score) for r in a] == [(r.chunk_id, r.rank, r.score) for r in b]


def test_fake_reranker_tie_stable_keeps_incoming_order() -> None:
    # Query token "zzz" matches nothing -> all overlaps are 0 (a tie). Incoming order wins.
    reranker = LexicalOverlapReranker()
    out = reranker.rerank("zzz", _candidates(), top_k=3)
    assert [r.chunk_id for r in out] == ["c1", "c2", "c3"]


def test_fake_reranker_top_k_zero_returns_empty() -> None:
    assert LexicalOverlapReranker().rerank("cat", _candidates(), top_k=0) == []


def test_identity_reranker_preserves_order_and_reranks() -> None:
    reranker = IdentityReranker()
    out = reranker.rerank("anything", _candidates(), top_k=2)
    assert [r.chunk_id for r in out] == ["c1", "c2"]
    assert [r.rank for r in out] == [1, 2]


def test_get_reranker_fake_is_lexical() -> None:
    assert isinstance(get_reranker(Settings(), fake=True), LexicalOverlapReranker)


def test_get_reranker_real_is_lazy_no_import_at_construction() -> None:
    # Constructing the real reranker must NOT import sentence_transformers (lazy on first use).
    reranker = get_reranker(Settings(), fake=False)
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker._model is None  # not loaded yet
