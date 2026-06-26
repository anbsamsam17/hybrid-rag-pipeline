"""Tests for :class:`DenseRetriever`.

Runs fully offline: deterministic :class:`HashingEmbedder` (no model download) + in-memory
Qdrant (no server/network). Skipped cleanly when ``qdrant_client`` is absent.
"""

from __future__ import annotations

import pytest

from rag.indexing.embeddings import HashingEmbedder
from rag.indexing.vector_store import QdrantVectorStore
from rag.ingestion.models import Chunk, make_chunk_id, make_doc_id
from rag.retrieval.dense import DenseRetriever

pytest.importorskip("qdrant_client")


def _make_chunk(text: str, start: int, end: int, ordinal: int) -> Chunk:
    rel_path = "doc.md"
    return Chunk(
        chunk_id=make_chunk_id(rel_path, start, end),
        doc_id=make_doc_id(rel_path),
        source_path=f"/x/{rel_path}",
        rel_path=rel_path,
        ordinal=ordinal,
        text=text,
        start=start,
        end=end,
    )


def _populated_store(embedder: HashingEmbedder) -> tuple[QdrantVectorStore, list[Chunk]]:
    texts = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
    chunks = []
    offset = 0
    for i, text in enumerate(texts):
        end = offset + len(text)
        chunks.append(_make_chunk(text, offset, end, i))
        offset = end
    vectors = embedder.embed_texts([c.text for c in chunks])
    store = QdrantVectorStore.in_memory("dense_retriever")
    store.ensure_collection(embedder.dim)
    store.upsert(chunks, vectors)
    return store, chunks


def test_dense_returns_ranked_chunk_id_score_best_first() -> None:
    embedder = HashingEmbedder(dim=64)
    store, chunks = _populated_store(embedder)
    retriever = DenseRetriever(embedder, store)

    results = retriever.retrieve("alpha beta gamma", k=3)
    assert results
    # Each result is a (chunk_id, score) pair.
    assert all(isinstance(cid, str) and isinstance(score, float) for cid, score in results)
    # Exact-match chunk ranks first under cosine.
    assert results[0][0] == chunks[0].chunk_id
    # Scores are non-increasing (best first).
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_respects_k() -> None:
    embedder = HashingEmbedder(dim=64)
    store, _ = _populated_store(embedder)
    retriever = DenseRetriever(embedder, store)
    assert len(retriever.retrieve("alpha", k=2)) <= 2


def test_dense_non_positive_k_returns_empty() -> None:
    embedder = HashingEmbedder(dim=64)
    store, _ = _populated_store(embedder)
    retriever = DenseRetriever(embedder, store)
    assert retriever.retrieve("alpha", k=0) == []


def test_dense_deterministic() -> None:
    # The ranked chunk_id ORDER is the load-bearing determinism contract; Qdrant's in-memory
    # backend can jitter cosine scores at ~1e-7 between calls, so scores are compared approx.
    embedder = HashingEmbedder(dim=64)
    store, _ = _populated_store(embedder)
    retriever = DenseRetriever(embedder, store)
    first = retriever.retrieve("delta epsilon", k=3)
    second = retriever.retrieve("delta epsilon", k=3)
    assert [cid for cid, _ in first] == [cid for cid, _ in second]
    for (_, s1), (_, s2) in zip(first, second, strict=True):
        assert s1 == pytest.approx(s2)
