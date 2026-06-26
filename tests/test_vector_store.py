"""Tests for the dense vector store.

The whole module is skipped if ``qdrant_client`` is absent. When present, an in-memory
client (``location=":memory:"``) is used so no server and no network are required. Vectors
come from the deterministic :class:`HashingEmbedder` (no model download).
"""

from __future__ import annotations

import uuid

import pytest

from rag.indexing.embeddings import HashingEmbedder
from rag.indexing.vector_store import QdrantVectorStore, point_id_for
from rag.ingestion.models import Chunk, make_chunk_id, make_doc_id

pytest.importorskip("qdrant_client")


def make_chunk(
    text: str, start: int, end: int, rel_path: str = "doc.md", ordinal: int = 0
) -> Chunk:
    """Build a Chunk directly for vector-store tests."""
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


def _chunks() -> list[Chunk]:
    texts = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
    chunks = []
    offset = 0
    for i, text in enumerate(texts):
        end = offset + len(text)
        chunks.append(make_chunk(text, offset, end, ordinal=i))
        offset = end
    return chunks


def test_point_id_deterministic_and_valid_uuid() -> None:
    cid = "abc123def456"
    first = point_id_for(cid)
    second = point_id_for(cid)
    assert first == second
    # Parses as a UUID (Qdrant accepts UUID-string ids).
    assert str(uuid.UUID(first)) == first
    # Distinct chunk ids -> distinct point ids.
    assert point_id_for("other") != first


def test_ensure_collection_and_upsert_count() -> None:
    chunks = _chunks()
    vectors = HashingEmbedder(dim=64).embed_texts([c.text for c in chunks])
    store = QdrantVectorStore.in_memory("test_collection")
    store.ensure_collection(64)
    store.upsert(chunks, vectors)
    assert store.count() == len(chunks)


def test_upsert_is_idempotent() -> None:
    chunks = _chunks()
    vectors = HashingEmbedder(dim=64).embed_texts([c.text for c in chunks])
    store = QdrantVectorStore.in_memory("idempotent")
    store.ensure_collection(64)
    store.upsert(chunks, vectors)
    first = store.count()
    # Re-upserting the same chunks overwrites in place (ids derived from chunk_id).
    store.upsert(chunks, vectors)
    assert store.count() == first == len(chunks)


def test_upsert_length_mismatch_raises() -> None:
    chunks = _chunks()
    store = QdrantVectorStore.in_memory("mismatch")
    store.ensure_collection(64)
    with pytest.raises(ValueError, match="length mismatch"):
        store.upsert(chunks, [[0.0] * 64])  # one vector for three chunks


def test_search_returns_chunk_ids() -> None:
    chunks = _chunks()
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.embed_texts([c.text for c in chunks])
    store = QdrantVectorStore.in_memory("search")
    store.ensure_collection(64)
    store.upsert(chunks, vectors)

    query_vec = embedder.embed_texts(["alpha beta gamma"])[0]
    results = store.search(query_vec, k=3)
    assert results
    # The exact-match chunk should rank first under cosine.
    assert results[0][0] == chunks[0].chunk_id
