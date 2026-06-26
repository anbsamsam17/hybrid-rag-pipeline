"""Tests for :meth:`QdrantVectorStore.get_payloads`.

In-memory Qdrant, deterministic :class:`HashingEmbedder`; skipped without ``qdrant_client``.
``get_payloads`` is what lets the hybrid retriever hydrate sparse-only candidates, so it must
return correct payloads keyed by ``chunk_id``, omit unknown ids, and dedup.
"""

from __future__ import annotations

import pytest

from rag.indexing.embeddings import HashingEmbedder
from rag.indexing.vector_store import QdrantVectorStore
from rag.ingestion.models import Chunk, make_chunk_id, make_doc_id

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
        heading_path=["Section", f"Sub{ordinal}"],
        metadata={"strategy": "recursive", "chunk_index_in_doc": ordinal},
    )


def _populated() -> tuple[QdrantVectorStore, list[Chunk]]:
    texts = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
    chunks = []
    offset = 0
    for i, text in enumerate(texts):
        end = offset + len(text)
        chunks.append(_make_chunk(text, offset, end, i))
        offset = end
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.embed_texts([c.text for c in chunks])
    store = QdrantVectorStore.in_memory("payloads")
    store.ensure_collection(64)
    store.upsert(chunks, vectors)
    return store, chunks


def test_get_payloads_returns_correct_payloads_by_chunk_id() -> None:
    store, chunks = _populated()
    ids = [c.chunk_id for c in chunks]
    payloads = store.get_payloads(ids)
    assert set(payloads.keys()) == set(ids)
    for chunk in chunks:
        payload = payloads[chunk.chunk_id]
        assert payload["text"] == chunk.text
        assert payload["rel_path"] == chunk.rel_path
        assert payload["heading_path"] == chunk.heading_path
        assert payload["metadata"] == chunk.metadata


def test_get_payloads_subset() -> None:
    store, chunks = _populated()
    one = chunks[1].chunk_id
    payloads = store.get_payloads([one])
    assert list(payloads.keys()) == [one]
    assert payloads[one]["text"] == chunks[1].text


def test_get_payloads_unknown_ids_omitted() -> None:
    store, chunks = _populated()
    payloads = store.get_payloads([chunks[0].chunk_id, "deadbeefdeadbeef"])
    assert chunks[0].chunk_id in payloads
    assert "deadbeefdeadbeef" not in payloads


def test_get_payloads_empty_input() -> None:
    store, _ = _populated()
    assert store.get_payloads([]) == {}


def test_get_payloads_dedups_repeated_ids() -> None:
    store, chunks = _populated()
    cid = chunks[0].chunk_id
    payloads = store.get_payloads([cid, cid, cid])
    assert list(payloads.keys()) == [cid]
