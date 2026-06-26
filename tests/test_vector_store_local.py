"""Tests for the no-server modes of QdrantVectorStore: mode validation + local on-disk.

These run fully offline: the local on-disk Qdrant (``QdrantClient(path=...)``) is embedded in
the process, so no Docker/server and no network are required. A deterministic HashingEmbedder
supplies vectors, so no model download either.
"""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.indexing.embeddings import HashingEmbedder
from rag.ingestion.models import Chunk

pytest.importorskip("qdrant_client")
from rag.indexing.vector_store import QdrantVectorStore  # noqa: E402


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d",
        source_path="d.md",
        rel_path="d.md",
        ordinal=0,
        text=text,
        start=0,
        end=len(text),
        heading_path=[],
        metadata={},
    )


def test_init_requires_exactly_one_mode() -> None:
    with pytest.raises(ValueError):
        QdrantVectorStore(collection="c")  # zero modes
    with pytest.raises(ValueError):
        QdrantVectorStore(collection="c", location=":memory:", url="http://x:6333")  # two modes


def test_from_settings_local_path_works_without_a_server(tmp_path) -> None:
    settings = Settings(qdrant_path=str(tmp_path / "qd"))
    store = QdrantVectorStore.from_settings(settings)

    emb = HashingEmbedder()
    chunks = [_chunk("a", "le trafic sur l'autoroute A7"), _chunk("b", "le viaduc de Millau")]
    vectors = emb.embed_texts([c.text for c in chunks])

    store.ensure_collection(emb.dim)
    store.upsert(chunks, vectors)

    assert store.count() == 2
    hits = store.search(emb.embed_texts(["A7"])[0], k=2)
    assert {cid for cid, _ in hits} <= {"a", "b"}
    # payload hydration also works in local-disk mode
    assert set(store.get_payloads(["a", "b"])) == {"a", "b"}


def test_from_settings_prefers_local_path_over_url(tmp_path) -> None:
    # qdrant_url points nowhere usable; because qdrant_path is set, we must NOT touch it.
    settings = Settings(qdrant_path=str(tmp_path / "qd2"), qdrant_url="http://unused.invalid:6333")
    store = QdrantVectorStore.from_settings(settings)
    store.ensure_collection(4)
    assert store.count() == 0  # proves it used the local path, not the (dead) url
