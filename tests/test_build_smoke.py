"""End-to-end smoke test for :func:`build_index` using injected fakes.

Skipped cleanly when either backend is absent (current env). When both are present, the
build runs fully offline: a deterministic :class:`HashingEmbedder` (no model download) and
an in-memory Qdrant store (no server, no network). All writes go to ``tmp_path``, never the
repo storage dir and never a real ``meta.json`` (so the protect hook is not tripped).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("rank_bm25")

from rag.config import Settings  # noqa: E402
from rag.indexing.build import build_index  # noqa: E402
from rag.indexing.embeddings import HashingEmbedder  # noqa: E402
from rag.indexing.meta import META_FILENAME  # noqa: E402
from rag.indexing.sparse import BM25_FILENAME, BM25Index  # noqa: E402
from rag.indexing.vector_store import QdrantVectorStore  # noqa: E402

NOTE_A = """\
---
title: Cats
---
# Cats

The cat sat on the warm mat in the afternoon sun and purred contentedly.
"""

NOTE_B = """\
# Physics

Quantum entanglement links distant particles so a measurement on one instantly
constrains the other, regardless of the separation between them.
"""

# A third, unrelated note keeps the corpus at >2 docs so the physics query terms have a
# positive BM25 IDF. With only 2 docs, a term in exactly one doc gets IDF=0 under BM25Okapi
# (log((N-df+0.5)/(df+0.5)) = log(1.5/1.5) = 0) and every score collapses to 0 — which would
# make `test_build_index_bm25_queryable` assert on an arbitrary tie, not on real relevance.
NOTE_C = """\
# Cooking

Slow roasting vegetables in olive oil brings out a sweet, caramelized flavor over time.
"""


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Write a tiny Obsidian-style vault and return its root."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cats.md").write_text(NOTE_A, encoding="utf-8")
    (corpus / "physics.md").write_text(NOTE_B, encoding="utf-8")
    (corpus / "cooking.md").write_text(NOTE_C, encoding="utf-8")
    return corpus


def _settings(corpus: Path, storage: Path) -> Settings:
    return Settings(
        corpus_dir=corpus,
        sample_dir=corpus,
        storage_dir=storage,
        chunk_strategy="recursive",
        chunk_size=256,
        chunk_overlap=32,
        qdrant_collection="smoke",
    )


def test_build_index_smoke(vault: Path, tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    settings = _settings(vault, storage)
    store = QdrantVectorStore.in_memory(settings.qdrant_collection)

    summary = build_index(settings, embedder=HashingEmbedder(), store=store)

    assert summary["n_documents"] == 3
    assert summary["n_chunks"] >= 3
    assert summary["embedding_dim"] == 256
    assert summary["vector_count"] == summary["n_chunks"]

    meta_path = storage / META_FILENAME
    bm25_path = storage / BM25_FILENAME
    assert meta_path.exists()
    assert bm25_path.exists()
    assert summary["meta_path"] == str(meta_path)
    assert summary["bm25_path"] == str(bm25_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["counts"]["n_documents"] == 3
    assert meta["counts"]["n_chunks"] == summary["n_chunks"]
    assert meta["embedding"]["dim"] == 256
    assert isinstance(meta["corpus_sha256"], str)


def test_build_index_bm25_queryable(vault: Path, tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    settings = _settings(vault, storage)
    store = QdrantVectorStore.in_memory(settings.qdrant_collection)
    build_index(settings, embedder=HashingEmbedder(), store=store)

    bm25 = BM25Index.load(storage)
    results = bm25.query("quantum entanglement particles", k=3)
    assert results
    # The physics note should win for a physics query.
    assert any("entangle" in tok for tok in bm25.corpus_tokens[bm25.chunk_ids.index(results[0][0])])


def test_build_index_idempotent_count(vault: Path, tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    settings = _settings(vault, storage)
    store = QdrantVectorStore.in_memory(settings.qdrant_collection)

    first = build_index(settings, embedder=HashingEmbedder(), store=store)
    # Rebuild into the same store: deterministic point ids -> count stays put.
    second = build_index(settings, embedder=HashingEmbedder(), store=store)
    assert first["vector_count"] == second["vector_count"]


def test_build_index_empty_corpus_writes_reloadable_paths(tmp_path: Path) -> None:
    """An empty corpus must still write paths that exist and reload (no advertised lie)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()  # no files -> no chunks
    storage = tmp_path / "storage"
    settings = _settings(corpus, storage)
    store = QdrantVectorStore.in_memory(settings.qdrant_collection)

    summary = build_index(settings, embedder=HashingEmbedder(), store=store)

    assert summary["n_chunks"] == 0
    assert summary["vector_count"] == 0
    meta_path = Path(summary["meta_path"])
    bm25_path = Path(summary["bm25_path"])
    # Both advertised artifacts actually exist on disk.
    assert meta_path.exists()
    assert bm25_path.exists()
    # The (empty) BM25 index reloads and queries safely.
    reloaded = BM25Index.load(storage)
    assert reloaded.chunk_ids == []
    assert reloaded.query("anything", k=5) == []


def test_build_index_rebuild_shrink_drops_orphans(vault: Path, tmp_path: Path) -> None:
    """A rebuild over a SHRUNK corpus must not leave orphan points (M2)."""
    storage = tmp_path / "storage"

    # First build: 3 distinct single-file chunks (each small note is one chunk).
    big_corpus = tmp_path / "big"
    big_corpus.mkdir()
    for i in range(3):
        (big_corpus / f"n{i}.md").write_text(
            f"# Note {i}\n\nUnique body text number {i}.\n", "utf-8"
        )
    store = QdrantVectorStore.in_memory("shrink")
    first = build_index(_settings(big_corpus, storage), embedder=HashingEmbedder(), store=store)
    assert first["n_chunks"] == 3
    assert store.count() == 3

    # Rebuild over a 2-chunk corpus into the SAME store: count must reflect the new corpus,
    # not accumulate the removed chunk's orphaned point.
    small_corpus = tmp_path / "small"
    small_corpus.mkdir()
    for i in range(2):
        (small_corpus / f"n{i}.md").write_text(
            f"# Note {i}\n\nUnique body text number {i}.\n", "utf-8"
        )
    second = build_index(_settings(small_corpus, storage), embedder=HashingEmbedder(), store=store)
    assert second["n_chunks"] == 2
    assert store.count() == 2
