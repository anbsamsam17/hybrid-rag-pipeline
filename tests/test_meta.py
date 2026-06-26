"""Tests for the reproducible build provenance (``meta.json``).

These build :class:`Chunk` objects directly and use no optional backends — absent libraries
simply record as ``None`` in ``library_versions``. ``write_meta`` targets ``tmp_path``,
never the repo storage dir (and never trips the meta.json protect hook).
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.config import Settings
from rag.indexing.meta import (
    META_FILENAME,
    build_meta,
    corpus_sha256,
    write_meta,
)
from rag.ingestion.models import Chunk, make_chunk_id, make_doc_id


def make_chunk(
    text: str, start: int, end: int, rel_path: str = "doc.md", ordinal: int = 0
) -> Chunk:
    """Build a Chunk directly for meta tests."""
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


def make_settings() -> Settings:
    return Settings(
        chunk_strategy="recursive",
        chunk_size=512,
        chunk_overlap=64,
        corpus_dir=Path("."),
        sample_dir=Path("."),
    )


def _chunks() -> list[Chunk]:
    return [
        make_chunk("first chunk text", 0, 16, ordinal=0),
        make_chunk("second chunk text", 16, 33, ordinal=1),
    ]


def test_build_meta_is_json_serializable() -> None:
    meta = build_meta(
        settings=make_settings(),
        chunks=_chunks(),
        n_documents=1,
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dim=384,
    )
    # Round-trips through json without raising.
    assert json.loads(json.dumps(meta)) == meta


def test_build_meta_has_required_keys() -> None:
    meta = build_meta(
        settings=make_settings(),
        chunks=_chunks(),
        n_documents=2,
        embedding_model="m",
        embedding_dim=8,
    )
    for key in (
        "schema_version",
        "created_at",
        "git_sha",
        "corpus_sha256",
        "counts",
        "embedding",
        "chunking",
        "vector_store",
        "library_versions",
    ):
        assert key in meta
    assert meta["counts"] == {"n_documents": 2, "n_chunks": 2}
    assert meta["embedding"] == {"model": "m", "dim": 8}
    assert meta["chunking"] == {"strategy": "recursive", "chunk_size": 512, "chunk_overlap": 64}
    assert meta["vector_store"]["distance"] == "cosine"
    assert "python" in meta["library_versions"]


def test_git_sha_is_str_or_none() -> None:
    meta = build_meta(
        settings=make_settings(),
        chunks=_chunks(),
        n_documents=1,
        embedding_model="m",
        embedding_dim=4,
    )
    assert meta["git_sha"] is None or isinstance(meta["git_sha"], str)


def test_corpus_sha256_deterministic() -> None:
    assert corpus_sha256(_chunks()) == corpus_sha256(_chunks())


def test_corpus_sha256_order_independent() -> None:
    chunks = _chunks()
    assert corpus_sha256(chunks) == corpus_sha256(list(reversed(chunks)))


def test_corpus_sha256_changes_when_text_changes() -> None:
    base = corpus_sha256(_chunks())
    mutated = [
        make_chunk("first chunk text CHANGED", 0, 16, ordinal=0),
        make_chunk("second chunk text", 16, 33, ordinal=1),
    ]
    assert corpus_sha256(mutated) != base


def test_corpus_sha256_no_concatenation_collision() -> None:
    a = [make_chunk("ab", 0, 2, rel_path="a.md"), make_chunk("c", 0, 1, rel_path="b.md")]
    b = [make_chunk("a", 0, 1, rel_path="a.md"), make_chunk("bc", 0, 2, rel_path="b.md")]
    assert corpus_sha256(a) != corpus_sha256(b)


def test_write_meta_roundtrip(tmp_path: Path) -> None:
    meta = build_meta(
        settings=make_settings(),
        chunks=_chunks(),
        n_documents=1,
        embedding_model="m",
        embedding_dim=4,
    )
    path = tmp_path / META_FILENAME
    write_meta(path, meta)
    assert path.exists()
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["corpus_sha256"] == meta["corpus_sha256"]
    assert reloaded["counts"] == meta["counts"]
