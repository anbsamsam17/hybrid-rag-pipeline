"""Tests for the sparse (BM25) index and the shared tokenizer.

The tokenizer tests run with zero dependencies. The BM25 build/save/load/query tests are
guarded by ``importorskip("rank_bm25")`` so they skip cleanly when the backend is absent,
and they persist to ``tmp_path`` (never the repo storage dir).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.indexing.sparse import BM25_FILENAME, BM25Index, tokenize
from rag.ingestion.models import Chunk, make_chunk_id, make_doc_id


def make_chunk(
    text: str, start: int, end: int, rel_path: str = "doc.md", ordinal: int = 0
) -> Chunk:
    """Build a Chunk directly (no chunker) for sparse-index tests."""
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


# --- tokenizer (no deps) ----------------------------------------------------


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_drops_empties_and_punctuation() -> None:
    assert tokenize("  a--b__c  ") == ["a", "b", "c"]
    assert tokenize("...") == []
    assert tokenize("") == []


def test_tokenize_keeps_alphanumeric() -> None:
    assert tokenize("BM25 ranks doc42 first") == ["bm25", "ranks", "doc42", "first"]


def test_tokenize_deterministic() -> None:
    text = "The Quick, brown FOX -- jumps! over 9 lazy dogs."
    assert tokenize(text) == tokenize(text)


# --- BM25 (requires rank_bm25) ---------------------------------------------


def _sample_chunks() -> list[Chunk]:
    texts = [
        "the cat sat on the warm mat in the sun",
        "quantum entanglement links distant particles",
        "a dog chased the cat around the yard",
    ]
    chunks = []
    offset = 0
    for i, text in enumerate(texts):
        end = offset + len(text)
        chunks.append(make_chunk(text, offset, end, ordinal=i))
        offset = end
    return chunks


def test_bm25_build_save_load_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("rank_bm25")
    chunks = _sample_chunks()
    index = BM25Index()
    index.build(chunks)
    path = index.save(tmp_path)
    assert path == tmp_path / BM25_FILENAME
    assert path.exists()

    reloaded = BM25Index.load(tmp_path)
    assert reloaded.chunk_ids == index.chunk_ids
    assert reloaded.corpus_tokens == index.corpus_tokens
    assert reloaded.params == index.params


def test_bm25_query_ranks_relevant_first(tmp_path: Path) -> None:
    pytest.importorskip("rank_bm25")
    chunks = _sample_chunks()
    index = BM25Index()
    index.build(chunks)

    results = index.query("quantum entanglement particles", k=3)
    assert results
    top_id = results[0][0]
    # The obviously-relevant quantum chunk is chunks[1].
    assert top_id == chunks[1].chunk_id


def test_bm25_query_after_reload_matches(tmp_path: Path) -> None:
    pytest.importorskip("rank_bm25")
    chunks = _sample_chunks()
    index = BM25Index()
    index.build(chunks)
    index.save(tmp_path)
    reloaded = BM25Index.load(tmp_path)

    query = "cat on the mat"
    assert index.query(query, k=3) == reloaded.query(query, k=3)


def test_bm25_query_tie_stable(tmp_path: Path) -> None:
    pytest.importorskip("rank_bm25")
    # Two identical-token chunks score equally for a matching query; corpus order wins.
    chunks = [
        make_chunk("apple apple apple", 0, 17, ordinal=0),
        make_chunk("apple apple apple", 17, 34, ordinal=1),
        make_chunk("banana banana banana", 34, 55, ordinal=2),
    ]
    index = BM25Index()
    index.build(chunks)
    results = index.query("apple", k=2)
    # Equal scores -> stable by original corpus index (0 before 1).
    assert [cid for cid, _ in results] == [chunks[0].chunk_id, chunks[1].chunk_id]


def test_bm25_empty_corpus_is_safe(tmp_path: Path) -> None:
    pytest.importorskip("rank_bm25")
    index = BM25Index()
    index.build([])
    index.save(tmp_path)
    reloaded = BM25Index.load(tmp_path)
    assert reloaded.query("anything", k=5) == []
