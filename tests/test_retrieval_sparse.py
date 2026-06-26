"""Tests for :class:`SparseRetriever` (the retrieval-stage BM25 adapter).

Named ``test_retrieval_sparse`` so it sits alongside the existing ``test_sparse.py`` (which
covers the indexing-stage :class:`BM25Index` itself). Skipped when ``rank_bm25`` is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import Settings
from rag.indexing.sparse import BM25Index
from rag.ingestion.models import Chunk, make_chunk_id, make_doc_id
from rag.retrieval.sparse import SparseRetriever

pytest.importorskip("rank_bm25")


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
        chunks.append(_make_chunk(text, offset, end, i))
        offset = end
    return chunks


def _built_index() -> tuple[BM25Index, list[Chunk]]:
    chunks = _sample_chunks()
    index = BM25Index()
    index.build(chunks)
    return index, chunks


def test_sparse_returns_ranked_chunk_id_score_best_first() -> None:
    index, chunks = _built_index()
    retriever = SparseRetriever(index)
    results = retriever.retrieve("quantum entanglement particles", k=3)
    assert results
    assert all(isinstance(cid, str) and isinstance(score, float) for cid, score in results)
    # The quantum chunk (chunks[1]) is the obvious winner.
    assert results[0][0] == chunks[1].chunk_id
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_sparse_respects_k() -> None:
    index, _ = _built_index()
    retriever = SparseRetriever(index)
    assert len(retriever.retrieve("cat", k=1)) <= 1


def test_sparse_non_positive_k_returns_empty() -> None:
    index, _ = _built_index()
    retriever = SparseRetriever(index)
    assert retriever.retrieve("cat", k=0) == []


def test_sparse_from_storage_roundtrip(tmp_path: Path) -> None:
    index, chunks = _built_index()
    index.save(tmp_path)
    settings = Settings(storage_dir=tmp_path)
    retriever = SparseRetriever.from_storage(settings)
    results = retriever.retrieve("quantum entanglement", k=3)
    assert results
    assert results[0][0] == chunks[1].chunk_id
