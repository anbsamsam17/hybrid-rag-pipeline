"""Indexing stage: turn :class:`~rag.ingestion.models.Chunk` objects into a searchable index.

Public surface:

* :func:`build_index` / :func:`main` — orchestrate load -> chunk -> embed -> upsert ->
  BM25 -> ``meta.json``; ``python -m rag.indexing.build`` runs the build.
* :class:`Embedder`, :class:`HashingEmbedder`, :class:`SentenceTransformerEmbedder`,
  :func:`get_embedder` — dense embedders (real + deterministic fake).
* :class:`VectorStore`, :class:`QdrantVectorStore`, :func:`point_id_for` — the dense store.
* :class:`BM25Index`, :func:`tokenize` — the sparse index and shared tokenizer.
* :func:`build_meta`, :func:`corpus_sha256`, :func:`write_meta` — reproducible provenance.

Every real backend (``sentence_transformers``, ``qdrant_client``, ``rank_bm25``) is
lazy-imported inside the methods that use it, so ``import rag.indexing`` succeeds with none
of them installed.
"""

from __future__ import annotations

from rag.indexing.build import build_index, main
from rag.indexing.embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)
from rag.indexing.meta import build_meta, corpus_sha256, write_meta
from rag.indexing.sparse import BM25Index, tokenize
from rag.indexing.vector_store import QdrantVectorStore, VectorStore, point_id_for

__all__ = [
    "build_index",
    "main",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "get_embedder",
    "build_meta",
    "corpus_sha256",
    "write_meta",
    "BM25Index",
    "tokenize",
    "QdrantVectorStore",
    "VectorStore",
    "point_id_for",
]
