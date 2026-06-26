"""Orchestrate the full index build: load -> chunk -> embed -> upsert -> BM25 -> meta.

:func:`build_index` is dependency-injectable: pass a fake embedder (e.g.
:class:`~rag.indexing.embeddings.HashingEmbedder`) and an in-memory store
(:meth:`QdrantVectorStore.in_memory`) to run the whole pipeline with no model download and
no Qdrant server. The defaults resolve to the real backends from ``settings`` and are
constructed *inside* the function so injecting fakes never imports a real backend.

``python -m rag.indexing.build`` (what ``make ingest`` calls) runs :func:`main`.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.config import Settings, get_settings
from rag.indexing.embeddings import Embedder, get_embedder
from rag.indexing.meta import META_FILENAME, build_meta, write_meta
from rag.indexing.sparse import BM25_FILENAME, BM25Index
from rag.indexing.vector_store import QdrantVectorStore, VectorStore
from rag.ingestion import chunk_corpus, load_corpus

logger = logging.getLogger(__name__)


def build_index(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> dict[str, Any]:
    """Build the dense + sparse index and write ``meta.json`` into ``settings.storage_dir``.

    ``embedder`` / ``store`` default to the real backends derived from ``settings`` but are
    injectable so tests run fully offline (HashingEmbedder + in-memory Qdrant). Returns a
    small summary dict with counts and the paths written.
    """
    settings.storage_dir.mkdir(parents=True, exist_ok=True)

    documents = load_corpus(settings.corpus_dir)
    chunks = chunk_corpus(documents, settings)
    logger.info("loaded %d documents -> %d chunks", len(documents), len(chunks))

    meta_path = settings.storage_dir / META_FILENAME
    bm25_path = settings.storage_dir / BM25_FILENAME

    # Lazy DI defaults: only resolve the real backends when none were injected.
    embedder = embedder or get_embedder(settings)

    if not chunks:
        # Empty corpus: write provenance + a reloadable (empty) BM25 index for auditability,
        # skip embed/upsert. Writing the empty index keeps the "bm25_path is always on disk
        # and BM25Index.load-able" contract (an empty build returns paths that exist).
        logger.warning(
            "no chunks produced from corpus_dir=%s; writing empty meta + BM25 index",
            settings.corpus_dir,
        )
        empty_bm25 = BM25Index()
        empty_bm25.build(chunks)
        empty_bm25.save(settings.storage_dir)
        meta = build_meta(
            settings=settings,
            chunks=chunks,
            n_documents=len(documents),
            embedding_model=settings.embedding_model,
            embedding_dim=0,
        )
        write_meta(meta_path, meta)
        return {
            "n_documents": len(documents),
            "n_chunks": 0,
            "embedding_dim": 0,
            "vector_count": 0,
            "meta_path": str(meta_path),
            "bm25_path": str(bm25_path),
        }

    store = store or QdrantVectorStore.from_settings(settings)

    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    dim = embedder.dim
    if len(vectors) != len(chunks):
        raise ValueError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")

    store.ensure_collection(dim)
    store.upsert(chunks, vectors)
    logger.info("upserted %d vectors (dim=%d) into the dense store", len(vectors), dim)

    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save(settings.storage_dir)
    logger.info("saved BM25 index to %s", bm25_path)

    meta = build_meta(
        settings=settings,
        chunks=chunks,
        n_documents=len(documents),
        embedding_model=settings.embedding_model,
        embedding_dim=dim,
    )
    write_meta(meta_path, meta)
    logger.info("wrote provenance to %s", meta_path)

    return {
        "n_documents": len(documents),
        "n_chunks": len(chunks),
        "embedding_dim": dim,
        "vector_count": store.count(),
        "meta_path": str(meta_path),
        "bm25_path": str(bm25_path),
    }


def main() -> None:
    """Entry point for ``python -m rag.indexing.build`` (used by ``make ingest``)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    summary = build_index(get_settings())
    logger.info("index build complete: %s", summary)


if __name__ == "__main__":
    main()
