"""Sparse (lexical, BM25) retriever: a thin adapter over :class:`BM25Index`.

Wraps the already-built :class:`~rag.indexing.sparse.BM25Index` so the retrieval stage sees
the same ``(chunk_id, score)`` contract as the dense path. The BM25 index owns tokenization
and tie-stable ranking; this adapter adds only a query guard and a loader that reads the
persisted ``bm25.json`` from ``settings.storage_dir``.
"""

from __future__ import annotations

import logging

from rag.config import Settings
from rag.indexing.sparse import BM25Index

logger = logging.getLogger(__name__)


class SparseRetriever:
    """Return the BM25 index's top-``k`` ``(chunk_id, score)`` hits for a query."""

    def __init__(self, bm25: BM25Index) -> None:
        """Wire the retriever to an already-built (or loaded) BM25 index."""
        self._bm25 = bm25

    @classmethod
    def from_storage(cls, settings: Settings) -> SparseRetriever:
        """Load the persisted ``bm25.json`` from ``settings.storage_dir`` and wrap it.

        Reuses :meth:`BM25Index.load`, which rebuilds the fitted model from the stored
        tokenized corpus (no pickle, deterministic scores on reload).
        """
        bm25 = BM25Index.load(settings.storage_dir)
        logger.info(
            "loaded BM25 index from %s (%d chunks)",
            settings.storage_dir,
            len(bm25.chunk_ids),
        )
        return cls(bm25)

    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs for ``query``, best first.

        Delegates to :meth:`BM25Index.query`, which is tie-stable by original corpus index.
        Returns ``[]`` for a non-positive ``k``.
        """
        if k <= 0:
            return []
        results = self._bm25.query(query, k)
        logger.debug("sparse retrieve: query=%r k=%d -> %d hits", query, k, len(results))
        return results
