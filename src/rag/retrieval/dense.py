"""Dense (semantic) retriever: embed the query, search the vector store.

A thin, dependency-injectable adapter over an :class:`~rag.indexing.embeddings.Embedder`
and a :class:`~rag.indexing.vector_store.VectorStore`. It owns no ranking logic of its own —
ordering and scores come straight from the store's ANN search — so the same retriever works
with the real ``sentence_transformers`` embedder + server Qdrant or the deterministic
:class:`~rag.indexing.embeddings.HashingEmbedder` + in-memory Qdrant used in tests.
"""

from __future__ import annotations

import logging

from rag.indexing.embeddings import Embedder
from rag.indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Embed a query and return the vector store's top-``k`` ``(chunk_id, score)`` hits."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        """Wire the retriever to an embedder and a (already-populated) vector store."""
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs for ``query``, best first.

        The query is embedded with the same embedder used at index time (so query and
        document vectors live in the same space), then handed to ``store.search``. Returns
        ``[]`` for a non-positive ``k`` without touching the store.
        """
        if k <= 0:
            return []
        vector = self._embedder.embed_texts([query])[0]
        results = self._store.search(vector, k)
        logger.debug("dense retrieve: query=%r k=%d -> %d hits", query, k, len(results))
        return results
