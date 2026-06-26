"""Hybrid retriever: dense + sparse -> RRF fusion -> hydrate -> optional rerank.

This is the project's headline path. :class:`HybridRetriever` runs the dense (semantic) and
sparse (lexical) retrievers, fuses their ranked lists with the hand-written
:func:`~rag.retrieval.fusion.reciprocal_rank_fusion`, hydrates each fused candidate with its
text/metadata from the vector-store payload — **including sparse-only candidates the dense
search never returned** — optionally reranks with a cross-encoder, assigns final 1-based
ranks, and returns the top-``k`` :class:`~rag.retrieval.models.RetrievalResult` objects.

Everything is dependency-injectable (embedder, store, bm25, reranker, settings) so the whole
path runs offline in tests with a :class:`~rag.indexing.embeddings.HashingEmbedder`, an
in-memory Qdrant store, and the deterministic fake reranker. Determinism is preserved
end-to-end: RRF is tie-stable, payload hydration preserves the fused order, and the reranker
breaks ties by incoming order.
"""

from __future__ import annotations

import logging

from rag.config import Settings
from rag.indexing.embeddings import Embedder
from rag.indexing.sparse import BM25Index
from rag.indexing.vector_store import VectorStore
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.models import RetrievalResult
from rag.retrieval.rerank import Reranker
from rag.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)

# How many fused candidates to hydrate + feed the reranker. A small multiple of the final
# top_k_rerank gives the cross-encoder enough recall to reorder from without hydrating the
# entire fused list. Bounded so a huge corpus can't blow up the rerank batch.
_RERANK_POOL_MULTIPLIER = 4


class HybridRetriever:
    """Combine dense + sparse retrieval via RRF, hydrate, optionally rerank, return top-k."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        bm25: BM25Index,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        """Wire all collaborators explicitly (no hidden globals) for offline-testable DI."""
        self._settings = settings
        self._store = store
        self._reranker = reranker
        self._dense = DenseRetriever(embedder, store)
        self._sparse = SparseRetriever(bm25)

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievalResult]:
        """Run the full hybrid path for ``query`` and return the top-``k`` results.

        Pipeline: dense ``top_k_dense`` + sparse ``top_k_sparse`` -> RRF fuse -> take the
        fused top-N -> hydrate text/metadata (payloads, incl. sparse-only ids) -> optional
        cross-encoder rerank (when ``settings.use_reranker``) -> assign final 1-based ranks
        -> return the top ``k``.

        Args:
            query: The user query string.
            k: Final result count. Defaults to ``settings.top_k_rerank``.

        Returns:
            Self-contained :class:`RetrievalResult` objects, best first, with ``rank`` 1..k.
        """
        final_k = self._settings.top_k_rerank if k is None else k
        if final_k <= 0:
            return []

        dense_hits = self._dense.retrieve(query, self._settings.top_k_dense)
        sparse_hits = self._sparse.retrieve(query, self._settings.top_k_sparse)

        dense_ids = [chunk_id for chunk_id, _ in dense_hits]
        sparse_ids = [chunk_id for chunk_id, _ in sparse_hits]
        dense_set = set(dense_ids)
        sparse_set = set(sparse_ids)

        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self._settings.rrf_k)
        if not fused:
            return []

        # Hydrate enough candidates to rerank from, but never the whole corpus.
        pool_size = max(final_k, final_k * _RERANK_POOL_MULTIPLIER)
        fused_pool = fused[:pool_size]
        pool_ids = [chunk_id for chunk_id, _ in fused_pool]

        payloads = self._store.get_payloads(pool_ids)

        results: list[RetrievalResult] = []
        for provisional_rank, (chunk_id, fused_score) in enumerate(fused_pool, start=1):
            payload = payloads.get(chunk_id)
            if payload is None:
                # A fused id with no payload can't be returned self-contained; skip loudly.
                logger.warning("no payload for fused chunk_id=%s; dropping from results", chunk_id)
                continue
            sources: list[str] = []
            if chunk_id in dense_set:
                sources.append("dense")
            if chunk_id in sparse_set:
                sources.append("sparse")
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=fused_score,
                    rank=provisional_rank,
                    text=str(payload.get("text", "")),
                    rel_path=str(payload.get("rel_path", "")),
                    heading_path=list(payload.get("heading_path") or []),
                    metadata=dict(payload.get("metadata") or {}),
                    sources=sources,
                )
            )

        if self._settings.use_reranker:
            return self._reranker.rerank(query, results, final_k)

        # No rerank: truncate to final_k and reassign clean 1-based ranks.
        return [
            result.model_copy(update={"rank": final_rank})
            for final_rank, result in enumerate(results[:final_k], start=1)
        ]
