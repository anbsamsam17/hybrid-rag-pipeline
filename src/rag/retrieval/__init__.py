"""Retrieval core: dense, sparse, hand-written RRF fusion (k=60), and cross-encoder reranking.

Public surface:

* :class:`HybridRetriever` — the headline path: dense + sparse -> RRF -> hydrate -> rerank.
* :func:`reciprocal_rank_fusion` — the hand-written, tie-stable RRF primitive.
* :class:`RetrievalResult` — the self-contained, frozen result model.
* :class:`DenseRetriever`, :class:`SparseRetriever` — the first-stage retrievers.
* :class:`Reranker`, :func:`get_reranker` — the rerank abstraction + factory (real + fake).

Every real backend (``sentence_transformers``, ``qdrant_client``, ``rank_bm25``) is
lazy-imported inside the code that uses it, so ``import rag.retrieval`` succeeds with none of
them installed.
"""

from __future__ import annotations

from rag.retrieval.dense import DenseRetriever
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.models import RetrievalResult
from rag.retrieval.rerank import (
    CrossEncoderReranker,
    IdentityReranker,
    LexicalOverlapReranker,
    Reranker,
    get_reranker,
)
from rag.retrieval.sparse import SparseRetriever

__all__ = [
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "RetrievalResult",
    "DenseRetriever",
    "SparseRetriever",
    "Reranker",
    "CrossEncoderReranker",
    "IdentityReranker",
    "LexicalOverlapReranker",
    "get_reranker",
]
