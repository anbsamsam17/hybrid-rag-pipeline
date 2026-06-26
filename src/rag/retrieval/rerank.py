"""Reranking stage: an abstraction plus a real cross-encoder and a deterministic fake.

After fusion narrows the candidate set, a reranker re-scores ``(query, chunk)`` *pairs*
jointly (a cross-encoder reads the query and the passage together, unlike the bi-encoder
used for first-stage dense retrieval) and returns the top-``top_k``. The :class:`Reranker`
ABC is the stable contract; two implementations sit behind it:

* :class:`CrossEncoderReranker` — the real model, **lazy-importing**
  ``sentence_transformers.CrossEncoder`` so importing this module never requires the package
  or triggers a model download. The model loads on first :meth:`rerank` call.
* :class:`LexicalOverlapReranker` — a deterministic, dependency-free fake for tests. It
  scores each candidate by token-set overlap with the query (reusing the shared
  :func:`~rag.indexing.sparse.tokenize`), so it reorders predictably and needs no model.

Every reranker re-ranks the candidates it is given, assigns each survivor a fresh 1-based
:pyattr:`RetrievalResult.rank` (1 = best), and truncates to ``top_k``. Ordering is
tie-stable: equal reranker scores keep the candidates' incoming order.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from rag.config import Settings
from rag.indexing.sparse import tokenize
from rag.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


def _rescored(candidate: RetrievalResult, score: float, rank: int) -> RetrievalResult:
    """Return a copy of ``candidate`` with a new ``score`` and 1-based ``rank``.

    ``RetrievalResult`` is frozen, so we rebuild via ``model_copy`` rather than mutate.
    """
    return candidate.model_copy(update={"score": score, "rank": rank})


class Reranker(ABC):
    """Re-score ``(query, candidate)`` pairs and return the best ``top_k`` results."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Return the top-``top_k`` candidates re-scored and re-ranked for ``query``.

        Implementations must be deterministic, assign a fresh 1-based ``rank`` to each
        returned result (1 = best), and break score ties by the candidates' incoming order.
        """


class LexicalOverlapReranker(Reranker):
    """Deterministic fake reranker: score by query/candidate token-set overlap.

    Dependency-free and reproducible — the test default so the suite never downloads a model.
    The score is the count of distinct query tokens present in the candidate's text; ties
    keep the incoming (fused) order, so the output is fully deterministic.
    """

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Re-score by distinct-token overlap with ``query``; tie-stable, top-``top_k``."""
        if top_k <= 0 or not candidates:
            return []
        query_tokens = set(tokenize(query))
        scored: list[tuple[float, int, RetrievalResult]] = []
        for incoming_index, candidate in enumerate(candidates):
            overlap = float(len(query_tokens & set(tokenize(candidate.text))))
            # incoming_index is the stable tie-break: equal overlap keeps fused order.
            scored.append((overlap, incoming_index, candidate))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            _rescored(candidate, score, rank=final_rank)
            for final_rank, (score, _, candidate) in enumerate(scored[:top_k], start=1)
        ]


class IdentityReranker(Reranker):
    """No-op reranker: keep the incoming order, just truncate and re-rank to ``top_k``.

    Useful as an explicit "reranking disabled but contract preserved" implementation and as
    a baseline in tests. Scores are left untouched; only ``rank`` is reassigned 1..top_k.
    """

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Truncate to ``top_k`` and reassign 1-based ranks; scores unchanged."""
        if top_k <= 0:
            return []
        return [
            candidate.model_copy(update={"rank": final_rank})
            for final_rank, candidate in enumerate(candidates[:top_k], start=1)
        ]


class CrossEncoderReranker(Reranker):
    """Real cross-encoder reranker (``sentence_transformers.CrossEncoder``, lazy-loaded)."""

    def __init__(self, model_name: str) -> None:
        """Store the model name; defer the import and model load until first use."""
        self._model_name = model_name
        self._model: object | None = None

    def _ensure_model(self) -> object:
        """Lazily import ``sentence_transformers`` and load the CrossEncoder once."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("loading cross-encoder reranker: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Score ``(query, text)`` pairs with the cross-encoder; tie-stable, top-``top_k``."""
        if top_k <= 0 or not candidates:
            return []
        model = self._ensure_model()
        pairs = [[query, candidate.text] for candidate in candidates]
        raw = model.predict(pairs)  # type: ignore[attr-defined]
        scores = [float(value) for value in raw]
        scored = list(zip(scores, range(len(candidates)), candidates, strict=True))
        # -score for descending; incoming index keeps ties stable (deterministic).
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            _rescored(candidate, score, rank=final_rank)
            for final_rank, (score, _, candidate) in enumerate(scored[:top_k], start=1)
        ]


def get_reranker(settings: Settings, *, fake: bool = False) -> Reranker:
    """Return a :class:`Reranker`.

    The real :class:`CrossEncoderReranker` (using ``settings.reranker_model``) is returned by
    default; pass ``fake=True`` for the deterministic, dependency-free
    :class:`LexicalOverlapReranker` used in tests and offline runs.
    """
    if fake:
        return LexicalOverlapReranker()
    return CrossEncoderReranker(settings.reranker_model)
