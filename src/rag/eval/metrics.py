"""Pure retrieval metrics: recall@k, nDCG@k, reciprocal rank, MRR (ADR-0004).

These functions are the load-bearing math of the repo's headline comparison, so they are
deliberately **pure**: stdlib + ``typing`` only, no I/O, no global state, no randomness, and
**no imports from** ``rag.retrieval`` / ``rag.verification`` / ``rag.indexing`` / ``numpy``.
They operate on plain ids (``ranked_ids: Sequence[str]`` best-first, ``relevant:
Collection[str]``) so the definitions are unit-testable against hand-computed fixtures and the
``eval`` module boundary stays one-directional (nothing in the repo imports ``eval``).

Conventions (fixed in ADR-0004), each chosen so the *wrong-but-plausible* variant is a test
failure rather than a silent inflation:

* **Rank is 1-based** (rank ``i`` == ``ranked_ids[i-1]``), matching ``RetrievalResult.rank``
  and the hand-written RRF discount ``1/(k+rank)``.
* **Shared pre-step:** deduplicate ``ranked_ids`` keeping first occurrence, *then* apply the
  ``[:k]`` cutoff. A retriever returning the same id twice must neither inflate a score nor
  waste a top-k slot. ``relevant`` is coerced to a ``set`` internally for O(1) membership.
* **recall@k** divides by ``|relevant|`` (not ``min(|relevant|, k)``) — honest, never capped
  up to 1.0 when ``|relevant| > k``.
* **nDCG@k** uses binary gains with the ``log2(rank + 1)`` discount; IDCG caps the ideal at
  ``min(|relevant|, k)`` so a perfect ranking scores exactly 1.0.
* **reciprocal_rank** takes no ``k`` (the caller truncates for RR@k); MRR is the plain mean
  of per-query reciprocal ranks, with misses counted as 0.

Totality: every function returns a finite float in ``[0, 1]`` for any input — empty
``ranked``/``relevant`` give ``0.0``, ``k > len(ranked)`` scores what is present with no
padding, and **no function ever returns NaN or inf**. The single exception is ``k <= 0``,
which raises :class:`ValueError` (a non-positive cutoff is a programming error and mirrors the
``k <= 0`` guard in :func:`rag.retrieval.fusion.reciprocal_rank_fusion`).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def _dedup_keep_first(ranked_ids: Sequence[str]) -> list[str]:
    """Return ``ranked_ids`` with duplicates removed, keeping each id's first occurrence.

    ``dict.fromkeys`` preserves insertion order and collapses repeats deterministically, so a
    repeated id keeps its best (earliest) rank and cannot consume a second top-k slot.
    """
    return list(dict.fromkeys(ranked_ids))


def _require_positive_k(k: int) -> None:
    """Raise :class:`ValueError` for a non-positive cutoff (mirrors the RRF ``k`` guard)."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def recall_at_k(ranked_ids: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of relevant ids found in the deduplicated top-``k`` retrieved results.

    ``recall@k = |relevant ∩ dedup(ranked)[:k]| / |relevant|``. The denominator is
    ``|relevant|`` (never ``min(|relevant|, k)``): when there are more relevant ids than slots,
    recall is honestly capped below 1.0 rather than silently inflated.

    Args:
        ranked_ids: Retrieved chunk ids, best-first. Duplicates are collapsed to first
            occurrence before the cutoff (shared pre-step).
        relevant: The golden relevant chunk ids; coerced to a ``set`` internally.
        k: Positive cutoff (``> 0``); ``k > len(ranked_ids)`` scores what is present.

    Returns:
        The recall in ``[0.0, 1.0]``; ``0.0`` when ``relevant`` or ``ranked_ids`` is empty.

    Raises:
        ValueError: If ``k <= 0``.
    """
    _require_positive_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = _dedup_keep_first(ranked_ids)[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(ranked_ids: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Binary nDCG@k with the standard ``log2(rank + 1)`` discount (rank 1-based).

    ``DCG@k = Σ_{i=1..k} rel_i / log2(i + 1)`` with ``rel_i ∈ {0, 1}``; ``IDCG@k`` is the DCG
    of the ideal ranking, i.e. ``Σ_{i=1..min(|relevant|, k)} 1 / log2(i + 1)``; ``nDCG = DCG /
    IDCG`` (``0.0`` when ``IDCG == 0``). The ``+ 1`` makes the rank-1 discount ``log2(2) = 1``,
    avoiding the classic divide-by-zero of a bare ``log2(i)``.

    Args:
        ranked_ids: Retrieved chunk ids, best-first; deduplicated then cut to ``k``.
        relevant: The golden relevant chunk ids; coerced to a ``set`` internally.
        k: Positive cutoff (``> 0``).

    Returns:
        The nDCG in ``[0.0, 1.0]``; ``0.0`` when ``relevant`` or ``ranked_ids`` is empty.

    Raises:
        ValueError: If ``k <= 0``.
    """
    _require_positive_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    top_k = _dedup_keep_first(ranked_ids)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(top_k, start=1)
        if chunk_id in relevant_set
    )

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def reciprocal_rank(ranked_ids: Sequence[str], relevant: Collection[str]) -> float:
    """Reciprocal of the 1-based rank of the first relevant id, or ``0.0`` if none.

    Takes no ``k``: for RR@k the caller passes ``dedup(ranked)[:k]``. The shared dedup pre-step
    still applies here, so a duplicate appearing before the first relevant id does not depress
    the rank.

    Args:
        ranked_ids: Retrieved chunk ids, best-first; deduplicated (no cutoff).
        relevant: The golden relevant chunk ids; coerced to a ``set`` internally.

    Returns:
        ``1 / rank_of_first_relevant`` in ``(0.0, 1.0]``, or ``0.0`` if no relevant id is
        present (or either input is empty). Never NaN/inf.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for rank, chunk_id in enumerate(_dedup_keep_first(ranked_ids), start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def mrr(rr_values: Sequence[float]) -> float:
    """Mean Reciprocal Rank: the arithmetic mean of per-query reciprocal ranks.

    Misses must already be encoded as ``0.0`` reciprocal ranks by the caller (they count, they
    are not dropped). ``mrr([]) == 0.0``.

    Args:
        rr_values: A :class:`~collections.abc.Sequence` of reciprocal ranks, one per query
            (each in ``[0.0, 1.0]``). Must be finite — a NaN/inf would silently poison the mean
            (``mean`` of anything with NaN is NaN), so it is rejected rather than propagated.

    Returns:
        The mean reciprocal rank in ``[0.0, 1.0]``; ``0.0`` for an empty sequence.

    Raises:
        ValueError: If any value in ``rr_values`` is NaN or inf.
    """
    if not rr_values:
        return 0.0
    if any(not math.isfinite(value) for value in rr_values):
        raise ValueError("rr_values must be finite (no NaN or inf)")
    return sum(rr_values) / len(rr_values)
