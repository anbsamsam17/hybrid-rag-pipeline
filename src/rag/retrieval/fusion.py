"""Hand-written Reciprocal Rank Fusion (RRF) — the project's signature fusion primitive.

RRF (Cormack, Clarke & Buettcher, 2009) merges several ranked lists *by rank only*, never
by raw score, so it fuses incomparable scorers (cosine similarity vs BM25) without any
tuning or score normalization. For each ranked list, a document at 1-based ``rank``
contributes ``1 / (k + rank)``; a document's fused score is the sum of its contributions
across all lists. The smoothing constant ``k`` (default 60) damps the influence of very
high ranks so no single list can dominate.

This implementation is **pure, dependency-free, and deterministic**. Two correctness
invariants are load-bearing and tested directly:

* **rank starts at 1** — the top item of a list contributes ``1/(k+1)``, never ``1/(k+0)``.
  An off-by-one here silently inflates every top item's weight.
* **tie-stable ordering** — documents with equal fused scores are returned in their
  *first-appearance order* across the input lists (the order in which they were first seen
  while scanning ``rankings`` left-to-right, then within each list top-to-bottom). Ordering
  never depends on ``set``/``dict`` iteration order leaking into the output.

Duplicates within a single list are collapsed to that document's **best (first) rank** in
that list, so a repeated id cannot be double-counted within one ranker's contribution.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists with Reciprocal Rank Fusion.

    Args:
        rankings: One ranked list of ``chunk_id`` strings per retriever, each ordered
            best-first. Lists may be empty, of differing lengths, and may contain ids that
            appear in some lists but not others. A duplicate id within one list uses that
            id's first (best) position in that list.
        k: RRF smoothing constant (``> 0``); the canonical value is 60. Each contribution is
            ``1 / (k + rank)`` with ``rank`` 1-based.

    Returns:
        ``(chunk_id, fused_score)`` pairs sorted by fused score descending. Ties are broken
        deterministically by first-appearance order across the inputs (see module docstring),
        so the result is fully reproducible and never leaks set/dict iteration order.

    Raises:
        ValueError: If ``k <= 0`` (the denominator must stay strictly positive).
    """
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")

    fused: dict[str, float] = {}
    # First-appearance index: assigned the first time an id is seen anywhere, giving a
    # deterministic, stable tie-break key independent of dict iteration order.
    first_seen: dict[str, int] = {}
    order = 0

    for ranking in rankings:
        seen_in_list: set[str] = set()
        for position, chunk_id in enumerate(ranking):
            # Collapse duplicates to the document's best (first) rank within THIS list.
            if chunk_id in seen_in_list:
                continue
            seen_in_list.add(chunk_id)

            # rank is 1-based: the top item (position 0) has rank 1 -> 1/(k+1).
            rank = position + 1
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)

            if chunk_id not in first_seen:
                first_seen[chunk_id] = order
                order += 1

    # Sort by fused score DESC, then by first-appearance ASC (stable, deterministic).
    return sorted(
        fused.items(),
        key=lambda item: (-item[1], first_seen[item[0]]),
    )
