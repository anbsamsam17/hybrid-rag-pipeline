"""Tests for the hand-written Reciprocal Rank Fusion (the signature primitive).

Pure math, zero dependencies — every expected fused score is computed by hand in the test so
a wrong RRF formula (off-by-one on rank, double-counted duplicates, score normalization
sneaking in) fails loudly. Tie-stability is asserted against first-appearance order so a
``set``/``dict`` iteration order leaking into the output would be caught.
"""

from __future__ import annotations

import math

import pytest

from rag.retrieval.fusion import reciprocal_rank_fusion


def test_rank_starts_at_one_top_item_is_k_plus_one() -> None:
    # A single list with one item: the top item's contribution is 1/(k+1), NOT 1/(k+0).
    out = reciprocal_rank_fusion([["a"]], k=60)
    assert out == [("a", 1.0 / 61.0)]
    # Guard against the off-by-one: it must not be 1/60.
    assert out[0][1] != pytest.approx(1.0 / 60.0)


def test_known_two_list_math() -> None:
    # list1: a(rank1), b(rank2), c(rank3)
    # list2: b(rank1), c(rank2), d(rank3)
    out = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert out["a"] == pytest.approx(1.0 / 61.0)
    assert out["b"] == pytest.approx(1.0 / 62.0 + 1.0 / 61.0)
    assert out["c"] == pytest.approx(1.0 / 63.0 + 1.0 / 62.0)
    assert out["d"] == pytest.approx(1.0 / 63.0)


def test_doc_in_both_lists_outranks_doc_in_one() -> None:
    # b appears in both lists; a only in one. b must fuse higher and sort first.
    ranked = reciprocal_rank_fusion([["a", "b"], ["b"]], k=60)
    ids = [cid for cid, _ in ranked]
    assert ids[0] == "b"
    scores = dict(ranked)
    assert scores["b"] > scores["a"]
    assert scores["b"] == pytest.approx(1.0 / 62.0 + 1.0 / 61.0)
    assert scores["a"] == pytest.approx(1.0 / 61.0)


def test_tie_stable_by_first_appearance() -> None:
    # Each id appears exactly once, all at rank 1 in their own single-item list -> identical
    # fused score 1/(k+1). Order must follow FIRST-APPEARANCE across the inputs: x, y, z.
    out = reciprocal_rank_fusion([["x"], ["y"], ["z"]], k=60)
    assert [cid for cid, _ in out] == ["x", "y", "z"]
    assert all(score == pytest.approx(1.0 / 61.0) for _, score in out)

    # Same ids, different first-appearance order -> output order tracks it deterministically.
    out2 = reciprocal_rank_fusion([["z"], ["y"], ["x"]], k=60)
    assert [cid for cid, _ in out2] == ["z", "y", "x"]


def test_tie_stable_cross_list_first_appearance() -> None:
    # a and b tie (each rank 1 once). a is first-seen in list1 before b in list2.
    out = reciprocal_rank_fusion([["a"], ["b"]], k=60)
    assert [cid for cid, _ in out] == ["a", "b"]


def test_tie_break_is_first_appearance_not_alphabetical_or_score_grouping() -> None:
    """Prove the tie-break is the explicit first-appearance key, not an incidental fallback.

    This test is deliberately *discriminating*: it decouples score order from first-appearance
    order so that the documented invariant ("ties resolve by first-appearance") can only hold
    if a real first-seen key drives the sort, rather than alphabetical order, a reversed key,
    or a set/dict-iteration leak.

    Input ``[["c", "d"], ["b"]]`` with ``k=60`` yields:

    * ``c`` — rank 1 in list1            -> ``1/61``
    * ``d`` — rank 2 in list1            -> ``1/62``   (strictly lower score)
    * ``b`` — rank 1 in list2            -> ``1/61``   (ties EXACTLY with ``c``)

    First-appearance order across the inputs is ``c, d, b``. The correct fused output is
    therefore ``["c", "b", "d"]``: the tied pair ``{c, b}`` resolves to ``c`` *before* ``b``
    (c was seen first), and the lower-scored ``d`` sinks to last even though it appeared
    *before* ``b``. Every naive tie-break disagrees with this exact ordering:

    * alphabetical (``key=... , item[0]``)        -> ``["b", "c", "d"]`` (b < c)
    * reversed first-seen (``..., -first_seen``)  -> ``["b", "c", "d"]``
    * a sorted-set group leak within tied scores  -> ``["b", "c", "d"]``

    so this single assertion fails any of them while passing the real implementation.
    """
    out = reciprocal_rank_fusion([["c", "d"], ["b"]], k=60)
    ids = [cid for cid, _ in out]
    # The load-bearing assertion: first-appearance wins the c/b tie, score sinks d last.
    assert ids == ["c", "b", "d"]
    # Make the construction self-validating: c and b really do tie, and d is strictly lower.
    scores = dict(out)
    assert scores["c"] == pytest.approx(scores["b"]), "test invalid: c and b must tie"
    assert scores["d"] < scores["c"], "test invalid: d must score strictly below the tie"

    # Reversing the inputs reverses the tied output (order is data-driven, not hard-coded):
    # now b is first-seen before c, so the tie resolves b-before-c.
    reversed_out = reciprocal_rank_fusion([["b"], ["c", "d"]], k=60)
    reversed_ids = [cid for cid, _ in reversed_out]
    assert reversed_ids == ["b", "c", "d"]


def test_duplicate_id_in_a_list_uses_best_rank_once() -> None:
    # "a" appears twice in one list (rank1 and rank2); only its best (first) rank counts,
    # and it is not double-added. So a -> 1/(k+1), b -> 1/(k+3) (b is at position 3).
    out = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=60))
    assert out["a"] == pytest.approx(1.0 / 61.0)
    assert out["b"] == pytest.approx(1.0 / 63.0)
    assert len(out) == 2


def test_empty_lists_handled() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []
    # One empty, one populated: the populated list still fuses normally.
    out = reciprocal_rank_fusion([[], ["a", "b"]], k=60)
    assert [cid for cid, _ in out] == ["a", "b"]


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([["a"]], k=-5)


def test_deterministic_across_calls() -> None:
    rankings = [["a", "b", "c"], ["c", "d", "a"], ["e"]]
    first = reciprocal_rank_fusion(rankings, k=60)
    second = reciprocal_rank_fusion(rankings, k=60)
    assert first == second


def test_sorted_descending_by_fused_score() -> None:
    out = reciprocal_rank_fusion([["a", "b", "c"], ["a", "b", "c"]], k=60)
    scores = [score for _, score in out]
    assert scores == sorted(scores, reverse=True)
    # a (rank1 twice) > b (rank2 twice) > c (rank3 twice).
    assert [cid for cid, _ in out] == ["a", "b", "c"]


def test_k_changes_relative_weighting_but_not_correctness() -> None:
    # With a tiny k, rank gaps matter more; top item still wins. Sanity, not magic numbers.
    out = reciprocal_rank_fusion([["a", "b"]], k=1)
    assert out[0][0] == "a"
    assert out[0][1] == pytest.approx(1.0 / 2.0)
    assert out[1][1] == pytest.approx(1.0 / 3.0)
    assert not math.isnan(out[0][1])
