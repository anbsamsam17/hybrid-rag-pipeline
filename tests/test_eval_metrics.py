"""Tests for the pure retrieval metrics (recall@k, nDCG@k, reciprocal rank, MRR).

Same rigor as ``test_fusion.py``: every expected value is computed by hand in the test so a
wrong-but-plausible definition fails loudly rather than shipping. The dangerous bugs these
target explicitly:

* recall capped to ``min(|relevant|, k)`` (inflates 1/3 → 1.0),
* an off-by-one in the nDCG ``log2`` discount (or a bare ``log2(i)`` that divides by zero),
* duplicates inflating a score or wasting a top-k slot,
* MRR averaging that drops misses instead of counting them as 0,
* an edge case leaking NaN/inf instead of a finite 0.0.
"""

from __future__ import annotations

import math

import pytest

from rag.eval.metrics import mrr, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_denominator_is_total_relevant_not_min_with_k() -> None:
    # 3 relevant docs, exactly one hit at rank 1, k=1. The honest answer is 1/3, NOT 1.0:
    # using min(|relevant|, k) as the denominator would silently inflate this to 1.0.
    ranked = ["r1", "x", "y"]
    relevant = {"r1", "r2", "r3"}
    assert recall_at_k(ranked, relevant, k=1) == pytest.approx(1.0 / 3.0)


def test_ndcg_and_recall_are_distinct_conventions_same_fixture() -> None:
    # |relevant| = 5, the top 3 retrieved are all relevant, k=3.
    # nDCG@3 == 1.0 (IDCG capped at min(5, 3) = 3 → a perfect top-3 scores 1.0)
    # recall@3 == 3/5 = 0.6 (denominator is the full relevant set).
    # Proving both on one fixture shows the two conventions are deliberately different.
    ranked = ["r1", "r2", "r3", "z"]
    relevant = {"r1", "r2", "r3", "r4", "r5"}
    assert ndcg_at_k(ranked, relevant, k=3) == pytest.approx(1.0)
    assert recall_at_k(ranked, relevant, k=3) == pytest.approx(0.6)


def test_dedup_keeps_first_occurrence_before_cutoff_recall() -> None:
    # ["a","a","b"] dedups to ["a","b"]; top-2 contains the relevant "b" → recall 1.0.
    # Cut-then-dedup would give ["a"] → 0.0, which is wrong (the duplicate must not eat a slot).
    assert recall_at_k(["a", "a", "b"], {"b"}, k=2) == pytest.approx(1.0)


def test_duplicate_relevant_id_does_not_double_ndcg_gain() -> None:
    # "b" is relevant and appears twice; after dedup it is counted once at rank 1.
    # DCG = 1/log2(2) = 1, IDCG (|relevant|=1) = 1 → nDCG exactly 1.0, never > 1.0.
    # Without dedup, b at ranks 1 and 2 would give DCG = 1 + 1/log2(3) ≈ 1.63 → inflated nDCG.
    assert ndcg_at_k(["b", "b", "c"], {"b"}, k=3) == pytest.approx(1.0)


def test_ndcg_dedup_frees_a_top_k_slot() -> None:
    # ["x","x","r"] with k=2: dedup-THEN-cut gives ["x","r"], so the relevant "r" lands at
    # rank 2 and inside the top-2 → nDCG@2 = (1/log2(3)) / (1/log2(2)) > 0. Cut-THEN-dedup
    # would give ["x"], dropping "r" → nDCG 0. Pins the pre-step ordering directly on nDCG.
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    result = ndcg_at_k(["x", "x", "r"], {"r"}, k=2)
    assert result == pytest.approx(expected)
    assert result > 0.0


def test_ndcg_rank_one_discount_is_log2_of_two() -> None:
    # A single relevant doc at rank 1: numerator term is 1/log2(1+1) = 1/log2(2) = 1.0.
    # IDCG is also 1.0 so nDCG == 1.0. This pins the "+1" in the discount (no divide-by-zero).
    assert ndcg_at_k(["r1", "x"], {"r1"}, k=2) == pytest.approx(1.0)


def test_ndcg_hand_computed_mixed_ranking() -> None:
    # ranked = [r1, x, r2], relevant = {r1, r2}, k=3.
    # DCG  = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3)      (ideal: two relevant at ranks 1,2)
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = dcg / idcg
    assert ndcg_at_k(["r1", "x", "r2"], {"r1", "r2"}, k=3) == pytest.approx(expected)


def test_reciprocal_rank_first_relevant_position() -> None:
    # First relevant at rank 1 → RR 1.0; at rank 2 → RR 0.5.
    assert reciprocal_rank(["r1", "x"], {"r1"}) == pytest.approx(1.0)
    assert reciprocal_rank(["x", "r1"], {"r1"}) == pytest.approx(0.5)


def test_reciprocal_rank_uses_dedup_pre_step() -> None:
    # ["x","x","r"] dedups to ["x","r"] → first relevant at rank 2 → RR 0.5 (not 1/3).
    assert reciprocal_rank(["x", "x", "r"], {"r"}) == pytest.approx(0.5)


def test_reciprocal_rank_no_relevant_is_zero() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_mrr_counts_misses_as_zero() -> None:
    # A miss contributes 0.0 and is NOT dropped: mean of [1.0, 0.5, 0.0] = 0.5.
    assert mrr([1.0, 0.5, 0.0]) == pytest.approx(0.5)


def test_mrr_empty_is_zero() -> None:
    assert mrr([]) == 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_mrr_rejects_nonfinite(bad) -> None:
    # A NaN/inf reciprocal rank would poison the mean (mean with NaN is NaN); reject it.
    with pytest.raises(ValueError, match="finite"):
        mrr([0.5, bad])


def test_k_greater_than_len_scores_present_without_padding() -> None:
    # k exceeds the list length: score what is present, no padding, no NaN.
    ranked = ["a", "b"]
    relevant = {"a"}
    assert recall_at_k(ranked, relevant, k=5) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, relevant, k=5) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
def test_empty_relevant_returns_zero_not_nan(metric) -> None:
    value = metric(["a", "b"], set(), k=3)
    assert value == 0.0
    assert not math.isnan(value)


@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
def test_empty_ranked_returns_zero_not_nan(metric) -> None:
    value = metric([], {"a"}, k=3)
    assert value == 0.0
    assert not math.isnan(value)


def test_empty_inputs_reciprocal_rank() -> None:
    assert reciprocal_rank([], {"a"}) == 0.0
    assert reciprocal_rank(["a"], set()) == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
@pytest.mark.parametrize("bad_k", [0, -1, -10])
def test_non_positive_k_raises(metric, bad_k) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        metric(["a", "b"], {"a"}, k=bad_k)
