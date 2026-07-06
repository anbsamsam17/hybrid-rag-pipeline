"""Tests for the frozen eval models' validation gardes (ADR-0004).

The models are the typed contract the harness will fill, so a malformed record must fail at
construction rather than dump an authoritative-looking but wrong number. These tests pin:

* metric scores constrained to ``[0, 1]`` (a value outside that range is a harness bug),
* ``RetrievalMetrics`` cutoffs consistent across ``recall`` / ``ndcg`` / ``k_values``,
* deterministic ascending-by-k ordering of the metric dicts,
* the ``GoldenItem`` contract (non-empty / unique relevant ids, non-empty id and query text).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.eval.models import GoldenItem, QueryMetrics, RetrievalMetrics


def test_golden_item_relevant_must_be_non_empty() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        GoldenItem(query_id="q1", query="why?", relevant_chunk_ids=())


def test_golden_item_relevant_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        GoldenItem(query_id="q1", query="why?", relevant_chunk_ids=("c1", "c1"))


@pytest.mark.parametrize("field", ["query_id", "query"])
def test_golden_item_rejects_empty_id_or_query(field) -> None:
    kwargs = {"query_id": "q1", "query": "why?", "relevant_chunk_ids": ("c1",)}
    kwargs[field] = ""
    with pytest.raises(ValidationError):
        GoldenItem(**kwargs)


def test_golden_item_stores_relevant_as_tuple() -> None:
    item = GoldenItem(query_id="q1", query="why?", relevant_chunk_ids=("c1", "c2"))
    assert item.relevant_chunk_ids == ("c1", "c2")
    assert isinstance(item.relevant_chunk_ids, tuple)


def test_query_metrics_rejects_recall_above_one() -> None:
    with pytest.raises(ValidationError):
        QueryMetrics(
            config="hybrid", query_id="q1", recall={1: 1.7}, ndcg={1: 0.5}, reciprocal_rank=0.5
        )


def test_query_metrics_rejects_reciprocal_rank_above_one() -> None:
    with pytest.raises(ValidationError):
        QueryMetrics(
            config="hybrid", query_id="q1", recall={1: 0.5}, ndcg={1: 0.5}, reciprocal_rank=9.0
        )


def test_query_metrics_sorts_k_keys_ascending() -> None:
    qm = QueryMetrics(
        config="hybrid",
        query_id="q1",
        recall={10: 0.8, 1: 0.2, 5: 0.6},
        ndcg={5: 0.5, 1: 0.1},
        reciprocal_rank=0.5,
    )
    assert list(qm.recall) == [1, 5, 10]
    assert list(qm.ndcg) == [1, 5]


def test_retrieval_metrics_rejects_mrr_above_one() -> None:
    with pytest.raises(ValidationError):
        RetrievalMetrics(
            config="hybrid",
            n_queries=3,
            k_values=(1,),
            recall={1: 0.5},
            ndcg={1: 0.5},
            mrr=2.0,
        )


def test_retrieval_metrics_rejects_cutoff_mismatch() -> None:
    # recall is reported at k=1 but k_values declares (5, 10): inconsistent, must fail.
    with pytest.raises(ValidationError, match="same cutoffs"):
        RetrievalMetrics(
            config="hybrid",
            n_queries=3,
            k_values=(5, 10),
            recall={1: 0.5},
            ndcg={1: 0.5},
            mrr=0.5,
        )


def test_retrieval_metrics_valid_aligned_cutoffs() -> None:
    rm = RetrievalMetrics(
        config="hybrid",
        n_queries=3,
        k_values=(5, 1),
        recall={5: 0.6, 1: 0.2},
        ndcg={1: 0.1, 5: 0.5},
        mrr=0.4,
    )
    # k_values sorted ascending; dict keys sorted ascending; sets agree.
    assert rm.k_values == (1, 5)
    assert list(rm.recall) == [1, 5]
    assert set(rm.recall) == set(rm.ndcg) == set(rm.k_values)
