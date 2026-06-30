"""Evaluation core + retrieval harness (ADR-0004 metrics/bootstrap, ADR-0005 harness).

This package owns the repo's headline-credibility math. It exposes the **metrics + statistics
core** (ADR-0004) and the **retrieval evaluation harness** (ADR-0005) that consumes them.
RAGAS and the ``attribution_rate`` aggregation require an LLM judge and land in a later
increment; they are intentionally absent here.

Public surface:

* Metrics (pure, backend-agnostic): :func:`~rag.eval.metrics.recall_at_k`,
  :func:`~rag.eval.metrics.ndcg_at_k`, :func:`~rag.eval.metrics.reciprocal_rank`,
  :func:`~rag.eval.metrics.mrr`.
* Statistics: :func:`~rag.eval.bootstrap.paired_bootstrap_ci`.
* Golden set: :func:`~rag.eval.golden.load_golden`.
* Harness: :func:`~rag.eval.harness.run_eval` (and ``python -m rag.eval.harness``).
* Frozen models: :class:`~rag.eval.models.GoldenItem`,
  :class:`~rag.eval.models.QueryMetrics`, :class:`~rag.eval.models.RetrievalMetrics`,
  :class:`~rag.eval.models.BootstrapResult`, :class:`~rag.eval.models.ConfigComparison`,
  :class:`~rag.eval.models.EvalProvenance`, :class:`~rag.eval.models.EvalReport`.

The dependency arrow is one-directional: ``eval`` imports ``retrieval``/``indexing``, but
nothing in the repo imports ``eval``, and the metrics/bootstrap modules import neither.
"""

from __future__ import annotations

from rag.eval.bootstrap import paired_bootstrap_ci
from rag.eval.golden import load_golden
from rag.eval.harness import run_eval
from rag.eval.metrics import mrr, ndcg_at_k, recall_at_k, reciprocal_rank
from rag.eval.models import (
    BootstrapResult,
    ConfigComparison,
    EvalProvenance,
    EvalReport,
    GoldenItem,
    QueryMetrics,
    RetrievalMetrics,
)

__all__ = [
    # metrics
    "recall_at_k",
    "ndcg_at_k",
    "reciprocal_rank",
    "mrr",
    # statistics
    "paired_bootstrap_ci",
    # golden set + harness
    "load_golden",
    "run_eval",
    # models
    "GoldenItem",
    "QueryMetrics",
    "RetrievalMetrics",
    "BootstrapResult",
    "ConfigComparison",
    "EvalProvenance",
    "EvalReport",
]
