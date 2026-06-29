"""Evaluation core: pure retrieval metrics + paired percentile bootstrap (ADR-0004).

This package owns the repo's headline-credibility math. This first increment exposes only the
**metrics + statistics core** — the harness, golden-set loading, RAGAS, and the
``attribution_rate`` aggregation land in a later increment and are intentionally absent here.

Public surface:

* Metrics (pure, backend-agnostic): :func:`~rag.eval.metrics.recall_at_k`,
  :func:`~rag.eval.metrics.ndcg_at_k`, :func:`~rag.eval.metrics.reciprocal_rank`,
  :func:`~rag.eval.metrics.mrr`.
* Statistics: :func:`~rag.eval.bootstrap.paired_bootstrap_ci`.
* Frozen models: :class:`~rag.eval.models.GoldenItem`,
  :class:`~rag.eval.models.QueryMetrics`, :class:`~rag.eval.models.RetrievalMetrics`,
  :class:`~rag.eval.models.BootstrapResult`, :class:`~rag.eval.models.ConfigComparison`.

The dependency arrow is one-directional: ``eval`` may later import ``retrieval``/``verification``,
but nothing in the repo imports ``eval``, and the metrics/bootstrap modules import neither.
"""

from __future__ import annotations

from rag.eval.bootstrap import paired_bootstrap_ci
from rag.eval.metrics import mrr, ndcg_at_k, recall_at_k, reciprocal_rank
from rag.eval.models import (
    BootstrapResult,
    ConfigComparison,
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
    # models
    "GoldenItem",
    "QueryMetrics",
    "RetrievalMetrics",
    "BootstrapResult",
    "ConfigComparison",
]
