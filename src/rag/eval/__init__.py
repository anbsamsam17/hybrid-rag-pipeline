"""Evaluation core + retrieval/attribution harnesses (ADR-0004/0005/0006).

This package owns the repo's headline-credibility math. It exposes the **metrics + statistics
core** (ADR-0004), the **retrieval evaluation harness** (ADR-0005), and the **attribution-rate
aggregation** (ADR-0006) that turns the measured per-answer ``attribution_rate`` into a
golden-set aggregate. RAGAS faithfulness/answer-relevance still require an LLM judge and land in
a later increment; they are intentionally absent here.

Public surface:

* Metrics (pure, backend-agnostic): :func:`~rag.eval.metrics.recall_at_k`,
  :func:`~rag.eval.metrics.ndcg_at_k`, :func:`~rag.eval.metrics.reciprocal_rank`,
  :func:`~rag.eval.metrics.mrr`.
* Statistics: :func:`~rag.eval.bootstrap.paired_bootstrap_ci`.
* Golden set: :func:`~rag.eval.golden.load_golden`.
* Harnesses: :func:`~rag.eval.harness.run_eval` (retrieval; ``python -m rag.eval.harness``) and
  :func:`~rag.eval.attribution.run_attribution_eval` (attribution;
  ``python -m rag.eval.attribution``), sharing the hermetic build via
  :func:`~rag.eval.harness.prepare_hermetic_eval`.
* Frozen models: :class:`~rag.eval.models.GoldenItem`,
  :class:`~rag.eval.models.QueryMetrics`, :class:`~rag.eval.models.RetrievalMetrics`,
  :class:`~rag.eval.models.BootstrapResult`, :class:`~rag.eval.models.ConfigComparison`,
  :class:`~rag.eval.models.EvalProvenance`, :class:`~rag.eval.models.EvalReport`,
  :class:`~rag.eval.models.AttributionQueryRecord`,
  :class:`~rag.eval.models.AttributionProvenance`, :class:`~rag.eval.models.AttributionReport`.

The dependency arrow is one-directional: ``eval`` imports ``retrieval``/``indexing``/
``generation``/``verification``, but nothing in the repo imports ``eval``, and the
metrics/bootstrap modules import neither.
"""

from __future__ import annotations

from rag.eval.attribution import run_attribution_eval
from rag.eval.bootstrap import paired_bootstrap_ci
from rag.eval.golden import load_golden
from rag.eval.harness import run_eval
from rag.eval.metrics import mrr, ndcg_at_k, recall_at_k, reciprocal_rank
from rag.eval.models import (
    AttributionProvenance,
    AttributionQueryRecord,
    AttributionReport,
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
    # golden set + harnesses
    "load_golden",
    "run_eval",
    "run_attribution_eval",
    # models
    "GoldenItem",
    "QueryMetrics",
    "RetrievalMetrics",
    "BootstrapResult",
    "ConfigComparison",
    "EvalProvenance",
    "EvalReport",
    "AttributionQueryRecord",
    "AttributionProvenance",
    "AttributionReport",
]
