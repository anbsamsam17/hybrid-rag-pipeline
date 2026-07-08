"""Evaluation core + retrieval/attribution harnesses (ADR-0004/0005/0006).

This package owns the repo's headline-credibility math. It exposes the **metrics + statistics
core** (ADR-0004), the **retrieval evaluation harness** (ADR-0005), the **attribution-rate
aggregation** (ADR-0006) that turns the measured per-answer ``attribution_rate`` into a
golden-set aggregate, the **corrective-vs-baseline eval** (ADR-0008), and the **RAGAS-style
generation-quality eval** (ADR-0009: faithfulness + answer_relevancy, reimplemented over the
Anthropic SDK with RAGAS credited as the spec).

Public surface:

* Metrics (pure, backend-agnostic): :func:`~rag.eval.metrics.recall_at_k`,
  :func:`~rag.eval.metrics.ndcg_at_k`, :func:`~rag.eval.metrics.reciprocal_rank`,
  :func:`~rag.eval.metrics.mrr`.
* Statistics: :func:`~rag.eval.bootstrap.paired_bootstrap_ci`.
* Golden set: :func:`~rag.eval.golden.load_golden`.
* Harnesses: :func:`~rag.eval.harness.run_eval` (retrieval; ``python -m rag.eval.harness``),
  :func:`~rag.eval.attribution.run_attribution_eval` (attribution;
  ``python -m rag.eval.attribution``), and :func:`~rag.eval.corrective.run_corrective_eval`
  (corrective-vs-baseline; ``python -m rag.eval.corrective``), all sharing the hermetic build via
  :func:`~rag.eval.harness.prepare_hermetic_eval`.
* Correctness judge (ADR-0008): :class:`~rag.eval.judge.AnswerCorrectnessJudge` Protocol +
  :class:`~rag.eval.judge.AnthropicAnswerCorrectnessJudge` /
  :class:`~rag.eval.judge.FakeAnswerCorrectnessJudge` + :func:`~rag.eval.judge.lexical_f1`.
* Generation-quality scorers (ADR-0009): :class:`~rag.eval.generation_scorers.FaithfulnessScorer`
  / :class:`~rag.eval.generation_scorers.AnswerRelevancyScorer` Protocols + their ``Anthropic*`` /
  ``Fake*`` implementations, and the orchestrator
  :func:`~rag.eval.generation_quality.run_generation_quality_eval`
  (``python -m rag.eval.generation_quality``).
* Frozen models: :class:`~rag.eval.models.GoldenItem`,
  :class:`~rag.eval.models.QueryMetrics`, :class:`~rag.eval.models.RetrievalMetrics`,
  :class:`~rag.eval.models.BootstrapResult`, :class:`~rag.eval.models.ConfigComparison`,
  :class:`~rag.eval.models.EvalProvenance`, :class:`~rag.eval.models.EvalReport`,
  :class:`~rag.eval.models.AttributionQueryRecord`,
  :class:`~rag.eval.models.AttributionProvenance`, :class:`~rag.eval.models.AttributionReport`,
  :class:`~rag.eval.models.CorrectiveQueryRecord`,
  :class:`~rag.eval.models.CorrectiveEvalProvenance`,
  :class:`~rag.eval.models.CorrectiveEvalReport`.

The dependency arrow is one-directional: ``eval`` imports ``retrieval``/``indexing``/
``generation``/``verification``, but nothing in the repo imports ``eval``, and the
metrics/bootstrap modules import neither.
"""

from __future__ import annotations

from rag.eval.attribution import run_attribution_eval
from rag.eval.bootstrap import paired_bootstrap_ci
from rag.eval.corrective import answer_once, run_corrective_eval
from rag.eval.generation_quality import run_generation_quality_eval
from rag.eval.generation_scorers import (
    AnswerRelevancyResult,
    AnswerRelevancyScorer,
    AnthropicAnswerRelevancyScorer,
    AnthropicFaithfulnessScorer,
    FaithfulnessResult,
    FaithfulnessScorer,
    FakeAnswerRelevancyScorer,
    FakeFaithfulnessScorer,
    StatementVerdict,
)
from rag.eval.golden import load_golden
from rag.eval.harness import run_eval
from rag.eval.judge import (
    AnswerCorrectnessJudge,
    AnthropicAnswerCorrectnessJudge,
    CorrectnessVerdict,
    FakeAnswerCorrectnessJudge,
    lexical_f1,
)
from rag.eval.metrics import mrr, ndcg_at_k, recall_at_k, reciprocal_rank
from rag.eval.models import (
    AttributionProvenance,
    AttributionQueryRecord,
    AttributionReport,
    BootstrapResult,
    ConfigComparison,
    CorrectiveEvalProvenance,
    CorrectiveEvalReport,
    CorrectiveQueryRecord,
    EvalProvenance,
    EvalReport,
    GenerationQualityProvenance,
    GenerationQualityQueryRecord,
    GenerationQualityReport,
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
    "run_corrective_eval",
    "run_generation_quality_eval",
    "answer_once",
    # correctness judge (ADR-0008)
    "AnswerCorrectnessJudge",
    "AnthropicAnswerCorrectnessJudge",
    "FakeAnswerCorrectnessJudge",
    "CorrectnessVerdict",
    "lexical_f1",
    # generation-quality scorers (ADR-0009)
    "FaithfulnessScorer",
    "AnthropicFaithfulnessScorer",
    "FakeFaithfulnessScorer",
    "FaithfulnessResult",
    "StatementVerdict",
    "AnswerRelevancyScorer",
    "AnthropicAnswerRelevancyScorer",
    "FakeAnswerRelevancyScorer",
    "AnswerRelevancyResult",
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
    "CorrectiveQueryRecord",
    "CorrectiveEvalProvenance",
    "CorrectiveEvalReport",
    "GenerationQualityQueryRecord",
    "GenerationQualityProvenance",
    "GenerationQualityReport",
]
