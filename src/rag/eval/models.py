"""Frozen Pydantic models for the evaluation core (ADR-0004).

These models are the typed contract around the metric + statistics core. They carry
**no logic** beyond validation and deterministic normalization, so a dumped record is
diffable and a stored eval run is reproducible after the fact:

* :class:`GoldenItem` — one labeled golden query. ``relevant_chunk_ids`` is stored as a
  ``tuple`` (not a ``set``/``frozenset``) so the model dump is ordered, deterministic, and
  diffable; the harness converts it to a ``set`` at the metric boundary. A validator forbids
  empty / duplicated relevant ids so a malformed golden row fails loudly at load.
* :class:`QueryMetrics` — per-query, per-config scores (``recall``/``ndcg`` keyed by ``k``).
* :class:`RetrievalMetrics` — the per-config aggregate (mean recall/ndcg by ``k`` plus MRR).
* :class:`BootstrapResult` — the output of a paired percentile bootstrap; it lives here (not
  in ``bootstrap.py``) so ``bootstrap.py`` imports it without a module cycle.
* :class:`ConfigComparison` — one baseline-vs-treatment comparison on a single metric.
* :class:`EvalProvenance` — the reproducibility/publishability header of one harness run
  (embedder class, git SHA, corpus SHA-256, ``k_values`` / ``seed`` / ``B`` / ``n``, and the
  ``publishable`` flag that goes ``False`` the moment a fake backend produced the rankings).
* :class:`EvalReport` — the immutable snapshot a ``make eval`` run dumps to
  ``eval_results.json``: provenance + per-config :class:`RetrievalMetrics` + the
  :class:`ConfigComparison` bootstrap table.

Every model is ``frozen``: an eval record is an immutable snapshot of one run, mirroring the
frozen :class:`~rag.retrieval.models.RetrievalResult`. ``recall``/``ndcg`` dicts are stored
with their ``k`` keys sorted ascending so two runs serialize byte-identically.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# A retrieval metric score is a fraction in [0, 1]; a value outside that range is a harness
# bug (e.g. recall divided by the wrong denominator), so it must fail validation rather than be
# stored as an authoritative-looking number. Applied to the scalar RR/MRR fields and to the
# VALUES of the recall/ndcg dicts in both QueryMetrics and RetrievalMetrics.
Score = Annotated[float, Field(ge=0.0, le=1.0)]


def _sorted_by_key(mapping: dict[int, Score]) -> dict[int, Score]:
    """Return ``mapping`` with integer keys inserted in ascending order (deterministic dump)."""
    return {key: mapping[key] for key in sorted(mapping)}


class GoldenItem(BaseModel):
    """One labeled golden query: the question plus its set of relevant chunk ids.

    ``relevant_chunk_ids`` is the ground-truth set used to score retrieval. It is stored as a
    ``tuple`` for a deterministic, diffable dump; the validator enforces the golden contract
    (non-empty, no duplicates) so a bad label cannot silently skew every metric downstream.
    """

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_chunk_ids: tuple[str, ...]
    reference_answer: str | None = None

    @field_validator("relevant_chunk_ids")
    @classmethod
    def _non_empty_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject an empty or duplicate-bearing relevant set (the golden contract, ADR-0004)."""
        if not value:
            raise ValueError("relevant_chunk_ids must be non-empty")
        if len(set(value)) != len(value):
            raise ValueError("relevant_chunk_ids must not contain duplicates")
        return value


class QueryMetrics(BaseModel):
    """Per-query, per-config metric scores: recall@k / nDCG@k (keyed by k) and RR."""

    model_config = ConfigDict(frozen=True)

    config: str
    query_id: str
    recall: dict[int, Score]
    ndcg: dict[int, Score]
    reciprocal_rank: Score

    @field_validator("recall", "ndcg")
    @classmethod
    def _sort_k_keys(cls, value: dict[int, Score]) -> dict[int, Score]:
        """Insert k -> score pairs in ascending k order for a deterministic dump."""
        return _sorted_by_key(value)


class RetrievalMetrics(BaseModel):
    """Per-config aggregate over the golden set: mean recall@k / nDCG@k (by k) plus MRR."""

    model_config = ConfigDict(frozen=True)

    config: str
    n_queries: int
    k_values: tuple[int, ...]
    recall: dict[int, Score]
    ndcg: dict[int, Score]
    mrr: Score

    @field_validator("k_values")
    @classmethod
    def _sort_k_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Sort the reported cutoffs ascending so the aggregate dumps deterministically."""
        return tuple(sorted(value))

    @field_validator("recall", "ndcg")
    @classmethod
    def _sort_k_keys(cls, value: dict[int, Score]) -> dict[int, Score]:
        """Insert k -> mean-score pairs in ascending k order for a deterministic dump."""
        return _sorted_by_key(value)

    @model_validator(mode="after")
    def _k_values_match_metric_keys(self) -> RetrievalMetrics:
        """Require recall, ndcg, and k_values to describe the SAME set of cutoffs.

        A mismatch (e.g. recall reported at k=1 but k_values=(5, 10)) means the aggregate was
        assembled inconsistently; reporting it would mislabel which cutoff a number belongs to.
        """
        recall_ks = set(self.recall)
        ndcg_ks = set(self.ndcg)
        declared_ks = set(self.k_values)
        if not (recall_ks == ndcg_ks == declared_ks):
            raise ValueError(
                "recall, ndcg, and k_values must cover the same cutoffs; got "
                f"recall={sorted(recall_ks)}, ndcg={sorted(ndcg_ks)}, "
                f"k_values={sorted(declared_ks)}"
            )
        return self


class BootstrapResult(BaseModel):
    """Output of a paired percentile bootstrap on a per-query metric difference.

    Records everything needed to defend and reproduce the interval: the observed means and
    diff, the percentile CI bounds, and the ``ci_level`` / ``n_resamples`` / ``seed`` /
    ``n_queries`` that produced them. ``significant`` is ``True`` iff 0 lies outside the CI.
    """

    model_config = ConfigDict(frozen=True)

    baseline_mean: float
    treatment_mean: float
    diff: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_resamples: int
    seed: int
    n_queries: int
    significant: bool


class ConfigComparison(BaseModel):
    """One baseline-vs-treatment comparison on a single named metric (e.g. ``"recall@5"``)."""

    model_config = ConfigDict(frozen=True)

    baseline: str
    treatment: str
    metric: str
    bootstrap: BootstrapResult


class EvalProvenance(BaseModel):
    """Reproducibility + publishability header for one harness run.

    Carries everything a skeptical reader needs to (a) reproduce the bootstrap intervals
    (``seed`` / ``n_resamples`` and the ``numpy`` version inside ``library_versions``) and
    (b) judge whether the numbers may be published. ``publishable`` is ``False`` whenever a
    fake/offline backend (e.g. :class:`~rag.indexing.embeddings.HashingEmbedder`) produced the
    rankings, so an offline/test run can never be mistaken for a real benchmark. ``n_queries``
    is recorded next to every interval downstream so a tight CI on a small ``n`` is never
    oversold.
    """

    model_config = ConfigDict(frozen=True)

    embedder_class: str
    reranker_class: str
    git_sha: str | None
    corpus_sha256: str
    corpus_dir: str
    library_versions: dict[str, str | None]
    k_values: tuple[int, ...]
    k_retrieve: int
    seed: int
    n_resamples: int
    n_queries: int
    headline_metric: str
    secondary_metric: str
    publishable: bool

    @field_validator("k_values")
    @classmethod
    def _sort_k_values(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Sort the reported cutoffs ascending so the provenance dumps deterministically."""
        return tuple(sorted(value))


class EvalReport(BaseModel):
    """Immutable snapshot of one full eval run: provenance + per-config metrics + comparisons.

    This is the object a ``make eval`` run dumps (sorted keys) to ``eval_results.json`` as the
    byte-diffable, reproducible artifact. ``configs`` and ``comparisons`` are stored as tuples
    whose element order is meaningful (the harness fixes the config order and the comparison
    order) and is preserved across dumps, so two reproducible runs serialize byte-identically.
    """

    model_config = ConfigDict(frozen=True)

    provenance: EvalProvenance
    configs: tuple[RetrievalMetrics, ...]
    comparisons: tuple[ConfigComparison, ...]


# --- Attribution-rate aggregation (ADR-0006) ---------------------------------------------------
# These models are the typed contract for the golden-set aggregate of the MEASURED per-answer
# attribution_rate produced by ``rag.verification.citations.verify_answer`` (grounded citations /
# total citations). They deliberately mirror the retrieval-eval models' discipline (frozen,
# diffable, publishable-flag) but are a SEPARATE, dedicated shape — an attribution run is a
# single-config LLM measurement with no bootstrap CI (LLM generation is not bit-exact
# reproducible), so reusing EvalProvenance/EvalReport would carry irrelevant seed/B/comparison
# fields and drop the LLM identity that makes an attribution number defensible.


class AttributionQueryRecord(BaseModel):
    """One golden query's attribution outcome (the per-query distribution the report exposes).

    ``attribution_rate`` is the MEASURED per-answer rate from ``verify_answer`` (``n_grounded /
    n_citations``, or ``0.0`` when there are no citations). ``abstained`` is exactly
    ``n_citations == 0``: it separates "the model cited nothing" from "the model cited but the
    span did not ground" (which scores ``0.0`` *with* citations), so a reader can never confuse an
    abstention with a grounding failure.
    """

    model_config = ConfigDict(frozen=True)

    query_id: str
    n_citations: int = Field(ge=0)
    n_grounded: int = Field(ge=0)
    attribution_rate: Score
    abstained: bool


class AttributionProvenance(BaseModel):
    """Reproducibility + publishability header for one attribution run.

    Records the LLM identity (``llm_class`` / ``llm_model``) alongside the retrieval backends and
    the corpus fingerprint, because an attribution number is only defensible if the reader knows
    which generator and which real backends produced it. ``publishable`` is ``True`` only when the
    LLM, embedder, AND reranker are all the real classes — any fake flips it to ``False`` so an
    offline/test run can never be mistaken for a benchmark. ``single_run`` records that this is a
    point estimate with no bootstrap CI (LLM generation is not bit-exact reproducible, so an
    intra-run bootstrap would manufacture false precision — see ADR-0006).
    """

    model_config = ConfigDict(frozen=True)

    llm_class: str
    llm_model: str
    embedder_class: str
    reranker_class: str
    top_k_rerank: int
    git_sha: str | None
    corpus_sha256: str
    corpus_dir: str
    library_versions: dict[str, str | None]
    n_queries: int
    single_run: bool = True
    publishable: bool


class AttributionReport(BaseModel):
    """Immutable snapshot of one attribution run over the golden set (ADR-0006).

    The HEADLINE is ``micro_attribution_rate`` — pooled ``total_grounded / total_citations`` —
    which is immune to the 0-citation → 0.0 convention (an abstaining query contributes no
    citations to either pool rather than a hard 0.0). ``macro_attribution_rate`` (mean of the
    per-query rates, abstentions counted as 0.0) is reported as SECONDARY, together with
    ``macro_attribution_rate_answered`` (macro over answered queries only) and ``n_abstained`` so
    the effect of abstentions is always visible and macro-only reporting — which conflates "cited
    but wrong" with "abstained" — is never the sole number. This measures GROUNDING (are cited
    spans supported), not correctness or completeness of the answer.
    """

    model_config = ConfigDict(frozen=True)

    provenance: AttributionProvenance
    config: str
    n_queries: int
    total_citations: int
    total_grounded: int
    micro_attribution_rate: Score
    macro_attribution_rate: Score
    macro_attribution_rate_answered: Score
    n_answered: int
    n_abstained: int
    per_query: tuple[AttributionQueryRecord, ...]
