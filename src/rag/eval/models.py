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


# --- Corrective-vs-baseline evaluation (ADR-0008) ----------------------------------------------
# The typed contract for the paired corrective-vs-baseline comparison over the golden set. The
# PRIMARY, pre-registered endpoint is the trace-only ``activation_rate`` (does the corrective loop
# fire at all?), judge-free and byte-stable under fakes; every SECONDARY endpoint (answer
# correctness, attribution regression guard, recall-on-final-contexts, cost) is directional and
# expected ~0 delta on this corpus. Like Attribution*, this is a SINGLE-config LLM measurement
# with no bootstrap CI (generation AND the LLM judge are non-reproducible), so ``single_run=True``
# and ``publishable`` gates on ALL real backends INCLUDING the judge.


class CorrectiveQueryRecord(BaseModel):
    """One golden query's corrective-vs-baseline outcome (the per-query distribution).

    Carries the trace-derived PRIMARY signals and the paired SECONDARY per-arm measurements.
    ``activated`` measures ONLY whether the corrective RETRY LOOP fired (``n_rewrites > 0`` or
    ``n_regenerations > 0``) — it is NOT "the layer did something", because the grade node can
    still FILTER the context set even when no retry fires. ``contexts_identical`` closes that gap:
    it is ``True`` iff the two arms generated over the exact same final context list (same
    chunk_ids in the same order); when ``activated`` is ``False`` but ``contexts_identical`` is
    also ``False``, the retry loop stayed idle yet grading still changed what generation saw, so
    the arms are NOT equivalent (``baseline_n_contexts`` vs ``corrective_n_contexts`` shows the
    filtering). ``final_query_changed`` is the EFFECTIVE rewrite signal (a rewrite actually altered
    the query text), reported separately because ``n_rewrites`` increments even on an unchanged
    reformulation. ``*_correct`` is ``None`` for a query with no ``reference_answer`` (excluded from
    the correctness rate and NOT counted in ``n_judged``).
    """

    model_config = ConfigDict(frozen=True)

    query_id: str
    activated: bool
    n_rewrites: int = Field(ge=0)
    n_regenerations: int = Field(ge=0)
    final_query_changed: bool
    terminated_reason: str
    rewrite_budget_exhausted: bool
    extra_llm_calls: int = Field(ge=0)
    contexts_identical: bool
    baseline_n_contexts: int = Field(ge=0)
    corrective_n_contexts: int = Field(ge=0)
    baseline_attr_rate: Score
    corrective_attr_rate: Score
    baseline_recall: dict[int, Score]
    corrective_recall: dict[int, Score]
    baseline_correct: bool | None
    corrective_correct: bool | None
    baseline_lexical_f1: Score
    corrective_lexical_f1: Score

    @field_validator("baseline_recall", "corrective_recall")
    @classmethod
    def _sort_k_keys(cls, value: dict[int, Score]) -> dict[int, Score]:
        """Insert k -> recall pairs in ascending k order for a deterministic dump."""
        return _sorted_by_key(value)


class CorrectiveEvalProvenance(BaseModel):
    """Reproducibility + publishability header for one corrective-vs-baseline run.

    Records the identity of BOTH LLM roles (generation ``baseline_llm_class`` and the corrective
    ``corrective_llm_class``) PLUS the correctness ``judge_class`` alongside the retrieval backends
    and corpus fingerprint, because a corrective-vs-baseline number is only defensible if a reader
    knows which generator, which corrective controller, and which judge produced it. The agentic
    budgets are recorded verbatim so the activation/cost accounting is reproducible. ``publishable``
    is ``True`` ONLY when the generation LLM, corrective LLM, judge, embedder, AND reranker are all
    the real classes — any fake flips it ``False``. ``single_run`` records the no-CI discipline
    (generation and the judge are non-reproducible; an intra-run bootstrap = false precision).
    """

    model_config = ConfigDict(frozen=True)

    baseline_llm_class: str
    corrective_llm_class: str
    judge_class: str
    embedder_class: str
    reranker_class: str
    llm_model: str
    judge_model: str
    top_k_rerank: int
    agentic_max_query_rewrites: int
    agentic_max_regenerations: int
    agentic_min_relevant_docs: int
    agentic_min_attribution_rate: float
    git_sha: str | None
    corpus_sha256: str
    corpus_dir: str
    library_versions: dict[str, str | None]
    n_queries: int
    n_judged: int
    single_run: bool = True
    publishable: bool


class CorrectiveEvalReport(BaseModel):
    """Immutable snapshot of one corrective-vs-baseline run over the golden set (ADR-0008).

    The HEADLINE is the pre-registered PRIMARY ``activation_rate`` — the fraction of queries where
    the corrective RETRY LOOP fired (a rewrite or a regeneration) — reported with its
    rewrite/regenerate/effective splits and the ``terminated_reason`` histogram. It is judge-free
    and byte-stable under fakes; on this corpus the honest expectation is ~0. ``activation_rate``
    alone does NOT prove a no-op, because the grade node can still filter the context set with the
    retry loop idle: ``contexts_identical_rate`` (also judge-free) reports the fraction of queries
    where both arms generated over the EXACT same final contexts. A TRUE no-op requires
    ``activation_rate == 0`` AND ``contexts_identical_rate == 1``; otherwise the honest framing is
    "the retry loop never fired, but grading still filtered contexts on N queries — see the recall/
    attribution/correctness deltas for the effect." Everything else is SECONDARY and directional:
    the cost (mean ``extra_llm_calls``), the attribution regression guard (``corrective_micro_attr``
    must NOT be below ``baseline_micro_attr``), recall-on-final-contexts per arm, and answer
    correctness (LLM-judge rate + deterministic lexical-F1 floor). At ~0 activation any correctness
    delta is generator+judge NOISE, never a win — this is stated in the rendered report.
    """

    model_config = ConfigDict(frozen=True)

    provenance: CorrectiveEvalProvenance
    config: str
    n_queries: int
    # PRIMARY (pre-registered, confirmatory) — retry-loop activation rate.
    activation_rate: Score
    n_activated: int
    n_rewrite_activated: int
    n_regenerate_activated: int
    n_final_query_changed: int
    terminated_reason_counts: dict[str, int]
    # PRIMARY (judge-free) — did grading change the context set even with the retry loop idle?
    n_contexts_identical: int
    contexts_identical_rate: Score
    # SECONDARY (directional, expected ~0 delta) — cost, attribution, recall, correctness.
    mean_extra_llm_calls: float
    baseline_micro_attr: Score
    corrective_micro_attr: Score
    attr_delta: float
    baseline_recall_mean: dict[int, Score]
    corrective_recall_mean: dict[int, Score]
    baseline_correctness_rate: Score
    corrective_correctness_rate: Score
    correctness_delta: float
    baseline_lexical_f1_mean: Score
    corrective_lexical_f1_mean: Score
    n_judged: int
    per_query: tuple[CorrectiveQueryRecord, ...]

    @field_validator("baseline_recall_mean", "corrective_recall_mean")
    @classmethod
    def _sort_k_keys(cls, value: dict[int, Score]) -> dict[int, Score]:
        """Insert k -> mean-recall pairs in ascending k order for a deterministic dump."""
        return _sorted_by_key(value)


# --- RAGAS-style generation quality (ADR-0009) -------------------------------------------------
# The typed contract for the golden-set aggregate of the two RAGAS-style generation-quality
# metrics reimplemented over the Anthropic SDK (RAGAS credited as the SPEC; these are NOT the
# canonical RAGAS library's output). Like Attribution*/Corrective*, this is a SINGLE-config LLM
# measurement with no bootstrap CI (statement-decomposition / NLI / question-generation are
# non-reproducible), so ``single_run=True`` and ``publishable`` gates on ALL real backends
# INCLUDING both scorers. faithfulness (all-claim grounding vs the full retrieved context) and
# answer_relevancy (does the answer address the question) measure DISTINCT things from each other
# and from ``attribution_rate`` (grounding of the citations the model MADE) — they are never
# summed or presented as one improving the other.
#
# ``Relevancy`` is deliberately NOT the ``Score`` [0, 1] type: cosine of embedded questions is
# theoretically in ``[-1, 1]``, and clamping a genuinely divergent answer to 0 would hide signal
# (ADR-0009 decision f). The noncommittal gate still forces an evasive answer to EXACTLY 0.0.
Relevancy = Annotated[float, Field(ge=-1.0, le=1.0)]


class GenerationQualityQueryRecord(BaseModel):
    """One golden query's generation-quality outcome (the per-query distribution the report shows).

    Carries both scored blocks for the SAME generated answer over the SAME retrieved contexts.
    ``n_statements`` is recorded per query so a suspicious ``n_statements == 1`` (a
    one-giant-statement decomposition that can mask unsupported sub-claims) is visible in the
    artifact rather than hidden inside a saturated aggregate. ``faith_abstained`` is exactly
    ``n_statements == 0`` (the answer made no factual claims to ground — EXCLUDED from the micro
    pool). ``noncommittal`` marks a refusal/evasive answer whose ``relevancy`` is forced to exactly
    ``0.0`` and which is INCLUDED as 0 in the answer-relevancy headline (evasiveness SHOULD score
    low). The two abstention conventions are deliberately asymmetric (ADR-0009 decision d).
    """

    model_config = ConfigDict(frozen=True)

    query_id: str
    # Faithfulness block (all-claim grounding vs the full retrieved context; LLM NLI).
    n_statements: int = Field(ge=0)
    n_supported: int = Field(ge=0)
    faithfulness: Score
    faith_abstained: bool
    # Answer-relevancy block (does the answer address the question; embedding cosine over
    # LLM-generated questions; needs NO ground truth).
    relevancy: Relevancy
    noncommittal: bool
    n_generated_questions: int = Field(ge=0)

    @model_validator(mode="after")
    def _supported_within_statements(self) -> GenerationQualityQueryRecord:
        """``n_supported`` can never exceed ``n_statements`` (a numerator > denominator bug)."""
        if self.n_supported > self.n_statements:
            raise ValueError(
                f"n_supported ({self.n_supported}) exceeds n_statements ({self.n_statements})"
            )
        return self


class GenerationQualityProvenance(BaseModel):
    """Reproducibility + publishability header for one generation-quality run (ADR-0009).

    Records the generator identity (``llm_class`` / ``llm_model``), BOTH scorer classes plus the
    resolved ``scorer_model`` (a model DIFFERENT from the generator by default, to blunt
    self-preference on the faithfulness NLI), the embedder identity used for answer-relevancy
    cosine (``embedder_class`` / ``embedding_model``), the reranker, the answering cutoff
    ``top_k_rerank``, and the answer-relevancy question count. ``publishable`` is ``True`` ONLY when
    the generator, BOTH scorers, the embedder, AND the reranker are all the real classes — any fake
    flips it ``False``. ``single_run`` records the no-CI discipline: decomposition / NLI /
    question-generation are non-reproducible, so an intra-run bootstrap would be false precision.
    """

    model_config = ConfigDict(frozen=True)

    llm_class: str
    llm_model: str
    faithfulness_scorer_class: str
    answer_relevancy_scorer_class: str
    scorer_model: str
    embedder_class: str
    embedding_model: str
    reranker_class: str
    top_k_rerank: int
    n_answer_relevancy_questions: int
    git_sha: str | None
    corpus_sha256: str
    corpus_dir: str
    library_versions: dict[str, str | None]
    n_queries: int
    single_run: bool = True
    publishable: bool


class GenerationQualityReport(BaseModel):
    """Immutable snapshot of one RAGAS-style generation-quality run over the golden set (ADR-0009).

    Faithfulness HEADLINE is ``micro_faithfulness`` — pooled ``total_supported /
    total_statements`` — immune to the 0-statement convention (a 0-statement answer contributes
    nothing to either pool). ``macro_faithfulness`` (mean of per-query faithfulness, abstentions
    counted 0.0), ``macro_faithfulness_answered`` (macro over answered queries only), and
    ``n_faith_abstained`` are SECONDARY and always reported so abstention is visible; macro is never
    reported alone. Answer-relevancy HEADLINE is ``macro_answer_relevancy`` — the mean over ALL
    queries with noncommittal answers included as 0 — reported with ``committal_answer_relevancy``
    (committal-only mean) and ``n_noncommittal`` alongside. These are RAGAS-STYLE reimplementations
    (RAGAS credited as spec), NOT canonical RAGAS-library output, and near-ceiling faithfulness on
    an easy corpus is a regime property, NOT a differentiator win.
    """

    model_config = ConfigDict(frozen=True)

    provenance: GenerationQualityProvenance
    config: str
    n_queries: int
    # Faithfulness block.
    total_statements: int
    total_supported: int
    micro_faithfulness: Score
    macro_faithfulness: Score
    macro_faithfulness_answered: Score
    n_faith_answered: int
    n_faith_abstained: int
    # Answer-relevancy block.
    macro_answer_relevancy: Relevancy
    committal_answer_relevancy: Relevancy
    n_noncommittal: int
    n_committal: int
    per_query: tuple[GenerationQualityQueryRecord, ...]

    @model_validator(mode="after")
    def _supported_within_statements(self) -> GenerationQualityReport:
        """Pooled ``total_supported`` can never exceed pooled ``total_statements``."""
        if self.total_supported > self.total_statements:
            raise ValueError(
                f"total_supported ({self.total_supported}) exceeds total_statements "
                f"({self.total_statements})"
            )
        return self
