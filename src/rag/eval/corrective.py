"""Corrective-vs-baseline evaluation over the golden set (ADR-0008), offline-fake-first.

Measures, honestly, whether the self-corrective LangGraph layer (ADR-0007) improves over the
single-pass baseline on the committed golden set — and at what cost. Because the baseline already
scores ``attribution_rate = 1.000`` on this easy corpus, attribution CANNOT be the discriminator;
the pre-registered PRIMARY, confirmatory endpoint is the trace-only **retry-loop activation rate**
(did a rewrite or a regeneration fire?), which is judge-free and byte-stable under fakes.

A crucial honesty caveat: activation measures ONLY the retry loop, but the corrective graph ALSO
grades and FILTERS the context set before generation (``corrective_rag._generate_node`` generates
over the graded-relevant subset). So ``activation == 0`` does NOT by itself prove a no-op: if
grading dropped off-topic filler, the corrective arm generates over fewer contexts than the
baseline's full retrieved set and the answers can diverge. The harness therefore ALSO reports the
judge-free ``contexts_identical_rate`` (both arms' final contexts identical). A TRUE no-op requires
``activation == 0`` AND ``contexts_identical_rate == 1``; otherwise the honest finding is "the
retry loop never fired, but grading still filtered contexts on N/50 queries — see the deltas."

Fair-comparison protocol (ADR-0008 decision 2) — ONE differing knob:

* ONE hermetic build via :func:`rag.eval.harness.prepare_hermetic_eval` (its anti-leakage +
  golden-coverage guards run before any number is trusted). ONE ``HybridRetriever(use_reranker=
  True)``, ONE generation ``LLMClient``, ONE ``CorrectiveLLM``, ONE ``AnswerCorrectnessJudge``.
  Iterate golden in the FROZEN file order; both arms retrieve at ``k = top_k_rerank`` (=5) so the
  first pass is identical.
* **Baseline arm = agentic OFF:** a TRUE single pass, :func:`answer_once`
  (``retrieve -> generate_answer -> verify_answer``). It is NOT approximated by
  ``run_corrective_rag`` with the budgets set to 0 — that still runs a grade call and filters the
  contexts by grading, so it would not be a single pass. The baseline therefore keeps ALL
  retrieved contexts.
* **Corrective arm = agentic ON:** :func:`~rag.agentic.corrective_rag.run_corrective_rag` with the
  real budgets from ``eval_settings``, reading its EXISTING frozen ``CorrectiveRAGResult`` trace —
  no new instrumentation in the agentic layer.

SECONDARY endpoints (all expected ~0 delta; directional, never headlined as a win): answer
correctness vs ``reference_answer`` (blind LLM judge + deterministic lexical-F1 floor), the
attribution micro-rate regression guard (corrective must NOT drop below baseline), recall on the
final contexts, and the trace-derived cost ``2*n_rewrites + n_regenerations + 1`` extra LLM calls
per query (at the expected 0/0 that is the honest "+1 grade call/query for nothing" number).

No CI (``single_run=True``): generation AND the LLM judge are non-reproducible, so an intra-run
bootstrap would be false precision. ``publishable`` is ``True`` ONLY when every backend — the
generation LLM, corrective LLM, judge, embedder, and reranker — is the real class; any fake flips
it ``False``. ``python -m rag.eval.corrective`` (what ``make eval-corrective`` calls) runs
:func:`main`; it is LLM-required and is deliberately NOT part of ``make eval``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from rag.agentic.corrective_rag import (
    AnthropicCorrectiveLLM,
    CorrectiveLLM,
    CorrectiveRAGRequest,
    CorrectiveRAGResult,
    run_corrective_rag,
)
from rag.config import Settings, get_settings
from rag.eval.harness import CONFIG_HYBRID_RERANK, prepare_hermetic_eval
from rag.eval.judge import (
    AnswerCorrectnessJudge,
    AnthropicAnswerCorrectnessJudge,
    lexical_f1,
)
from rag.eval.metrics import recall_at_k
from rag.eval.models import (
    CorrectiveEvalProvenance,
    CorrectiveEvalReport,
    CorrectiveQueryRecord,
)
from rag.generation.generate import generate_answer
from rag.generation.llm import LLMClient, get_llm_client
from rag.generation.models import Answer
from rag.indexing.embeddings import Embedder
from rag.indexing.meta import META_FILENAME
from rag.indexing.vector_store import VectorStore
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.models import RetrievalResult
from rag.retrieval.rerank import Reranker, get_reranker
from rag.verification.citations import verify_answer
from rag.verification.models import VerificationReport

logger = logging.getLogger(__name__)

# Both arms answer at the REAL answering config (hybrid + rerank); only agentic on/off differs.
CONFIG: str = f"corrective-vs-baseline ({CONFIG_HYBRID_RERANK})"
CORRECTIVE_RESULTS_FILENAME = "corrective_results.json"

# Recall-on-final-contexts cutoffs. Both arms retrieve at k = top_k_rerank (=5), so cutoffs beyond
# 5 add nothing; (1, 3, 5) capture whether a rewrite changed what reached the final context set.
K_VALUES: tuple[int, ...] = (1, 3, 5)

# Publishability requires EVERY backend to be the real class — any fake flips it False. Kept as
# explicit class-name constants (no private cross-module import) so a rename fails a test rather
# than silently drifting the flag. The judge is included: a judged correctness number is only
# publishable when a real judge produced it.
_PUBLISHABLE_LLM = "AnthropicLLMClient"
_PUBLISHABLE_CORRECTIVE = "AnthropicCorrectiveLLM"
_PUBLISHABLE_JUDGE = "AnthropicAnswerCorrectnessJudge"
_PUBLISHABLE_EMBEDDER = "SentenceTransformerEmbedder"
_PUBLISHABLE_RERANKER = "CrossEncoderReranker"


def answer_once(
    query: str,
    k: int,
    *,
    retriever: HybridRetriever,
    llm: LLMClient,
    settings: Settings,
) -> tuple[Answer, VerificationReport, list[RetrievalResult]]:
    """The TRUE single-pass baseline: ``retrieve -> generate_answer -> verify_answer``.

    This is the exact sequence :class:`~rag.api.service.RagService` and the attribution harness
    run, factored here so the baseline arm has one shared copy. It performs NO grading and NO
    filtering — it returns ALL retrieved contexts — which is what makes it a fair single-pass
    baseline against the corrective graph (whose grade node may filter the context set). It does
    not re-implement retrieval/fusion/rerank/attribution; it only composes their existing
    contracts.

    Args:
        query: The user question.
        k: Final retrieval count (the corrective arm uses the SAME ``k`` on its first pass).
        retriever: The SHARED hybrid+rerank retriever both arms use (one differing knob).
        llm: The SHARED generation client both arms use.
        settings: Eval-scoped settings.

    Returns:
        ``(answer, report, contexts)`` where ``contexts`` is the full retrieved list generation
        and verification both saw (never a re-retrieval).
    """
    contexts = retriever.retrieve(query, k=k)
    answer = generate_answer(query, contexts, llm=llm, settings=settings)
    report = verify_answer(answer, contexts)
    return answer, report, contexts


def compute_extra_llm_calls(n_rewrites: int, n_regenerations: int) -> int:
    """Trace-derived extra LLM calls the corrective arm cost over the single-pass baseline.

    ``2*n_rewrites + n_regenerations + 1`` (ADR-0008 decision 4c): per-pass the corrective arm
    makes ``grade_calls = n_rewrites + 1`` and ``rewrite_calls = n_rewrites`` (=> ``2*n_rewrites +
    1`` grade/rewrite calls) and ``generate_calls = n_regenerations + 1`` vs the baseline's single
    generate (=> ``n_regenerations`` extra). At the expected ``n_rewrites = n_regenerations = 0``
    this is exactly **+1** — the honest "+1 grade call per query, even when it does nothing" cost.
    """
    return 2 * n_rewrites + n_regenerations + 1


@dataclass(frozen=True)
class _TraceFields:
    """The PRIMARY (judge-free) signals derived purely from one ``CorrectiveRAGResult`` trace."""

    activated: bool
    final_query_changed: bool
    extra_llm_calls: int


def _trace_fields(result: CorrectiveRAGResult) -> _TraceFields:
    """Derive the trace-only PRIMARY fields from a corrective result (pure, judge-free).

    ``activated`` is CONTROL-FLOW activation (a rewrite OR a regeneration happened);
    ``final_query_changed`` is EFFECTIVE activation (the query text actually changed). They are
    tracked separately because ``n_rewrites`` increments even on a reformulation that did not
    change the query, so control-flow activation can be a false positive for "the layer did
    something useful" — the effective signal guards against overclaiming.
    """
    activated = result.n_rewrites > 0 or result.n_regenerations > 0
    final_query_changed = result.final_query != result.original_query
    extra = compute_extra_llm_calls(result.n_rewrites, result.n_regenerations)
    return _TraceFields(
        activated=activated, final_query_changed=final_query_changed, extra_llm_calls=extra
    )


def _n_grounded(report: VerificationReport) -> int:
    """Count grounded citations in a verification report (the micro-attribution numerator)."""
    return sum(1 for check in report.checks if check.grounded)


def _is_publishable(
    *,
    llm: LLMClient,
    corrective: CorrectiveLLM,
    judge: AnswerCorrectnessJudge,
    embedder: Embedder,
    reranker: Reranker,
) -> bool:
    """Publishable only when the LLM, corrective LLM, judge, embedder, AND reranker are all real."""
    return (
        type(llm).__name__ == _PUBLISHABLE_LLM
        and type(corrective).__name__ == _PUBLISHABLE_CORRECTIVE
        and type(judge).__name__ == _PUBLISHABLE_JUDGE
        and type(embedder).__name__ == _PUBLISHABLE_EMBEDDER
        and type(reranker).__name__ == _PUBLISHABLE_RERANKER
    )


def _build_provenance(
    eval_settings: Settings,
    *,
    llm: LLMClient,
    corrective: CorrectiveLLM,
    judge: AnswerCorrectnessJudge,
    embedder: Embedder,
    reranker: Reranker,
    n_queries: int,
    n_judged: int,
) -> CorrectiveEvalProvenance:
    """Assemble the provenance header, reading corpus SHA + lib versions from the eval meta.json."""
    meta = json.loads((eval_settings.storage_dir / META_FILENAME).read_text(encoding="utf-8"))
    return CorrectiveEvalProvenance(
        baseline_llm_class=type(llm).__name__,
        corrective_llm_class=type(corrective).__name__,
        judge_class=type(judge).__name__,
        embedder_class=type(embedder).__name__,
        reranker_class=type(reranker).__name__,
        llm_model=eval_settings.llm_model,
        # The judge exposes its resolved model on ``.model`` (a distinct model from the generator
        # by default, to blunt self-preference); fall back to the generator model if a stub judge
        # has no ``.model`` attribute.
        judge_model=getattr(judge, "model", eval_settings.llm_model),
        top_k_rerank=eval_settings.top_k_rerank,
        agentic_max_query_rewrites=eval_settings.agentic_max_query_rewrites,
        agentic_max_regenerations=eval_settings.agentic_max_regenerations,
        agentic_min_relevant_docs=eval_settings.agentic_min_relevant_docs,
        agentic_min_attribution_rate=eval_settings.agentic_min_attribution_rate,
        git_sha=meta.get("git_sha"),
        corpus_sha256=str(meta.get("corpus_sha256", "")),
        corpus_dir=str(eval_settings.corpus_dir),
        library_versions=dict(meta.get("library_versions", {})),
        n_queries=n_queries,
        n_judged=n_judged,
        single_run=True,
        publishable=_is_publishable(
            llm=llm, corrective=corrective, judge=judge, embedder=embedder, reranker=reranker
        ),
    )


def _write_artifact(eval_settings: Settings, report: CorrectiveEvalReport) -> None:
    """Dump ``report`` to ``storage_dir/eval/corrective_results.json`` (sorted, byte-diffable).

    Distinct filename from ``eval_results.json`` / ``attribution_results.json`` so the artifacts
    never collide. Sorted keys so the deterministic-fake run is byte-identical and diffable; a real
    run is ``single_run`` and NOT bit-exact (generation + judge are non-reproducible).
    """
    path = eval_settings.storage_dir / CORRECTIVE_RESULTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("wrote corrective artifact to %s", path)


def run_corrective_eval(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    corrective: CorrectiveLLM | None = None,
    judge: AnswerCorrectnessJudge | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    reranker: Reranker | None = None,
) -> CorrectiveEvalReport:
    """Run the paired corrective-vs-baseline comparison over the golden set (ADR-0008).

    One hermetic build; two arms differing in exactly one knob (agentic off vs on). Per golden
    query, in the frozen file order, at ``k = eval_settings.top_k_rerank`` so both arms retrieve
    identically on the first pass:

    * **Baseline (agentic OFF):** :func:`answer_once` — a true single pass over ALL retrieved
      contexts.
    * **Corrective (agentic ON):** :func:`~rag.agentic.corrective_rag.run_corrective_rag` with the
      real budgets, reading its frozen ``CorrectiveRAGResult`` trace.

    Computes the PRIMARY activation rate (trace-only) and the SECONDARY per-arm attribution
    (micro/pooled, a regression guard), recall-on-final-contexts, cost, and answer correctness
    (blind LLM judge + deterministic lexical-F1 floor). Writes ``corrective_results.json`` and
    returns the report. The anti-leakage + golden-coverage guards (inherited via
    ``prepare_hermetic_eval``) raise before any number is trusted; ``reference_answer`` lives in
    the protected golden set and is NEVER indexed.

    Args:
        settings: Pipeline settings; ``sample_dir`` / ``golden_path`` / chunking must match the
            config that minted the golden set (the coverage guard enforces it).
        llm: Injected generation client; defaults (lazily) to the real ``AnthropicLLMClient``.
        corrective: Injected corrective controller; defaults to the real ``AnthropicCorrectiveLLM``.
        judge: Injected correctness judge; defaults to the real ``AnthropicAnswerCorrectnessJudge``.
        embedder: Injected embedder; defaults to the real ``bge-small`` embedder.
        store: Injected vector store; defaults to a Qdrant store from the eval settings.
        reranker: Injected reranker; defaults to the real cross-encoder.

    Returns:
        The :class:`CorrectiveEvalReport` (also written to ``corrective_results.json``). Any fake
        backend (including a fake judge) sets ``publishable=False``.
    """
    # Shared hermetic build + anti-leakage guards (single source of truth with run_eval).
    prepared = prepare_hermetic_eval(settings, embedder=embedder, store=store)
    eval_settings = prepared.eval_settings
    embedder = prepared.embedder
    store = prepared.store
    bm25 = prepared.bm25
    golden = prepared.golden

    # Lazy DI: resolve the REAL backends only when none were injected (loud if a dep is absent).
    llm = llm or get_llm_client(settings)
    reranker = reranker or get_reranker(settings)
    corrective = corrective or AnthropicCorrectiveLLM(eval_settings)
    judge = judge or AnthropicAnswerCorrectnessJudge(eval_settings)

    # ONE retriever, forced hybrid + rerank; BOTH arms share this identical instance. Only the
    # presence of the grade/rewrite/regenerate graph differs between the arms.
    retriever = HybridRetriever(
        embedder=embedder,
        store=store,
        bm25=bm25,
        reranker=reranker,
        settings=eval_settings.model_copy(update={"use_reranker": True}),
    )
    k = eval_settings.top_k_rerank

    records: list[CorrectiveQueryRecord] = []
    n_activated = 0
    n_rewrite_activated = 0
    n_regenerate_activated = 0
    n_final_query_changed = 0
    total_extra_calls = 0
    baseline_total_citations = baseline_total_grounded = 0
    corrective_total_citations = corrective_total_grounded = 0
    baseline_recall_sums = {kk: 0.0 for kk in K_VALUES}
    corrective_recall_sums = {kk: 0.0 for kk in K_VALUES}
    baseline_correct_count = corrective_correct_count = 0
    baseline_f1_sum = corrective_f1_sum = 0.0
    n_judged = 0
    n_contexts_identical = 0
    terminated_counts: dict[str, int] = {}

    for item in golden:  # frozen file order
        relevant = set(item.relevant_chunk_ids)

        # --- Baseline arm: TRUE single pass (no grading, no filtering) ---
        b_answer, b_report, b_contexts = answer_once(
            item.query, k, retriever=retriever, llm=llm, settings=eval_settings
        )
        # --- Corrective arm: the real agentic graph with real budgets ---
        c_result = run_corrective_rag(
            CorrectiveRAGRequest(query=item.query, k=k),
            retriever=retriever,
            llm=llm,
            corrective=corrective,
            settings=eval_settings,
        )

        trace = _trace_fields(c_result)
        terminated_counts[c_result.terminated_reason] = (
            terminated_counts.get(c_result.terminated_reason, 0) + 1
        )

        # Attribution (measured, per arm) — pooled into the micro rate.
        b_citations = b_report.n_citations
        b_grounded = _n_grounded(b_report)
        c_citations = c_result.report.n_citations
        c_grounded = _n_grounded(c_result.report)

        # Recall on the FINAL contexts each arm handed to generation (baseline = all retrieved,
        # corrective = the graded/rewritten final set), against the golden relevant ids.
        b_ranked = [ctx.chunk_id for ctx in b_contexts]
        c_ranked = [ctx.chunk_id for ctx in c_result.contexts]
        b_recall = {kk: recall_at_k(b_ranked, relevant, kk) for kk in K_VALUES}
        c_recall = {kk: recall_at_k(c_ranked, relevant, kk) for kk in K_VALUES}

        # Did the graph change the context set even with the retry loop idle? Compare the ORDERED
        # chunk_id lists: any filtering (subset) OR reordering makes the arms non-identical, so
        # activation == 0 does not imply the arms saw the same contexts (the M1 honesty guard).
        contexts_identical = b_ranked == c_ranked

        # Correctness (SECONDARY): judged blind, per arm, only when a reference exists.
        reference = (item.reference_answer or "").strip()
        if reference:
            n_judged += 1
            b_f1 = lexical_f1(reference, b_answer.text)
            c_f1 = lexical_f1(reference, c_result.answer.text)
            # Judge sees ONLY (query, reference, candidate) — never the arm label (blind).
            b_correct: bool | None = judge.judge(item.query, reference, b_answer.text).correct
            c_correct: bool | None = judge.judge(
                item.query, reference, c_result.answer.text
            ).correct
        else:
            b_f1 = c_f1 = 0.0
            b_correct = c_correct = None

        records.append(
            CorrectiveQueryRecord(
                query_id=item.query_id,
                activated=trace.activated,
                n_rewrites=c_result.n_rewrites,
                n_regenerations=c_result.n_regenerations,
                final_query_changed=trace.final_query_changed,
                terminated_reason=c_result.terminated_reason,
                rewrite_budget_exhausted=c_result.rewrite_budget_exhausted,
                extra_llm_calls=trace.extra_llm_calls,
                contexts_identical=contexts_identical,
                baseline_n_contexts=len(b_contexts),
                corrective_n_contexts=len(c_result.contexts),
                baseline_attr_rate=b_report.attribution_rate,
                corrective_attr_rate=c_result.report.attribution_rate,
                baseline_recall=b_recall,
                corrective_recall=c_recall,
                baseline_correct=b_correct,
                corrective_correct=c_correct,
                baseline_lexical_f1=b_f1,
                corrective_lexical_f1=c_f1,
            )
        )

        # Accumulate aggregates.
        n_activated += int(trace.activated)
        n_rewrite_activated += int(c_result.n_rewrites > 0)
        n_regenerate_activated += int(c_result.n_regenerations > 0)
        n_final_query_changed += int(trace.final_query_changed)
        n_contexts_identical += int(contexts_identical)
        total_extra_calls += trace.extra_llm_calls
        baseline_total_citations += b_citations
        baseline_total_grounded += b_grounded
        corrective_total_citations += c_citations
        corrective_total_grounded += c_grounded
        for kk in K_VALUES:
            baseline_recall_sums[kk] += b_recall[kk]
            corrective_recall_sums[kk] += c_recall[kk]
        if reference:
            baseline_correct_count += int(bool(b_correct))
            corrective_correct_count += int(bool(c_correct))
            baseline_f1_sum += b_f1
            corrective_f1_sum += c_f1

    n_queries = len(records)

    # PRIMARY: retry-loop activation rate + the judge-free contexts-identical rate (together they
    # decide whether the layer is a TRUE no-op: activation == 0 AND contexts_identical_rate == 1).
    activation_rate = n_activated / n_queries if n_queries else 0.0
    contexts_identical_rate = n_contexts_identical / n_queries if n_queries else 0.0

    # SECONDARY: cost, attribution (micro/pooled, immune to the 0-citation convention), recall.
    mean_extra_llm_calls = total_extra_calls / n_queries if n_queries else 0.0
    baseline_micro = (
        baseline_total_grounded / baseline_total_citations if baseline_total_citations else 0.0
    )
    corrective_micro = (
        corrective_total_grounded / corrective_total_citations
        if corrective_total_citations
        else 0.0
    )
    baseline_recall_mean = {kk: baseline_recall_sums[kk] / n_queries for kk in K_VALUES}
    corrective_recall_mean = {kk: corrective_recall_sums[kk] / n_queries for kk in K_VALUES}

    # SECONDARY: correctness (LLM-judge rate + deterministic lexical-F1 floor), over judged queries.
    baseline_correctness_rate = baseline_correct_count / n_judged if n_judged else 0.0
    corrective_correctness_rate = corrective_correct_count / n_judged if n_judged else 0.0
    baseline_lexical_f1_mean = baseline_f1_sum / n_judged if n_judged else 0.0
    corrective_lexical_f1_mean = corrective_f1_sum / n_judged if n_judged else 0.0

    provenance = _build_provenance(
        eval_settings,
        llm=llm,
        corrective=corrective,
        judge=judge,
        embedder=embedder,
        reranker=reranker,
        n_queries=n_queries,
        n_judged=n_judged,
    )
    report = CorrectiveEvalReport(
        provenance=provenance,
        config=CONFIG,
        n_queries=n_queries,
        activation_rate=activation_rate,
        n_activated=n_activated,
        n_rewrite_activated=n_rewrite_activated,
        n_regenerate_activated=n_regenerate_activated,
        n_final_query_changed=n_final_query_changed,
        terminated_reason_counts=dict(sorted(terminated_counts.items())),
        n_contexts_identical=n_contexts_identical,
        contexts_identical_rate=contexts_identical_rate,
        mean_extra_llm_calls=mean_extra_llm_calls,
        baseline_micro_attr=baseline_micro,
        corrective_micro_attr=corrective_micro,
        attr_delta=corrective_micro - baseline_micro,
        baseline_recall_mean=baseline_recall_mean,
        corrective_recall_mean=corrective_recall_mean,
        baseline_correctness_rate=baseline_correctness_rate,
        corrective_correctness_rate=corrective_correctness_rate,
        correctness_delta=corrective_correctness_rate - baseline_correctness_rate,
        baseline_lexical_f1_mean=baseline_lexical_f1_mean,
        corrective_lexical_f1_mean=corrective_lexical_f1_mean,
        n_judged=n_judged,
        per_query=tuple(records),
    )
    _write_artifact(eval_settings, report)
    return report


def _fmt(value: float) -> str:
    """Format a rate/float to a fixed 3 decimals for the console report."""
    return f"{value:.3f}"


def render_corrective_report(report: CorrectiveEvalReport) -> str:
    """Render the console corrective-vs-baseline report (the only publishable surface).

    Prints the provenance header (both LLM roles + judge + backends + corpus fingerprint), the
    HEADLINE retry-loop activation rate + the judge-free contexts-identical rate with their splits
    + ``terminated_reason`` histogram, the SECONDARY deltas (cost, attribution regression guard,
    recall, correctness), and the honesty caveats: a no-op is only asserted when the retry loop
    stayed idle AND grading changed no context set; at ~0 activation any correctness delta is
    generator+judge NOISE (never a win); the self-preference note when judge == generator; no CI
    in v1; and the non-publishable banner on a fake run. MEASURED aggregates only.
    """
    p = report.provenance
    rule = "=" * 92
    lines: list[str] = [rule, "CORRECTIVE-VS-BASELINE EVALUATION - hybrid-rag-pipeline", rule]

    git = (p.git_sha or "unknown")[:12]
    sha = (p.corpus_sha256 or "unknown")[:16]
    lines.append(f"config={report.config}  n={p.n_queries}  n_judged={p.n_judged}  git_sha={git}")
    lines.append(
        f"baseline_llm={p.baseline_llm_class} ({p.llm_model})  "
        f"corrective_llm={p.corrective_llm_class}  judge={p.judge_class} ({p.judge_model})"
    )
    lines.append(
        f"embedder={p.embedder_class}  reranker={p.reranker_class}  top_k_rerank={p.top_k_rerank}  "
        f"budgets: R={p.agentic_max_query_rewrites} G={p.agentic_max_regenerations} "
        f"min_relevant={p.agentic_min_relevant_docs} min_attr={p.agentic_min_attribution_rate}"
    )
    lines.append(f"corpus_sha256={sha}  corpus_dir={p.corpus_dir}")

    if not p.publishable:
        lines += [
            "",
            "!! NOT PUBLISHABLE: a fake/offline backend produced these numbers "
            f"(llm={p.baseline_llm_class}, corrective={p.corrective_llm_class}, "
            f"judge={p.judge_class}, embedder={p.embedder_class}, reranker={p.reranker_class}).",
            "!! They exercise the harness only; they must NOT reach the README/docs.",
        ]

    hist = ", ".join(
        f"{reason}={count}" for reason, count in report.terminated_reason_counts.items()
    )
    n = report.n_queries
    n_filtered = n - report.n_contexts_identical
    lines += [
        "",
        f"HEADLINE (PRIMARY, pre-registered)  retry-loop activation_rate = "
        f"{_fmt(report.activation_rate)}  ({report.n_activated}/{n} queries fired a "
        "rewrite/regeneration)",
        f"          rewrite-activated = {report.n_rewrite_activated}/{n}   "
        f"regenerate-activated = {report.n_regenerate_activated}/{n}   "
        f"final_query_changed (effective) = {report.n_final_query_changed}/{n}",
        f"          terminated_reason histogram: {hist}",
        f"          contexts_identical_rate = {_fmt(report.contexts_identical_rate)}  "
        f"({report.n_contexts_identical}/{n} queries: both arms generated over the SAME final "
        "contexts; grading FILTERED the rest)",
    ]
    if report.n_activated == 0 and report.n_contexts_identical == n:
        lines.append(
            f"          => TRUE NO-OP on this corpus: the retry loop never fired (0/{n}) AND "
            "grading changed no context set. Every secondary delta is confirmatory colour, not a "
            "win."
        )
    elif report.n_activated == 0:
        lines.append(
            f"          => NOT a no-op: the retry loop never fired (0/{n}), but grading still "
            f"FILTERED contexts on {n_filtered}/{n} queries, so the arms did NOT see the same "
            "contexts. Read the recall/attribution/correctness deltas below for the effect."
        )

    lines += [
        "",
        "SECONDARY (directional; expected ~0 delta; NEVER headlined as a win)",
        "-" * len(rule),
        f"cost   mean extra LLM calls/query vs baseline = {_fmt(report.mean_extra_llm_calls)}  "
        "(= 2*n_rewrites + n_regenerations + 1; at 0 activation this is the +1 grade call/query "
        "you pay for nothing)",
        f"attr   micro attribution_rate  baseline={_fmt(report.baseline_micro_attr)}  "
        f"corrective={_fmt(report.corrective_micro_attr)}  delta={report.attr_delta:+.3f}  "
        "(REGRESSION GUARD: corrective must NOT be below baseline)",
    ]
    if report.corrective_micro_attr < report.baseline_micro_attr:
        lines.append(
            "       !! ATTRIBUTION REGRESSION: corrective micro attribution is BELOW baseline - "
            "the corrective layer reduced grounding; investigate before publishing."
        )
    lines += [
        f"recall recall@k on final contexts  baseline={_fmt_recall(report.baseline_recall_mean)}  "
        f"corrective={_fmt_recall(report.corrective_recall_mean)}  "
        "(non-zero delta only if a rewrite changed retrieval)",
        f"correct LLM-judge rate  baseline={_fmt(report.baseline_correctness_rate)}  "
        f"corrective={_fmt(report.corrective_correctness_rate)}  "
        f"delta={report.correctness_delta:+.3f}  (n_judged={report.n_judged})",
        f"        lexical-F1 floor  baseline={_fmt(report.baseline_lexical_f1_mean)}  "
        f"corrective={_fmt(report.corrective_lexical_f1_mean)}  (deterministic)",
        "",
        "Honesty: at ~0 retry activation the corrective answer is a 2nd draw from the SAME "
        "generator, over contexts that may be a graded SUBSET of the baseline's (see "
        "contexts_identical_rate above), so ANY correctness/attribution delta is generator+judge "
        "NOISE, not corrective lift - reported as directional only, never a win/loss.",
        f"No CI: single_run={p.single_run} point estimates - generation AND the LLM judge are "
        "non-reproducible, so an intra-run bootstrap would be false precision.",
    ]
    if p.judge_model == p.llm_model:
        lines.append(
            "Self-preference: judge model == generator model - only the DELTA (both arms share the "
            "judge) is trustworthy here, NOT the absolute correctness rate. Configure a distinct "
            "judge model to harden the absolute number."
        )

    if not p.publishable:
        lines.append("")
        lines.append(
            "Reminder: NON-PUBLISHABLE run (fake backend) - the numbers above are illustrative."
        )
    lines.append(rule)
    return "\n".join(lines)


def _fmt_recall(recall: dict[int, float]) -> str:
    """Format a recall-by-k dict compactly for the console (e.g. ``@1=1.000 @3=1.000 @5=1.000``)."""
    return " ".join(f"@{kk}={_fmt(recall[kk])}" for kk in sorted(recall))


def main() -> None:
    """Entry point for ``python -m rag.eval.corrective`` (used by ``make eval-corrective``)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    report = run_corrective_eval(get_settings())
    print(render_corrective_report(report))


if __name__ == "__main__":
    main()
