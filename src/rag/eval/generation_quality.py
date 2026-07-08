"""RAGAS-style generation-quality evaluation over the golden set (ADR-0009), offline-fake-first.

This increment adds the third generation-quality signal alongside the measured
``attribution_rate`` (ADR-0006): two RAGAS-STYLE metrics reimplemented over the Anthropic SDK
(``rag.eval.generation_scorers``), scored on the SAME generated answers over the SAME
hybrid+rerank retrieval:

* **faithfulness** — grounding of ALL the answer's atomic statements against the FULL retrieved
  context (LLM decompose + NLI). Denominator = statements. HEADLINE = MICRO (pooled)
  ``total_supported / total_statements``, immune to the 0-statement convention.
* **answer_relevancy** — does the answer address the QUESTION (embedding cosine over LLM-generated
  questions; needs NO ground truth). HEADLINE = MACRO mean over ALL queries, a noncommittal answer
  INCLUDED as 0.

These are RAGAS-STYLE reimplementations — **RAGAS credited as the spec, not the canonical library's
output** — and the three metrics are positioned so nothing is double-counted:

* ``attribution_rate`` (ADR-0006) — grounding of the citations the model MADE (denominator =
  citations; deterministic lexical check). "Are the model's own citations honest?"
* ``faithfulness`` (here) — grounding of ALL statements vs the FULL context, cited or not
  (denominator = statements; LLM NLI). Broader scope; different denominator + checker.
* ``answer_relevancy`` (here) — does the answer address the question. An orthogonal axis.

The three use different denominators / checkers and are NEVER summed or presented as one improving
another. On this easy corpus faithfulness is pre-registered as near-ceiling — a regime property,
NOT a differentiator win, and a saturated ``1.000`` is dangerously indistinguishable from an
always-"supported" bug, which the negative-fixture test guards against.

Reuses the shared hermetic build + anti-leakage guards
(:func:`rag.eval.harness.prepare_hermetic_eval`),
the DI-clean :func:`rag.generation.generate.generate_answer`, and the real answering config (hybrid
+ rerank at ``settings.top_k_rerank``). No CI (``single_run=True``): decomposition / NLI /
question-generation are non-reproducible, so an intra-run bootstrap would be false precision.
``publishable`` is ``True`` ONLY when the generator, BOTH scorers, the embedder, and the reranker
are all the real classes; any fake flips it ``False``. ``python -m rag.eval.generation_quality``
(what ``make eval-ragas`` calls) runs :func:`main`; it is LLM-required and deliberately NOT part of
``make eval``.
"""

from __future__ import annotations

import json
import logging
from statistics import fmean

from rag.config import Settings, get_settings
from rag.eval.generation_scorers import (
    AnswerRelevancyScorer,
    AnthropicAnswerRelevancyScorer,
    AnthropicFaithfulnessScorer,
    FaithfulnessScorer,
)
from rag.eval.harness import CONFIG_HYBRID_RERANK, prepare_hermetic_eval
from rag.eval.models import (
    GenerationQualityProvenance,
    GenerationQualityQueryRecord,
    GenerationQualityReport,
)
from rag.generation.generate import generate_answer
from rag.generation.llm import LLMClient, get_llm_client
from rag.indexing.embeddings import Embedder
from rag.indexing.meta import META_FILENAME
from rag.indexing.vector_store import VectorStore
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerank import Reranker, get_reranker

logger = logging.getLogger(__name__)

# The generation-quality run scores exactly the real answering config: hybrid + rerank.
CONFIG: str = CONFIG_HYBRID_RERANK
GENERATION_QUALITY_RESULTS_FILENAME = "generation_quality_results.json"

# Publishability requires EVERY backend to be the real class — any fake flips it False. Kept as
# explicit class-name constants (no private cross-module import) so a rename fails a test rather
# than silently drifting the flag. BOTH scorers are included: a judged faithfulness/relevancy
# number is only publishable when the real scorers produced it.
_PUBLISHABLE_LLM = "AnthropicLLMClient"
_PUBLISHABLE_FAITHFULNESS = "AnthropicFaithfulnessScorer"
_PUBLISHABLE_ANSWER_RELEVANCY = "AnthropicAnswerRelevancyScorer"
_PUBLISHABLE_EMBEDDER = "SentenceTransformerEmbedder"
_PUBLISHABLE_RERANKER = "CrossEncoderReranker"


def _is_publishable(
    *,
    llm: LLMClient,
    faithfulness_scorer: FaithfulnessScorer,
    answer_relevancy_scorer: AnswerRelevancyScorer,
    embedder: Embedder,
    scorer_embedder: Embedder,
    reranker: Reranker,
) -> bool:
    """Publishable only when EVERY real backend is a real class.

    ``embedder`` is the orchestrator embedder that drove RETRIEVAL; ``scorer_embedder`` is the
    embedder the answer-relevancy scorer ACTUALLY embedded the cosine with. Both must be the real
    class: gating only on the orchestrator embedder would let a real-retrieval + fake-embedded-
    relevancy run publish (the scorer holds its OWN embedder). In the published ``main()`` path they
    are the same instance, so this is byte-identical there and strictly closes the gap otherwise.
    """
    return (
        type(llm).__name__ == _PUBLISHABLE_LLM
        and type(faithfulness_scorer).__name__ == _PUBLISHABLE_FAITHFULNESS
        and type(answer_relevancy_scorer).__name__ == _PUBLISHABLE_ANSWER_RELEVANCY
        and type(embedder).__name__ == _PUBLISHABLE_EMBEDDER
        and type(scorer_embedder).__name__ == _PUBLISHABLE_EMBEDDER
        and type(reranker).__name__ == _PUBLISHABLE_RERANKER
    )


def _build_provenance(
    eval_settings: Settings,
    *,
    llm: LLMClient,
    faithfulness_scorer: FaithfulnessScorer,
    answer_relevancy_scorer: AnswerRelevancyScorer,
    embedder: Embedder,
    reranker: Reranker,
    n_queries: int,
) -> GenerationQualityProvenance:
    """Assemble the provenance header, reading corpus SHA + lib versions from the eval meta.json.

    ``scorer_model`` is read from the faithfulness scorer's resolved ``.model`` (the
    correctness-sensitive NLI judge); the answer-relevancy scorer shares the same
    :func:`~rag.eval.judge.default_judge_model` default. A fake scorer records its fake sentinel.
    ``embedder_class`` / ``embedding_model`` are read from the answer-relevancy scorer's ACTUAL
    embedder (the one that produced the cosine), not the orchestrator embedder — so a fake embedder
    inside the scorer is recorded honestly and gates ``publishable`` off.
    """
    meta = json.loads((eval_settings.storage_dir / META_FILENAME).read_text(encoding="utf-8"))
    # The embedder the relevancy cosine was ACTUALLY computed with (falls back to the orchestrator
    # embedder only for a stub scorer that does not expose one).
    scorer_embedder = getattr(answer_relevancy_scorer, "embedder", embedder)
    scorer_embedding_model = getattr(
        answer_relevancy_scorer, "embedding_model", eval_settings.embedding_model
    )
    return GenerationQualityProvenance(
        llm_class=type(llm).__name__,
        llm_model=eval_settings.llm_model,
        faithfulness_scorer_class=type(faithfulness_scorer).__name__,
        answer_relevancy_scorer_class=type(answer_relevancy_scorer).__name__,
        scorer_model=getattr(faithfulness_scorer, "model", eval_settings.llm_model),
        embedder_class=type(scorer_embedder).__name__,
        embedding_model=scorer_embedding_model,
        reranker_class=type(reranker).__name__,
        top_k_rerank=eval_settings.top_k_rerank,
        n_answer_relevancy_questions=eval_settings.ragas_answer_relevancy_n_questions,
        git_sha=meta.get("git_sha"),
        corpus_sha256=str(meta.get("corpus_sha256", "")),
        corpus_dir=str(eval_settings.corpus_dir),
        library_versions=dict(meta.get("library_versions", {})),
        n_queries=n_queries,
        single_run=True,
        publishable=_is_publishable(
            llm=llm,
            faithfulness_scorer=faithfulness_scorer,
            answer_relevancy_scorer=answer_relevancy_scorer,
            embedder=embedder,
            scorer_embedder=scorer_embedder,
            reranker=reranker,
        ),
    )


def _write_artifact(eval_settings: Settings, report: GenerationQualityReport) -> None:
    """Dump ``report`` to ``storage_dir/eval/generation_quality_results.json`` (sorted, diffable).

    Distinct filename from ``eval_results.json`` / ``attribution_results.json`` /
    ``corrective_results.json`` so the artifacts never collide. Sorted keys so a deterministic-fake
    run is byte-identical and diffable; a real run is ``single_run`` and NOT bit-exact.
    """
    path = eval_settings.storage_dir / GENERATION_QUALITY_RESULTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("wrote generation-quality artifact to %s", path)


def run_generation_quality_eval(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    reranker: Reranker | None = None,
    faithfulness_scorer: FaithfulnessScorer | None = None,
    answer_relevancy_scorer: AnswerRelevancyScorer | None = None,
) -> GenerationQualityReport:
    """Score RAGAS-style faithfulness + answer_relevancy over the golden set (ADR-0009).

    Per golden query, in the frozen file order: retrieve with hybrid + rerank at
    ``settings.top_k_rerank`` (the REAL answering config), generate a grounded answer over those
    contexts ONCE, then score the SAME answer over the SAME contexts with BOTH scorers (no
    re-retrieval, no re-generation). Each query yields a :class:`GenerationQualityQueryRecord`; the
    run aggregates them into the faithfulness MICRO headline + macro/abstention split and the
    answer-relevancy MACRO headline + committal/noncommittal split.

    The scorers are BLIND by construction — they receive only ``(query, answer_text, contexts)``,
    never ``reference_answer`` or ``relevant_chunk_ids`` — so neither metric can leak or be gamed by
    the golden labels. The anti-leakage + golden-coverage guards (inherited via
    ``prepare_hermetic_eval``) raise before any number is trusted; the golden set is never indexed.

    Args:
        settings: Pipeline settings; ``sample_dir`` / ``golden_path`` / chunking must match the
            config that minted the golden set (the coverage guard enforces it).
        llm: Injected generation client; defaults (lazily) to the real ``AnthropicLLMClient``.
        embedder: Injected embedder (also the answer-relevancy embedder); defaults to bge-small.
        store: Injected vector store; defaults to a Qdrant store from the eval settings.
        reranker: Injected reranker; defaults to the real cross-encoder.
        faithfulness_scorer: Injected scorer; defaults to :class:`AnthropicFaithfulnessScorer`.
        answer_relevancy_scorer: Injected scorer; defaults to
            :class:`AnthropicAnswerRelevancyScorer` embedding with the SAME resolved embedder.

    Returns:
        The :class:`GenerationQualityReport` (also written to ``generation_quality_results.json``).
        Any fake backend (including either scorer) sets ``publishable=False``.

    Raises:
        EvalLeakageError: If the golden set lives under the indexed corpus, or the eval corpus
            resolves to the private prod corpus.
        GoldenCoverageError: If any golden relevant id is missing from the built index.
        ImportError: If a real backend is required (none injected) but its dependency is absent —
            a loud, actionable failure, never a silent fake fallback.
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
    faithfulness_scorer = faithfulness_scorer or AnthropicFaithfulnessScorer(eval_settings)
    # The answer-relevancy scorer embeds with the SAME resolved embedder instance the index uses.
    answer_relevancy_scorer = answer_relevancy_scorer or AnthropicAnswerRelevancyScorer(
        eval_settings, embedder
    )

    # The real answering path: hybrid + rerank, k = top_k_rerank (=5), use_reranker forced on.
    retriever = HybridRetriever(
        embedder=embedder,
        store=store,
        bm25=bm25,
        reranker=reranker,
        settings=eval_settings.model_copy(update={"use_reranker": True}),
    )
    k = eval_settings.top_k_rerank

    records: list[GenerationQualityQueryRecord] = []
    total_statements = 0
    total_supported = 0
    for item in golden:  # frozen file order
        contexts = retriever.retrieve(item.query, k=k)
        answer = generate_answer(item.query, contexts, llm=llm, settings=eval_settings)
        # Score the SAME answer over the SAME contexts with BOTH scorers. Blind: only the query,
        # the answer text, and the retrieved contexts are passed — never the reference/relevant ids.
        faith = faithfulness_scorer.score(item.query, answer.text, contexts)
        relevance = answer_relevancy_scorer.score(item.query, answer.text, contexts)
        records.append(
            GenerationQualityQueryRecord(
                query_id=item.query_id,
                n_statements=faith.n_statements,
                n_supported=faith.n_supported,
                faithfulness=faith.faithfulness,
                faith_abstained=(faith.n_statements == 0),
                relevancy=relevance.relevancy,
                noncommittal=relevance.noncommittal,
                n_generated_questions=len(relevance.generated_questions),
            )
        )
        total_statements += faith.n_statements
        total_supported += faith.n_supported

    n_queries = len(records)

    # --- Faithfulness aggregation ---
    # Micro (headline): pooled supported / statements, immune to the 0-statement convention.
    micro_faithfulness = total_supported / total_statements if total_statements else 0.0
    # Macro (secondary): mean of per-query faithfulness; abstentions already contribute 0.0.
    macro_faithfulness = fmean(record.faithfulness for record in records) if records else 0.0
    answered = [record.faithfulness for record in records if not record.faith_abstained]
    macro_faithfulness_answered = fmean(answered) if answered else 0.0
    n_faith_abstained = sum(1 for record in records if record.faith_abstained)
    n_faith_answered = n_queries - n_faith_abstained

    # --- Answer-relevancy aggregation ---
    # Macro over ALL queries (headline): a noncommittal answer is included as its forced 0.0.
    macro_answer_relevancy = fmean(record.relevancy for record in records) if records else 0.0
    committal = [record.relevancy for record in records if not record.noncommittal]
    committal_answer_relevancy = fmean(committal) if committal else 0.0
    n_noncommittal = sum(1 for record in records if record.noncommittal)
    n_committal = n_queries - n_noncommittal

    provenance = _build_provenance(
        eval_settings,
        llm=llm,
        faithfulness_scorer=faithfulness_scorer,
        answer_relevancy_scorer=answer_relevancy_scorer,
        embedder=embedder,
        reranker=reranker,
        n_queries=n_queries,
    )
    report = GenerationQualityReport(
        provenance=provenance,
        config=CONFIG,
        n_queries=n_queries,
        total_statements=total_statements,
        total_supported=total_supported,
        micro_faithfulness=micro_faithfulness,
        macro_faithfulness=macro_faithfulness,
        macro_faithfulness_answered=macro_faithfulness_answered,
        n_faith_answered=n_faith_answered,
        n_faith_abstained=n_faith_abstained,
        macro_answer_relevancy=macro_answer_relevancy,
        committal_answer_relevancy=committal_answer_relevancy,
        n_noncommittal=n_noncommittal,
        n_committal=n_committal,
        per_query=tuple(records),
    )
    _write_artifact(eval_settings, report)
    return report


def _fmt(value: float) -> str:
    """Format a rate/score to a fixed 3 decimals for the console report."""
    return f"{value:.3f}"


def render_generation_quality_report(report: GenerationQualityReport) -> str:
    """Render the console generation-quality report (the only publishable surface).

    Prints the provenance header (generator + both scorers + embedder + corpus fingerprint), the
    RAGAS-style label, the faithfulness MICRO headline + macro/abstention split, the answer-
    relevancy MACRO headline + committal/noncommittal split, the three-metric positioning (no
    double-counting), and the honesty caveats: near-ceiling faithfulness on this corpus is a regime
    property not a win; the self-preference note when the scorer model == the generator model; no
    CI in v1; and the non-publishable banner on a fake run. MEASURED aggregates only — no invention.
    """
    p = report.provenance
    rule = "=" * 92
    lines: list[str] = [
        rule,
        "GENERATION-QUALITY EVALUATION (RAGAS-style) - hybrid-rag-pipeline",
        rule,
    ]

    git = (p.git_sha or "unknown")[:12]
    sha = (p.corpus_sha256 or "unknown")[:16]
    lines.append(
        "metrics: faithfulness + answer_relevancy - RAGAS-STYLE reimplementation over the "
        "Anthropic SDK (RAGAS credited as the spec; NOT the canonical RAGAS library's output)."
    )
    lines.append(f"config={report.config}  llm={p.llm_class} ({p.llm_model})  git_sha={git}")
    lines.append(
        f"faithfulness_scorer={p.faithfulness_scorer_class}  "
        f"answer_relevancy_scorer={p.answer_relevancy_scorer_class}  scorer_model={p.scorer_model}"
    )
    lines.append(
        f"embedder={p.embedder_class} ({p.embedding_model})  reranker={p.reranker_class}  "
        f"top_k_rerank={p.top_k_rerank}  N_arq={p.n_answer_relevancy_questions}  n={p.n_queries}"
    )
    lines.append(f"corpus_sha256={sha}  corpus_dir={p.corpus_dir}")

    if not p.publishable:
        lines += [
            "",
            "!! NOT PUBLISHABLE: a fake/offline backend produced these numbers "
            f"(llm={p.llm_class}, faithfulness={p.faithfulness_scorer_class}, "
            f"answer_relevancy={p.answer_relevancy_scorer_class}, embedder={p.embedder_class}, "
            f"reranker={p.reranker_class}).",
            "!! They exercise the harness only; they must NOT reach the README/docs.",
        ]

    lines += [
        "",
        "FAITHFULNESS (all answer claims grounded vs the FULL retrieved context; LLM NLI)",
        f"HEADLINE  micro faithfulness = {_fmt(report.micro_faithfulness)}  "
        f"(pooled: {report.total_supported}/{report.total_statements} supported statements)",
        f"secondary macro faithfulness = {_fmt(report.macro_faithfulness)}  "
        "(mean of per-query faithfulness; abstentions count 0.0)",
        f"          macro over answered  = {_fmt(report.macro_faithfulness_answered)}"
        f"  (n_answered={report.n_faith_answered})",
        f"          abstentions          = {report.n_faith_abstained}/{report.n_queries} "
        "answers made no verifiable claim (n_statements == 0)",
        "",
        "ANSWER_RELEVANCY (does the answer address the QUESTION; embedding cosine over generated "
        "questions; NO ground truth)",
        f"HEADLINE  macro answer_relevancy = {_fmt(report.macro_answer_relevancy)}  "
        "(mean over ALL queries; a noncommittal answer is included as 0)",
        f"secondary committal-only mean   = {_fmt(report.committal_answer_relevancy)}  "
        f"(n_committal={report.n_committal})",
        f"          noncommittal          = {report.n_noncommittal}/{report.n_queries} "
        "answers were evasive/refusals (forced relevancy 0.0)",
        "",
        "Three-metric positioning (NOT double-counted): attribution_rate = grounding of the "
        "citations the model MADE (denominator = citations); faithfulness = grounding of ALL "
        "statements vs the FULL context, cited or not (denominator = statements, LLM NLI); "
        "answer_relevancy = does the answer address the question (no ground truth). Different "
        "denominators + checkers - never summed, never one 'improving' another.",
        "Regime: on this small on-topic corpus with a grounded generator, faithfulness is expected "
        "near-ceiling - a regime property, NOT a differentiator win. A saturated 1.000 is checked "
        "against an always-'supported' bug by a mandatory fabricated-claim test fixture.",
        f"No CI: single_run={p.single_run} point estimates - decomposition / NLI / question-gen "
        "are non-reproducible, so an intra-run bootstrap would be false precision.",
    ]
    if p.scorer_model == p.llm_model:
        lines.append(
            "Self-preference: scorer model == generator model - the faithfulness NLI may favour "
            "the generator's own phrasing, inflating the ABSOLUTE faithfulness. Configure a "
            "distinct scorer model (the default already differs) to harden the number."
        )

    lines += ["", "PER-QUERY DISTRIBUTION", "-" * len(rule)]
    for record in report.per_query:
        faith_flag = " [faith_abstained]" if record.faith_abstained else ""
        rel_flag = " [noncommittal]" if record.noncommittal else ""
        lines.append(
            f"{record.query_id:<10} faithfulness={_fmt(record.faithfulness)} "
            f"({record.n_supported}/{record.n_statements}){faith_flag}  "
            f"relevancy={_fmt(record.relevancy)}{rel_flag}"
        )

    if not p.publishable:
        lines.append("")
        lines.append(
            "Reminder: NON-PUBLISHABLE run (fake backend) - the numbers above are illustrative."
        )
    lines.append(rule)
    return "\n".join(lines)


def main() -> None:
    """Entry point for ``python -m rag.eval.generation_quality`` (used by ``make eval-ragas``)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    report = run_generation_quality_eval(get_settings())
    print(render_generation_quality_report(report))


if __name__ == "__main__":
    main()
