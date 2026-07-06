"""Attribution-rate aggregation over the golden set (ADR-0006), offline-fake-first.

This increment turns the per-answer, MEASURED ``attribution_rate`` that
:func:`rag.verification.citations.verify_answer` already produces into a defensible golden-set
aggregate. It reuses the deterministic verifier and the DI-clean
:func:`rag.generation.generate.generate_answer` as-is (it re-implements neither), and it runs the
REAL answering configuration: hybrid + rerank at ``settings.top_k_rerank`` (=5), scoring the
SAME contexts that were handed to generation — never a re-retrieval.

Normative design (frozen by rag-architect, ADR-0006):

* **Micro (pooled) is the headline.** ``micro = total_grounded / total_citations`` is immune to
  the 0-citation → 0.0 convention: an abstaining query contributes nothing to either pool instead
  of a hard 0.0. The macro (mean of per-query rates, abstentions counted as 0.0) is SECONDARY,
  reported next to ``macro_over_answered`` and ``n_abstained`` so the effect of abstentions is
  always visible. Reporting macro alone — which conflates "cited but ungrounded" with "abstained"
  — is forbidden. ``abstained = (n_citations == 0)``.
* **No confidence interval in v1.** LLM generation is not bit-exact reproducible (even without
  ``temperature``), so an intra-run bootstrap would manufacture false precision. We report point
  estimates + the per-query distribution + ``single_run=True``; the paired bootstrap
  (``rag.eval.bootstrap``) does not apply to a single-config measurement.
* **Publishable only when fully real.** ``publishable`` is ``True`` iff the LLM, embedder, and
  reranker are all the real classes (``AnthropicLLMClient`` / ``SentenceTransformerEmbedder`` /
  ``CrossEncoderReranker``); any fake flips it to ``False``. Provenance records ``llm_class`` and
  ``llm_model`` so a judged number is defensible after the fact.
* **Hermetic + leak-guarded.** The eval-scoped index build and the anti-leakage guards are shared
  with the retrieval harness via :func:`rag.eval.harness.prepare_hermetic_eval` — one copy, no
  drift. Real ``make eval-attribution`` resolves the real backends lazily and fails LOUDLY if a
  backend or the API key is absent; there is no fake fallback. Tests inject the deterministic
  fakes and assert structure / invariants / determinism, never a real LLM value.

``attribution_rate`` measures GROUNDING (are cited spans actually supported), NOT the correctness
or completeness of the answer. ``python -m rag.eval.attribution`` (what ``make eval-attribution``
calls) runs :func:`main`.
"""

from __future__ import annotations

import json
import logging
from statistics import fmean

from rag.config import Settings, get_settings
from rag.eval.harness import CONFIG_HYBRID_RERANK, prepare_hermetic_eval
from rag.eval.models import (
    AttributionProvenance,
    AttributionQueryRecord,
    AttributionReport,
)
from rag.generation.generate import generate_answer
from rag.generation.llm import LLMClient, get_llm_client
from rag.indexing.embeddings import Embedder
from rag.indexing.meta import META_FILENAME
from rag.indexing.vector_store import VectorStore
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerank import Reranker, get_reranker
from rag.verification.citations import verify_answer

logger = logging.getLogger(__name__)

# The attribution run scores exactly the real answering config: hybrid + rerank.
CONFIG: str = CONFIG_HYBRID_RERANK
ATTRIBUTION_RESULTS_FILENAME = "attribution_results.json"

# Publishability requires all three stages to be the real classes; any fake → publishable=False.
# These mirror the harness's retrieval publishability but add the LLM identity that a judged
# attribution number turns on. Kept as explicit class-name constants (no cross-module private
# import) so a rename fails a test rather than silently drifting the flag.
_PUBLISHABLE_LLM = "AnthropicLLMClient"
_PUBLISHABLE_EMBEDDER = "SentenceTransformerEmbedder"
_PUBLISHABLE_RERANKER = "CrossEncoderReranker"


def _is_publishable(llm: LLMClient, embedder: Embedder, reranker: Reranker) -> bool:
    """Publishable only when the LLM, embedder, AND reranker are all the real classes."""
    return (
        type(llm).__name__ == _PUBLISHABLE_LLM
        and type(embedder).__name__ == _PUBLISHABLE_EMBEDDER
        and type(reranker).__name__ == _PUBLISHABLE_RERANKER
    )


def _build_provenance(
    eval_settings: Settings,
    *,
    llm: LLMClient,
    embedder: Embedder,
    reranker: Reranker,
    n_queries: int,
) -> AttributionProvenance:
    """Assemble the provenance header, reading corpus SHA + lib versions from the eval meta.json.

    ``llm_model`` is recorded from ``eval_settings.llm_model`` (the model the real client would
    call); on a fake run it is still recorded for completeness but ``publishable`` is ``False``.
    """
    meta = json.loads((eval_settings.storage_dir / META_FILENAME).read_text(encoding="utf-8"))
    return AttributionProvenance(
        llm_class=type(llm).__name__,
        llm_model=eval_settings.llm_model,
        embedder_class=type(embedder).__name__,
        reranker_class=type(reranker).__name__,
        top_k_rerank=eval_settings.top_k_rerank,
        git_sha=meta.get("git_sha"),
        corpus_sha256=str(meta.get("corpus_sha256", "")),
        corpus_dir=str(eval_settings.corpus_dir),
        library_versions=dict(meta.get("library_versions", {})),
        n_queries=n_queries,
        single_run=True,
        publishable=_is_publishable(llm, embedder, reranker),
    )


def _write_artifact(eval_settings: Settings, report: AttributionReport) -> None:
    """Dump ``report`` to ``storage_dir/eval/attribution_results.json`` (sorted, byte-diffable).

    The filename is distinct from the retrieval harness's ``eval_results.json`` so the two
    artifacts never collide. Written via plain Python I/O with sorted keys so a deterministic run
    (fakes) is byte-identical and diffable in review; a real LLM run is NOT bit-exact reproducible
    and is marked ``single_run`` accordingly.
    """
    path = eval_settings.storage_dir / ATTRIBUTION_RESULTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("wrote attribution artifact to %s", path)


def run_attribution_eval(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    reranker: Reranker | None = None,
) -> AttributionReport:
    """Aggregate the measured per-answer ``attribution_rate`` over the golden set.

    Per golden query, in the frozen file order: retrieve with hybrid + rerank at
    ``settings.top_k_rerank`` (the REAL answering config, NOT ``K_RETRIEVE``), generate a grounded
    answer over those contexts, then verify the answer against the SAME contexts (no
    re-retrieval). Each query yields an :class:`AttributionQueryRecord`; the run aggregates them
    into micro (headline, pooled) and macro (secondary) rates plus the abstention split.

    Args:
        settings: Pipeline settings; ``sample_dir`` / ``golden_path`` / chunking config must match
            the config that minted the golden set (the coverage guard enforces it).
        llm: Injected LLM client; defaults (lazily) to the real :class:`AnthropicLLMClient`. A
            fake flips ``publishable`` to ``False``.
        embedder: Injected embedder; defaults (lazily) to the real ``bge-small`` embedder.
        store: Injected vector store; defaults to a Qdrant store from the eval settings.
        reranker: Injected reranker; defaults to the real cross-encoder.

    Returns:
        The :class:`AttributionReport` (also written to ``attribution_results.json``).

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

    # Lazy DI: resolve the REAL LLM + reranker only when none were injected (loud if absent).
    llm = llm or get_llm_client(settings)
    reranker = reranker or get_reranker(settings)

    # The real answering path: hybrid + rerank, k = top_k_rerank (=5), use_reranker forced on.
    # Every config read here comes from eval_settings — the single source of config truth for the
    # run (it equals settings for these fields today, but reading one object avoids any drift).
    retriever = HybridRetriever(
        embedder=embedder,
        store=store,
        bm25=bm25,
        reranker=reranker,
        settings=eval_settings.model_copy(update={"use_reranker": True}),
    )
    k = eval_settings.top_k_rerank

    records: list[AttributionQueryRecord] = []
    total_citations = 0
    total_grounded = 0
    for item in golden:  # frozen file order
        contexts = retriever.retrieve(item.query, k=k)
        answer = generate_answer(item.query, contexts, llm=llm, settings=eval_settings)
        # Verify against the SAME contexts objects passed to generation (never re-retrieve).
        report = verify_answer(answer, contexts)
        n_citations = report.n_citations
        n_grounded = sum(1 for check in report.checks if check.grounded)
        records.append(
            AttributionQueryRecord(
                query_id=item.query_id,
                n_citations=n_citations,
                n_grounded=n_grounded,
                attribution_rate=report.attribution_rate,
                abstained=(n_citations == 0),
            )
        )
        total_citations += n_citations
        total_grounded += n_grounded

    n_queries = len(records)
    n_abstained = sum(1 for record in records if record.abstained)
    n_answered = n_queries - n_abstained

    # Micro (headline): pooled grounded / citations, immune to the 0-citation convention.
    micro = total_grounded / total_citations if total_citations else 0.0
    # Macro (secondary): mean of per-query rates; abstentions already contribute 0.0.
    macro = fmean(record.attribution_rate for record in records) if records else 0.0
    # Macro over answered-only makes the abstention effect on the macro visible.
    answered_rates = [r.attribution_rate for r in records if not r.abstained]
    macro_answered = fmean(answered_rates) if answered_rates else 0.0

    provenance = _build_provenance(
        eval_settings, llm=llm, embedder=embedder, reranker=reranker, n_queries=n_queries
    )
    report_out = AttributionReport(
        provenance=provenance,
        config=CONFIG,
        n_queries=n_queries,
        total_citations=total_citations,
        total_grounded=total_grounded,
        micro_attribution_rate=micro,
        macro_attribution_rate=macro,
        macro_attribution_rate_answered=macro_answered,
        n_answered=n_answered,
        n_abstained=n_abstained,
        per_query=tuple(records),
    )
    _write_artifact(eval_settings, report_out)
    return report_out


def _fmt_rate(value: float) -> str:
    """Format an attribution rate to a fixed 3 decimals for the console report."""
    return f"{value:.3f}"


def render_attribution_report(report: AttributionReport) -> str:
    """Render the console attribution report (the only publishable surface).

    Prints the provenance header (LLM + backends + corpus fingerprint), the headline MICRO rate,
    the secondary MACRO / macro-over-answered / abstention split, the per-query distribution
    summary, and the honesty caveats (grounding-not-correctness; no CI in v1; non-publishable
    banner on a fake run). Numbers are the MEASURED aggregates only — nothing is invented.
    """
    p = report.provenance
    rule = "=" * 92
    lines: list[str] = [rule, "ATTRIBUTION-RATE EVALUATION - hybrid-rag-pipeline", rule]

    git = (p.git_sha or "unknown")[:12]
    sha = (p.corpus_sha256 or "unknown")[:16]
    lines.append(f"config={report.config}  llm={p.llm_class} ({p.llm_model})  git_sha={git}")
    lines.append(
        f"embedder={p.embedder_class}  reranker={p.reranker_class}  "
        f"top_k_rerank={p.top_k_rerank}  n={p.n_queries}"
    )
    lines.append(f"corpus_sha256={sha}  corpus_dir={p.corpus_dir}")

    if not p.publishable:
        lines += [
            "",
            "!! NOT PUBLISHABLE: a fake/offline backend produced the answers "
            f"(llm={p.llm_class}, embedder={p.embedder_class}, reranker={p.reranker_class}).",
            "!! These numbers exercise the harness only; they must NOT reach the README/docs.",
        ]

    lines += [
        "",
        f"HEADLINE  micro attribution_rate = {_fmt_rate(report.micro_attribution_rate)}  "
        f"(pooled: {report.total_grounded}/{report.total_citations} grounded citations)",
        f"secondary macro attribution_rate = {_fmt_rate(report.macro_attribution_rate)}  "
        "(mean of per-query rates; abstentions count 0.0)",
        f"          macro over answered      = {_fmt_rate(report.macro_attribution_rate_answered)}"
        f"  (n_answered={report.n_answered})",
        f"          abstentions              = {report.n_abstained}/{report.n_queries} "
        "queries cited nothing (n_citations == 0)",
        "",
        "Micro is the headline: it is immune to the 0-citation -> 0.0 convention (an abstaining "
        "query adds nothing to either pool). Macro alone would conflate 'cited but ungrounded' "
        "with 'abstained', so it is never reported alone.",
        "Scope: attribution_rate measures GROUNDING (are cited spans supported), NOT the "
        "correctness or completeness of the answer.",
        f"No CI: this is a single_run={p.single_run} point estimate - LLM generation is not "
        "bit-exact reproducible, so an intra-run bootstrap would be false precision.",
        "",
        "PER-QUERY DISTRIBUTION",
        "-" * len(rule),
    ]
    for record in report.per_query:
        flag = " [abstained]" if record.abstained else ""
        lines.append(
            f"{record.query_id:<10} rate={_fmt_rate(record.attribution_rate)}  "
            f"grounded={record.n_grounded}/{record.n_citations}{flag}"
        )

    if not p.publishable:
        lines.append("")
        lines.append(
            "Reminder: NON-PUBLISHABLE run (fake backend) - the numbers above are illustrative."
        )
    lines.append(rule)
    return "\n".join(lines)


def main() -> None:
    """Entry point for ``python -m rag.eval.attribution`` (used by ``make eval-attribution``)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    report = run_attribution_eval(get_settings())
    print(render_attribution_report(report))


if __name__ == "__main__":
    main()
