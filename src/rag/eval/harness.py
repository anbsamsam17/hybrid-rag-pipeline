"""Retrieval evaluation harness (ADR-0005): hermetic index, single-list scoring, comparisons.

This is the increment that *consumes* the metric + bootstrap core (ADR-0004) and turns the
committed golden set into the repo's headline deliverable: a per-config table
(dense-only / sparse-only / hybrid / hybrid+rerank) of recall@k / nDCG@k / MRR, plus a
**paired** bootstrap CI95 on the differences that matter. Retrieval-only here; RAGAS and the
``attribution_rate`` aggregation require an LLM judge and land in a later increment.

Normative design (frozen by rag-architect):

* **Single ranked list per (config, query).** Each config is run **once** at
  ``K_RETRIEVE = max(k_values) = 10`` and projected to a best-first ``list[chunk_id]``; recall
  and nDCG are then evaluated at every cutoff off that one list. No re-sort, no tie-break is
  applied on top of the retrievers (they are already tie-stable) — re-sorting would leak
  dict/set iteration order into the ranks. MRR is therefore ``RR@K_RETRIEVE``: a relevant
  chunk ranked beyond 10 counts as a miss.
* **Hermetic, eval-scoped index + DI.** The index is built on the public ``sample_dir`` into an
  ``*_eval`` collection so the eval never touches the production index; in Qdrant on-disk mode
  the vectors are additionally redirected under ``storage_dir/eval/qdrant`` for true on-disk
  isolation (in server mode the eval-scoped collection name suffices). The BM25 index + the
  ``meta.json`` provenance live under ``storage_dir/eval``. ``make eval`` resolves the real
  backends lazily here (bge-small embedder, Qdrant, cross-encoder reranker) and fails loudly if
  they are absent; tests inject a :class:`HashingEmbedder`, an in-memory Qdrant store, and a
  :class:`LexicalOverlapReranker` to run fully offline.
* **Anti-leakage guards that raise, never warn** (see :class:`EvalLeakageError`,
  :class:`GoldenCoverageError`): the golden set may not live under the indexed corpus; the eval
  corpus (``sample_dir``) may not be the private prod corpus (``corpus_dir``); and **every**
  golden relevant id must be present in the built index (else the chunking config drifted from
  the one that minted the golden set).
* **Paired bootstrap.** Per-query vectors are built in the frozen golden order so index ``i``
  is the same query in both arms; the headline metric is nDCG@10 and the secondary is
  recall@5, both bootstrapped at ``B=10000`` / ``seed=12345``.
* **Honesty.** With ``|relevant| == 1`` per query, recall@k is binary and recall@10 saturates
  near 1.0 — nDCG@10 and MRR are the discriminating signals, and ``n`` is printed next to every
  interval so a tight CI on a small ``n`` is never oversold. A run on a fake backend is marked
  ``publishable=False``; only a real ``make eval`` produces publishable numbers.

``python -m rag.eval.harness`` (what ``make eval`` calls) runs :func:`main`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

from rag.config import Settings, get_settings
from rag.eval.bootstrap import paired_bootstrap_ci
from rag.eval.golden import load_golden
from rag.eval.metrics import mrr, ndcg_at_k, recall_at_k, reciprocal_rank
from rag.eval.models import (
    ConfigComparison,
    EvalProvenance,
    EvalReport,
    GoldenItem,
    RetrievalMetrics,
)
from rag.indexing.build import build_index
from rag.indexing.embeddings import Embedder, get_embedder
from rag.indexing.meta import META_FILENAME
from rag.indexing.sparse import BM25Index
from rag.indexing.vector_store import QdrantVectorStore, VectorStore
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerank import Reranker, get_reranker
from rag.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)

# --- Config identifiers (rows of the comparison table) ----------------------------------------
CONFIG_DENSE = "dense-only"
CONFIG_SPARSE = "sparse-only"
CONFIG_HYBRID = "hybrid"
CONFIG_HYBRID_RERANK = "hybrid+rerank"
CONFIGS: tuple[str, ...] = (CONFIG_DENSE, CONFIG_SPARSE, CONFIG_HYBRID, CONFIG_HYBRID_RERANK)

# --- Frozen knobs (ADR-0005) -------------------------------------------------------------------
K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
K_RETRIEVE: int = max(K_VALUES)  # one retrieval per (config, query), scored at every cutoff
HEADLINE_K: int = 10
SECONDARY_K: int = 5
HEADLINE_METRIC: str = f"ndcg@{HEADLINE_K}"
SECONDARY_METRIC: str = f"recall@{SECONDARY_K}"
SEED: int = 12345
N_RESAMPLES: int = 10000
EVAL_RESULTS_FILENAME = "eval_results.json"

# The four comparisons that license a claim, each run on both the headline and secondary metric.
COMPARISON_PAIRS: tuple[tuple[str, str], ...] = (
    (CONFIG_DENSE, CONFIG_HYBRID),
    (CONFIG_SPARSE, CONFIG_HYBRID),
    (CONFIG_HYBRID, CONFIG_HYBRID_RERANK),
    (CONFIG_DENSE, CONFIG_HYBRID_RERANK),
)

# Pre-registered PRIMARY comparison (chosen before seeing numbers): nDCG@10, the full-stack
# uplift dense-only -> hybrid+rerank. The other 7 (pairs x metrics) are secondary/exploratory.
# Pre-registering one primary endpoint guards against multiplicity: with 8 comparisons there is
# a ~34% chance of at least one spurious "significant" under the null, so only the primary
# carries confirmatory weight; the rest are directional.
PRIMARY_BASELINE = CONFIG_DENSE
PRIMARY_TREATMENT = CONFIG_HYBRID_RERANK
PRIMARY_METRIC = HEADLINE_METRIC
N_COMPARISONS = len(COMPARISON_PAIRS) * 2  # 4 pairs x {headline, secondary}

# Only the real backends produce numbers that may be published; anything else is illustrative.
_PUBLISHABLE_EMBEDDER = "SentenceTransformerEmbedder"
_PUBLISHABLE_RERANKER = "CrossEncoderReranker"


class EvalGuardError(RuntimeError):
    """Base class for an anti-leakage / integrity guard that refuses to let an eval proceed."""


class EvalLeakageError(EvalGuardError):
    """Raised when the eval would score retrieval against labels it indexed (test-on-train)."""


class GoldenCoverageError(EvalGuardError):
    """Raised when a golden relevant id is absent from the built index (join/chunking drift)."""


@dataclass
class _ConfigRun:
    """Mutable accumulator of per-query metric vectors for one config, in golden order."""

    config: str
    recall: dict[int, list[float]] = field(default_factory=lambda: {k: [] for k in K_VALUES})
    ndcg: dict[int, list[float]] = field(default_factory=lambda: {k: [] for k in K_VALUES})
    rr: list[float] = field(default_factory=list)


def _check_golden_not_in_corpus(golden_path: Path, corpus_dir: Path) -> None:
    """Guard 1: the golden set must not live under the indexed corpus (test-on-train leak)."""
    golden = golden_path.resolve()
    corpus = corpus_dir.resolve()
    if golden.is_relative_to(corpus) or corpus == golden.parent:
        raise EvalLeakageError(
            f"golden set {golden} lives under the indexed corpus {corpus} — the eval would "
            "score retrieval against labels it indexed (test-on-train leak). Keep the golden "
            "set outside corpus_dir / sample_dir."
        )


def _check_eval_corpus_is_public(eval_corpus_dir: Path, private_corpus_dir: Path) -> None:
    """Guard 3: the eval must index the PUBLIC sample corpus, never the private prod corpus.

    The eval corpus is ``settings.sample_dir``. If that has been pointed at the private
    production corpus (``settings.corpus_dir``), the harness would index proprietary docs and
    could emit ``publishable`` numbers over a non-public, non-reproducible corpus — so an
    equality raises rather than runs. (Comparing the eval corpus to ``sample_dir`` would be a
    tautology, since the eval corpus *is* ``sample_dir``; the meaningful check is against the
    private corpus.)
    """
    if eval_corpus_dir.resolve() == private_corpus_dir.resolve():
        raise EvalLeakageError(
            f"eval corpus {eval_corpus_dir} resolves to the PRIVATE prod corpus "
            f"{private_corpus_dir} (settings.sample_dir == settings.corpus_dir); refusing to "
            "run — indexing the private corpus would yield non-public, non-reproducible numbers "
            "that could be mislabeled publishable. Point sample_dir at the public sample corpus."
        )


def _check_golden_coverage(golden: list[GoldenItem], bm25: BM25Index) -> None:
    """Guard 4 (crucial): every golden relevant id must be present in the built index.

    A missing id silently scores ~0 recall for that query and quietly drags the headline
    numbers down, so it is a hard fail listing the absent ids rather than a warning.
    """
    indexed_ids = set(bm25.chunk_ids)
    golden_ids: set[str] = set()
    for item in golden:
        golden_ids.update(item.relevant_chunk_ids)
    missing = sorted(golden_ids - indexed_ids)
    if missing:
        raise GoldenCoverageError(
            "golden id not in index — chunking config likely drifted from the config that "
            "minted golden.jsonl, or wrong corpus indexed. missing ids: " + ", ".join(missing)
        )


def _is_publishable(embedder: Embedder, reranker: Reranker) -> bool:
    """Numbers are publishable only when BOTH the embedder and reranker are the real models."""
    return (
        type(embedder).__name__ == _PUBLISHABLE_EMBEDDER
        and type(reranker).__name__ == _PUBLISHABLE_RERANKER
    )


def _ranked_ids_by_config(
    query: str,
    *,
    dense: DenseRetriever,
    sparse: SparseRetriever,
    hybrid_norerank: HybridRetriever,
    hybrid_rerank: HybridRetriever,
) -> dict[str, list[str]]:
    """Project each config's single ``K_RETRIEVE`` retrieval to a best-first ``list[chunk_id]``.

    No re-sort and no extra tie-break is applied: the retrievers are already tie-stable, and
    re-ordering here would leak dict/set iteration order into the ranks. ``k=K_RETRIEVE`` is
    passed explicitly to the hybrids so the result count never silently follows
    ``settings.top_k_rerank``.
    """
    return {
        CONFIG_DENSE: [chunk_id for chunk_id, _ in dense.retrieve(query, K_RETRIEVE)],
        CONFIG_SPARSE: [chunk_id for chunk_id, _ in sparse.retrieve(query, K_RETRIEVE)],
        CONFIG_HYBRID: [r.chunk_id for r in hybrid_norerank.retrieve(query, k=K_RETRIEVE)],
        CONFIG_HYBRID_RERANK: [r.chunk_id for r in hybrid_rerank.retrieve(query, k=K_RETRIEVE)],
    }


def _score_configs(
    golden: list[GoldenItem],
    *,
    dense: DenseRetriever,
    sparse: SparseRetriever,
    hybrid_norerank: HybridRetriever,
    hybrid_rerank: HybridRetriever,
) -> dict[str, _ConfigRun]:
    """Build per-config, per-query metric vectors in the frozen golden order (bootstrap axis)."""
    runs = {name: _ConfigRun(config=name) for name in CONFIGS}
    for item in golden:
        relevant = set(item.relevant_chunk_ids)
        ranked_by_config = _ranked_ids_by_config(
            item.query,
            dense=dense,
            sparse=sparse,
            hybrid_norerank=hybrid_norerank,
            hybrid_rerank=hybrid_rerank,
        )
        for name in CONFIGS:
            ranked = ranked_by_config[name]
            run = runs[name]
            for k in K_VALUES:
                run.recall[k].append(recall_at_k(ranked, relevant, k))
                run.ndcg[k].append(ndcg_at_k(ranked, relevant, k))
            run.rr.append(reciprocal_rank(ranked, relevant))
    return runs


def _aggregate(run: _ConfigRun, n_queries: int) -> RetrievalMetrics:
    """Collapse a config's per-query vectors into the per-config aggregate (means + MRR)."""
    return RetrievalMetrics(
        config=run.config,
        n_queries=n_queries,
        k_values=K_VALUES,
        recall={k: fmean(run.recall[k]) for k in K_VALUES},
        ndcg={k: fmean(run.ndcg[k]) for k in K_VALUES},
        mrr=mrr(run.rr),
    )


def _bootstrap_comparisons(runs: dict[str, _ConfigRun]) -> tuple[ConfigComparison, ...]:
    """Paired bootstrap CI95 for each (pair, metric); per-query vectors share the golden order."""
    selectors = {
        HEADLINE_METRIC: lambda name: runs[name].ndcg[HEADLINE_K],
        SECONDARY_METRIC: lambda name: runs[name].recall[SECONDARY_K],
    }
    comparisons: list[ConfigComparison] = []
    for metric_name, select in selectors.items():
        for baseline, treatment in COMPARISON_PAIRS:
            result = paired_bootstrap_ci(
                select(baseline),
                select(treatment),
                n_resamples=N_RESAMPLES,
                seed=SEED,
            )
            comparisons.append(
                ConfigComparison(
                    baseline=baseline,
                    treatment=treatment,
                    metric=metric_name,
                    bootstrap=result,
                )
            )
    return tuple(comparisons)


def _build_provenance(
    eval_settings: Settings,
    *,
    embedder: Embedder,
    reranker: Reranker,
    n_queries: int,
) -> EvalProvenance:
    """Assemble the provenance header, reading corpus SHA + lib versions from the eval meta.json."""
    meta = json.loads((eval_settings.storage_dir / META_FILENAME).read_text(encoding="utf-8"))
    return EvalProvenance(
        embedder_class=type(embedder).__name__,
        reranker_class=type(reranker).__name__,
        git_sha=meta.get("git_sha"),
        corpus_sha256=str(meta.get("corpus_sha256", "")),
        corpus_dir=str(eval_settings.corpus_dir),
        library_versions=dict(meta.get("library_versions", {})),
        k_values=K_VALUES,
        k_retrieve=K_RETRIEVE,
        seed=SEED,
        n_resamples=N_RESAMPLES,
        n_queries=n_queries,
        headline_metric=HEADLINE_METRIC,
        secondary_metric=SECONDARY_METRIC,
        publishable=_is_publishable(embedder, reranker),
    )


def _write_artifact(eval_settings: Settings, report: EvalReport) -> None:
    """Dump ``report`` to ``storage_dir/eval/eval_results.json`` as byte-diffable, sorted JSON.

    Written via plain Python I/O (the protected-path hook only guards golden.jsonl / meta.json),
    with sorted keys so a reproducible run is bit-identical and diffable in review.
    """
    path = eval_settings.storage_dir / EVAL_RESULTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("wrote eval artifact to %s", path)


def run_eval(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    reranker: Reranker | None = None,
) -> EvalReport:
    """Run the full retrieval eval over the golden set and return an :class:`EvalReport`.

    Builds a hermetic, eval-scoped index on ``settings.sample_dir`` (an ``*_eval`` collection;
    in Qdrant on-disk mode the vectors are also redirected under ``storage_dir/eval/qdrant``),
    scores the four configs with a single ``K_RETRIEVE`` retrieval each, computes per-config
    metrics + paired bootstrap CI95 on the headline (nDCG@10) and secondary (recall@5) metrics,
    writes the JSON artifact, and returns the report. The anti-leakage guards (§4) raise before
    any number is trusted.

    Args:
        settings: Pipeline settings. ``sample_dir`` / ``golden_path`` / chunking config must
            match the config that minted the golden set, or the coverage guard fails.
        embedder: Injected embedder; defaults (lazily) to the real ``bge-small`` embedder. A
            :class:`HashingEmbedder` flips ``publishable`` to ``False``.
        store: Injected vector store; defaults to a Qdrant store from the eval settings.
        reranker: Injected reranker; defaults to the real cross-encoder.

    Returns:
        The :class:`EvalReport` for this run (also written to ``eval_results.json``).

    Raises:
        EvalLeakageError: If the golden set lives under the indexed corpus, or the eval corpus
            (``sample_dir``) resolves to the private prod corpus (``corpus_dir``).
        GoldenCoverageError: If any golden relevant id is missing from the built index.
        ImportError: If a real backend is required (none injected) but its dependency is absent
            — a loud, actionable failure, never a silent fake fallback.
    """
    # Hermetic eval scope: sample corpus, an *_eval collection, an eval-only storage dir. In
    # Qdrant on-disk mode we ALSO redirect qdrant_path under storage_dir/eval/qdrant so the eval
    # vectors are truly isolated on disk (and can't deadlock on a prod-locked on-disk store); in
    # server mode (qdrant_url) the eval-scoped collection name is sufficient isolation.
    eval_storage_dir = settings.storage_dir / "eval"
    update: dict[str, object] = {
        "corpus_dir": settings.sample_dir,
        "qdrant_collection": settings.qdrant_collection + "_eval",
        "storage_dir": eval_storage_dir,
    }
    if settings.qdrant_path:
        update["qdrant_path"] = str(eval_storage_dir / "qdrant")
    eval_settings = settings.model_copy(update=update)

    # Guards 1 + 3 are path-only and run BEFORE any build, so we never index a leaky/private
    # corpus: the golden set must not live under the indexed corpus, and the eval corpus
    # (sample_dir) must not be the private prod corpus. Then load the golden set up front so a
    # malformed label also fails before the build.
    _check_golden_not_in_corpus(settings.golden_path, eval_settings.corpus_dir)
    _check_eval_corpus_is_public(eval_settings.corpus_dir, settings.corpus_dir)
    golden = load_golden(settings.golden_path)
    n_queries = len(golden)

    # Lazy DI: resolve the REAL backends only when none were injected (loud failure if absent).
    embedder = embedder or get_embedder(settings)
    store = store or QdrantVectorStore.from_settings(eval_settings)
    reranker = reranker or get_reranker(settings)

    build_index(eval_settings, embedder=embedder, store=store)
    bm25 = BM25Index.load(eval_settings.storage_dir)

    # Guard 4: every golden relevant id must actually be present in the built index.
    _check_golden_coverage(golden, bm25)

    dense = DenseRetriever(embedder, store)
    sparse = SparseRetriever(bm25)
    hybrid_norerank = HybridRetriever(
        embedder=embedder,
        store=store,
        bm25=bm25,
        reranker=reranker,
        settings=eval_settings.model_copy(update={"use_reranker": False}),
    )
    hybrid_rerank = HybridRetriever(
        embedder=embedder,
        store=store,
        bm25=bm25,
        reranker=reranker,
        settings=eval_settings.model_copy(update={"use_reranker": True}),
    )

    runs = _score_configs(
        golden,
        dense=dense,
        sparse=sparse,
        hybrid_norerank=hybrid_norerank,
        hybrid_rerank=hybrid_rerank,
    )

    configs = tuple(_aggregate(runs[name], n_queries) for name in CONFIGS)
    comparisons = _bootstrap_comparisons(runs)
    provenance = _build_provenance(
        eval_settings, embedder=embedder, reranker=reranker, n_queries=n_queries
    )

    report = EvalReport(provenance=provenance, configs=configs, comparisons=comparisons)
    _write_artifact(eval_settings, report)
    return report


def _fmt_float(value: float) -> str:
    """Format a metric value to a fixed 3 decimals for the console table."""
    return f"{value:.3f}"


def render_report(report: EvalReport) -> str:
    """Render the console report (the ONLY publishable surface): provenance, table, comparisons.

    The console is the single source of publishable numbers. It prints the provenance header,
    the per-config table, the honesty caveats, and one line per :class:`ConfigComparison` with
    ``n`` next to every interval. A non-publishable (fake-backend) run is banner-flagged so its
    numbers cannot be mistaken for a benchmark.
    """
    p = report.provenance
    rule = "=" * 92
    lines: list[str] = [rule, "RETRIEVAL EVALUATION - hybrid-rag-pipeline", rule]

    git = (p.git_sha or "unknown")[:12]
    sha = (p.corpus_sha256 or "unknown")[:16]
    lines.append(f"embedder={p.embedder_class}  reranker={p.reranker_class}  git_sha={git}")
    lines.append(f"corpus_sha256={sha}  corpus_dir={p.corpus_dir}")
    lines.append(
        f"k_values={p.k_values}  k_retrieve={p.k_retrieve}  seed={p.seed}  "
        f"B={p.n_resamples}  n={p.n_queries}  numpy={p.library_versions.get('numpy')}"
    )

    if not p.publishable:
        lines += [
            "",
            "!! NOT PUBLISHABLE: rankings came from a fake/offline backend "
            f"(embedder={p.embedder_class}).",
            "!! These numbers exercise the harness only; they must NOT reach the README/docs.",
        ]

    header = (
        f"{'config':<16} | {'R@1':>6} | {'R@5':>6} | {'R@10':>6} | "
        f"{'nDCG@5':>7} | {'nDCG@10':>8} | {'MRR':>6}"
    )
    lines += ["", header, "-" * len(header)]
    for cfg in report.configs:
        lines.append(
            f"{cfg.config:<16} | {_fmt_float(cfg.recall[1]):>6} | {_fmt_float(cfg.recall[5]):>6} "
            f"| {_fmt_float(cfg.recall[10]):>6} | {_fmt_float(cfg.ndcg[5]):>7} "
            f"| {_fmt_float(cfg.ndcg[10]):>8} | {_fmt_float(cfg.mrr):>6}"
        )

    lines += [
        "",
        "Caveat: every golden query has exactly 1 relevant chunk, so recall@k is binary and "
        "recall@10 saturates near 1.0 - do NOT headline it.",
        f"Caveat: nDCG@{HEADLINE_K} and MRR are the discriminating signals. MRR is RR@"
        f"{p.k_retrieve} (a relevant chunk ranked beyond {p.k_retrieve} counts as a miss = 0).",
        "",
        f"PAIRED BOOTSTRAP COMPARISONS (percentile CI95, B={p.n_resamples}, seed={p.seed}, "
        f"n={p.n_queries})",
        "-" * len(rule),
    ]
    for comp in report.comparisons:
        b = comp.bootstrap
        verdict = "significant" if b.significant else "ns (not distinguishable at this n)"
        # CI bounds + diff at 4 decimals: at 3dp a genuinely significant CI whose lower bound is
        # a positive epsilon would print "[+0.000, ...] (significant)" and read as a contradiction.
        is_primary = (
            comp.baseline == PRIMARY_BASELINE
            and comp.treatment == PRIMARY_TREATMENT
            and comp.metric == PRIMARY_METRIC
        )
        tag = "[PRIMARY]    " if is_primary else "[exploratory]"
        lines.append(
            f"{tag} {comp.treatment} vs {comp.baseline} - {comp.metric}: "
            f"diff={b.diff:+.4f} CI95=[{b.ci_low:+.4f}, {b.ci_high:+.4f}] "
            f"({verdict}), n={b.n_queries}"
        )

    lines += [
        "",
        f"Pre-registered PRIMARY endpoint: {PRIMARY_METRIC}, {PRIMARY_TREATMENT} vs "
        f"{PRIMARY_BASELINE} (marked [PRIMARY]). Only it carries confirmatory weight.",
        f"Multiplicity: {N_COMPARISONS} comparisons are reported ({len(COMPARISON_PAIRS)} pairs "
        "x 2 metrics); under the null there is a ~34% chance of >=1 spurious 'significant', so "
        "treat every [exploratory] 'significant' as directional, not confirmatory.",
    ]
    if any(not c.bootstrap.significant for c in report.comparisons):
        lines.append(
            f"Honesty: comparisons marked 'ns' are NOT distinguishable at n={p.n_queries} and "
            "must not be reported as wins; at this n even 'significant' CIs are wide."
        )
    if not p.publishable:
        lines.append(
            "Reminder: NON-PUBLISHABLE run (fake backend) - the numbers above are illustrative."
        )
    lines.append(rule)
    return "\n".join(lines)


def main() -> None:
    """Entry point for ``python -m rag.eval.harness`` (used by ``make eval``)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    report = run_eval(get_settings())
    print(render_report(report))


if __name__ == "__main__":
    main()
