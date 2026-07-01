"""Tests for the retrieval evaluation harness (ADR-0005), run fully OFFLINE.

These exercise the genuine build -> retrieve -> score -> bootstrap path with the proven offline
backends (:class:`HashingEmbedder` + in-memory Qdrant + :class:`LexicalOverlapReranker`) — no
torch, no network, no API key. They assert **structure, invariants, determinism, and that the
anti-leakage guards raise** — never precise metric values and never a config ordering, because
the fake bag-of-tokens embedder produces uninterpretable rankings whose exact numbers carry no
meaning (and are marked non-publishable for exactly that reason).
"""

from __future__ import annotations

from pathlib import Path

import pytest

# The harness build path needs the lightweight index backends; skip the whole module cleanly
# (not a silent pass) if they are unavailable.
pytest.importorskip("qdrant_client")
pytest.importorskip("rank_bm25")

from rag.config import PROJECT_ROOT, Settings  # noqa: E402
from rag.eval.golden import load_golden  # noqa: E402
from rag.eval.harness import (  # noqa: E402
    CONFIG_SPARSE,
    CONFIGS,
    HEADLINE_METRIC,
    K_RETRIEVE,
    K_VALUES,
    N_RESAMPLES,
    PRIMARY_BASELINE,
    PRIMARY_METRIC,
    PRIMARY_TREATMENT,
    SECONDARY_METRIC,
    SEED,
    EvalLeakageError,
    GoldenCoverageError,
    render_report,
    run_eval,
)
from rag.eval.models import EvalReport  # noqa: E402
from rag.indexing.embeddings import HashingEmbedder  # noqa: E402
from rag.indexing.vector_store import QdrantVectorStore  # noqa: E402
from rag.retrieval.rerank import LexicalOverlapReranker  # noqa: E402

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "golden.jsonl"

# Derive n from the committed golden set — NEVER hard-code it. The offline tests bind to the
# real data/eval/golden.jsonl, so growing the golden set (16 -> 50 -> ...) must keep these
# assertions correct without an edit; a hard-coded n would silently rot the moment it changes.
N_GOLDEN = len(load_golden(GOLDEN_PATH))


def _settings(storage: Path, **overrides: object) -> Settings:
    """Eval settings pinned to the committed sample corpus + golden set.

    Chunking is pinned to the config that minted golden.jsonl (recursive / 512 / 64) so the
    golden-coverage guard passes; storage is redirected to a temp dir so nothing leaks into the
    repo's real ``storage/``.
    """
    base: dict[str, object] = {
        "corpus_dir": PROJECT_ROOT / "data" / "corpus",  # unused by eval (it uses sample_dir)
        "sample_dir": SAMPLE_DIR,
        "golden_path": GOLDEN_PATH,
        "storage_dir": storage,
        "chunk_strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "qdrant_collection": "eval_harness_test",
    }
    base.update(overrides)
    return Settings(**base)


def _run(settings: Settings) -> EvalReport:
    """Run the harness with the offline fakes (fresh in-memory store each call)."""
    return run_eval(
        settings,
        embedder=HashingEmbedder(),
        store=QdrantVectorStore.in_memory("eval_harness_test"),
        reranker=LexicalOverlapReranker(),
    )


def test_run_eval_offline_structure(tmp_path: Path) -> None:
    report = _run(_settings(tmp_path / "storage"))

    assert isinstance(report, EvalReport)
    # All four configs present; assert the SET, never the order.
    assert {cfg.config for cfg in report.configs} == set(CONFIGS)
    for cfg in report.configs:
        assert cfg.k_values == tuple(sorted(K_VALUES))
        assert set(cfg.recall) == set(cfg.ndcg) == set(K_VALUES)
        assert cfg.n_queries == report.provenance.n_queries

    prov = report.provenance
    assert prov.n_queries == N_GOLDEN
    assert prov.k_retrieve == K_RETRIEVE == max(K_VALUES)
    assert prov.seed == SEED
    assert prov.n_resamples == N_RESAMPLES
    assert prov.headline_metric == HEADLINE_METRIC
    assert prov.secondary_metric == SECONDARY_METRIC
    assert prov.corpus_sha256  # came from the eval meta.json the build wrote


def test_run_eval_marks_fake_backend_not_publishable(tmp_path: Path) -> None:
    report = _run(_settings(tmp_path / "storage"))
    assert report.provenance.publishable is False
    assert report.provenance.embedder_class == "HashingEmbedder"
    assert report.provenance.reranker_class == "LexicalOverlapReranker"


def test_run_eval_comparison_invariants(tmp_path: Path) -> None:
    report = _run(_settings(tmp_path / "storage"))

    names = set(CONFIGS)
    # 4 pairs x 2 metrics (headline + secondary).
    assert len(report.comparisons) == 2 * 4
    assert {comp.metric for comp in report.comparisons} == {HEADLINE_METRIC, SECONDARY_METRIC}
    for comp in report.comparisons:
        assert comp.baseline in names
        assert comp.treatment in names
        assert comp.baseline != comp.treatment
        boot = comp.bootstrap
        assert boot.n_queries == N_GOLDEN
        assert boot.n_resamples == N_RESAMPLES
        assert boot.seed == SEED
        # significance must be consistent with the recorded interval (no contradictory flag).
        assert boot.significant == (not boot.ci_low <= 0.0 <= boot.ci_high)


def test_run_eval_join_is_wired_sparse_recall_nonzero(tmp_path: Path) -> None:
    # The headline-credibility smoke test: prove the golden<->results join is actually wired.
    # BM25 (sparse) is lexical and real even offline, so it reliably surfaces the golden chunks;
    # recall@10 > 0 proves the harness matches retrieved ids to golden ids by stable chunk_id.
    # A regression that broke the join would score 0 everywhere and trip this. We assert ">0"
    # only -- never an exact value, never a config ordering (the fake embedder is meaningless).
    report = _run(_settings(tmp_path / "storage"))
    sparse = next(cfg for cfg in report.configs if cfg.config == CONFIG_SPARSE)
    assert sparse.recall[10] > 0.0
    assert sparse.mrr > 0.0


def test_run_eval_is_deterministic(tmp_path: Path) -> None:
    # Same corpus -> same chunk_ids -> same metrics; seeded bootstrap -> identical report.
    report_a = _run(_settings(tmp_path / "a"))
    report_b = _run(_settings(tmp_path / "b"))
    assert report_a.model_dump() == report_b.model_dump()


def test_run_eval_writes_byte_stable_artifact(tmp_path: Path) -> None:
    settings_a = _settings(tmp_path / "a")
    settings_b = _settings(tmp_path / "b")
    _run(settings_a)
    _run(settings_b)

    artifact_a = (settings_a.storage_dir / "eval" / "eval_results.json").read_bytes()
    artifact_b = (settings_b.storage_dir / "eval" / "eval_results.json").read_bytes()
    # Two reproducible runs serialize byte-identically (sorted keys, no timestamp in the report).
    assert artifact_a == artifact_b
    # And the artifact round-trips back into a valid EvalReport.
    EvalReport.model_validate_json(artifact_a.decode("utf-8"))


def test_render_report_states_n_and_flags_non_publishable(tmp_path: Path) -> None:
    text = render_report(_run(_settings(tmp_path / "storage")))
    assert f"n={N_GOLDEN}" in text
    assert "NOT PUBLISHABLE" in text
    # Every config appears as a table row.
    for name in CONFIGS:
        assert name in text
    # Each comparison metric label is rendered.
    assert HEADLINE_METRIC in text
    assert SECONDARY_METRIC in text


def test_render_report_marks_primary_and_multiplicity(tmp_path: Path) -> None:
    text = render_report(_run(_settings(tmp_path / "storage")))
    # The pre-registered primary endpoint is marked and the multiplicity caveat is present, so
    # exploratory "significant" flags can't be oversold as confirmatory.
    assert "[PRIMARY]" in text
    assert "Multiplicity" in text
    assert PRIMARY_TREATMENT in text
    assert PRIMARY_BASELINE in text
    assert PRIMARY_METRIC in text


def test_guard_golden_under_corpus_raises(tmp_path: Path) -> None:
    # Place a golden file INSIDE the indexed corpus -> the test-on-train guard must fire,
    # before any index is built.
    corpus = tmp_path / "sample"
    corpus.mkdir()
    (corpus / "doc.md").write_text("# Doc\nSome text about motorway speed limits.\n", "utf-8")
    leaky_golden = corpus / "golden.jsonl"
    leaky_golden.write_text(
        '{"query_id": "q1", "query": "a?", "relevant_chunk_ids": ["x"]}\n', "utf-8"
    )

    settings = _settings(tmp_path / "storage", sample_dir=corpus, golden_path=leaky_golden)
    with pytest.raises(EvalLeakageError, match="test-on-train"):
        _run(settings)


def test_guard_golden_coverage_failure_raises(tmp_path: Path) -> None:
    # Index an unrelated corpus while keeping the real golden set: none of its ids can be
    # present, so the coverage guard must hard-fail with the canonical message.
    corpus = tmp_path / "othercorpus"
    corpus.mkdir()
    (corpus / "unrelated.md").write_text(
        "# Unrelated\nThis note is about cooking pasta and has no driving content at all.\n",
        "utf-8",
    )

    settings = _settings(tmp_path / "storage", sample_dir=corpus)  # golden_path stays the real one
    with pytest.raises(GoldenCoverageError, match="golden id not in index"):
        _run(settings)


def test_guard_eval_corpus_is_private_raises(tmp_path: Path) -> None:
    # Point the eval corpus (sample_dir) at the PRIVATE prod corpus (corpus_dir): guard 3 must
    # fire BEFORE any indexing, so proprietary docs are never indexed and never scored.
    private = tmp_path / "private"
    private.mkdir()
    (private / "secret.md").write_text("# Secret\nProprietary internal content.\n", "utf-8")

    settings = _settings(tmp_path / "storage", sample_dir=private, corpus_dir=private)
    with pytest.raises(EvalLeakageError, match="PRIVATE prod corpus"):
        _run(settings)
