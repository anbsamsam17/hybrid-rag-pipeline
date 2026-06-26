"""End-to-end tests for :class:`HybridRetriever`, fully offline.

Builds a tiny in-memory index from a temp vault via :func:`build_index` with a
:class:`HashingEmbedder` (no model download) + in-memory Qdrant (no server), wraps the
persisted BM25 index, and uses the deterministic fake reranker. Skipped without both
backends.

The headline assertion is the **hybrid-value** test: a query whose rare exact token the
hashing (fake) dense embedder ranks poorly is surfaced by BM25, so the fused/hybrid result
includes it — demonstrating why hybrid beats dense-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("rank_bm25")

from rag.config import Settings  # noqa: E402
from rag.indexing.build import build_index  # noqa: E402
from rag.indexing.embeddings import HashingEmbedder  # noqa: E402
from rag.indexing.sparse import BM25Index  # noqa: E402
from rag.indexing.vector_store import QdrantVectorStore  # noqa: E402
from rag.retrieval.dense import DenseRetriever  # noqa: E402
from rag.retrieval.hybrid import HybridRetriever  # noqa: E402
from rag.retrieval.models import RetrievalResult  # noqa: E402
from rag.retrieval.rerank import LexicalOverlapReranker  # noqa: E402

# A vault engineered so dense (hashing) and sparse (BM25) DISAGREE on the rare-token note:
#
# * The query's common words ("system report data ...") appear in EVERY note, so their BM25
#   IDF is ~0 and they carry no sparse signal; the rare token "zqxlemmatron" appears in only
#   the TARGET note, giving it the highest IDF, so BM25 ranks the target #1.
# * The TARGET note is long and padded with unique vocabulary the query never mentions, so
#   the hashing embedder's L2-normalized token-set vector barely weights the rare token; the
#   query (mostly common words) is therefore MORE similar to the shorter decoy notes, and
#   dense ranks the target LAST.
#
# Net: the rare-exact-token note is surfaced by sparse but ranked poorly by dense — exactly
# the case where hybrid (RRF over both) beats dense-only. The construction validity is
# re-asserted inside the test, so the vault drifting would fail loudly rather than silently
# weaken the claim.
_COMMON = "the system report data shows results for the team this period overall"
_TARGET_PAD = " ".join(f"uniqueword{i}" for i in range(40))

NOTE_TARGET = f"# Codename\n\nzqxlemmatron {_TARGET_PAD} {_COMMON}\n"
NOTE_DECOY_A = "# A\n\n" + _COMMON + " " + " ".join(f"avocab{i}" for i in range(8)) + "\n"
NOTE_DECOY_B = "# B\n\n" + _COMMON + " " + " ".join(f"bvocab{i}" for i in range(8)) + "\n"
NOTE_DECOY_C = "# C\n\n" + _COMMON + " " + " ".join(f"cvocab{i}" for i in range(8)) + "\n"

# Query: the rare token + common (zero-IDF) words. Used by the hybrid-value tests.
RARE_QUERY = "zqxlemmatron system report data results team period"


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "codename.md").write_text(NOTE_TARGET, encoding="utf-8")
    (corpus / "a.md").write_text(NOTE_DECOY_A, encoding="utf-8")
    (corpus / "b.md").write_text(NOTE_DECOY_B, encoding="utf-8")
    (corpus / "c.md").write_text(NOTE_DECOY_C, encoding="utf-8")
    return corpus


def _settings(corpus: Path, storage: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "corpus_dir": corpus,
        "sample_dir": corpus,
        "storage_dir": storage,
        "chunk_strategy": "recursive",
        # Large chunk so each note is exactly one chunk (keeps the IDF/length reasoning above
        # exact: one chunk per note, rare token in exactly one chunk).
        "chunk_size": 1024,
        "chunk_overlap": 0,
        "qdrant_collection": "hybrid_test",
    }
    base.update(overrides)
    return Settings(**base)


def _build(settings: Settings) -> tuple[QdrantVectorStore, BM25Index]:
    store = QdrantVectorStore.in_memory(settings.qdrant_collection)
    build_index(settings, embedder=HashingEmbedder(), store=store)
    bm25 = BM25Index.load(settings.storage_dir)
    return store, bm25


def _hybrid(settings: Settings, store: QdrantVectorStore, bm25: BM25Index) -> HybridRetriever:
    return HybridRetriever(
        embedder=HashingEmbedder(),
        store=store,
        bm25=bm25,
        reranker=LexicalOverlapReranker(),
        settings=settings,
    )


def test_retrieve_returns_populated_results(vault: Path, tmp_path: Path) -> None:
    settings = _settings(vault, tmp_path / "storage")
    store, bm25 = _build(settings)
    retriever = _hybrid(settings, store, bm25)

    results = retriever.retrieve(RARE_QUERY)
    assert results
    assert all(isinstance(r, RetrievalResult) for r in results)
    # Self-contained: text is hydrated from the payload (the headline contract).
    assert all(r.text for r in results)
    assert all(r.rel_path for r in results)
    # Final ranks are a clean 1..n sequence.
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    # Sources are recorded and are a subset of the two retrievers.
    for r in results:
        assert r.sources
        assert set(r.sources) <= {"dense", "sparse"}


def test_retrieve_respects_k(vault: Path, tmp_path: Path) -> None:
    settings = _settings(vault, tmp_path / "storage")
    store, bm25 = _build(settings)
    retriever = _hybrid(settings, store, bm25)
    # The 4-chunk vault always has >= k retrievable chunks for k in {1, 2}, so the count is
    # deterministic: exactly k. `==` (not `<=`) so a retriever silently returning FEWER than
    # k (off-by-one truncation, a dropped payload-less id) fails loudly.
    assert len(retriever.retrieve(RARE_QUERY, k=2)) == 2
    assert len(retriever.retrieve(RARE_QUERY, k=1)) == 1


def test_retrieve_default_k_is_top_k_rerank(vault: Path, tmp_path: Path) -> None:
    settings = _settings(vault, tmp_path / "storage", top_k_rerank=2)
    store, bm25 = _build(settings)
    retriever = _hybrid(settings, store, bm25)
    # Default k = top_k_rerank = 2, and the vault yields >= 2 chunks -> exactly 2.
    assert len(retriever.retrieve(RARE_QUERY)) == 2


def _fused_order(settings: Settings, store: QdrantVectorStore, bm25: BM25Index) -> list[str]:
    """Reproduce the pre-rerank fused chunk-id order the retriever computes internally.

    Mirrors :meth:`HybridRetriever.retrieve` up to (but not including) the rerank step, so a
    test can assert the reranker actually CHANGED this order rather than merely passing it
    through. Uses the same dense/sparse/RRF parameters from ``settings``.
    """
    from rag.retrieval.fusion import reciprocal_rank_fusion
    from rag.retrieval.sparse import SparseRetriever

    dense_ids = [
        cid
        for cid, _ in DenseRetriever(HashingEmbedder(), store).retrieve(
            RARE_QUERY, settings.top_k_dense
        )
    ]
    sparse_ids = [
        cid for cid, _ in SparseRetriever(bm25).retrieve(RARE_QUERY, settings.top_k_sparse)
    ]
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=settings.rrf_k)
    return [cid for cid, _ in fused]


def test_default_path_reranker_reorders_fused_candidates(vault: Path, tmp_path: Path) -> None:
    """DEFAULT path (use_reranker=True) end-to-end: the reranker REORDERS the fused list.

    This exercises the shipped default config and proves the rerank stage actually ran and
    changed order — not just that counts are right. With ``RARE_QUERY`` the fused (RRF) order
    ranks the rare-token TARGET chunk *second*, but the lexical-overlap reranker scores TARGET
    highest (it contains the rare token PLUS every common query token, overlap 7), so the
    reranker promotes it to rank 1. We assert (a) the fused order did NOT already have TARGET
    first (so a reorder is observable), (b) the reranked order DOES, and (c) the final score
    is the reranker's integer overlap count, proving the RRF float score was replaced.
    """
    settings = _settings(vault, tmp_path / "storage", top_k_rerank=4, use_reranker=True)
    store, bm25 = _build(settings)
    target = _target_chunk_id(bm25)

    fused_ids = _fused_order(settings, store, bm25)
    assert fused_ids[0] != target, "test invalid: fused order already put TARGET first (no reorder)"
    assert target in fused_ids

    retriever = _hybrid(settings, store, bm25)
    results = retriever.retrieve(RARE_QUERY)
    reranked_ids = [r.chunk_id for r in results]

    # The reranker ran and CHANGED the order relative to the pre-rerank fused order.
    assert reranked_ids != fused_ids, "reranker did not reorder the fused candidates"
    # ...specifically promoting the rare-token TARGET to the top.
    assert reranked_ids[0] == target
    # The top result's score is the reranker's overlap count (7 distinct query tokens, all in
    # TARGET), not the tiny RRF fused score -> proves the reranker score replaced the RRF one.
    assert results[0].score == pytest.approx(7.0)
    assert results[0].score > 1.0  # an RRF fused score here is ~1/61; the rerank score dwarfs it
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_no_rerank_path_truncates_to_k_with_clean_ranks(vault: Path, tmp_path: Path) -> None:
    """NO-rerank path (use_reranker=False): results are truncated to k with 1-based ranks.

    This routes the ``results[:final_k]`` branch in ``HybridRetriever.retrieve`` that the
    shipped default (use_reranker=True) never exercises. The 4-chunk vault has more than k=2
    retrievable chunks, so the branch must truncate to EXACTLY 2 and reassign a clean 1..2
    rank sequence. An over-returning truncation (``results`` instead of ``results[:final_k]``)
    would return 4 here and fail.
    """
    settings = _settings(vault, tmp_path / "storage", top_k_rerank=2, use_reranker=False)
    store, bm25 = _build(settings)
    retriever = _hybrid(settings, store, bm25)

    results = retriever.retrieve(RARE_QUERY)
    assert len(results) == 2  # truncated to k from the >2-chunk corpus
    assert [r.rank for r in results] == [1, 2]  # clean, 1-based, contiguous
    # Sanity: without a corpus cap, k=4 returns all four chunks (so the cap above is real).
    assert len(retriever.retrieve(RARE_QUERY, k=4)) == 4


def test_deterministic_across_runs(vault: Path, tmp_path: Path) -> None:
    settings = _settings(vault, tmp_path / "storage")
    store, bm25 = _build(settings)
    retriever = _hybrid(settings, store, bm25)
    a = retriever.retrieve(RARE_QUERY)
    b = retriever.retrieve(RARE_QUERY)
    assert [(r.chunk_id, r.rank, r.score) for r in a] == [(r.chunk_id, r.rank, r.score) for r in b]


def _target_chunk_id(bm25: BM25Index) -> str:
    """Return the chunk_id of the note carrying the rare token (the BM25 ground truth)."""
    for chunk_id, tokens in zip(bm25.chunk_ids, bm25.corpus_tokens, strict=True):
        if "zqxlemmatron" in tokens:
            return chunk_id
    raise AssertionError("rare-token chunk not found in BM25 corpus")


def test_hybrid_beats_dense_only_on_rare_exact_token(vault: Path, tmp_path: Path) -> None:
    """Hybrid surfaces a rare-exact-token chunk that dense-only (hashing) misses.

    Construction validity is asserted inline: BM25 ranks the rare-token chunk #1, while the
    hashing dense embedder ranks it OUTSIDE the final top-k. The hybrid result must still
    include it (RRF lifts it via the sparse list), proving hybrid > dense-only here.
    """
    # Keep the final k tight so "dense misses it" is meaningful.
    settings = _settings(vault, tmp_path / "storage", top_k_rerank=2, use_reranker=False)
    store, bm25 = _build(settings)
    target = _target_chunk_id(bm25)

    # 1. Sparse/BM25 surfaces the rare-token chunk at the very top (rare token dominates IDF).
    sparse_top = bm25.query(RARE_QUERY, k=4)
    assert sparse_top[0][0] == target, "test invalid: BM25 did not rank the rare-token chunk #1"

    # 2. Dense-only (hashing) ranks the target OUTSIDE the final top-k -> dense alone misses.
    dense = DenseRetriever(HashingEmbedder(), store)
    dense_ids = [cid for cid, _ in dense.retrieve(RARE_QUERY, settings.top_k_dense)]
    dense_top_k = dense_ids[: settings.top_k_rerank]
    assert target not in dense_top_k, "test invalid: dense already had the target in top-k"

    # 3. Hybrid (RRF over dense+sparse) DOES include the target in its top-k.
    retriever = _hybrid(settings, store, bm25)
    hybrid_ids = [r.chunk_id for r in retriever.retrieve(RARE_QUERY)]
    assert target in hybrid_ids

    # 4. And the target's result is correctly attributed to (at least) the sparse retriever.
    target_result = next(r for r in retriever.retrieve(RARE_QUERY) if r.chunk_id == target)
    assert "sparse" in target_result.sources


def test_sparse_only_candidate_text_is_hydrated(vault: Path, tmp_path: Path) -> None:
    """A candidate found ONLY by sparse must still come back with its text populated.

    Regression guard for the 'losing sparse-only candidates' text' bug: the payload for a
    sparse-only id must be fetched via ``get_payloads``, not left empty. We shrink
    ``top_k_dense`` below the corpus size so the dense-last target falls outside dense
    retrieval entirely (truly sparse-only), then assert it returns with text + the right
    source attribution.
    """
    # 4 chunks total; the target is dense-ranked LAST, so top_k_dense=3 excludes it from the
    # dense list entirely -> the only way it reaches the fused pool is via sparse.
    settings = _settings(
        vault,
        tmp_path / "storage",
        top_k_dense=3,
        top_k_sparse=20,
        top_k_rerank=4,
        use_reranker=False,
    )
    store, bm25 = _build(settings)
    target = _target_chunk_id(bm25)

    dense = DenseRetriever(HashingEmbedder(), store)
    dense_ids = {cid for cid, _ in dense.retrieve(RARE_QUERY, settings.top_k_dense)}
    assert target not in dense_ids, "test invalid: dense top_k still contained the target"

    retriever = _hybrid(settings, store, bm25)
    results = retriever.retrieve(RARE_QUERY)
    target_result = next((r for r in results if r.chunk_id == target), None)
    assert target_result is not None
    # Sparse-only: dense never surfaced it, so the only source is sparse...
    assert target_result.sources == ["sparse"]
    # ...yet its text was still hydrated from the payload (the bug this guards against).
    assert "zqxlemmatron" in target_result.text.lower()
