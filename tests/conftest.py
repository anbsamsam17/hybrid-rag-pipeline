"""Shared offline fixtures for the API tests.

These build a fully-wired :class:`~rag.api.service.RagService` with **only fakes**: a
:class:`~rag.indexing.embeddings.HashingEmbedder` (no model download), an in-memory Qdrant
store (no server, no network), the deterministic fake reranker
(:class:`~rag.retrieval.rerank.LexicalOverlapReranker`), and a
:class:`~rag.generation.llm.FakeLLMClient` (no API key, no SDK). A tiny vault is indexed
through the real :func:`~rag.indexing.build.build_index` so the tests exercise the genuine
retrieval/generation/verification path end-to-end — just with offline backends.

The optional lightweight backends (``qdrant_client`` / ``rank_bm25``) are checked **inside
the fixtures that actually need them**, not at module import. This keeps the skip scoped: a
missing backend skips only the API tests that build a real index, while unrelated tests that
import this conftest (config, ingestion-chunking, etc.) collect and run normally. A skip here
means the API contract for those tests went unverified — it is not a silent pass.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

# A tiny, self-contained vault: each note is short so the FakeLLMClient's leading-slice quote
# is a real substring of the top chunk -> verification grounds it -> attribution_rate == 1.0.
NOTE_RETRIEVAL = """\
# Hybrid Retrieval

Hybrid retrieval combines dense embeddings with sparse BM25 and fuses them with
reciprocal rank fusion before a cross-encoder reranks the candidates.
"""

NOTE_CITATIONS = """\
# Verified Citations

Every generated claim is checked against its cited source span so the pipeline reports a
measured attribution rate rather than a declared one.
"""

NOTE_EVAL = """\
# Evaluation

Retrieval quality is measured with recall at k, nDCG at k, and MRR, compared across
dense-only, sparse-only, and hybrid configurations with bootstrap confidence intervals.
"""


def _require_index_backends() -> None:
    """Skip the calling test/fixture if an offline index backend is unavailable.

    Scoped to the fixtures that build a real in-memory index, so a missing optional backend
    skips only those API tests — never unrelated tests that merely share this conftest.
    """
    pytest.importorskip("qdrant_client")
    pytest.importorskip("rank_bm25")


def make_settings(corpus: Path, storage: Path, **overrides: object):
    """Build offline-friendly settings pointing at a temp corpus + storage dir."""
    from rag.config import Settings

    base: dict[str, object] = {
        "corpus_dir": corpus,
        "sample_dir": corpus,
        "storage_dir": storage,
        "chunk_strategy": "recursive",
        "chunk_size": 1024,
        "chunk_overlap": 0,
        "qdrant_collection": "api_test",
    }
    base.update(overrides)
    return Settings(**base)


def build_offline_service(corpus: Path, storage: Path, *, llm: object | None = None, **overrides):
    """Index ``corpus`` with fakes and return a query+ingest-capable :class:`RagService`.

    Args:
        corpus: Directory of source notes to index.
        storage: Directory for the build artifacts (meta.json, bm25.json).
        llm: LLM client to wire into the service; defaults to :class:`FakeLLMClient` (grounded,
            ``attribution_rate == 1.0``). Pass :class:`FabricatingFakeLLMClient` to exercise
            the ungrounded path (measured ``attribution_rate < 1.0``).
        **overrides: Settings overrides forwarded to :func:`make_settings`.
    """
    from rag.api.service import RagService
    from rag.generation.llm import FakeLLMClient
    from rag.indexing.build import build_index
    from rag.indexing.embeddings import HashingEmbedder
    from rag.indexing.sparse import BM25Index
    from rag.indexing.vector_store import QdrantVectorStore
    from rag.retrieval import HybridRetriever
    from rag.retrieval.rerank import LexicalOverlapReranker

    settings = make_settings(corpus, storage, **overrides)
    store = QdrantVectorStore.in_memory(settings.qdrant_collection)
    build_index(settings, embedder=HashingEmbedder(), store=store)
    bm25 = BM25Index.load(settings.storage_dir)
    retriever = HybridRetriever(
        embedder=HashingEmbedder(),
        store=store,
        bm25=bm25,
        reranker=LexicalOverlapReranker(),
        settings=settings,
    )
    return RagService(
        retriever=retriever,
        llm=llm if llm is not None else FakeLLMClient(),
        settings=settings,
        embedder=HashingEmbedder(),
        store=store,
    )


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Write a tiny multi-note vault and return its root."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "retrieval.md").write_text(NOTE_RETRIEVAL, encoding="utf-8")
    (corpus / "citations.md").write_text(NOTE_CITATIONS, encoding="utf-8")
    (corpus / "evaluation.md").write_text(NOTE_EVAL, encoding="utf-8")
    return corpus


@pytest.fixture
def service(vault: Path, tmp_path: Path):
    """A fully-offline :class:`RagService` over the indexed vault.

    Requires the optional index backends; skips cleanly (scoped to this fixture) if absent.
    """
    _require_index_backends()
    return build_offline_service(vault, tmp_path / "storage")


@pytest.fixture
def client(service) -> Iterator:
    """A FastAPI ``TestClient`` over an app with the offline service injected."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    from rag.api import create_app

    app = create_app(service=service)
    with fastapi_testclient.TestClient(app) as test_client:
        yield test_client
