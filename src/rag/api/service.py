"""``RagService`` — the wired RAG application, decoupled from FastAPI.

The service holds the already-wired collaborators (``retriever``, ``llm``, ``settings``) and
exposes the two operations the API needs: :meth:`answer_query` (retrieve -> generate ->
verify -> assemble) and :meth:`ingest` (build the dense + sparse index). Keeping this layer
HTTP-agnostic is what makes the whole stack testable: a test constructs a ``RagService`` on
fakes (HashingEmbedder, in-memory Qdrant, FakeLLMClient, fake reranker) and asserts behavior
with no FastAPI at all, and the API tests inject that same service into the app factory.

Construction has two doors:

* :meth:`__init__` — inject fully-built collaborators (the test/offline door).
* :meth:`from_settings` — build the REAL components from ``Settings`` (the production door).
  Heavy/networked backends are constructed but never *used* at build time: the embedder and
  reranker lazy-load their models on first call, the Anthropic client lazy-imports the SDK on
  first generate, and the BM25 index is loaded from the persisted artifact on disk. Nothing
  here touches the network or downloads a model just by existing, so ``create_app()`` at
  import time stays cheap and import-safe.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from rag.api.models import (
    CitationOut,
    IngestResponse,
    QueryResponse,
    ResultOut,
)
from rag.config import Settings, get_settings
from rag.generation import generate_answer
from rag.generation.llm import LLMClient, get_llm_client
from rag.indexing.build import build_index
from rag.indexing.embeddings import Embedder, get_embedder
from rag.indexing.sparse import BM25Index
from rag.indexing.vector_store import QdrantVectorStore, VectorStore
from rag.retrieval import HybridRetriever, get_reranker
from rag.verification import verify_answer

logger = logging.getLogger(__name__)


class RagService:
    """Holds the wired RAG components and runs the end-to-end query and ingest flows."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        llm: LLMClient,
        settings: Settings,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        """Wire the service from already-built collaborators (the DI / test door).

        Args:
            retriever: The hybrid retriever used by :meth:`answer_query`.
            llm: The generation client used by :meth:`answer_query`.
            settings: Pipeline settings (default ``k``, corpus dir, etc.).
            embedder: Embedder used by :meth:`ingest`. Optional: only needed if the service
                will serve ``/ingest``; tests that only hit ``/query`` may omit it.
            store: Vector store used by :meth:`ingest`. Same optionality as ``embedder``.
        """
        self._retriever = retriever
        self._llm = llm
        self._settings = settings
        self._embedder = embedder
        self._store = store

    @property
    def settings(self) -> Settings:
        """The settings this service was wired with (read-only accessor)."""
        return self._settings

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RagService:
        """Build a service over the REAL components derived from ``settings`` (production).

        Everything heavy is lazy: the real embedder/reranker only load their models on first
        use, the Anthropic client only imports the SDK on first generate, and the BM25 index
        is read from the persisted ``bm25.json`` in ``settings.storage_dir`` (a prior
        ``/ingest`` / ``make ingest`` must have produced it). Constructing this service does
        not hit the network or download anything, so it is safe to call at app-import time.
        """
        settings = settings or get_settings()

        embedder = get_embedder(settings)
        store = QdrantVectorStore.from_settings(settings)
        bm25 = _load_bm25(settings)
        reranker = get_reranker(settings)
        retriever = HybridRetriever(
            embedder=embedder,
            store=store,
            bm25=bm25,
            reranker=reranker,
            settings=settings,
        )
        llm = get_llm_client(settings)
        return cls(
            retriever=retriever,
            llm=llm,
            settings=settings,
            embedder=embedder,
            store=store,
        )

    def answer_query(self, query: str, k: int | None = None) -> QueryResponse:
        """Run the full RAG path for ``query`` and assemble a :class:`QueryResponse`.

        Pipeline: :meth:`HybridRetriever.retrieve` -> :func:`generate_answer` (over the
        injected ``llm``) -> :func:`verify_answer` -> assemble. The response's
        ``attribution_rate`` is taken straight from the verification report (a MEASURED
        number, never declared), and ``latency_ms`` is the server-measured wall time.

        Args:
            query: The user question (already validated/stripped at the edge).
            k: Final result count; ``None`` defers to ``settings.top_k_rerank``.

        Returns:
            A fully-populated :class:`QueryResponse`.
        """
        start = time.perf_counter()

        results = self._retriever.retrieve(query, k=k)
        answer = generate_answer(query, results, llm=self._llm, settings=self._settings)
        report = verify_answer(answer, results)

        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "answered query: results=%d citations=%d attribution_rate=%.3f latency_ms=%.1f",
            len(results),
            len(answer.citations),
            report.attribution_rate,
            latency_ms,
        )

        return QueryResponse(
            answer=answer.text,
            citations=[
                CitationOut(
                    chunk_id=citation.chunk_id,
                    rel_path=citation.rel_path,
                    supporting_quote=citation.supporting_quote,
                )
                for citation in answer.citations
            ],
            attribution_rate=report.attribution_rate,
            results=[
                ResultOut(
                    chunk_id=result.chunk_id,
                    rel_path=result.rel_path,
                    score=result.score,
                    rank=result.rank,
                    text=result.text,
                    sources=list(result.sources),
                )
                for result in results
            ],
            latency_ms=latency_ms,
        )

    def ingest(self, corpus_dir: str | None = None) -> IngestResponse:
        """Build the dense + sparse index and return counts + artifact paths.

        Reuses :func:`rag.indexing.build.build_index` with the service's embedder/store (so a
        test ingests with a HashingEmbedder + in-memory store, and production uses the real
        backends). ``corpus_dir`` overrides ``settings.corpus_dir`` for this call only — the
        override is applied via ``model_copy`` so the service's own settings are untouched.

        Raises:
            RuntimeError: if this service was constructed without an embedder/store (i.e. a
                query-only test service). The app maps this to a clean ``500``.
        """
        if self._embedder is None or self._store is None:
            raise RuntimeError(
                "RagService was built without an embedder/store; ingest is unavailable"
            )

        settings = self._settings
        if corpus_dir is not None:
            settings = settings.model_copy(update={"corpus_dir": Path(corpus_dir)})

        start = time.perf_counter()
        summary: dict[str, Any] = build_index(settings, embedder=self._embedder, store=self._store)
        latency_ms = (time.perf_counter() - start) * 1000.0

        paths = [str(path) for path in (summary.get("meta_path"), summary.get("bm25_path")) if path]
        logger.info(
            "ingest complete: n_documents=%d n_chunks=%d latency_ms=%.1f",
            int(summary["n_documents"]),
            int(summary["n_chunks"]),
            latency_ms,
        )
        return IngestResponse(
            n_documents=int(summary["n_documents"]),
            n_chunks=int(summary["n_chunks"]),
            paths=paths,
            latency_ms=latency_ms,
        )


def _load_bm25(settings: Settings) -> BM25Index:
    """Load the persisted BM25 index, or return an empty one if no artifact exists yet.

    The retrieval path needs a BM25Index; the real one lives in ``settings.storage_dir`` after
    a build. If the service is constructed before any ``/ingest`` (cold start), we return an
    empty index rather than crash at construction — queries then run dense-only until an
    ``/ingest`` repopulates it, and the failure mode is a warning, not an import-time error.
    """
    try:
        return BM25Index.load(settings.storage_dir)
    except FileNotFoundError:
        logger.warning(
            "no persisted BM25 index in %s; starting with an empty sparse index "
            "(run /ingest to populate it)",
            settings.storage_dir,
        )
        return BM25Index()
