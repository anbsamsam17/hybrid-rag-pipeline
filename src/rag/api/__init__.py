"""FastAPI service: the "make it usable" layer over the hybrid RAG pipeline.

Public surface:

* :func:`create_app` — the APP FACTORY. Returns a wired ``FastAPI`` app and accepts an
  injected :class:`RagService` so the whole HTTP surface is testable offline (fakes +
  in-memory index + ``TestClient``). With no injection it builds the real service LAZILY, so
  ``uvicorn rag.api.app:app`` works without doing heavy work at import time.
* :class:`RagService` — the HTTP-agnostic wiring of retriever + llm + settings, exposing
  ``answer_query`` (retrieve -> generate -> verify) and ``ingest`` (build the index).

Endpoints: ``GET /health``, ``POST /query``, ``POST /ingest``, and the optional SSE
``POST /query/stream``. See :mod:`rag.api.app` and :mod:`rag.api.service`.

``fastapi`` is a runtime dependency, but importing this package is still cheap: no real
backend (embedder model, Qdrant client, Anthropic SDK) is constructed until a request needs
it.
"""

from __future__ import annotations

from rag.api.app import create_app
from rag.api.service import RagService

__all__ = [
    "create_app",
    "RagService",
]
