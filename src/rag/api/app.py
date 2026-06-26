"""FastAPI app factory + endpoints for the hybrid RAG service.

:func:`create_app` is the single key to testability. It returns a fully-wired ``FastAPI``
app and accepts an **injected** :class:`~rag.api.service.RagService`, so tests pass a service
built on fakes (HashingEmbedder + in-memory Qdrant + FakeLLMClient + fake reranker) and drive
the whole HTTP surface with a ``TestClient`` — no network, no model download, no Qdrant
server. When no service is injected, the app builds the real one **lazily**: nothing heavy is
constructed at import time, the real ``RagService`` is built on the first request that needs
it (and cached on ``app.state``), so ``uvicorn rag.api.app:app`` works in production while
``import rag.api.app`` stays cheap and side-effect-free.

Endpoints:

* ``GET  /health``       -> ``{"status": "ok"}`` (liveness; never touches the service).
* ``POST /query``        -> :class:`~rag.api.models.QueryResponse`.
* ``POST /ingest``       -> :class:`~rag.api.models.IngestResponse` (build the index).
* ``POST /query/stream`` -> Server-Sent Events (bonus; requires ``sse-starlette``).

Cross-cutting concerns: a request-id + latency logging middleware (``logging``, never
``print``), and exception handlers that map errors to clean status codes — ``422`` for
pydantic validation (empty/whitespace/invalid input is rejected here, before any handler
runs) and ``500`` for anything unexpected — and NEVER leak a stack trace or an internal
exception message into the response body (the traceback is logged server-side instead). An
intentional domain-level ``400`` is raised explicitly at the call site via
:class:`fastapi.HTTPException` with a curated, non-internal message; the built-in
``ValueError`` type is deliberately NOT mapped, so a deep ``ValueError`` from
retrieval/generation/verification surfaces as a clean ``500`` (server fault) rather than a
mislabeled ``400`` that leaks internals.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag.api.models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from rag.api.service import RagService
from rag.config import Settings

logger = logging.getLogger(__name__)

# Header carrying the per-request correlation id (echoed back to the client).
REQUEST_ID_HEADER = "X-Request-ID"


def _get_service(request: Request) -> RagService:
    """Return the app's :class:`RagService`, building the real one lazily on first need.

    If a service was injected at :func:`create_app` time it is used as-is (the test path).
    Otherwise the real, settings-derived service is constructed on first use and cached on
    ``app.state`` — keeping app import cheap while ``uvicorn rag.api.app:app`` still serves
    real traffic. Construction is lazy and import-safe (no model download, no network).
    """
    service: RagService | None = getattr(request.app.state, "service", None)
    if service is None:
        settings: Settings | None = getattr(request.app.state, "settings", None)
        logger.info("no injected service; building real RagService from settings (lazy)")
        service = RagService.from_settings(settings)
        request.app.state.service = service
    return service


def create_app(settings: Settings | None = None, *, service: RagService | None = None) -> FastAPI:
    """Build and return the FastAPI app.

    Args:
        settings: Settings to use when the real service is built lazily. Ignored when
            ``service`` is injected. ``None`` defers to ``get_settings()`` at first use.
        service: A fully-wired :class:`RagService` to inject (tests pass one built on fakes).
            When ``None``, the real service is constructed lazily on the first request that
            needs it — so importing this module / calling ``create_app()`` stays cheap.

    Returns:
        A configured ``FastAPI`` application.
    """
    app = FastAPI(
        title="Hybrid RAG Pipeline API",
        version="0.1.0",
        summary="Hybrid retrieval + verified-citation generation over a document corpus.",
    )
    # Stash injected (or absent) dependencies on app.state for the lazy builder + handlers.
    app.state.service = service
    app.state.settings = settings

    _register_middleware(app)
    _register_exception_handlers(app)
    _register_routes(app)
    return app


def _register_middleware(app: FastAPI) -> None:
    """Attach the request-id + latency structured-logging middleware."""

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[[Request], Awaitable]):
        """Assign/propagate a request id, time the request, and log it (no ``print``)."""
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Let the registered exception handlers shape the body; just log with context.
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.exception(
                "request failed: id=%s method=%s path=%s latency_ms=%.1f",
                request_id,
                request.method,
                request.url.path,
                latency_ms,
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000.0
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request: id=%s method=%s path=%s status=%d latency_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        return response


def _error_body(request: Request, message: str, *, detail: object | None = None) -> dict:
    """Build a uniform, stack-trace-free error body carrying the request id."""
    body: dict[str, object] = {
        "error": message,
        "request_id": getattr(request.state, "request_id", None),
    }
    if detail is not None:
        body["detail"] = detail
    return body


def _register_exception_handlers(app: FastAPI) -> None:
    """Register handlers that map errors to clean status codes with no leaked traceback."""

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """422: malformed/invalid request body or params (pydantic validation).

        ``exc.errors()`` can embed non-JSON objects (e.g. the original ``ValueError`` raised
        by a field validator in each entry's ``ctx``); ``jsonable_encoder`` flattens those to
        strings so the body always serializes — no leaked, unserializable exception object.
        """
        return JSONResponse(
            status_code=422,
            content=_error_body(request, "validation error", detail=jsonable_encoder(exc.errors())),
        )

    @app.exception_handler(HTTPException)
    async def on_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        """Pass through an INTENTIONAL, explicitly-raised HTTP error with a curated message.

        Endpoints raise :class:`fastapi.HTTPException` for genuine domain-level errors (e.g. a
        ``400`` bad request) with a SAFE, non-internal ``detail`` chosen at the call site. The
        built-in ``ValueError`` type is *not* mapped, so a deep ``ValueError`` from
        retrieval/generation/verification never reaches here — it falls through to the generic
        ``500`` handler below instead of being mislabeled a ``400`` and leaking internals.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(request, str(exc.detail)),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def on_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """500: anything unexpected. The traceback is LOGGED, never put in the response."""
        logger.exception(
            "unhandled error: id=%s type=%s",
            getattr(request.state, "request_id", None),
            type(exc).__name__,
        )
        return JSONResponse(status_code=500, content=_error_body(request, "internal server error"))


def _register_routes(app: FastAPI) -> None:
    """Register the HTTP routes on ``app``."""

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe. Cheap and dependency-free — never builds the service."""
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    async def query(request: Request, payload: QueryRequest) -> QueryResponse:
        """Run the full RAG path for a query and return the verified, cited answer.

        ``payload`` is already validated (non-empty query, ``k`` bounds) by pydantic; an
        invalid body never reaches here (the validation handler returns ``422``). The
        retrieval/generation/verification work is synchronous and CPU-bound, so it runs in a
        thread pool to keep the event loop responsive.
        """
        service = _get_service(request)
        return await _run_in_threadpool(service.answer_query, payload.query, payload.k)

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(request: Request, payload: IngestRequest) -> IngestResponse:
        """(Re)build the dense + sparse index, optionally over a supplied ``corpus_dir``."""
        service = _get_service(request)
        return await _run_in_threadpool(service.ingest, payload.corpus_dir)

    @app.post("/query/stream")
    async def query_stream(request: Request, payload: QueryRequest):
        """BONUS: stream the answer as Server-Sent Events (requires ``sse-starlette``).

        ``sse_starlette`` is imported lazily so the rest of the API works even when it is not
        installed (the import error surfaces only when this endpoint is actually called). The
        answer is computed once, then emitted as a small event sequence
        (``meta`` -> ``answer`` -> ``done``) so a UI can render progressively.
        """
        from sse_starlette.sse import EventSourceResponse

        service = _get_service(request)
        result = await _run_in_threadpool(service.answer_query, payload.query, payload.k)

        async def event_stream():
            """Yield the response as discrete SSE events (one computed answer, chunked)."""
            # ``result`` is already a validated QueryResponse (answer_query returns one), so it
            # is used directly — no redundant model_validate of trusted, in-process data.
            meta_json = result.model_copy(update={"answer": ""}).model_dump_json()
            yield {"event": "meta", "data": meta_json}
            yield {"event": "answer", "data": result.answer}
            yield {"event": "done", "data": result.model_dump_json()}

        return EventSourceResponse(event_stream())


async def _run_in_threadpool(func, /, *args):
    """Run a sync callable off the event loop, surfacing its exceptions to the handlers.

    The RAG path (retrieval/generation/verification, index build) is synchronous and
    potentially slow; running it via Starlette's threadpool keeps the async server responsive
    while letting exceptions propagate to the registered exception handlers unchanged.
    """
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(func, *args)


# Module-level app so ``uvicorn rag.api.app:app`` works. No service is injected, so the real
# components are built LAZILY on the first request — importing this module stays cheap and
# does not require an API key, a Qdrant server, or a model download.
app = create_app()
