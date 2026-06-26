"""``GET /health`` returns 200 ``{"status": "ok"}`` and never touches the service.

Offline: the app is built with an injected fake service, but ``/health`` is dependency-free
so it would pass even without one. We also assert the request-id middleware echoes a header.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag.api import create_app


def test_health_ok() -> None:
    """Health is a cheap liveness probe with a fixed body and no service dependency."""
    app = create_app()  # no service injected; /health must not build the real one
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_sets_request_id_header() -> None:
    """The middleware assigns and echoes a request id on every response."""
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.headers.get("X-Request-ID")
