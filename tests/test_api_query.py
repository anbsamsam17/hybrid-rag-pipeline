"""``POST /query`` over a tiny in-memory-indexed vault, fully offline.

The ``client`` fixture (see ``conftest.py``) injects a :class:`RagService` built on a
HashingEmbedder + in-memory Qdrant (via the real ``build_index``) + FakeLLMClient + fake
reranker — so these assert the genuine retrieve -> generate -> verify -> assemble path with
no network, model, or server. The FakeLLMClient quotes a real substring of the top chunk, so
the measured ``attribution_rate`` is exactly ``1.0``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from .conftest import build_offline_service

# The secret a failing collaborator carries: assertions prove it never reaches the response.
_SECRET_INTERNAL_DETAIL = "INTERNAL secret span mismatch at vector index 42 — do not leak"


class _RaisingService:
    """A stub :class:`RagService` whose downstream raises a chosen exception on every query.

    Duck-typed for the ``/query`` route (the app only calls ``answer_query``): exercises the
    failure path without any real retrieval/generation, proving the API maps a deep exception
    to a clean ``500`` and never leaks its message or a traceback.
    """

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def answer_query(self, query: str, k: int | None = None):
        """Always raise the configured exception (simulates a downstream fault)."""
        raise self._exc


def _client_for(service: object) -> TestClient:
    """Build a TestClient over an app with ``service`` injected; surface 500s as responses."""
    from rag.api import create_app

    app = create_app(service=service)
    # raise_server_exceptions=False so a 500 comes back as a response (not re-raised in-test).
    return TestClient(app, raise_server_exceptions=False)


def test_query_returns_grounded_answer(client: TestClient) -> None:
    """A well-formed query yields an answer, >=1 grounded citation, and full attribution."""
    response = client.post("/query", json={"query": "what is hybrid retrieval"})
    assert response.status_code == 200

    body = response.json()
    assert body["answer"]  # non-empty answer text
    assert len(body["citations"]) >= 1
    # The fake cites a REAL substring of the top chunk -> verification measures 1.0.
    assert body["attribution_rate"] == 1.0

    # Each citation carries provenance + the supporting quote.
    for citation in body["citations"]:
        assert citation["chunk_id"]
        assert citation["rel_path"]
        assert citation["supporting_quote"]

    # Results are populated and self-contained (text hydrated from the payload).
    assert body["results"]
    for result in body["results"]:
        assert result["text"]
        assert result["rel_path"]
        assert result["rank"] >= 1
        assert set(result["sources"]) <= {"dense", "sparse"}

    assert body["latency_ms"] >= 0.0


def test_query_respects_k(client: TestClient) -> None:
    """``k`` caps the number of returned results (3-note vault -> exactly k for k in 1,2)."""
    for k in (1, 2):
        response = client.post("/query", json={"query": "hybrid retrieval evaluation", "k": k})
        assert response.status_code == 200
        assert len(response.json()["results"]) == k


def test_query_empty_string_is_rejected(client: TestClient) -> None:
    """An empty query is rejected at the edge (pydantic min_length -> 422)."""
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422


def test_query_whitespace_only_is_rejected(client: TestClient) -> None:
    """A whitespace-only query is rejected by the field validator (-> 422)."""
    response = client.post("/query", json={"query": "   "})
    assert response.status_code == 422


def test_query_missing_field_is_422(client: TestClient) -> None:
    """A body missing the required ``query`` field is a validation error."""
    response = client.post("/query", json={"nope": 1})
    assert response.status_code == 422


def test_query_malformed_body_is_422(client: TestClient) -> None:
    """Wrong-typed / extra fields fail validation cleanly with no stack trace leaked."""
    # k must be an int; a string is invalid.
    bad_type = client.post("/query", json={"query": "ok", "k": "lots"})
    assert bad_type.status_code == 422
    assert "error" in bad_type.json()
    # extra="forbid" on QueryRequest rejects unknown fields.
    extra = client.post("/query", json={"query": "ok", "surprise": True})
    assert extra.status_code == 422


def test_query_error_body_has_no_traceback(client: TestClient) -> None:
    """Validation errors return a structured body, never a Python traceback string."""
    response = client.post("/query", json={"query": ""})
    body = response.json()
    assert "Traceback" not in response.text
    assert body["error"] == "validation error"
    assert "request_id" in body


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError(_SECRET_INTERNAL_DETAIL),
        # A deep ValueError MUST NOT be misclassified as a 400 nor leak its message: the
        # blanket ValueError->400 handler is gone, so this falls through to the generic 500.
        ValueError(_SECRET_INTERNAL_DETAIL),
    ],
    ids=["runtime_error", "value_error"],
)
def test_query_downstream_failure_is_clean_500(exc: Exception) -> None:
    """A downstream exception yields a uniform 500 with no leaked message and no traceback.

    Discriminating on two axes: (1) status is exactly 500 — a ValueError is NOT turned into a
    400; (2) neither the secret internal message nor the word "Traceback" appears anywhere in
    the response, and the body is the uniform ``{"error": "internal server error", ...}``.
    """
    test_client = _client_for(_RaisingService(exc))
    response = test_client.post("/query", json={"query": "what is hybrid retrieval"})

    # A ValueError no longer becomes a 400 — it is a server fault (500), like any exception.
    assert response.status_code == 500
    # No internal detail and no traceback leak into the body or any header value.
    assert _SECRET_INTERNAL_DETAIL not in response.text
    assert "Traceback" not in response.text

    body = response.json()
    assert body == {"error": "internal server error", "request_id": body.get("request_id")}
    assert body["request_id"]  # the middleware-assigned correlation id is present


def test_query_surfaces_measured_attribution_below_one(vault: Path, tmp_path: Path) -> None:
    """The API carries a MEASURED ``attribution_rate < 1.0`` for an ungrounded citation.

    Uses ``FabricatingFakeLLMClient`` (cites a real chunk but with a fabricated quote), so
    verification marks the citation unsupported and the measured rate drops below 1.0. This is
    discriminating against a hard-coded/declared rate: the grounded fake yields exactly 1.0
    (see ``test_query_returns_grounded_answer``), while this same HTTP path yields < 1.0 only
    because the rate is measured end-to-end through verification, not asserted.
    """
    pytest.importorskip("qdrant_client")
    pytest.importorskip("rank_bm25")

    from rag.generation.llm import FabricatingFakeLLMClient

    service = build_offline_service(vault, tmp_path / "storage", llm=FabricatingFakeLLMClient())
    test_client = _client_for(service)

    response = test_client.post("/query", json={"query": "what is hybrid retrieval"})
    assert response.status_code == 200

    body = response.json()
    # A citation is still emitted (real chunk_id) but its quote is fabricated -> ungrounded.
    assert len(body["citations"]) >= 1
    # The headline anti-hallucination signal: measured rate is in [0, 1) — strictly below 1.0.
    assert 0.0 <= body["attribution_rate"] < 1.0
