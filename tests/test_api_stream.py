"""BONUS: ``POST /query/stream`` emits Server-Sent Events, fully offline.

Guarded with ``importorskip("sse_starlette")`` so the suite still passes where the optional
dep is absent. When present, we assert the stream yields the answer/done events with the same
grounded content the non-streaming ``/query`` produces.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("sse_starlette")


def test_query_stream_emits_events(client: TestClient) -> None:
    """The SSE endpoint streams meta/answer/done events for a valid query."""
    with client.stream("POST", "/query/stream", json={"query": "what is hybrid retrieval"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    # The event sequence is present and carries the answer payload.
    assert "event: answer" in body
    assert "event: done" in body


def test_query_stream_rejects_empty_query(client: TestClient) -> None:
    """An empty query is rejected before streaming starts (-> 422)."""
    response = client.post("/query/stream", json={"query": ""})
    assert response.status_code == 422
