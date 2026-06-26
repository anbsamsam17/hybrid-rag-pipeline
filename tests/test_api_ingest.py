"""``POST /ingest`` builds the index over a supplied ``corpus_dir``, fully offline.

The injected service ingests with a HashingEmbedder + in-memory Qdrant store, so the build
runs with no model download and no Qdrant server. We point ``corpus_dir`` at a freshly
written temp corpus and assert the returned counts match what we wrote.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def _write_corpus(root: Path, n: int) -> Path:
    """Write ``n`` single-chunk notes into a fresh corpus dir and return it."""
    corpus = root / "ingest_corpus"
    corpus.mkdir()
    for i in range(n):
        (corpus / f"note{i}.md").write_text(
            f"# Note {i}\n\nUnique body text number {i} about retrieval and evaluation.\n",
            encoding="utf-8",
        )
    return corpus


def test_ingest_reports_correct_counts(client: TestClient, tmp_path: Path) -> None:
    """Ingesting a 3-note corpus returns n_documents == 3 and matching chunk counts."""
    corpus = _write_corpus(tmp_path, 3)
    response = client.post("/ingest", json={"corpus_dir": str(corpus)})
    assert response.status_code == 200

    body = response.json()
    assert body["n_documents"] == 3
    # Each tiny note is a single chunk under chunk_size=1024.
    assert body["n_chunks"] == 3
    # The build advertises the artifact paths it wrote (meta.json + bm25.json), which exist.
    assert body["paths"]
    for path in body["paths"]:
        assert Path(path).exists()
    assert body["latency_ms"] >= 0.0


def test_ingest_default_corpus_dir(client: TestClient) -> None:
    """Omitting ``corpus_dir`` ingests the service's settings.corpus_dir (the seeded vault)."""
    response = client.post("/ingest", json={})
    assert response.status_code == 200
    body = response.json()
    # The conftest vault has 3 notes.
    assert body["n_documents"] == 3
    assert body["n_chunks"] >= 3


def test_ingest_empty_corpus_reports_zero(client: TestClient, tmp_path: Path) -> None:
    """An empty corpus dir ingests cleanly with zero counts (no crash)."""
    empty = tmp_path / "empty_corpus"
    empty.mkdir()
    response = client.post("/ingest", json={"corpus_dir": str(empty)})
    assert response.status_code == 200
    body = response.json()
    assert body["n_documents"] == 0
    assert body["n_chunks"] == 0


def test_ingest_malformed_body_is_422(client: TestClient) -> None:
    """A wrong-typed corpus_dir is a validation error (-> 422), not a 500."""
    response = client.post("/ingest", json={"corpus_dir": 123})
    assert response.status_code == 422
