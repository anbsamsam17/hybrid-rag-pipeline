"""Reproducible build provenance: the ``meta.json`` written next to every index.

``meta.json`` is what makes a build auditable and reproducible: it records a deterministic
fingerprint of the corpus, the git SHA, library versions, and the chunking/embedding config.
A rebuild from the same inputs must yield an identical ``corpus_sha256`` (and identical
retrieval metrics downstream); the single intentionally non-deterministic field is
``created_at`` (provenance only — it is excluded from the corpus hash).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any

from rag.config import Settings
from rag.ingestion.models import Chunk

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
META_FILENAME = "meta.json"

# Key libraries whose installed versions are recorded for reproducibility. Absent ones are
# recorded as ``None`` (graceful), matching environments where backends are not installed.
_TRACKED_LIBS = (
    "qdrant-client",
    "rank-bm25",
    "sentence-transformers",
    "numpy",
    "pydantic",
)


def corpus_sha256(chunks: list[Chunk]) -> str:
    """Deterministic fingerprint of a corpus over its ``(chunk_id, text)`` pairs.

    Fingerprints ``chunk_id`` + ``text`` ONLY; ``ordinal``, ``heading_path``, ``rel_path``,
    and ``metadata`` are **excluded by design**. This is sufficient because ``chunk_id`` is
    ``sha256(rel_path:start:end)`` -- so ``rel_path`` and span boundaries are transitively
    captured, and any change to a chunk's text or span changes the digest. Two corpora that
    differ *only* in heading breadcrumb or non-span metadata hash identically; if
    ``heading_path``/``metadata`` ever become retrieval-relevant, fold a canonical JSON of
    them into the hash here.

    The pairs are **sorted** (so the hash is independent of input ordering) and each is fed
    to the hasher with a length-prefixed, NUL/0x01-delimited encoding. The length prefix and
    delimiters make the encoding unambiguous, preventing concatenation collisions (e.g.
    ``"ab"+"c"`` vs ``"a"+"bc"``). Changing any chunk's id or text changes the digest.
    """
    hasher = hashlib.sha256()
    for chunk_id, text in sorted((c.chunk_id, c.text) for c in chunks):
        hasher.update(f"{chunk_id}\x00{len(text)}\x00{text}\x01".encode())
    return hasher.hexdigest()


def _git_sha() -> str | None:
    """Return the current git commit SHA, or ``None`` if unavailable.

    Gracefully handles a missing ``git`` binary, a non-repo working directory, and any
    subprocess error (returns ``None`` instead of raising).
    """
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _lib_versions() -> dict[str, str | None]:
    """Map ``{"python": version, <lib>: version | None}`` for the tracked libraries."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in _TRACKED_LIBS:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_meta(
    *,
    settings: Settings,
    chunks: list[Chunk],
    n_documents: int,
    embedding_model: str,
    embedding_dim: int,
) -> dict[str, Any]:
    """Assemble the provenance dict for a build (JSON-serializable).

    ``created_at`` (ISO-8601 UTC) is the only non-deterministic field and is deliberately
    excluded from :func:`corpus_sha256` so the corpus fingerprint stays reproducible.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "git_sha": _git_sha(),
        "corpus_sha256": corpus_sha256(chunks),
        "counts": {
            "n_documents": n_documents,
            "n_chunks": len(chunks),
        },
        "embedding": {
            "model": embedding_model,
            "dim": embedding_dim,
        },
        "chunking": {
            "strategy": settings.chunk_strategy,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "vector_store": {
            "collection": settings.qdrant_collection,
            "distance": "cosine",
        },
        "library_versions": _lib_versions(),
    }


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    """Write ``meta`` to ``path`` as stable, sorted, UTF-8 JSON (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
