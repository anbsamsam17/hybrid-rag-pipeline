"""Dense vector store backed by Qdrant (lazy-imported), with in-memory test support.

The :class:`VectorStore` protocol is the stable contract the build path and the (future)
retrieval stage code against. :class:`QdrantVectorStore` is the concrete implementation; it
**lazy-imports** ``qdrant_client`` inside the constructor so importing this module never
requires the package, and it supports an in-memory client (``location=":memory:"``) so the
test suite needs no running server and no network.

Point ids are **derived deterministically from the chunk id** (UUIDv5 over a fixed
namespace). Qdrant ids must be ``int`` or UUID, and a stable, namespaced UUID makes
re-upserting the same corpus *idempotent*: points overwrite in place rather than
accumulating, so ``count()`` is reproducible across rebuilds. Cosine distance matches the
L2-normalized embeddings produced by both embedders.
"""

from __future__ import annotations

import logging
import uuid
from typing import Protocol, runtime_checkable

from rag.config import Settings
from rag.ingestion.models import Chunk

logger = logging.getLogger(__name__)

# Fixed, project-wide namespace so chunk_id -> point-id is stable forever.
# This is a hard-coded constant (NEVER uuid.uuid4()): regenerating it would change every
# point id and break the idempotency / reproducibility guarantee.
POINT_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def point_id_for(chunk_id: str) -> str:
    """Deterministic Qdrant point id (UUIDv5 string) derived from a ``chunk_id``.

    A pure function of ``chunk_id`` over a fixed namespace: collision-resistant, stable
    across processes/OS, and idempotent under re-upsert.
    """
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


def _payload(chunk: Chunk) -> dict[str, object]:
    """Build the JSON-safe Qdrant payload stored alongside a chunk's vector."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "rel_path": chunk.rel_path,
        "ordinal": chunk.ordinal,
        "heading_path": chunk.heading_path,
        "text": chunk.text,
        "metadata": chunk.metadata,
    }


@runtime_checkable
class VectorStore(Protocol):
    """Minimal dense-store contract used by the build path and retrieval stage."""

    def ensure_collection(self, dim: int) -> None:
        """Create (or recreate) the collection sized for ``dim``-dimensional vectors."""
        ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Upsert one point per chunk; ids are derived from ``chunk_id`` (idempotent)."""
        ...

    def count(self) -> int:
        """Return the exact number of points currently in the collection."""
        ...

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(chunk_id, score)`` pairs for a query vector, best first.

        Retrieval hook for the later retrieval stage; the build path does not call it.
        """
        ...

    def get_payloads(self, chunk_ids: list[str]) -> dict[str, dict[str, object]]:
        """Return ``{chunk_id: payload}`` for the given chunk ids (missing ids omitted).

        Lets the hybrid retriever hydrate text/metadata for fused candidates — including
        sparse-only ones never returned by :meth:`search` — without re-reading the corpus.
        """
        ...


class QdrantVectorStore:
    """Concrete :class:`VectorStore` over Qdrant (server URL or in-memory)."""

    def __init__(
        self,
        *,
        collection: str,
        url: str | None = None,
        location: str | None = None,
    ) -> None:
        """Connect to Qdrant via ``location`` (e.g. ``":memory:"``) or a server ``url``.

        ``qdrant_client`` is imported lazily here, so this module imports cleanly without
        the package installed; only constructing a store requires it.
        """
        if url is None and location is None:
            raise ValueError("exactly one of `url` or `location` must be provided")
        if url is not None and location is not None:
            raise ValueError("provide only one of `url` or `location`, not both")

        from qdrant_client import QdrantClient

        self.collection = collection
        if location is not None:
            self._client = QdrantClient(location=location)
        else:
            self._client = QdrantClient(url=url)

    @classmethod
    def in_memory(cls, collection: str) -> QdrantVectorStore:
        """Construct an in-process store (no server, no network) for tests/offline use."""
        return cls(collection=collection, location=":memory:")

    @classmethod
    def from_settings(cls, settings: Settings) -> QdrantVectorStore:
        """Construct a server-backed store from ``settings`` (url + collection)."""
        return cls(collection=settings.qdrant_collection, url=settings.qdrant_url)

    def ensure_collection(self, dim: int) -> None:
        """Create a fresh collection with cosine distance sized for ``dim`` vectors.

        Always recreates the collection (delete-if-exists + create) so a rebuild yields a
        clean, fully reproducible ``count()`` with **no orphan points**: if the corpus
        shrank between builds, points for removed chunks would otherwise linger (the
        deterministic ids only overwrite chunks that are still present). Recreating also
        prevents silently mixing vector dimensions across builds.

        Note: a fake (256-dim) and a real (e.g. bge-small 384-dim) embedder are **not**
        collection-compatible; recreating here means switching embedders rebuilds cleanly.

        Uses the non-deprecated ``collection_exists``/``delete_collection``/
        ``create_collection`` flow (``recreate_collection`` is deprecated on
        ``qdrant-client >= 1.11``).
        """
        from qdrant_client import models as qm

        if self._client.collection_exists(self.collection):
            logger.info("recreating collection %s (clean rebuild, dim=%d)", self.collection, dim)
            self._client.delete_collection(self.collection)
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Upsert ``chunks`` with their ``vectors`` (length-checked, deterministic ids)."""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks/vectors length mismatch: {len(chunks)} != {len(vectors)}")
        if not chunks:
            return

        from qdrant_client import models as qm

        points = [
            qm.PointStruct(
                id=point_id_for(chunk.chunk_id),
                vector=vector,
                payload=_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self.collection, points=points, wait=True)

    def count(self) -> int:
        """Return the exact point count in the collection."""
        return int(self._client.count(self.collection, exact=True).count)

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        """Return top-``k`` ``(chunk_id, score)`` pairs for ``vector`` (best first).

        Uses the non-deprecated ``query_points`` API (``search`` is deprecated on
        ``qdrant-client >= 1.11``) and requests payloads explicitly. Hits whose payload is
        missing or carries no ``chunk_id`` are skipped rather than raising, so a stray point
        upserted outside this store cannot crash the retrieval hot path.
        """
        response = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=k,
            with_payload=True,
        )
        results: list[tuple[str, float]] = []
        for hit in response.points:
            chunk_id = (hit.payload or {}).get("chunk_id")
            if chunk_id is None:
                logger.warning("search hit %s has no chunk_id payload; skipping", hit.id)
                continue
            results.append((str(chunk_id), float(hit.score)))
        return results

    def get_payloads(self, chunk_ids: list[str]) -> dict[str, dict[str, object]]:
        """Return ``{chunk_id: payload}`` for ``chunk_ids`` (deduped; missing ids omitted).

        Translates each ``chunk_id`` to its deterministic point id via :func:`point_id_for`
        and retrieves the payloads in one round-trip with the non-deprecated ``retrieve``
        API (``with_payload=True``, no vectors). The returned dict is keyed by the payload's
        own ``chunk_id`` (the source of truth), so a point whose id we asked for but whose
        payload lacks a ``chunk_id`` is skipped rather than mis-keyed. Order is irrelevant —
        callers index by ``chunk_id`` — so a dict is the right return type here.

        Used by the hybrid retriever to hydrate text/metadata for **every** fused candidate,
        including sparse-only ids the dense ``search`` never returned.
        """
        if not chunk_ids:
            return {}

        # Dedup while preserving determinism; map point id -> chunk_id for safe lookup.
        unique_ids = list(dict.fromkeys(chunk_ids))
        point_ids = [point_id_for(chunk_id) for chunk_id in unique_ids]

        records = self._client.retrieve(
            collection_name=self.collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        payloads: dict[str, dict[str, object]] = {}
        for record in records:
            payload = record.payload or {}
            chunk_id = payload.get("chunk_id")
            if chunk_id is None:
                logger.warning("payload for point %s has no chunk_id; skipping", record.id)
                continue
            payloads[str(chunk_id)] = dict(payload)
        return payloads
