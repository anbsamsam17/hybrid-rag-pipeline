"""Dense text embedders for the indexing stage.

Two implementations sit behind one :class:`Embedder` contract:

* :class:`SentenceTransformerEmbedder` — the real, model-backed embedder. It
  **lazy-imports** ``sentence_transformers`` so importing this module (and therefore
  ``rag.indexing``) never triggers a model download or even requires the package to be
  installed. The model is loaded on first use.
* :class:`HashingEmbedder` — a deterministic, dependency-free fake used by the test
  suite and any offline path. The same text always maps to the same vector, with no
  randomness, no network, and no numpy (Python ``float`` arithmetic keeps the result
  bit-stable across platforms and processes).

Both produce **L2-normalized** vectors, so cosine similarity equals the dot product and
the downstream :class:`~rag.indexing.vector_store.VectorStore` behaves identically
regardless of which embedder produced the vectors.
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod

from rag.config import Settings
from rag.indexing.sparse import tokenize

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Maps texts to fixed-dimension dense vectors. Backend-agnostic contract."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output vector dimensionality (stable for the life of the instance)."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in input order.

        ``len(out) == len(texts)`` and every vector has length :pyattr:`dim`.
        """


class HashingEmbedder(Embedder):
    """Deterministic, dependency-free fake embedder (the offline/test default).

    Each text is tokenized with the shared :func:`rag.indexing.sparse.tokenize` and folded
    into a fixed-dimension float vector via *signed* feature hashing (a sign bit drawn from
    the same digest reduces collision bias). The vector is then L2-normalized. The mapping
    is a pure function of ``(text, dim)``: identical text yields an identical vector across
    calls, processes, and operating systems.
    """

    def __init__(self, dim: int = 256) -> None:
        """Create a hashing embedder producing ``dim``-dimensional vectors."""
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = dim

    @property
    def dim(self) -> int:
        """Output vector dimensionality."""
        return self._dim

    def _embed_one(self, text: str) -> list[float]:
        """Hash a single text into an L2-normalized vector of length :pyattr:`dim`."""
        vec = [0.0] * self._dim
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self._dim
            sign = 1.0 if digest[8] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vec))
        if norm == 0.0:
            # Empty / token-less text -> all-zeros (never divide by zero).
            return vec
        return [value / norm for value in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed every text deterministically, preserving input order."""
        return [self._embed_one(text) for text in texts]


class SentenceTransformerEmbedder(Embedder):
    """Real embedder backed by ``sentence_transformers`` (lazy-loaded on first use)."""

    def __init__(self, model_name: str) -> None:
        """Store the model name; defer all heavy work until the model is first needed."""
        self._model_name = model_name
        self._model: object | None = None

    def _ensure_model(self) -> object:
        """Lazily import ``sentence_transformers`` and load the model exactly once."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading sentence-transformers model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        """Output dimensionality reported by the loaded model."""
        model = self._ensure_model()
        return int(model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode texts with the model, returning L2-normalized vectors as plain lists."""
        model = self._ensure_model()
        vectors = model.encode(  # type: ignore[attr-defined]
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [list(map(float, row)) for row in vectors.tolist()]


def get_embedder(settings: Settings, *, fake: bool = False) -> Embedder:
    """Return an :class:`Embedder`.

    The real :class:`SentenceTransformerEmbedder` (using ``settings.embedding_model``) is
    returned by default; pass ``fake=True`` for the deterministic, dependency-free
    :class:`HashingEmbedder` used in tests and offline runs.
    """
    if fake:
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)
