"""Sparse (BM25) index: deterministic tokenizer + a JSON-persisted BM25 index.

The tokenizer is the *single* tokenizer for the whole package — the
:class:`~rag.indexing.embeddings.HashingEmbedder` reuses it, so dense and sparse paths
agree on what a "token" is. It is pure Python with no dependencies, so the tokenizer test
runs with zero optional backends installed.

:class:`BM25Index` lazy-imports ``rank_bm25`` only inside ``build``/``load`` (so importing
this module never requires it) and persists to **JSON, not pickle**: we store the tokenized
corpus, the parallel ``chunk_id`` order, and the BM25 parameters, then *rebuild* the fitted
model on load. JSON is portable, diffable, and safe (no code execution on load); BM25 stats
are a deterministic function of the tokenized corpus, so a reload reproduces identical
scores.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from rag.ingestion.models import Chunk

logger = logging.getLogger(__name__)

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

BM25_FILENAME = "bm25.json"
SCHEMA_VERSION = 1
TOKENIZER_ID = "lower_alnum_v1"

# BM25 Okapi defaults (Robertson/Sparck-Jones); recorded in the persisted file.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    """Deterministic tokenizer: lowercase, split on runs of non-alphanumeric, drop empties.

    Pure Python, no dependencies. This is THE single tokenizer used across the package
    (the hashing embedder reuses it). Identical input always yields identical tokens.
    """
    return [token for token in _TOKEN_SPLIT_RE.split(text.lower()) if token]


class BM25Index:
    """A BM25 sparse index over chunk texts, persisted as JSON and rebuilt on load."""

    def __init__(self, *, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        """Create an empty index with the given BM25 parameters."""
        self.chunk_ids: list[str] = []
        self.corpus_tokens: list[list[str]] = []
        self.params: dict[str, float] = {"k1": k1, "b": b}
        self._bm25: Any | None = None

    def build(self, chunks: list[Chunk]) -> None:
        """Fit BM25 over ``chunks``, preserving their (deterministic) input order.

        ``rank_bm25`` is imported lazily here so the module imports without it installed.
        """
        from rank_bm25 import BM25Okapi

        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.corpus_tokens = [tokenize(chunk.text) for chunk in chunks]
        # BM25Okapi requires a non-empty corpus; guard so build() on no chunks is a no-op.
        if self.corpus_tokens:
            self._bm25 = BM25Okapi(
                self.corpus_tokens,
                k1=self.params["k1"],
                b=self.params["b"],
            )
        else:
            self._bm25 = None
        logger.info("built BM25 index over %d chunks", len(self.chunk_ids))

    def save(self, directory: Path) -> Path:
        """Persist the tokenized corpus + ids + params to ``directory/bm25.json``.

        Returns the path written. JSON (not pickle) keeps the artifact portable and safe;
        ``sort_keys`` stabilizes the *top-level* byte output for diffing, while the
        ``chunk_ids`` / ``corpus_tokens`` lists are never reordered (their order is the
        load-bearing corpus order).
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / BM25_FILENAME
        payload = {
            "schema_version": SCHEMA_VERSION,
            "tokenizer": TOKENIZER_ID,
            "params": self.params,
            "chunk_ids": self.chunk_ids,
            "corpus_tokens": self.corpus_tokens,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        """Load ``directory/bm25.json`` and rebuild the fitted BM25 model from tokens."""
        path = directory / BM25_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported BM25 schema_version {version!r} (expected {SCHEMA_VERSION})"
            )
        params = payload["params"]
        index = cls(k1=float(params["k1"]), b=float(params["b"]))
        index.chunk_ids = list(payload["chunk_ids"])
        index.corpus_tokens = [list(tokens) for tokens in payload["corpus_tokens"]]
        if index.corpus_tokens:
            from rank_bm25 import BM25Okapi

            index._bm25 = BM25Okapi(
                index.corpus_tokens,
                k1=index.params["k1"],
                b=index.params["b"],
            )
        return index

    def query(self, text: str, k: int) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(chunk_id, score)`` pairs for ``text``, best first.

        Ties are broken by original corpus index (never by ``chunk_id`` string and never an
        unstable sort on score alone), so equal-scored results keep a stable, reproducible
        order — important for downstream rank fusion.
        """
        if self._bm25 is None or not self.chunk_ids:
            return []
        scores = self._bm25.get_scores(tokenize(text))
        ranked = sorted(
            enumerate(scores),
            key=lambda pair: (-float(pair[1]), pair[0]),
        )
        top = ranked[: max(0, k)]
        return [(self.chunk_ids[index], float(score)) for index, score in top]
