"""Ingestion stage: document loading and hand-written, deterministic chunking.

Public surface:

* :func:`load_corpus` — walk a corpus dir into :class:`Document` objects (Obsidian vault
  first, plus generic text and optional PDF/DOCX).
* :func:`get_chunker` — factory returning a strategy (``fixed``/``recursive``/``semantic``).
* :func:`chunk_document` / :func:`chunk_corpus` — chunk one document or a whole corpus
  using the size/overlap/strategy from :class:`~rag.config.Settings`.
* :class:`Document`, :class:`Chunk` — the typed data models.
"""

from __future__ import annotations

from rag.ingestion.chunking import chunk_corpus, chunk_document, get_chunker
from rag.ingestion.loaders import load_corpus
from rag.ingestion.models import Chunk, Document

__all__ = [
    "Document",
    "Chunk",
    "load_corpus",
    "chunk_document",
    "chunk_corpus",
    "get_chunker",
]
