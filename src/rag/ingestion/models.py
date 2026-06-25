"""Typed data models for the ingestion stage.

A :class:`Document` is one source file after loading + cleaning; a :class:`Chunk` is a
contiguous slice of that document's ``text`` produced by a chunker. Both are **frozen**
Pydantic models because immutability is part of the determinism contract: a chunk's id
is derived from its fields, so those fields must never drift after construction.

Identity is fully deterministic and anchored on ``rel_path`` (the path relative to the
corpus root, POSIX-normalized) so the same corpus produces identical ids on Windows and
Linux:

* ``doc_id  = sha256(rel_path).hexdigest()[:16]``
* ``chunk_id = sha256(f"{rel_path}:{start}:{end}").hexdigest()[:16]``

Offsets (``start`` / ``end``) are **character** offsets into ``Document.text`` — Python
slicing is codepoint-based, so unicode is handled correctly as long as we never convert
to bytes for offset math.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _short_sha(value: str) -> str:
    """Return the first 16 hex chars of the UTF-8 SHA-256 of ``value``.

    The explicit ``utf-8`` encode keeps ids platform-independent and unicode-safe.
    This is the single source of the id formula — never inline it elsewhere.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def make_doc_id(rel_path: str) -> str:
    """Deterministic document id derived solely from the POSIX ``rel_path``."""
    return _short_sha(rel_path)


def make_chunk_id(rel_path: str, start: int, end: int) -> str:
    """Deterministic chunk id derived from ``rel_path`` and the char span.

    Formula (load-bearing — tested directly): ``sha256(f"{rel_path}:{start}:{end}")[:16]``.
    """
    return _short_sha(f"{rel_path}:{start}:{end}")


class Document(BaseModel):
    """A single loaded source document with cleaned, offset-stable body text."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="sha256(rel_path)[:16] — deterministic from rel path.")
    source_path: str = Field(description="Absolute path as a string (JSON-safe, not Path).")
    rel_path: str = Field(
        description="Path relative to the corpus root, POSIX-normalized — the id anchor."
    )
    title: str = Field(description="Frontmatter title -> first H1 -> filename stem.")
    text: str = Field(
        description="Clean body text after frontmatter strip + wikilink render; "
        "the exact string chunk offsets index into."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="{folder, tags, wikilinks, frontmatter, source_format}.",
    )


class Chunk(BaseModel):
    """A contiguous slice of a :class:`Document` produced by a chunker.

    Invariant **T** (fixed/recursive, and header-aware which reuses the recursive packer):
    ``text == parent_document.text[start:end]``.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="sha256(f'{rel_path}:{start}:{end}')[:16].")
    doc_id: str = Field(description="Parent Document.doc_id.")
    source_path: str = Field(description="Copied from the parent document.")
    rel_path: str = Field(description="Copied from the parent document (the id-relevant path).")
    ordinal: int = Field(description="0-based position of this chunk within the document.")
    text: str = Field(description="The chunk body; equals doc.text[start:end].")
    start: int = Field(description="Inclusive character offset into Document.text.")
    end: int = Field(description="Exclusive character offset into Document.text.")
    heading_path: list[str] = Field(
        default_factory=list,
        description="Markdown heading breadcrumb, e.g. ['Intro', 'Setup']; [] for "
        "fixed/recursive strategies.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Inherited document metadata + {strategy, chunk_index_in_doc}.",
    )
