"""Typed result model for the retrieval stage.

A :class:`RetrievalResult` is the single, self-contained unit the retrieval stage hands to
generation/verification: it carries not only the fused/reranked ``score`` and final 1-based
``rank`` but the chunk's **text and provenance** (``rel_path``, ``heading_path``,
``metadata``) so downstream code never has to re-fetch a payload. It also records which
retrievers surfaced it (``sources``: ``"dense"`` / ``"sparse"``), which is what makes the
hybrid contribution auditable (a result found only by sparse proves the lexical path
mattered).

The model is **frozen** because a result is an immutable record of one retrieval run; the
ranker assigns ``rank``/``score`` once and nothing downstream should mutate them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RetrievalResult(BaseModel):
    """One self-contained retrieved chunk with its final score, rank, text and provenance."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Stable chunk id (matches Chunk.chunk_id).")
    score: float = Field(
        description="Final ranking score: RRF fused score, or the reranker score when "
        "reranking ran. Higher is better; not comparable across configs."
    )
    rank: int = Field(
        ge=1,
        description="Final 1-based rank in the returned list (1 = best). Assigned once, last.",
    )
    text: str = Field(description="The chunk body, fetched from the vector-store payload.")
    rel_path: str = Field(description="Source path relative to the corpus root (provenance).")
    heading_path: list[str] = Field(
        default_factory=list,
        description="Markdown heading breadcrumb for the chunk, e.g. ['Intro', 'Setup'].",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata copied from the stored payload.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Which retrievers surfaced this chunk: subset of {'dense', 'sparse'}, "
        "in a deterministic order (dense before sparse).",
    )
