"""Pydantic request/response models for the FastAPI service.

These are the *wire contract* of the API — the JSON shapes a client sends and receives.
They are deliberately distinct from the internal domain models
(:class:`~rag.retrieval.models.RetrievalResult`, :class:`~rag.generation.models.Answer`,
:class:`~rag.verification.models.VerificationReport`): the service layer maps domain objects
into these so the HTTP surface can evolve without leaking internal field churn, and so an
internal model gaining a field never silently changes the public payload.

Every model documents its fields (they become the OpenAPI schema). Request models validate
input at the edge — e.g. ``QueryRequest.query`` must be non-empty after stripping — so a
malformed call fails with a clean ``422`` before any retrieval/generation runs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- /query -----------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """A user question plus an optional override of the final result count ``k``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description="The user question. Must be non-empty after stripping whitespace.",
    )
    k: int | None = Field(
        default=None,
        gt=0,
        le=100,
        description="Final number of retrieved results to return. Defaults to "
        "settings.top_k_rerank when omitted. Bounded to keep the rerank batch sane.",
    )

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        """Reject all-whitespace queries (``min_length`` alone accepts ``"   "``)."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty or whitespace-only")
        return stripped


class CitationOut(BaseModel):
    """One citation in the response: the cited chunk plus the quote that supports it."""

    chunk_id: str = Field(description="Stable id of the cited chunk.")
    rel_path: str = Field(description="Source path of the cited chunk (provenance).")
    supporting_quote: str = Field(
        description="The exact snippet from the cited chunk that backs the claim."
    )


class ResultOut(BaseModel):
    """One retrieved chunk surfaced to the client, with score, rank, text and sources."""

    chunk_id: str = Field(description="Stable chunk id.")
    rel_path: str = Field(description="Source path relative to the corpus root (provenance).")
    score: float = Field(description="Final ranking score (RRF or reranker); higher is better.")
    rank: int = Field(ge=1, description="Final 1-based rank in the returned list (1 = best).")
    text: str = Field(description="The chunk body, hydrated from the vector-store payload.")
    sources: list[str] = Field(
        default_factory=list,
        description="Which retrievers surfaced this chunk: subset of {'dense', 'sparse'}.",
    )


class QueryResponse(BaseModel):
    """The full answer payload: text, citations, measured attribution rate, and results."""

    answer: str = Field(description="The generated, citation-grounded answer text.")
    citations: list[CitationOut] = Field(
        default_factory=list,
        description="Per-claim citations carried through from generation.",
    )
    attribution_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="MEASURED grounded-citations / total-citations from verification "
        "(never declared). 0.0 when the answer made no citations.",
    )
    results: list[ResultOut] = Field(
        default_factory=list,
        description="The retrieved chunks the answer was generated over, best first.",
    )
    latency_ms: float = Field(
        ge=0.0, description="Server-measured end-to-end wall time for this query, in ms."
    )


# --- /ingest ----------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Trigger an index (re)build. ``corpus_dir`` overrides ``settings.corpus_dir``."""

    model_config = ConfigDict(extra="forbid")

    corpus_dir: str | None = Field(
        default=None,
        description="Corpus directory to index. Defaults to settings.corpus_dir when omitted.",
    )


class IngestResponse(BaseModel):
    """The result of an index build: document/chunk counts and the artifact paths written."""

    n_documents: int = Field(ge=0, description="Number of source documents loaded.")
    n_chunks: int = Field(ge=0, description="Number of chunks produced and indexed.")
    paths: list[str] = Field(
        default_factory=list,
        description="Artifact paths written by the build (meta.json, bm25.json).",
    )
    latency_ms: float = Field(ge=0.0, description="Server-measured wall time for the build, in ms.")
