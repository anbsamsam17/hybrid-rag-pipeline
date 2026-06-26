"""Frozen Pydantic models for the verification stage.

Verification turns an :class:`~rag.generation.models.Answer` into a **measured**
:class:`VerificationReport`: per-citation grounded/not-grounded decisions plus the
aggregate ``attribution_rate``. The headline discipline of this repo is that
``attribution_rate`` is *computed against source spans*, never declared — so these models
carry enough detail (per-check ``method`` and ``reason``) to defend any number after the
fact.

* :class:`CitationCheck` — one citation's verdict: ``grounded`` (bool), the ``method`` that
  decided it (e.g. ``"chunk_not_in_context"``, ``"normalized_substring"``,
  ``"token_overlap"``), and a human-readable ``reason``.
* :class:`VerificationReport` — the aggregate: ``attribution_rate`` (grounded / total),
  the ordered ``checks``, the ``unsupported`` chunk ids, and ``n_citations``.

Both are frozen — a report is an immutable record of one verification run.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CitationCheck(BaseModel):
    """The verdict for a single citation, with the deciding method and a reason."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="The cited chunk id this verdict is about.")
    grounded: bool = Field(
        description="True iff the cited chunk is in context AND its supporting_quote is "
        "lexically grounded in that chunk's text."
    )
    method: str = Field(
        description="Which check produced the verdict: 'chunk_not_in_context', "
        "'normalized_substring', 'token_overlap', or 'no_overlap'."
    )
    reason: str = Field(description="Human-readable explanation of the verdict.")


class VerificationReport(BaseModel):
    """Aggregate verification result: a measured attribution_rate plus per-citation checks."""

    model_config = ConfigDict(frozen=True)

    attribution_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Grounded citations / total citations. An answer with 0 citations is "
        "defined as 0.0 (see verify_answer docstring) — nothing was attributed, so no "
        "attribution was verified.",
    )
    checks: list[CitationCheck] = Field(
        default_factory=list,
        description="One CitationCheck per citation, in the answer's citation order.",
    )
    unsupported: list[str] = Field(
        default_factory=list,
        description="chunk_ids of citations that were NOT grounded, in order of appearance.",
    )
    n_citations: int = Field(
        ge=0, description="Total number of citations examined (== len(checks))."
    )
