"""Frozen Pydantic models for the generation stage: ``Citation`` and ``Answer``.

These are the *contract* between generation and verification. The whole pipeline's
signal is "every claim is checked against its cited source span", so a citation must
carry three things the verifier can act on deterministically:

* ``chunk_id`` — which retrieved chunk the claim is attributed to. The verifier first
  checks this id is actually among the provided contexts (a fabricated/hallucinated id
  is the #1 way a model games attribution — see :mod:`rag.verification.citations`).
* ``rel_path`` — provenance for display; carried through from the cited
  :class:`~rag.retrieval.models.RetrievalResult` so the answer is self-contained.
* ``supporting_quote`` — the exact snippet from that chunk's text that backs the claim.
  This is the string the lexical grounding check runs against; it is the load-bearing
  field for a *measured* (not declared) ``attribution_rate``.

Both models are **frozen**: an ``Answer`` is an immutable record of one generation run,
mirroring the frozen :class:`RetrievalResult`. Nothing downstream mutates it; the
verifier reads it and produces a separate :class:`~rag.verification.models.VerificationReport`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """One claim's attribution: a cited chunk plus the exact quote that supports it."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        description="Stable id of the cited chunk; MUST match a RetrievalResult.chunk_id "
        "in the contexts handed to generation (else verification marks it ungrounded)."
    )
    rel_path: str = Field(
        description="Source path of the cited chunk, relative to the corpus root (provenance)."
    )
    supporting_quote: str = Field(
        description="The exact snippet from the cited chunk's text that backs the claim. "
        "Verification checks this is lexically grounded in that chunk's text."
    )


class Answer(BaseModel):
    """A grounded, cited answer: free text (with optional inline markers) plus citations."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        description="The answer text. May contain inline citation markers like [1]. If the "
        "context does not contain the answer, this should say so explicitly."
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Per-claim citations. May be empty when the context does not support an "
        "answer; verification handles the 0-citation case explicitly.",
    )
