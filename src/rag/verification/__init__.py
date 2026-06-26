"""Attribution checking: verify each generated claim against its cited source span.

Public surface:

* :func:`verify_answer` — the deterministic, lexical attribution checker. Returns a
  measured :class:`VerificationReport` (no LLM, no network, no locale dependence).
* :class:`VerificationReport`, :class:`CitationCheck` — the frozen result models.
* :data:`OVERLAP_THRESHOLD` — the token-overlap grounding threshold (default 0.6).

The lexical method is the deterministic default; an optional NLI / LLM-judge method is a
documented future extension that slots in per-citation via the ``method`` field without
changing this default (see :mod:`rag.verification.citations`).
"""

from __future__ import annotations

from rag.verification.citations import OVERLAP_THRESHOLD, verify_answer
from rag.verification.models import CitationCheck, VerificationReport

__all__ = [
    "verify_answer",
    "OVERLAP_THRESHOLD",
    "VerificationReport",
    "CitationCheck",
]
