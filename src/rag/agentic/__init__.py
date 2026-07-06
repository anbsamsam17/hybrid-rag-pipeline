"""Optional self-corrective RAG implemented as a LangGraph StateGraph.

Public surface for ``rag.agentic``. Importing this package is cheap and dependency-light:
``langgraph`` is only imported lazily inside :func:`build_corrective_rag`, and ``anthropic``
only inside :class:`AnthropicCorrectiveLLM`'s first call — mirroring the rest of the repo's
lazy-import discipline. This layer is optional and off by default (``settings.agentic_enabled``);
the base retrieve -> generate -> verify path works without it.
"""

from __future__ import annotations

from rag.agentic.corrective_rag import (
    AnthropicCorrectiveLLM,
    CorrectiveLLM,
    CorrectiveRAGRequest,
    CorrectiveRAGResult,
    CorrectiveRAGState,
    DocRelevanceGrade,
    FakeCorrectiveLLM,
    GradeResponse,
    RewrittenQuery,
    build_corrective_rag,
    route_after_grade,
    route_after_verify,
    run_corrective_rag,
)

__all__ = [
    "AnthropicCorrectiveLLM",
    "CorrectiveLLM",
    "CorrectiveRAGRequest",
    "CorrectiveRAGResult",
    "CorrectiveRAGState",
    "DocRelevanceGrade",
    "FakeCorrectiveLLM",
    "GradeResponse",
    "RewrittenQuery",
    "build_corrective_rag",
    "route_after_grade",
    "route_after_verify",
    "run_corrective_rag",
]
