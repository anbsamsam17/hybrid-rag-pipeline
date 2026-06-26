"""Thin generation orchestrator.

:func:`generate_answer` is a deliberately thin pass-through over an injected
:class:`~rag.generation.llm.LLMClient`. The citation-enforced prompt is built inside the
client (via :func:`~rag.generation.prompts.build_grounding_prompt`), so this layer only
wires the dependency-injected client and settings together — keeping the module boundary
clean and the path fully offline-testable with a :class:`~rag.generation.llm.FakeLLMClient`.

Settings is accepted (and currently unused beyond being passed through the client) so the
signature is stable as generation grows config-dependent (e.g. max-context caps), matching
the rest of the codebase's "pass settings as a param" convention.
"""

from __future__ import annotations

import logging

from rag.config import Settings
from rag.generation.llm import LLMClient
from rag.generation.models import Answer
from rag.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


def generate_answer(
    query: str,
    contexts: list[RetrievalResult],
    *,
    llm: LLMClient,
    settings: Settings,
) -> Answer:
    """Generate a grounded, cited :class:`Answer` for ``query`` over ``contexts``.

    Args:
        query: The user's question.
        contexts: Retrieved chunks, best first.
        llm: The (dependency-injected) client that performs generation.
        settings: Pipeline settings (passed for forward-compatibility / consistency).

    Returns:
        The :class:`Answer` produced by ``llm``. Verification is a separate, explicit step
        (:func:`rag.verification.citations.verify_answer`) — generation never self-certifies.
    """
    logger.debug("generate_answer: model=%s contexts=%d", settings.llm_model, len(contexts))
    return llm.generate_answer(query, contexts)
