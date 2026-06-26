"""Citation-enforced generation: structured Pydantic Answer/Citation outputs.

Public surface:

* :class:`Answer`, :class:`Citation` — the frozen output models.
* :func:`build_grounding_prompt` — the citation-enforced prompt builder.
* :class:`LLMClient` — the generation Protocol.
* :class:`AnthropicLLMClient` — the real client (lazy-imports ``anthropic``, follows the
  ``CLAUDE.md`` SDK rules: adaptive thinking + effort, structured output via
  ``messages.parse(output_format=Answer)``, no temperature/top_p/top_k/budget_tokens, no
  prefill).
* :class:`FakeLLMClient`, :class:`FabricatingFakeLLMClient` — deterministic, dependency-free
  fakes for offline tests (grounded and unsupported paths respectively).
* :func:`get_llm_client` — real/fake factory.
* :func:`generate_answer` — thin orchestrator over a client + the prompt.

``anthropic`` is lazy-imported inside :class:`AnthropicLLMClient`, so ``import rag.generation``
succeeds without the SDK installed.
"""

from __future__ import annotations

from rag.generation.generate import generate_answer
from rag.generation.llm import (
    AnthropicLLMClient,
    FabricatingFakeLLMClient,
    FakeLLMClient,
    LLMClient,
    get_llm_client,
)
from rag.generation.models import Answer, Citation
from rag.generation.prompts import build_grounding_prompt

__all__ = [
    "Answer",
    "Citation",
    "build_grounding_prompt",
    "LLMClient",
    "AnthropicLLMClient",
    "FakeLLMClient",
    "FabricatingFakeLLMClient",
    "get_llm_client",
    "generate_answer",
]
