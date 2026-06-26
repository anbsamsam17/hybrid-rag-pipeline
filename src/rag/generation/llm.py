"""LLM clients for citation-enforced generation.

Three things live here behind one :class:`LLMClient` Protocol:

* :class:`AnthropicLLMClient` — the REAL client. It lazy-imports the official ``anthropic``
  SDK (so ``import rag.generation`` works with the SDK uninstalled), and obeys the
  project's BINDING Anthropic SDK rules (see ``CLAUDE.md`` §"LLM / Anthropic SDK rules"):
  default model ``settings.llm_model``, **adaptive thinking** + **effort**, and
  **structured output** straight into the :class:`~rag.generation.models.Answer` Pydantic
  model via ``client.messages.parse(output_format=Answer)`` → ``response.parsed_output``.
  It NEVER passes ``temperature``/``top_p``/``top_k``/``budget_tokens`` and NEVER uses
  assistant-prefill (all 400 on the 4.x models this repo targets).

* :class:`FakeLLMClient` — a DETERMINISTIC, dependency-free, network-free, key-free fake for
  tests and offline runs. Given contexts, it quotes the TOP context with a real substring,
  so verifying the fake's own output passes (``attribution_rate == 1.0``). A subclass knob
  (:class:`FabricatingFakeLLMClient`) instead emits a fabricated quote so the unsupported
  path is testable (``attribution_rate < 1.0``).

* :func:`get_llm_client` — factory: real by default, fake on request.

The Protocol method is ``generate_answer(query, contexts) -> Answer`` — the prompt
construction lives in :mod:`rag.generation.prompts`, so both clients format contexts the
same way and the orchestrator in :mod:`rag.generation.generate` stays a thin pass-through.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from rag.config import Settings
from rag.generation.models import Answer, Citation
from rag.generation.prompts import build_grounding_prompt
from rag.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

# Max output tokens for the (non-streaming) structured-output call. Answers are small
# JSON objects; this is comfortably under the SDK's non-streaming timeout guard.
_MAX_TOKENS = 4096
# Effort for the generation call. "high" is the default; generation is correctness-
# sensitive (a wrong citation destroys the headline signal) so we keep it explicit.
_EFFORT = "high"

# Snippet length the fakes quote from the top context. Long enough to be a meaningful,
# non-trivial substring; short enough to stay inside short test fixtures.
_FAKE_QUOTE_CHARS = 60


@runtime_checkable
class LLMClient(Protocol):
    """Generate a grounded, cited :class:`Answer` for a query over retrieved contexts."""

    def generate_answer(self, query: str, contexts: list[RetrievalResult]) -> Answer:
        """Return a grounded :class:`Answer` (text + citations) for ``query``/``contexts``."""
        ...


class AnthropicLLMClient:
    """Real Anthropic-SDK client: adaptive thinking + effort + structured Answer output.

    The ``anthropic`` import is deferred to the first :meth:`generate_answer` call, so
    constructing this client (and importing the package) never requires the SDK or touches
    the network. Construct it freely; it only calls out when you generate.
    """

    def __init__(self, settings: Settings) -> None:
        """Store config; defer SDK import and client construction until first use."""
        self._settings = settings
        self._client: object | None = None

    def _ensure_client(self) -> object:
        """Lazily import ``anthropic`` and build the SDK client once."""
        if self._client is None:
            import anthropic  # lazy: package imports fine without the SDK installed

            # api_key may be None here; the SDK also reads ANTHROPIC_API_KEY from the env.
            self._client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        return self._client

    def generate_answer(self, query: str, contexts: list[RetrievalResult]) -> Answer:
        """Generate a grounded :class:`Answer` via the Anthropic Messages API.

        Uses ``client.messages.parse(output_format=Answer)`` so the SDK validates the
        response straight into the frozen :class:`Answer` model and returns it on
        ``response.parsed_output``. Follows ``CLAUDE.md`` exactly: ``settings.llm_model``,
        adaptive thinking, an effort setting, and NO ``temperature``/``top_p``/``top_k``/
        ``budget_tokens`` and NO assistant-prefill.
        """
        client = self._ensure_client()
        prompt = build_grounding_prompt(query, contexts)

        logger.info(
            "generating answer: model=%s contexts=%d", self._settings.llm_model, len(contexts)
        )
        # EXACT structured-output call per CLAUDE.md. messages.parse validates the model
        # output against the Answer pydantic schema and exposes it on .parsed_output.
        response = client.messages.parse(  # type: ignore[attr-defined]
            model=self._settings.llm_model,
            max_tokens=_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": _EFFORT},
            output_format=Answer,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = response.parsed_output
        if parsed is None:
            # parsed_output is None when the model refused or hit max_tokens before valid
            # JSON. Surface an explicit, ungrounded (0-citation) answer rather than crash.
            logger.warning("messages.parse returned no parsed_output; returning empty answer")
            return Answer(
                text="The model did not return a parseable grounded answer.",
                citations=[],
            )
        return parsed


class FakeLLMClient:
    """Deterministic, dependency-free fake: cites the TOP context with a real substring.

    No network, no API key, no SDK. Given non-empty contexts, returns an :class:`Answer`
    whose text references the top chunk and whose single citation quotes a REAL substring
    of that chunk's text — so running verification on the fake's own output yields
    ``attribution_rate == 1.0``. With no contexts, returns an explicit 0-citation answer.
    """

    def generate_answer(self, query: str, contexts: list[RetrievalResult]) -> Answer:
        """Return a deterministic grounded answer citing the top context, or a refusal."""
        if not contexts:
            return Answer(
                text="The provided context does not contain an answer to this question.",
                citations=[],
            )
        top = contexts[0]
        quote = self._quote(top)
        text = f"Based on [{1}] ({top.chunk_id}): {quote}"
        return Answer(
            text=text,
            citations=[
                Citation(chunk_id=top.chunk_id, rel_path=top.rel_path, supporting_quote=quote)
            ],
        )

    def _quote(self, ctx: RetrievalResult) -> str:
        """Return a real substring of the context text (deterministic, leading slice)."""
        return ctx.text[:_FAKE_QUOTE_CHARS]


class FabricatingFakeLLMClient(FakeLLMClient):
    """Fake variant that emits a FABRICATED quote (not present in the cited chunk).

    Exercises the *unsupported* verification path: the cited ``chunk_id`` is real, but the
    supporting quote is not lexically grounded in that chunk, so verification marks the
    citation ungrounded and ``attribution_rate < 1.0``. Used by tests of the failure path.
    """

    # A sentinel quote engineered to share no meaningful tokens with any plausible chunk.
    _FABRICATED_QUOTE = "zzqxv totally fabricated unsupported claim wzzqxv"

    def _quote(self, ctx: RetrievalResult) -> str:
        """Return a fabricated quote that is NOT a substring/overlap of the chunk text."""
        return self._FABRICATED_QUOTE


def get_llm_client(settings: Settings, *, fake: bool = False) -> LLMClient:
    """Return an :class:`LLMClient`.

    The real :class:`AnthropicLLMClient` is returned by default (it lazy-imports the SDK on
    first use); pass ``fake=True`` for the deterministic, dependency-free
    :class:`FakeLLMClient` used in tests and offline runs.
    """
    if fake:
        return FakeLLMClient()
    return AnthropicLLMClient(settings)
