"""Tests for citation-enforced generation, fully offline (FakeLLMClient + pure prompt).

No API key, no network, no SDK: only the deterministic fakes and the pure prompt builder
are exercised. The real :class:`AnthropicLLMClient` is verified only for lazy construction
(it must not import ``anthropic`` at build time).
"""

from __future__ import annotations

from rag.config import Settings
from rag.generation.llm import (
    AnthropicLLMClient,
    FabricatingFakeLLMClient,
    FakeLLMClient,
    get_llm_client,
)
from rag.generation.models import Answer, Citation
from rag.generation.prompts import build_grounding_prompt
from rag.retrieval.models import RetrievalResult


def _ctx(chunk_id: str, text: str, rank: int, rel_path: str = "doc.md") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=1.0,
        rank=rank,
        text=text,
        rel_path=rel_path,
        heading_path=["Intro"],
        metadata={},
        sources=["dense"],
    )


def _contexts() -> list[RetrievalResult]:
    return [
        _ctx("c1", "The mitochondria is the powerhouse of the cell.", rank=1, rel_path="bio.md"),
        _ctx("c2", "Photosynthesis converts sunlight into chemical energy.", rank=2),
    ]


# --- FakeLLMClient.generate_answer ---------------------------------------------------------


def test_fake_returns_answer_with_at_least_one_citation() -> None:
    answer = FakeLLMClient().generate_answer("what is the powerhouse?", _contexts())
    assert isinstance(answer, Answer)
    assert len(answer.citations) >= 1
    assert all(isinstance(c, Citation) for c in answer.citations)


def test_fake_cites_top_context_with_real_substring() -> None:
    contexts = _contexts()
    answer = FakeLLMClient().generate_answer("q", contexts)
    citation = answer.citations[0]
    # Cites the TOP context...
    assert citation.chunk_id == contexts[0].chunk_id
    assert citation.rel_path == contexts[0].rel_path
    # ...with a quote that is a REAL substring of that chunk's text.
    assert citation.supporting_quote in contexts[0].text


def test_fake_no_contexts_returns_zero_citations() -> None:
    answer = FakeLLMClient().generate_answer("q", [])
    assert answer.citations == []
    assert answer.text  # explicit "no answer in context" message, not empty


def test_fake_is_deterministic() -> None:
    contexts = _contexts()
    a = FakeLLMClient().generate_answer("q", contexts)
    b = FakeLLMClient().generate_answer("q", contexts)
    assert a == b  # frozen pydantic models compare by value


def test_fabricating_fake_quote_is_not_in_chunk() -> None:
    contexts = _contexts()
    answer = FabricatingFakeLLMClient().generate_answer("q", contexts)
    citation = answer.citations[0]
    # Cited id is REAL (so the failure is in grounding, not chunk-existence)...
    assert citation.chunk_id == contexts[0].chunk_id
    # ...but the quote is fabricated: not a substring of the chunk.
    assert citation.supporting_quote not in contexts[0].text


# --- Answer model shape --------------------------------------------------------------------


def test_answer_models_are_frozen() -> None:
    import pydantic
    import pytest

    answer = FakeLLMClient().generate_answer("q", _contexts())
    with pytest.raises(pydantic.ValidationError):
        answer.text = "mutated"  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError):
        answer.citations[0].chunk_id = "x"  # type: ignore[misc]


# --- build_grounding_prompt ----------------------------------------------------------------


def test_prompt_contains_query_and_each_context_id_and_text() -> None:
    query = "what is the powerhouse of the cell?"
    contexts = _contexts()
    prompt = build_grounding_prompt(query, contexts)
    assert query in prompt
    for ctx in contexts:
        assert ctx.chunk_id in prompt  # the model can cite by id
        assert ctx.text in prompt  # the source text is present to quote from


def test_prompt_enforces_citation_and_no_guessing() -> None:
    prompt = build_grounding_prompt("q", _contexts()).lower()
    # The contract the verifier later measures must be stated in the prompt.
    assert "only" in prompt  # answer ONLY from context
    assert "chunk_id" in prompt  # cite by chunk_id
    assert "quote" in prompt  # include the exact supporting quote


def test_prompt_is_deterministic() -> None:
    contexts = _contexts()
    assert build_grounding_prompt("q", contexts) == build_grounding_prompt("q", contexts)


def test_prompt_handles_no_contexts() -> None:
    prompt = build_grounding_prompt("q", [])
    assert "q" in prompt
    assert "none" in prompt.lower()


# --- factory + lazy real client ------------------------------------------------------------


def test_get_llm_client_fake_is_fake() -> None:
    assert isinstance(get_llm_client(Settings(), fake=True), FakeLLMClient)


def test_get_llm_client_real_is_lazy_no_import_at_construction() -> None:
    # Constructing the real client must NOT import anthropic (lazy on first generate call).
    client = get_llm_client(Settings(), fake=False)
    assert isinstance(client, AnthropicLLMClient)
    assert client._client is None  # SDK client not built yet
