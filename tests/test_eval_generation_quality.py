"""Tests for the RAGAS-style generation-quality eval (ADR-0009), run fully OFFLINE.

Mirrors ``tests/test_eval_attribution.py`` / ``tests/test_eval_corrective.py``: the full-run tests
exercise the genuine hermetic build -> retrieve (hybrid+rerank) -> generate -> score path with the
proven offline backends (:class:`HashingEmbedder` + in-memory Qdrant +
:class:`LexicalOverlapReranker`) and the DETERMINISTIC fakes (:class:`FakeLLMClient`,
:class:`FakeFaithfulnessScorer`, :class:`FakeAnswerRelevancyScorer`) — no torch, no network, no API
key, no real LLM/scorer value ever asserted.

They assert STRUCTURE, INVARIANTS, DETERMINISM, and the load-bearing design guards:
* a fabricated/unsupported answer drives faithfulness **< 1.0** (the guard against an always-
  "supported" scorer stuck at 1.0),
* a grounded answer scores faithfulness 1.0,
* a noncommittal/refusal answer scores answer_relevancy 0.0 and is counted in ``n_noncommittal``,
* the scorers are BLIND (see only ``(question, answer, contexts)`` — never ``reference_answer`` /
  ``relevant_chunk_ids``),
* ``publishable`` flips ``False`` on ANY fake (including either scorer),
* the micro-vs-macro-vs-abstention aggregation math is exact on a hand-checked fixture,
* the real scorers fail CLOSED (unparseable ⇒ not supported / relevancy 0.0).
"""

from __future__ import annotations

from pathlib import Path

import pytest

# The generation-quality build path needs the lightweight index backends; skip the whole module
# cleanly (not a silent pass) if they are unavailable.
pytest.importorskip("qdrant_client")
pytest.importorskip("rank_bm25")

from rag.config import PROJECT_ROOT, Settings  # noqa: E402
from rag.eval.generation_quality import (  # noqa: E402
    _PUBLISHABLE_ANSWER_RELEVANCY,
    _PUBLISHABLE_EMBEDDER,
    _PUBLISHABLE_FAITHFULNESS,
    _PUBLISHABLE_LLM,
    _PUBLISHABLE_RERANKER,
    CONFIG,
    GENERATION_QUALITY_RESULTS_FILENAME,
    _is_publishable,
    render_generation_quality_report,
    run_generation_quality_eval,
)
from rag.eval.generation_scorers import (  # noqa: E402
    AnswerRelevancyResult,
    AnswerRelevancyScorer,
    AnthropicAnswerRelevancyScorer,
    AnthropicFaithfulnessScorer,
    FaithfulnessResult,
    FaithfulnessScorer,
    FakeAnswerRelevancyScorer,
    FakeFaithfulnessScorer,
    StatementVerdict,
    _DecomposedStatements,
    _FaithfulnessVerdicts,
    cosine_similarity,
    faithfulness_value,
    is_noncommittal,
    split_statements,
    token_overlap_fraction,
)
from rag.eval.golden import load_golden  # noqa: E402
from rag.eval.judge import default_judge_model  # noqa: E402
from rag.eval.models import (  # noqa: E402
    GenerationQualityProvenance,
    GenerationQualityQueryRecord,
    GenerationQualityReport,
)
from rag.generation.llm import (  # noqa: E402
    AnthropicLLMClient,
    FabricatingFakeLLMClient,
    FakeLLMClient,
)
from rag.indexing.embeddings import HashingEmbedder, SentenceTransformerEmbedder  # noqa: E402
from rag.indexing.vector_store import QdrantVectorStore  # noqa: E402
from rag.retrieval.models import RetrievalResult  # noqa: E402
from rag.retrieval.rerank import CrossEncoderReranker, LexicalOverlapReranker  # noqa: E402

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "golden.jsonl"

# Derive n from the committed golden set — NEVER hard-code it.
GOLDEN = load_golden(GOLDEN_PATH)
N_GOLDEN = len(GOLDEN)


# --- Small deterministic collaborators ---------------------------------------------------------


def _ctx(chunk_id: str, text: str, rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=1.0,
        rank=rank,
        text=text,
        rel_path=f"{chunk_id}.md",
        heading_path=[],
        metadata={},
        sources=["dense"],
    )


class _StubResponse:
    """Duck-types the ``messages.parse`` response with a ``.parsed_output`` (None = unparseable)."""

    def __init__(self, parsed: object) -> None:
        self.parsed_output = parsed


class _StubMessages:
    def __init__(self, responses: list[_StubResponse]) -> None:
        self._responses = responses
        self._index = 0

    def parse(self, **_kwargs: object) -> _StubResponse:
        response = self._responses[self._index]
        self._index += 1
        return response


class _StubClient:
    """A stand-in Anthropic client that replays canned ``messages.parse`` responses in order."""

    def __init__(self, responses: list[_StubResponse]) -> None:
        self.messages = _StubMessages(responses)


class _StubFaithfulnessScorer:
    """Query-keyed faithfulness stub forcing micro != macro != macro_answered + one abstention.

    * ``half_query`` -> 2 statements, 1 supported (per-answer faithfulness 0.5),
    * ``abstain_query`` -> 0 statements (abstention; faithfulness 0.0, faith_abstained),
    * every other query -> 1 statement, 1 supported (faithfulness 1.0).
    """

    model = "stub-faith"

    def __init__(self, *, half_query: str, abstain_query: str) -> None:
        self._half = half_query
        self._abstain = abstain_query

    def score(
        self, question: str, answer: str, contexts: list[RetrievalResult]
    ) -> FaithfulnessResult:
        if question == self._abstain:
            return FaithfulnessResult(
                statements=(), n_statements=0, n_supported=0, faithfulness=0.0
            )
        if question == self._half:
            verdicts = (
                StatementVerdict(statement="a", supported=True, reason="r"),
                StatementVerdict(statement="b", supported=False, reason="r"),
            )
            return FaithfulnessResult(
                statements=verdicts, n_statements=2, n_supported=1, faithfulness=0.5
            )
        return FaithfulnessResult(
            statements=(StatementVerdict(statement="a", supported=True, reason="r"),),
            n_statements=1,
            n_supported=1,
            faithfulness=1.0,
        )


class _StubAnswerRelevancyScorer:
    """Query-keyed relevancy stub: one noncommittal (0.0), the rest a fixed committal value."""

    model = "stub-rel"

    def __init__(self, *, noncommittal_query: str, value: float = 0.8) -> None:
        self._noncommittal = noncommittal_query
        self._value = value

    def score(
        self, question: str, answer: str, contexts: list[RetrievalResult]
    ) -> AnswerRelevancyResult:
        if question == self._noncommittal:
            return AnswerRelevancyResult(
                generated_questions=(), similarities=(), noncommittal=True, relevancy=0.0
            )
        return AnswerRelevancyResult(
            generated_questions=("q1",),
            similarities=(self._value,),
            noncommittal=False,
            relevancy=self._value,
        )


class _RecordingFaithfulnessScorer:
    """Records every ``(question, answer, context_ids)`` it is scored with (blindness proof)."""

    model = "recording-faith"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def score(
        self, question: str, answer: str, contexts: list[RetrievalResult]
    ) -> FaithfulnessResult:
        self.calls.append((question, answer, tuple(ctx.chunk_id for ctx in contexts)))
        return FaithfulnessResult(
            statements=(StatementVerdict(statement="s", supported=True, reason="r"),),
            n_statements=1,
            n_supported=1,
            faithfulness=1.0,
        )


class _RecordingAnswerRelevancyScorer:
    """Records every ``(question, answer, context_ids)`` it is scored with (blindness proof)."""

    model = "recording-rel"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def score(
        self, question: str, answer: str, contexts: list[RetrievalResult]
    ) -> AnswerRelevancyResult:
        self.calls.append((question, answer, tuple(ctx.chunk_id for ctx in contexts)))
        return AnswerRelevancyResult(
            generated_questions=("q",), similarities=(0.5,), noncommittal=False, relevancy=0.5
        )


def _settings(storage: Path, **overrides: object) -> Settings:
    """Eval settings pinned to the committed sample corpus + golden set (config that minted it)."""
    base: dict[str, object] = {
        "corpus_dir": PROJECT_ROOT / "data" / "corpus",  # unused by eval (it uses sample_dir)
        "sample_dir": SAMPLE_DIR,
        "golden_path": GOLDEN_PATH,
        "storage_dir": storage,
        "chunk_strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "qdrant_collection": "eval_ragas_test",
    }
    base.update(overrides)
    return Settings(**base)


def _run(
    storage: Path,
    *,
    llm: object | None = None,
    faithfulness: object | None = None,
    answer_relevancy: object | None = None,
) -> GenerationQualityReport:
    """Run the generation-quality eval fully offline (fresh in-memory store each call)."""
    return run_generation_quality_eval(
        _settings(storage),
        llm=llm if llm is not None else FakeLLMClient(),  # type: ignore[arg-type]
        embedder=HashingEmbedder(),
        store=QdrantVectorStore.in_memory("eval_ragas_test"),
        reranker=LexicalOverlapReranker(),
        faithfulness_scorer=faithfulness if faithfulness is not None else FakeFaithfulnessScorer(),  # type: ignore[arg-type]
        answer_relevancy_scorer=(
            answer_relevancy if answer_relevancy is not None else FakeAnswerRelevancyScorer()
        ),  # type: ignore[arg-type]
    )


# --- pure helpers (corpus-independent, hand-checked) -------------------------------------------


def test_faithfulness_value_conventions() -> None:
    assert faithfulness_value(0, 0) == 0.0  # documented 0-statement convention
    assert faithfulness_value(1, 2) == 0.5
    assert faithfulness_value(3, 3) == 1.0
    assert faithfulness_value(0, 4) == 0.0


def test_split_statements_and_token_overlap() -> None:
    assert split_statements("The sky is blue. The grass is green!") == [
        "The sky is blue",
        "The grass is green",
    ]
    assert split_statements("   ") == []
    ctx_tokens = set(["the", "sky", "is", "blue", "and", "the", "grass", "is", "green"])
    assert token_overlap_fraction("the sky is blue", ctx_tokens) == 1.0
    assert token_overlap_fraction("elephants pilot jets", ctx_tokens) == 0.0
    assert token_overlap_fraction("", ctx_tokens) == 0.0


def test_cosine_similarity_defensive_and_bounded() -> None:
    # A zero vector (the hashing embedder's token-less output) never raises -> 0.0.
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0  # signed hashing CAN go negative
    assert -1.0 <= cosine_similarity([3.0, 4.0], [4.0, 3.0]) <= 1.0


def test_is_noncommittal_lexical_heuristic() -> None:
    assert is_noncommittal("") is True
    assert is_noncommittal("   ") is True
    assert is_noncommittal("The provided context does not contain an answer.") is True
    assert is_noncommittal("Paris is the capital of France.") is False


# --- FakeFaithfulnessScorer: grounded 1.0, fabricated < 1.0, abstention --------------------------


def test_fake_faithfulness_grounded_answer_scores_1_0() -> None:
    contexts = [_ctx("d1", "the sky is blue and the grass is green in spring")]
    result = FakeFaithfulnessScorer().score("q", "The sky is blue. The grass is green.", contexts)
    assert result.n_statements == 2
    assert result.n_supported == 2
    assert result.faithfulness == 1.0


def test_fake_faithfulness_fabricated_statement_drops_below_1_0() -> None:
    """MANDATORY negative fixture: an unsupported statement MUST push faithfulness below 1.0.

    A scorer stuck at 1.0 regardless of input (an always-"supported" bug) would fail here — this
    is the guard that makes a saturated faithfulness auditable rather than authoritative-by-fiat.
    """
    contexts = [_ctx("d1", "the sky is blue and the grass is green")]
    result = FakeFaithfulnessScorer().score(
        "q", "The sky is blue. Elephants pilot fighter jets over the ocean.", contexts
    )
    assert result.n_statements == 2
    assert result.n_supported == 1  # only the grounded statement
    assert result.faithfulness < 1.0
    assert result.faithfulness == 0.5


def test_fake_faithfulness_zero_statement_answer_abstains() -> None:
    result = FakeFaithfulnessScorer().score("q", "", [_ctx("d1", "the sky is blue")])
    assert result.n_statements == 0
    assert result.n_supported == 0
    assert result.faithfulness == 0.0  # documented 0-statement convention


# --- FakeAnswerRelevancyScorer: noncommittal 0.0, committal in [-1, 1] --------------------------


def test_fake_answer_relevancy_noncommittal_forced_zero() -> None:
    result = FakeAnswerRelevancyScorer().score(
        "what is X?", "The provided context does not contain an answer to this question.", []
    )
    assert result.noncommittal is True
    assert result.relevancy == 0.0
    assert result.generated_questions == ()


def test_fake_answer_relevancy_committal_in_range() -> None:
    result = FakeAnswerRelevancyScorer().score(
        "what is the capital of france?", "Paris is the capital of France.", []
    )
    assert result.noncommittal is False
    assert -1.0 <= result.relevancy <= 1.0
    assert len(result.generated_questions) == 3  # RAGAS default N


def test_fake_answer_relevancy_no_content_tokens_scores_zero() -> None:
    # A non-refusal answer with no tokens (pure punctuation) cannot generate questions -> 0.0.
    result = FakeAnswerRelevancyScorer().score("q", "!!! ???", [])
    assert result.noncommittal is False
    assert result.generated_questions == ()
    assert result.relevancy == 0.0


# --- Protocol conformance + real scorers are lazy / model selection -----------------------------


def test_scorer_classes_satisfy_protocols_and_real_is_lazy() -> None:
    assert isinstance(FakeFaithfulnessScorer(), FaithfulnessScorer)
    assert isinstance(FakeAnswerRelevancyScorer(), AnswerRelevancyScorer)
    faith = AnthropicFaithfulnessScorer(Settings())
    relevancy = AnthropicAnswerRelevancyScorer(Settings(), HashingEmbedder())
    assert isinstance(faith, FaithfulnessScorer)
    assert isinstance(relevancy, AnswerRelevancyScorer)
    # SDK client not built at construction time (no key needed).
    assert faith._client is None
    assert relevancy._client is None


def test_scorer_default_model_differs_from_generator() -> None:
    settings = Settings(llm_model="claude-sonnet-4-6")
    expected = default_judge_model("claude-sonnet-4-6")
    assert expected != settings.llm_model  # anti self-preference default
    assert AnthropicFaithfulnessScorer(settings).model == expected
    assert AnthropicAnswerRelevancyScorer(settings, HashingEmbedder()).model == expected
    # Explicit override honored (e.g. cheaper bulk scoring).
    assert AnthropicFaithfulnessScorer(settings, model="claude-sonnet-4-6").model == (
        "claude-sonnet-4-6"
    )
    # N defaults from settings.
    assert AnthropicAnswerRelevancyScorer(settings, HashingEmbedder()).n_questions == 3


# --- real scorers fail CLOSED (unparseable ⇒ not supported / relevancy 0.0) via a stub client ---


def test_real_faithfulness_verify_unparseable_fails_closed() -> None:
    scorer = AnthropicFaithfulnessScorer(Settings())
    scorer._client = _StubClient(
        [
            _StubResponse(_DecomposedStatements(statements=["a", "b"])),
            _StubResponse(None),  # verification is unparseable
        ]
    )
    result = scorer.score("q", "some answer here.", [_ctx("d", "unrelated context text")])
    assert result.n_statements == 2
    assert result.n_supported == 0  # fail-closed: every statement counts NOT supported
    assert result.faithfulness == 0.0


def test_real_faithfulness_decompose_unparseable_falls_back_to_one_statement() -> None:
    scorer = AnthropicFaithfulnessScorer(Settings())
    scorer._client = _StubClient(
        [
            _StubResponse(None),  # decomposition is unparseable
            _StubResponse(
                _FaithfulnessVerdicts(
                    verdicts=[StatementVerdict(statement="x", supported=False, reason="r")]
                )
            ),
        ]
    )
    result = scorer.score("q", "a fabricated claim", [_ctx("d", "unrelated context text")])
    # Fell back to a single statement (the whole answer) so it can never silently abstain to 0/0.
    assert result.n_statements == 1
    assert result.n_supported == 0
    assert result.faithfulness == 0.0


def test_real_faithfulness_missing_verdict_fails_closed() -> None:
    # Decomposition yields 2 statements but verification returns only 1 verdict -> the missing one
    # is failed closed (NOT supported), never silently marked grounded.
    scorer = AnthropicFaithfulnessScorer(Settings())
    scorer._client = _StubClient(
        [
            _StubResponse(_DecomposedStatements(statements=["a", "b"])),
            _StubResponse(
                _FaithfulnessVerdicts(
                    verdicts=[StatementVerdict(statement="a", supported=True, reason="r")]
                )
            ),
        ]
    )
    result = scorer.score("q", "two claims.", [_ctx("d", "text")])
    assert result.n_statements == 2
    assert result.n_supported == 1  # the second (missing) verdict failed closed
    assert result.faithfulness == 0.5


def test_real_answer_relevancy_question_gen_unparseable_fails_closed() -> None:
    scorer = AnthropicAnswerRelevancyScorer(Settings(), HashingEmbedder())
    scorer._client = _StubClient([_StubResponse(None)])  # question-gen unparseable
    result = scorer.score("q", "a committal substantive answer", [])
    assert result.relevancy == 0.0  # never silently high
    assert result.noncommittal is False
    assert result.generated_questions == ()


def test_real_answer_relevancy_empty_answer_is_noncommittal_without_sdk() -> None:
    # An empty answer is trivially noncommittal and must NOT touch the SDK (no client set).
    scorer = AnthropicAnswerRelevancyScorer(Settings(), HashingEmbedder())
    result = scorer.score("q", "   ", [])
    assert result.noncommittal is True
    assert result.relevancy == 0.0
    assert scorer._client is None  # lazy: no SDK call for an empty answer


# --- full offline run: grounded 1.0 / fabricated < 1.0 -----------------------------------------


def test_grounded_fake_full_run_faithfulness_1_0(tmp_path: Path) -> None:
    report = _run(tmp_path / "storage", llm=FakeLLMClient())
    assert isinstance(report, GenerationQualityReport)
    assert report.config == CONFIG
    assert report.n_queries == N_GOLDEN
    assert len(report.per_query) == N_GOLDEN
    # The grounded fake quotes a REAL substring of the top context for every query -> every claim
    # is supported: micro == macro == 1.0, nothing abstains.
    assert report.micro_faithfulness == 1.0
    assert report.macro_faithfulness == 1.0
    assert report.macro_faithfulness_answered == 1.0
    assert report.n_faith_abstained == 0
    assert report.total_statements > 0
    assert report.total_supported == report.total_statements
    for record in report.per_query:
        assert record.n_statements >= 1
        assert record.n_supported == record.n_statements
        assert record.faithfulness == 1.0


def test_fabricated_fake_full_run_faithfulness_below_1_0(tmp_path: Path) -> None:
    """MANDATORY through-pipeline negative fixture: fabricated answers drive faithfulness < 1.0."""
    report = _run(tmp_path / "storage", llm=FabricatingFakeLLMClient())
    assert report.micro_faithfulness < 1.0
    assert report.macro_faithfulness < 1.0
    assert report.total_supported < report.total_statements


# --- publishability + provenance ---------------------------------------------------------------


def _named(name: str) -> object:
    """A dummy whose class ``__name__`` is ``name``, to unit-test the publishable gate per axis."""
    return type(name, (), {})()


def test_publishable_constants_bind_to_real_class_names() -> None:
    # The comment/ADR claim "a rename fails a test, not a silent flag drift" is only true if the
    # constants are bound to the REAL classes' __name__ — a string-literal-only test would let a
    # rename silently flip publishable False on real runs. Bind them here so a rename fails HERE.
    assert AnthropicLLMClient.__name__ == _PUBLISHABLE_LLM
    assert AnthropicFaithfulnessScorer.__name__ == _PUBLISHABLE_FAITHFULNESS
    assert AnthropicAnswerRelevancyScorer.__name__ == _PUBLISHABLE_ANSWER_RELEVANCY
    assert SentenceTransformerEmbedder.__name__ == _PUBLISHABLE_EMBEDDER
    assert CrossEncoderReranker.__name__ == _PUBLISHABLE_RERANKER


def test_publishable_true_only_when_every_backend_is_real() -> None:
    real = {
        "llm": _named("AnthropicLLMClient"),
        "faithfulness_scorer": _named("AnthropicFaithfulnessScorer"),
        "answer_relevancy_scorer": _named("AnthropicAnswerRelevancyScorer"),
        "embedder": _named("SentenceTransformerEmbedder"),
        "scorer_embedder": _named("SentenceTransformerEmbedder"),
        "reranker": _named("CrossEncoderReranker"),
    }
    assert _is_publishable(**real) is True  # type: ignore[arg-type]

    # Flipping ANY single axis to a fake class name flips publishable off — including EITHER scorer
    # AND the answer-relevancy scorer's OWN embedder (scorer_embedder), distinct from the
    # orchestrator/retrieval embedder.
    for axis, fake_name in [
        ("llm", "FakeLLMClient"),
        ("faithfulness_scorer", "FakeFaithfulnessScorer"),
        ("answer_relevancy_scorer", "FakeAnswerRelevancyScorer"),
        ("embedder", "HashingEmbedder"),
        ("scorer_embedder", "HashingEmbedder"),
        ("reranker", "LexicalOverlapReranker"),
    ]:
        broken = dict(real)
        broken[axis] = _named(fake_name)
        assert _is_publishable(**broken) is False, axis  # type: ignore[arg-type]


def test_real_scorer_with_fake_embedder_inside_flips_publishable_false() -> None:
    # A real answer-relevancy scorer CLASS that embeds the cosine with a FAKE embedder must NOT be
    # publishable, even when the orchestrator/retrieval embedder is real. publishable gates on the
    # embedder the scorer ACTUALLY used, which the scorer exposes on ``.embedder``.
    scorer = AnthropicAnswerRelevancyScorer(Settings(), HashingEmbedder())
    assert type(scorer.embedder).__name__ == "HashingEmbedder"  # fake embedder inside the scorer
    result = _is_publishable(
        llm=_named("AnthropicLLMClient"),  # type: ignore[arg-type]
        faithfulness_scorer=_named("AnthropicFaithfulnessScorer"),  # type: ignore[arg-type]
        answer_relevancy_scorer=_named("AnthropicAnswerRelevancyScorer"),  # type: ignore[arg-type]
        embedder=_named(
            "SentenceTransformerEmbedder"
        ),  # real orchestrator/retrieval embedder  # type: ignore[arg-type]
        scorer_embedder=scorer.embedder,  # ...but a FAKE embedder inside the scorer
        reranker=_named("CrossEncoderReranker"),  # type: ignore[arg-type]
    )
    assert result is False


def test_full_fake_run_is_not_publishable(tmp_path: Path) -> None:
    prov = _run(tmp_path / "storage").provenance
    assert prov.publishable is False
    assert prov.llm_class == "FakeLLMClient"
    assert prov.faithfulness_scorer_class == "FakeFaithfulnessScorer"
    assert prov.answer_relevancy_scorer_class == "FakeAnswerRelevancyScorer"
    assert prov.embedder_class == "HashingEmbedder"
    assert prov.reranker_class == "LexicalOverlapReranker"


def test_provenance_is_coherent_with_settings(tmp_path: Path) -> None:
    prov = _run(tmp_path / "storage").provenance
    assert prov.n_queries == N_GOLDEN
    assert prov.top_k_rerank == 5  # the real answering config, not K_RETRIEVE=10
    assert prov.n_answer_relevancy_questions == 3
    # embedder identity is read from the answer-relevancy scorer's ACTUAL embedder (not settings):
    # a fake-embedded relevancy records the fake, never the real bge-small model name.
    assert prov.embedder_class == "HashingEmbedder"
    assert prov.embedding_model == "fake-hashing"
    assert prov.embedding_model != Settings().embedding_model  # never the misleading real name
    assert prov.single_run is True
    # The fake scorer records its explicit fake sentinel, never the generator model.
    assert prov.scorer_model == "fake-token-overlap"
    assert prov.scorer_model != prov.llm_model
    assert prov.corpus_sha256  # a real corpus fingerprint was recorded


# --- aggregation math (micro vs macro vs abstention; relevancy macro incl. noncommittal) --------


def test_aggregation_micro_macro_abstention_noncommittal_hand_checked(tmp_path: Path) -> None:
    """THE honesty test: homogeneous fakes leave micro==macro and no abstention, so a bug that
    reported micro AS macro would pass everything else. Query-keyed stubs break the symmetry with
    hand-computable exact values.
    """
    queries = [item.query for item in GOLDEN]
    half_query = queries[1]
    abstain_query = queries[-1]
    # Guard the design: the special queries are distinct and each unique in the golden set.
    assert half_query != abstain_query
    assert queries.count(half_query) == 1
    assert queries.count(abstain_query) == 1

    report = _run(
        tmp_path / "storage",
        faithfulness=_StubFaithfulnessScorer(half_query=half_query, abstain_query=abstain_query),
        answer_relevancy=_StubAnswerRelevancyScorer(noncommittal_query=abstain_query, value=0.8),
    )

    n_default = N_GOLDEN - 2  # all-1/1 queries (excluding the half + the abstain)
    expected_total_statements = n_default * 1 + 2 + 0
    expected_total_supported = n_default * 1 + 1 + 0
    expected_micro = expected_total_supported / expected_total_statements
    expected_macro = (n_default * 1.0 + 0.5 + 0.0) / N_GOLDEN
    expected_macro_answered = (n_default * 1.0 + 0.5) / (N_GOLDEN - 1)

    assert report.total_statements == expected_total_statements
    assert report.total_supported == expected_total_supported
    assert report.n_faith_abstained == 1
    assert report.n_faith_answered == N_GOLDEN - 1
    assert report.micro_faithfulness == pytest.approx(expected_micro)
    assert report.macro_faithfulness == pytest.approx(expected_macro)
    assert report.macro_faithfulness_answered == pytest.approx(expected_macro_answered)

    # Load-bearing: micro, macro, and macro-over-answered are all DISTINCT here (a micro==macro bug
    # would collapse the first two). Micro (pooled) is higher: the 0.5 query carries 2 statements
    # in the pool but only 1/50 of the macro mean.
    assert report.micro_faithfulness != pytest.approx(report.macro_faithfulness)
    assert report.macro_faithfulness != pytest.approx(report.macro_faithfulness_answered)
    assert report.micro_faithfulness > report.macro_faithfulness

    # Answer-relevancy: macro over ALL includes the noncommittal as 0; committal-only excludes it.
    expected_rel_macro = (n_default + 1) * 0.8 / N_GOLDEN  # 49 committal * 0.8, 1 noncommittal * 0
    assert report.n_noncommittal == 1
    assert report.n_committal == N_GOLDEN - 1
    assert report.macro_answer_relevancy == pytest.approx(expected_rel_macro)
    assert report.committal_answer_relevancy == pytest.approx(0.8)
    assert report.macro_answer_relevancy != pytest.approx(report.committal_answer_relevancy)

    # Per-query records reflect the exact design, keyed by the ORIGINAL golden query_id.
    by_id = {record.query_id: record for record in report.per_query}
    half = by_id[GOLDEN[1].query_id]
    abstained = by_id[GOLDEN[-1].query_id]
    assert half.n_statements == 2
    assert half.n_supported == 1
    assert half.faithfulness == pytest.approx(0.5)
    assert half.faith_abstained is False
    assert abstained.n_statements == 0
    assert abstained.faith_abstained is True
    assert abstained.noncommittal is True
    assert abstained.relevancy == 0.0


def test_n_statements_recorded_per_query(tmp_path: Path) -> None:
    # n_statements is recorded per query so a suspicious n_statements==1 (one-giant-statement
    # decomposition) is visible in the artifact rather than hidden in the aggregate.
    report = _run(tmp_path / "storage")
    for record in report.per_query:
        assert record.n_statements >= 0
        assert record.n_supported <= record.n_statements
    # The grounded fake produces at least one statement per (non-abstaining) answer.
    assert all(record.n_statements >= 1 for record in report.per_query)


# --- blindness: scorers never receive reference_answer / relevant_chunk_ids --------------------


def test_scorers_are_blind_to_reference_and_relevant_ids(tmp_path: Path) -> None:
    faith_recorder = _RecordingFaithfulnessScorer()
    rel_recorder = _RecordingAnswerRelevancyScorer()
    _run(tmp_path / "storage", faithfulness=faith_recorder, answer_relevancy=rel_recorder)

    golden_queries = {item.query for item in GOLDEN}
    references = {(item.reference_answer or "").strip() for item in GOLDEN}

    for recorder in (faith_recorder.calls, rel_recorder.calls):
        assert len(recorder) == N_GOLDEN  # scored once per golden query
        for question, answer, _context_ids in recorder:
            # The scorer sees ONLY the golden question + the generated answer (out of harness
            # control) + retrieved context ids. NEVER a golden reference_answer as an argument.
            assert question in golden_queries
            assert question not in references
            assert answer.strip() not in references

    # The harness passes each golden query IN ORDER — the query, never the reference.
    for index, (question, _answer, _ids) in enumerate(faith_recorder.calls):
        assert question == GOLDEN[index].query
        assert question != (GOLDEN[index].reference_answer or "").strip()


# --- determinism / byte-stable artifact --------------------------------------------------------


def test_run_is_deterministic(tmp_path: Path) -> None:
    report_a = _run(tmp_path / "a")
    report_b = _run(tmp_path / "b")
    # Deterministic fakes end-to-end -> identical report objects.
    assert report_a.model_dump() == report_b.model_dump()


def test_writes_byte_stable_distinct_artifact(tmp_path: Path) -> None:
    storage_a = tmp_path / "a"
    storage_b = tmp_path / "b"
    _run(storage_a)
    _run(storage_b)

    assert GENERATION_QUALITY_RESULTS_FILENAME == "generation_quality_results.json"
    artifact_a = (storage_a / "eval" / GENERATION_QUALITY_RESULTS_FILENAME).read_bytes()
    artifact_b = (storage_b / "eval" / GENERATION_QUALITY_RESULTS_FILENAME).read_bytes()
    # Two deterministic runs serialize byte-identically (sorted keys, no timestamp).
    assert artifact_a == artifact_b
    # And the artifact round-trips back into a valid report.
    GenerationQualityReport.model_validate_json(artifact_a.decode("utf-8"))


# --- console report ----------------------------------------------------------------------------


def test_render_report_labels_ragas_style_and_flags_non_publishable(tmp_path: Path) -> None:
    text = render_generation_quality_report(_run(tmp_path / "storage"))
    assert "NOT PUBLISHABLE" in text
    # Never claim canonical RAGAS output.
    assert "RAGAS-STYLE" in text or "RAGAS-style" in text
    assert "NOT the canonical RAGAS library" in text
    assert "micro faithfulness" in text  # faithfulness headline is micro
    assert "macro" in text  # reported alongside, never alone
    assert "answer_relevancy" in text
    assert "noncommittal" in text.lower()
    assert "double-counted" in text  # the three-metric positioning caveat
    assert f"n={N_GOLDEN}" in text
    # With a distinct scorer model (the default), no self-preference note is printed.
    assert "Self-preference" not in text


def _minimal_report(*, scorer_model: str, llm_model: str) -> GenerationQualityReport:
    """A tiny valid report for exercising render branches without a full hermetic run."""
    provenance = GenerationQualityProvenance(
        llm_class="AnthropicLLMClient",
        llm_model=llm_model,
        faithfulness_scorer_class="AnthropicFaithfulnessScorer",
        answer_relevancy_scorer_class="AnthropicAnswerRelevancyScorer",
        scorer_model=scorer_model,
        embedder_class="SentenceTransformerEmbedder",
        embedding_model="BAAI/bge-small-en-v1.5",
        reranker_class="CrossEncoderReranker",
        top_k_rerank=5,
        n_answer_relevancy_questions=3,
        git_sha="deadbeef",
        corpus_sha256="abc123",
        corpus_dir="/tmp/sample",
        library_versions={},
        n_queries=1,
        single_run=True,
        publishable=True,
    )
    record = GenerationQualityQueryRecord(
        query_id="q1",
        n_statements=1,
        n_supported=1,
        faithfulness=1.0,
        faith_abstained=False,
        relevancy=0.9,
        noncommittal=False,
        n_generated_questions=3,
    )
    return GenerationQualityReport(
        provenance=provenance,
        config=CONFIG,
        n_queries=1,
        total_statements=1,
        total_supported=1,
        micro_faithfulness=1.0,
        macro_faithfulness=1.0,
        macro_faithfulness_answered=1.0,
        n_faith_answered=1,
        n_faith_abstained=0,
        macro_answer_relevancy=0.9,
        committal_answer_relevancy=0.9,
        n_noncommittal=0,
        n_committal=1,
        per_query=(record,),
    )


def test_render_surfaces_self_preference_note_only_when_scorer_equals_generator() -> None:
    same = render_generation_quality_report(
        _minimal_report(scorer_model="claude-opus-4-8", llm_model="claude-opus-4-8")
    )
    assert "Self-preference" in same  # scorer model == generator model -> the note surfaces
    distinct = render_generation_quality_report(
        _minimal_report(scorer_model="claude-opus-4-8", llm_model="claude-sonnet-4-6")
    )
    assert "Self-preference" not in distinct
