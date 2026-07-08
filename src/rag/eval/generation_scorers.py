"""RAGAS-style generation-quality scorers (ADR-0009), offline-fake-first.

Two generation-quality signals — **faithfulness** (are ALL the answer's claims grounded in the
full retrieved context) and **answer_relevancy** (does the answer address the question) — are
reimplemented over the official ``anthropic`` SDK, faithful to the published RAGAS algorithms.
**RAGAS is credited as the SPEC; these are NOT the canonical RAGAS library's output** — we own the
decomposition / NLI / question-generation prompts, so the numbers are "RAGAS-style". This is the
deliberate ADR-0009 decision: the ``ragas`` library drives LLMs through a LangChain wrapper that
sets ``temperature`` and offers no adaptive-thinking / effort surface, a direct violation of this
repo's binding SDK rules (all banned params 400 on the 4.x models), and it is not
offline-deterministic. Reimplementing keeps every LLM call the ONE idiom used across the repo
(lazy ``anthropic``, ``messages.parse``, adaptive thinking, high effort, fail-closed) and lets the
whole eval run byte-stably in CI with no key.

Each metric sits behind one Protocol, mirroring :mod:`rag.eval.judge` exactly:

* :class:`FaithfulnessScorer` — ``AnthropicFaithfulnessScorer`` performs the RAGAS TWO-step:
  (i) decompose the answer into atomic statements, (ii) verify each is inferable from the FULL
  retrieved context, both via ``client.messages.parse`` with adaptive thinking + high effort,
  never the banned params, never prefill, fail-closed (unparseable verification ⇒ NOT supported).
  :class:`FakeFaithfulnessScorer` is deterministic: sentence-split the answer, ground each
  statement by a public token-overlap floor against the pooled context (reusing the ONE package
  tokenizer :func:`rag.indexing.sparse.tokenize`, never reaching into ``verification`` internals),
  so a fabricated claim scores unsupported.
* :class:`AnswerRelevancyScorer` — ``AnthropicAnswerRelevancyScorer`` generates ``N`` candidate
  questions (+ a ``noncommittal`` flag) from the answer via ONE ``messages.parse`` call, embeds the
  original + generated questions with the injected :class:`~rag.indexing.embeddings.Embedder` (the
  same real bge-small the index uses; $0 API cost), and returns ``mean_i cos(q, q_i)``, forced to
  exactly ``0.0`` for a noncommittal answer. :class:`FakeAnswerRelevancyScorer` templates
  deterministic questions and embeds with :class:`~rag.indexing.embeddings.HashingEmbedder`.

**Blind by construction (anti-leakage, ADR-0009 decision 3):** neither scorer signature exposes
``reference_answer`` or ``relevant_chunk_ids``. Faithfulness sees only ``(question, answer,
retrieved contexts)``; answer-relevancy structurally needs NO ground truth. A scorer physically
cannot leak or be gamed by the golden labels.

**Self-preference:** both scorers default to :func:`~rag.eval.judge.default_judge_model` (imported,
never re-implemented) so the scorer model DIFFERS from the generator by default, blunting
self-preference on the faithfulness verdicts; the resolved model is exposed on ``.model`` for
provenance and is constructor-overridable.
"""

from __future__ import annotations

import logging
import math
import re
from statistics import fmean
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from rag.config import Settings
from rag.eval.judge import default_judge_model
from rag.indexing.embeddings import Embedder, HashingEmbedder
from rag.indexing.sparse import tokenize
from rag.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

# --- Bounded structured-output sizes + effort (correctness-sensitive judging stays explicit) ----
_DECOMPOSE_MAX_TOKENS = 1024
_VERIFY_MAX_TOKENS = 2048
_QUESTION_GEN_MAX_TOKENS = 1024
_EFFORT = "high"

# Documented convention for a 0-statement answer's per-answer faithfulness. It matches the
# attribution macro convention (an abstention counts 0.0 in the macro mean); the MICRO headline is
# immune to it because a 0-statement answer adds nothing to either pool.
_ZERO_STATEMENT_FAITHFULNESS = 0.0

# The deterministic FAKE scorers have no real model; recorded verbatim in provenance so a reader
# never mistakes an offline number for a judged one (mirrors judge.py's fake-model sentinel).
_FAKE_FAITHFULNESS_MODEL = "fake-token-overlap"
_FAKE_ANSWER_RELEVANCY_MODEL = "fake-lexical-cosine"
# The FAKE answer-relevancy scorer embeds with HashingEmbedder; recorded verbatim so a fake-embedded
# relevancy can never be mistaken for a real bge-small one in provenance.
_FAKE_EMBEDDING_MODEL = "fake-hashing"

# Token-set overlap fraction at/above which the deterministic FAKE faithfulness scorer calls a
# statement "supported". 0.5 = "most of the statement's tokens appear in the context". It is a
# deterministic FLOOR for offline runs, NOT the publishable faithfulness signal (that is the real
# LLM NLI); the negative fixtures sit at the extremes (1.0 grounded, 0.0 fabricated) so the exact
# threshold never decides the load-bearing tests.
_FAKE_SUPPORT_THRESHOLD = 0.5

# Lexical refusal markers for the FAKE noncommittal heuristic (an empty/refusal answer genuinely
# fails to address the question -> relevancy 0.0). Matched case-insensitively as substrings.
_REFUSAL_MARKERS: tuple[str, ...] = (
    "does not contain",
    "does not answer",
    "not contain an answer",
    "cannot answer",
    "can't answer",
    "no answer",
    "i don't know",
    "i do not know",
    "unable to answer",
    "not provide",
    "no information",
)

# Deterministic question-template prefixes for the FAKE answer-relevancy scorer.
_QUESTION_PREFIXES: tuple[str, ...] = ("what", "how", "why", "when", "where", "which", "who")

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


# --- Public result models (mirroring judge.py: the scorer's typed return types live here) -------


class StatementVerdict(BaseModel):
    """One atomic statement's NLI verdict against the retrieved context (frozen).

    ``statement`` is the atomic claim decomposed from the answer; ``supported`` is ``True`` iff the
    context entails it (fail-closed: an unverifiable/unparseable verdict is ``False``); ``reason``
    is a one-sentence justification kept for after-the-fact defense of a judged number. Doubles as
    the ``messages.parse`` item schema for the verification step.
    """

    model_config = ConfigDict(frozen=True)

    statement: str = Field(description="The atomic factual statement extracted from the answer.")
    supported: bool = Field(
        description="True iff the statement can be directly inferred from the retrieved context."
    )
    reason: str = Field(description="One short sentence explaining the verdict.")


class FaithfulnessResult(BaseModel):
    """One answer's RAGAS-style faithfulness outcome (frozen; the scorer return type).

    ``faithfulness`` is ``n_supported / n_statements`` (all in ``[0, 1]``), or the documented
    0-statement convention (``0.0``) when the answer made no verifiable claim — in which case the
    aggregator treats it as an abstention (excluded from the MICRO pool). ``statements`` carries the
    per-statement verdicts so a saturated ``1.000`` can be audited claim-by-claim rather than
    trusted blindly.
    """

    model_config = ConfigDict(frozen=True)

    statements: tuple[StatementVerdict, ...]
    n_statements: int = Field(ge=0)
    n_supported: int = Field(ge=0)
    faithfulness: float = Field(ge=0.0, le=1.0)


# --- messages.parse output schemas (typed; never a regex over a string) ------------------------


class _DecomposedStatements(BaseModel):
    """``messages.parse`` schema for the faithfulness DECOMPOSITION step (atomic claims)."""

    model_config = ConfigDict(frozen=True)

    statements: list[str] = Field(
        description="The answer decomposed into atomic, self-contained factual statements, in "
        "the order they appear. Empty if the answer makes no verifiable factual claim."
    )


class _FaithfulnessVerdicts(BaseModel):
    """``messages.parse`` schema for the faithfulness VERIFICATION step (NLI, one per statement)."""

    model_config = ConfigDict(frozen=True)

    verdicts: list[StatementVerdict] = Field(
        description="One verdict per input statement, in the SAME order as the input statements."
    )


class _GeneratedQuestions(BaseModel):
    """``messages.parse`` schema for the answer-relevancy QUESTION-GENERATION step."""

    model_config = ConfigDict(frozen=True)

    questions: list[str] = Field(
        description="Candidate questions the given answer would be a direct answer to."
    )
    noncommittal: bool = Field(
        description="True iff the answer is evasive/refuses (e.g. 'I don't know', 'the context "
        "does not contain the answer') rather than committing to a substantive answer."
    )


# --- Public result models for answer-relevancy (faithfulness result lives in eval.models) -------


class AnswerRelevancyResult(BaseModel):
    """One answer's RAGAS-style answer-relevancy outcome (frozen; the scorer return type).

    ``relevancy`` is the mean cosine of the original question against each generated question, in a
    documented ``[-1, 1]`` range (unclamped except by float precision), FORCED to exactly ``0.0``
    when ``noncommittal`` (an evasive answer genuinely fails to address the question). ``0.0`` is
    also the fail-closed value when question-generation is unparseable or yields no question — a
    relevancy is never silently high.
    """

    model_config = ConfigDict(frozen=True)

    generated_questions: tuple[str, ...]
    similarities: tuple[float, ...]
    noncommittal: bool
    relevancy: float = Field(ge=-1.0, le=1.0)


# --- Protocols (blind by construction: no reference_answer / relevant_chunk_ids) ---------------


@runtime_checkable
class FaithfulnessScorer(Protocol):
    """Score whether ALL of ``answer``'s claims are grounded in the retrieved ``contexts``.

    Owned by ``eval/``; blind by construction — the signature exposes only the question, the
    answer text, and the retrieved contexts, NEVER the golden reference or relevant chunk ids.
    """

    def score(
        self, question: str, answer: str, contexts: list[RetrievalResult]
    ) -> FaithfulnessResult:
        """Return a typed :class:`FaithfulnessResult` for ``answer`` over ``contexts``."""
        ...


@runtime_checkable
class AnswerRelevancyScorer(Protocol):
    """Score whether ``answer`` addresses ``question`` (needs NO ground truth).

    ``contexts`` is accepted for signature uniformity with :class:`FaithfulnessScorer`; the RAGAS
    answer-relevancy algorithm does not use it. Blind by construction (no reference / relevant ids).
    """

    def score(
        self, question: str, answer: str, contexts: list[RetrievalResult]
    ) -> AnswerRelevancyResult:
        """Return a typed :class:`AnswerRelevancyResult` for ``answer`` against ``question``."""
        ...


# --- Pure, dependency-free helpers (shared by the fakes; auditable in-repo) ---------------------


def join_contexts(contexts: list[RetrievalResult]) -> str:
    """Pool the retrieved context texts into one string (the FULL context faithfulness checks)."""
    return "\n\n".join(ctx.text for ctx in contexts)


def split_statements(text: str) -> list[str]:
    """Split ``text`` into atomic statements by a simple, public sentence rule (deterministic).

    Splits on runs of ``.``/``!``/``?`` and drops empty fragments. This is the FAKE scorer's
    dependency-free stand-in for the LLM decomposition step; it is intentionally naive (it is a
    deterministic floor, not the publishable decomposition).
    """
    return [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(text) if segment.strip()]


def token_overlap_fraction(statement: str, context_tokens: set[str]) -> float:
    """Fraction of ``statement``'s distinct tokens that appear in ``context_tokens`` (``[0, 1]``).

    Uses the single package tokenizer (:func:`rag.indexing.sparse.tokenize`) so the fake dense and
    sparse paths agree on what a token is. Returns ``0.0`` for a token-less statement.
    """
    statement_tokens = set(tokenize(statement))
    if not statement_tokens:
        return 0.0
    return len(statement_tokens & context_tokens) / len(statement_tokens)


def is_noncommittal(answer: str) -> bool:
    """Deterministic lexical refusal heuristic: empty or a known refusal phrase ⇒ noncommittal."""
    text = answer.strip().lower()
    if not text:
        return True
    return any(marker in text for marker in _REFUSAL_MARKERS)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Defensive cosine of two equal-length vectors, clamped to ``[-1, 1]``.

    Both real (bge-small) and fake (:class:`HashingEmbedder`) embedders L2-normalize, so cosine
    equals the dot product — but we normalize here anyway (the hashing embedder returns an
    all-zeros vector for token-less text, which is NOT unit-norm), and guard divide-by-zero by
    returning ``0.0``. Clamped so float rounding can never produce a value outside ``[-1, 1]``.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def faithfulness_value(n_supported: int, n_statements: int) -> float:
    """Per-answer faithfulness: ``n_supported / n_statements``; the 0-statement convention else."""
    if n_statements == 0:
        return _ZERO_STATEMENT_FAITHFULNESS
    return n_supported / n_statements


def _templated_questions(answer: str, n_questions: int) -> list[str]:
    """Deterministically template ``n_questions`` distinct questions from the answer's tokens."""
    tokens = tokenize(answer)
    if not tokens:
        return []
    base = " ".join(tokens)
    return [f"{_QUESTION_PREFIXES[i % len(_QUESTION_PREFIXES)]} {base}" for i in range(n_questions)]


# --- Faithfulness scorers ----------------------------------------------------------------------


class AnthropicFaithfulnessScorer:
    """Real ``FaithfulnessScorer``: lazy ``anthropic`` SDK + RAGAS two-step over ``messages.parse``.

    Mirrors :class:`rag.eval.judge.AnthropicAnswerCorrectnessJudge`: constructing this object never
    imports the SDK or touches the network (no key needed at build time); the import happens on the
    first :meth:`score` call. Both LLM steps use adaptive thinking + high effort and NEVER pass
    ``temperature`` / ``top_p`` / ``top_k`` / ``budget_tokens`` and NEVER use assistant-prefill.

    By default the scorer runs on a model DIFFERENT from the generator (:func:`default_judge_model`)
    to blunt self-preference on the NLI verdicts; ``model`` overrides it. The resolved model is
    exposed on ``.model`` so the run's provenance records exactly what scored.
    """

    def __init__(self, settings: Settings, *, model: str | None = None) -> None:
        """Store config; defer SDK import and client construction until first use."""
        self._settings = settings
        self.model = model or default_judge_model(settings.llm_model)
        self._client: object | None = None

    def _ensure_client(self) -> object:
        """Lazily import ``anthropic`` and build the SDK client once."""
        if self._client is None:
            import anthropic  # lazy: package imports fine without the SDK installed

            self._client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        return self._client

    def score(
        self, question: str, answer: str, contexts: list[RetrievalResult]
    ) -> FaithfulnessResult:
        """Decompose ``answer`` into atomic statements, then NLI-verify each against the context."""
        statements = self._decompose(question, answer)
        if not statements:
            # Legitimately no factual claim (or an empty answer): a 0-statement abstention, NOT a
            # grounding failure. Excluded from the micro pool by the aggregator.
            return FaithfulnessResult(
                statements=(),
                n_statements=0,
                n_supported=0,
                faithfulness=faithfulness_value(0, 0),
            )
        verdicts = self._verify(statements, join_contexts(contexts))
        n_supported = sum(1 for verdict in verdicts if verdict.supported)
        return FaithfulnessResult(
            statements=tuple(verdicts),
            n_statements=len(verdicts),
            n_supported=n_supported,
            faithfulness=faithfulness_value(n_supported, len(verdicts)),
        )

    def _decompose(self, question: str, answer: str) -> list[str]:
        """Decompose the answer into atomic statements via one structured Messages call.

        An empty answer yields no statements (a genuine abstention). An UNPARSEABLE decomposition
        fails closed to a SINGLE statement (the whole answer) so a fabricated answer that fails to
        decompose still gets verified — it can never silently abstain to a 0/0 that the micro pool
        ignores.
        """
        if not answer.strip():
            return []
        client = self._ensure_client()
        prompt = _build_decompose_prompt(question, answer)
        logger.info("faithfulness decompose: model=%s", self.model)
        response = client.messages.parse(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=_DECOMPOSE_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": _EFFORT},
            output_format=_DecomposedStatements,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = response.parsed_output
        if parsed is None:
            logger.warning("decompose parsed_output is None; failing closed to one statement")
            return [answer.strip()]
        return [statement.strip() for statement in parsed.statements if statement.strip()]

    def _verify(self, statements: list[str], context_text: str) -> list[StatementVerdict]:
        """NLI-verify each statement against the full context; fail closed on any missing verdict.

        The verdicts are matched to the input statements BY POSITION (the prompt pins the order and
        one-verdict-per-statement). A count mismatch or an unparseable response counts the affected
        statements as NOT supported — an unverifiable claim is never silently marked grounded.
        """
        client = self._ensure_client()
        prompt = _build_verify_prompt(statements, context_text)
        logger.info("faithfulness verify: model=%s statements=%d", self.model, len(statements))
        response = client.messages.parse(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=_VERIFY_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": _EFFORT},
            output_format=_FaithfulnessVerdicts,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = response.parsed_output
        if parsed is None:
            logger.warning("verify parsed_output is None; failing closed (all unsupported)")
            return [
                StatementVerdict(
                    statement=statement,
                    supported=False,
                    reason="unparseable verification; failing closed",
                )
                for statement in statements
            ]
        returned = list(parsed.verdicts)
        out: list[StatementVerdict] = []
        for index, statement in enumerate(statements):
            if index < len(returned):
                verdict = returned[index]
                # Trust OUR statement text (the decomposition), the model's supported/reason.
                out.append(
                    StatementVerdict(
                        statement=statement,
                        supported=verdict.supported,
                        reason=verdict.reason,
                    )
                )
            else:
                out.append(
                    StatementVerdict(
                        statement=statement,
                        supported=False,
                        reason="no verdict returned for this statement; failing closed",
                    )
                )
        return out


class FakeFaithfulnessScorer:
    """Deterministic, dependency-free, key-free ``FaithfulnessScorer`` for tests/offline runs.

    Sentence-splits the answer (:func:`split_statements`) and marks each statement supported iff a
    public token-overlap floor (:func:`token_overlap_fraction` ``>= threshold``) against the pooled
    context holds. It is a pure function of ``(answer, contexts)`` so it is byte-stable across runs.
    A grounded answer (a real substring of a context) overlaps fully ⇒ faithfulness ``1.0``; a
    fabricated statement shares no context tokens ⇒ unsupported ⇒ faithfulness ``< 1.0`` (the
    load-bearing negative-fixture guard against an always-"supported" scorer).
    """

    #: No real model backs the fake; exposed (like the real scorer's ``.model``) for provenance.
    model = _FAKE_FAITHFULNESS_MODEL

    def __init__(self, *, threshold: float = _FAKE_SUPPORT_THRESHOLD) -> None:
        """Store the support threshold (defaults to the 0.5 'most tokens overlap' floor)."""
        self._threshold = threshold

    def score(
        self, question: str, answer: str, contexts: list[RetrievalResult]
    ) -> FaithfulnessResult:
        """Return a deterministic faithfulness result from statement/context token overlap."""
        context_tokens = set(tokenize(join_contexts(contexts)))
        statements = split_statements(answer)
        if not statements:
            return FaithfulnessResult(
                statements=(),
                n_statements=0,
                n_supported=0,
                faithfulness=faithfulness_value(0, 0),
            )
        verdicts: list[StatementVerdict] = []
        for statement in statements:
            overlap = token_overlap_fraction(statement, context_tokens)
            supported = overlap >= self._threshold
            comparator = ">=" if supported else "<"
            verdicts.append(
                StatementVerdict(
                    statement=statement,
                    supported=supported,
                    reason=f"token-overlap {overlap:.3f} {comparator} {self._threshold:.2f}",
                )
            )
        n_supported = sum(1 for verdict in verdicts if verdict.supported)
        return FaithfulnessResult(
            statements=tuple(verdicts),
            n_statements=len(verdicts),
            n_supported=n_supported,
            faithfulness=faithfulness_value(n_supported, len(verdicts)),
        )


# --- Answer-relevancy scorers ------------------------------------------------------------------


class AnthropicAnswerRelevancyScorer:
    """Real ``AnswerRelevancyScorer``: lazy SDK question-generation + injected-embedder cosine.

    Generates ``N`` candidate questions (+ a ``noncommittal`` flag) from the answer via ONE
    ``messages.parse`` call (adaptive thinking + high effort, never the banned params, never
    prefill), embeds the ORIGINAL question and the ``N`` generated questions with ONE injected
    :class:`~rag.indexing.embeddings.Embedder` instance, and returns the mean cosine. A noncommittal
    answer is forced to exactly ``0.0``; an unparseable question-gen response fails closed to
    ``0.0``. ``N`` defaults to ``settings.ragas_answer_relevancy_n_questions``.
    """

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        *,
        model: str | None = None,
        n_questions: int | None = None,
    ) -> None:
        """Store config + the injected embedder; defer SDK import until first use.

        ``embedder`` is exposed publicly (not ``_embedder``) so the run's provenance /
        publishability gate on the embedder that ACTUALLY produced the relevancy cosine — not an
        orchestrator-level one that may differ from it.
        """
        self._settings = settings
        self.embedder = embedder
        self.model = model or default_judge_model(settings.llm_model)
        self.n_questions = (
            n_questions if n_questions is not None else settings.ragas_answer_relevancy_n_questions
        )
        self._client: object | None = None

    @property
    def embedding_model(self) -> str:
        """Honest model identity of the embedder that produced the relevancy cosine.

        Reads the injected embedder's own model name defensively; a non-real embedder (which has no
        ``_model_name``) degrades to its class name, so a ``HashingEmbedder`` slipped into this
        scorer can never be recorded in provenance as the real bge-small model.
        """
        return getattr(self.embedder, "_model_name", type(self.embedder).__name__)

    def _ensure_client(self) -> object:
        """Lazily import ``anthropic`` and build the SDK client once."""
        if self._client is None:
            import anthropic  # lazy: package imports fine without the SDK installed

            self._client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        return self._client

    def score(
        self, question: str, answer: str, contexts: list[RetrievalResult]
    ) -> AnswerRelevancyResult:
        """Generate questions from ``answer`` and return their mean cosine to ``question``."""
        generated = self._generate_questions(answer)
        if generated is None:
            # Fail-closed: an unparseable question-gen response scores 0.0, never silently high.
            logger.warning("question-gen parsed_output is None; failing closed (relevancy 0.0)")
            return AnswerRelevancyResult(
                generated_questions=(), similarities=(), noncommittal=False, relevancy=0.0
            )
        questions, noncommittal = generated
        if noncommittal:
            # Evasive/refusal answer genuinely fails to address the question -> forced 0.0.
            return AnswerRelevancyResult(
                generated_questions=tuple(questions),
                similarities=(),
                noncommittal=True,
                relevancy=0.0,
            )
        if not questions:
            return AnswerRelevancyResult(
                generated_questions=(), similarities=(), noncommittal=False, relevancy=0.0
            )
        vectors = self.embedder.embed_texts([question, *questions])
        question_vector = vectors[0]
        similarities = tuple(cosine_similarity(question_vector, vector) for vector in vectors[1:])
        relevancy = fmean(similarities) if similarities else 0.0
        return AnswerRelevancyResult(
            generated_questions=tuple(questions),
            similarities=similarities,
            noncommittal=False,
            relevancy=relevancy,
        )

    def _generate_questions(self, answer: str) -> tuple[list[str], bool] | None:
        """Generate questions + a noncommittal flag; ``None`` on an unparseable response."""
        if not answer.strip():
            # An empty answer is trivially noncommittal (no LLM call needed).
            return ([], True)
        client = self._ensure_client()
        prompt = _build_question_gen_prompt(answer, self.n_questions)
        logger.info("answer-relevancy question-gen: model=%s n=%d", self.model, self.n_questions)
        response = client.messages.parse(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=_QUESTION_GEN_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": _EFFORT},
            output_format=_GeneratedQuestions,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = response.parsed_output
        if parsed is None:
            return None
        questions = [question.strip() for question in parsed.questions if question.strip()]
        return (questions, parsed.noncommittal)


class FakeAnswerRelevancyScorer:
    """Deterministic, dependency-free, key-free ``AnswerRelevancyScorer`` for tests/offline runs.

    Templates deterministic questions from the answer's tokens (:func:`_templated_questions`),
    detects noncommittal answers by a public lexical refusal heuristic (:func:`is_noncommittal`),
    and embeds the original + generated questions with an injected embedder (defaulting to the
    dependency-free :class:`~rag.indexing.embeddings.HashingEmbedder`) to a mean cosine. A
    noncommittal answer is forced to exactly ``0.0``. Being a pure function of its inputs it is
    byte-stable across runs.
    """

    #: No real model backs the fake; exposed (like the real scorer's ``.model``) for provenance.
    model = _FAKE_ANSWER_RELEVANCY_MODEL
    #: Non-real embedding-model sentinel so a fake relevancy is never recorded as real bge-small.
    embedding_model = _FAKE_EMBEDDING_MODEL

    def __init__(self, embedder: Embedder | None = None, *, n_questions: int = 3) -> None:
        """Store the embedder (default :class:`HashingEmbedder`) and the question count.

        ``embedder`` is exposed publicly so the provenance / publishability gate on the fake's
        actual (non-real) embedder, exactly like the real scorer.
        """
        self.embedder = embedder or HashingEmbedder()
        self.n_questions = n_questions

    def score(
        self, question: str, answer: str, contexts: list[RetrievalResult]
    ) -> AnswerRelevancyResult:
        """Return a deterministic answer-relevancy result from templated-question cosine."""
        if is_noncommittal(answer):
            return AnswerRelevancyResult(
                generated_questions=(), similarities=(), noncommittal=True, relevancy=0.0
            )
        questions = _templated_questions(answer, self.n_questions)
        if not questions:
            return AnswerRelevancyResult(
                generated_questions=(), similarities=(), noncommittal=False, relevancy=0.0
            )
        vectors = self.embedder.embed_texts([question, *questions])
        question_vector = vectors[0]
        similarities = tuple(cosine_similarity(question_vector, vector) for vector in vectors[1:])
        relevancy = fmean(similarities) if similarities else 0.0
        return AnswerRelevancyResult(
            generated_questions=tuple(questions),
            similarities=similarities,
            noncommittal=False,
            relevancy=relevancy,
        )


# --- Prompt builders (pure functions of blind inputs only; no reference / relevant ids) --------


def _build_decompose_prompt(question: str, answer: str) -> str:
    """Pure prompt builder for the RAGAS-style statement decomposition (question + answer only)."""
    return "\n".join(
        [
            "You decompose an answer into atomic, self-contained factual statements for a "
            "faithfulness check. Break the ANSWER into the smallest standalone factual claims it "
            "makes; resolve pronouns using the QUESTION for context. Do NOT add facts not in the "
            "answer. If the answer makes no verifiable factual claim (e.g. it refuses or says the "
            "context lacks the answer), return an empty list.",
            "",
            f"QUESTION: {question}",
            f"ANSWER: {answer}",
            "",
            "Return the list of atomic statements, in the order they appear in the answer.",
        ]
    )


def _build_verify_prompt(statements: list[str], context_text: str) -> str:
    """Pure prompt builder for the RAGAS-style NLI verification (statements + full context only)."""
    numbered = "\n".join(f"{index + 1}. {statement}" for index, statement in enumerate(statements))
    return "\n".join(
        [
            "You are a strict natural-language-inference verifier for a faithfulness metric. For "
            "each STATEMENT, decide whether it can be directly inferred from the CONTEXT. A "
            "statement is supported ONLY if the context entails it; if the context is silent, "
            "ambiguous, or contradicts it, it is NOT supported. Judge each statement "
            "independently.",
            "",
            "CONTEXT:",
            context_text,
            "",
            "STATEMENTS:",
            numbered,
            "",
            "Return one verdict per statement, in the SAME order as the statements above, each "
            "with supported (bool) and a one-sentence reason. Return exactly as many verdicts as "
            "there are statements.",
        ]
    )


def _build_question_gen_prompt(answer: str, n_questions: int) -> str:
    """Pure prompt builder for the RAGAS-style answer-relevancy question generation (answer)."""
    return "\n".join(
        [
            f"Given an ANSWER, generate {n_questions} distinct questions that the answer would be "
            "a direct and complete answer to. The questions must be answerable purely from the "
            "answer text. Also decide whether the answer is noncommittal — evasive or a refusal "
            "(e.g. 'I don't know', 'the context does not contain the answer') rather than a "
            "substantive answer.",
            "",
            f"ANSWER: {answer}",
            "",
            f"Return exactly {n_questions} questions and the noncommittal flag.",
        ]
    )
