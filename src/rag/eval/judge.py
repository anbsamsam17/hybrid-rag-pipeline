"""Answer-correctness judge for the corrective-vs-baseline eval (ADR-0008), offline-fake-first.

The corrective-vs-baseline harness (``rag.eval.corrective``) needs to decide, per golden query,
whether a candidate answer is *correct* relative to the row's ``reference_answer``. That is the
one place a real lift *would* show — but at the ~0 activation this corpus produces it is
generator+judge noise (ADR-0008), so it is a SECONDARY, directional endpoint, never a
confirmatory win.

This module owns that judge behind one :class:`AnswerCorrectnessJudge` Protocol, mirroring the
:class:`~rag.agentic.corrective_rag.CorrectiveLLM` pattern exactly:

* :class:`AnthropicAnswerCorrectnessJudge` — the REAL judge. It lazy-imports the official
  ``anthropic`` SDK (constructing it never imports the SDK or needs an API key), uses
  ``client.messages.parse(output_format=CorrectnessVerdict)`` for a TYPED verdict (never a
  regex over a string), and obeys the binding SDK rules: ``settings.llm_model`` (correctness-
  sensitive -> opus/sonnet-4.x), **adaptive thinking** + **high effort**, and NEVER
  ``temperature`` / ``top_p`` / ``top_k`` / ``budget_tokens`` and NEVER assistant-prefill (all
  400 on the 4.x models this repo targets).
* :class:`FakeAnswerCorrectnessJudge` — a DETERMINISTIC, dependency-free, key-free fake that
  scores the token-F1 of the reference against the candidate (via the package
  :func:`~rag.indexing.sparse.tokenize`) and thresholds it, so the whole eval runs + is
  byte-stable offline.

**Anti-leakage / blindness (ADR-0008 decision 7):** the judge is shown ONLY
``(query, reference_answer, candidate_answer)`` — never the corpus, never ``relevant_chunk_ids``,
and **never which arm produced the answer**. The Protocol signature has no arm parameter, so
blindness holds by construction: a judge physically cannot be biased toward the corrective arm.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from rag.config import Settings
from rag.indexing.sparse import tokenize

logger = logging.getLogger(__name__)

# Small structured output (a bool + a score + a one-line reason); comfortably bounded.
_JUDGE_MAX_TOKENS = 1024
# Correctness-sensitive judging -> keep effort explicit and high (never silently pass-mark).
_EFFORT = "high"
# Token-F1 at/above which the deterministic FAKE judge calls a candidate "correct". 0.5 is the
# balanced point of the harmonic mean; the fake is a deterministic FLOOR for offline runs, not
# the publishable correctness signal (that is the real LLM judge).
_FAKE_CORRECT_THRESHOLD = 0.5
# The deterministic FAKE judge has no real model; recorded verbatim in provenance so a reader
# never mistakes an offline correctness number for a judged one.
_FAKE_JUDGE_MODEL = "fake-lexical-f1"


def default_judge_model(generator_model: str) -> str:
    """Pick a correctness-judge model that DIFFERS from the generator (anti self-preference).

    An LLM judge scoring answers from the same model it judges can exhibit self-preference bias,
    inflating the ABSOLUTE correctness rate. Judging on a different model hardens that absolute
    number (it does not change the corrective-vs-baseline DELTA, since both arms share the same
    generator and judge). Both are correctness-sensitive 4.x models per CLAUDE.md: if the generator
    is opus, judge on sonnet; otherwise judge on opus (the stronger default when the generator is
    the cheaper sonnet). Callers can still override the model explicitly.
    """
    return "claude-sonnet-4-6" if "opus" in generator_model.lower() else "claude-opus-4-8"


class CorrectnessVerdict(BaseModel):
    """A judge's typed verdict on one candidate answer (frozen; the ``messages.parse`` schema).

    ``correct`` is the headline boolean the correctness *rate* aggregates; ``score`` is a
    ``[0, 1]`` confidence the judge assigns; ``reason`` is a one-sentence justification kept for
    after-the-fact defense of a judged number. Never carries an arm label (blind judging).
    """

    model_config = ConfigDict(frozen=True)

    correct: bool = Field(
        description="True iff the candidate answer conveys the reference answer's key facts."
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in correctness, 0..1 (1 = certainly correct).",
    )
    reason: str = Field(description="One short sentence explaining the verdict.")


@runtime_checkable
class AnswerCorrectnessJudge(Protocol):
    """Judge whether ``candidate_answer`` correctly answers ``query`` per ``reference_answer``.

    Owned by ``eval/``; blind by construction — the signature exposes ONLY the three strings a
    fair, unbiased correctness judgment needs, and NEVER the arm label / corpus / relevant ids.
    """

    def judge(self, query: str, reference_answer: str, candidate_answer: str) -> CorrectnessVerdict:
        """Return a typed :class:`CorrectnessVerdict` for the candidate against the reference."""
        ...


def lexical_f1(reference: str, candidate: str) -> float:
    """Deterministic token-set F1 of ``reference`` against ``candidate`` (pure, dependency-free).

    Tokenizes both with the single package tokenizer (:func:`~rag.indexing.sparse.tokenize`),
    then returns the harmonic mean of precision (shared / candidate tokens) and recall (shared /
    reference tokens) over the token SETS. This is a deterministic lexical FLOOR reported next to
    the LLM-judge correctness rate (ADR-0008): short factual reference answers ("Dipped
    headlights.") make it informative, and being byte-stable it lets a fully-offline run exercise
    the correctness path without an LLM. It is NOT a semantic measure — a correct paraphrase that
    reuses no reference tokens scores 0.0, which is exactly why the LLM judge is the headline and
    this is only the floor.

    Returns:
        The token-F1 in ``[0.0, 1.0]``; ``0.0`` when either side has no tokens or they share none.
    """
    ref_tokens = set(tokenize(reference))
    cand_tokens = set(tokenize(candidate))
    if not ref_tokens or not cand_tokens:
        return 0.0
    shared = len(ref_tokens & cand_tokens)
    if shared == 0:
        return 0.0
    precision = shared / len(cand_tokens)
    recall = shared / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


class AnthropicAnswerCorrectnessJudge:
    """Real ``AnswerCorrectnessJudge``: lazy ``anthropic`` SDK + structured ``messages.parse``.

    Mirrors :class:`rag.agentic.corrective_rag.AnthropicCorrectiveLLM` /
    :class:`rag.generation.llm.AnthropicLLMClient` exactly: constructing this object never imports
    the SDK or touches the network (no API key required at build time); the import happens on the
    first :meth:`judge` call. The call uses adaptive thinking + high effort and NEVER passes
    ``temperature`` / ``top_p`` / ``top_k`` / ``budget_tokens`` and NEVER uses assistant-prefill.

    By default the judge runs on a model DIFFERENT from the generator (:func:`default_judge_model`)
    to blunt self-preference bias in the absolute correctness rate; ``model`` overrides it. The
    resolved model is exposed on ``.model`` so the run's provenance records exactly what judged.
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

    def judge(self, query: str, reference_answer: str, candidate_answer: str) -> CorrectnessVerdict:
        """Judge the candidate against the reference via one structured Messages call.

        The prompt is a pure function of ``(query, reference_answer, candidate_answer)`` ONLY —
        it never sees which arm produced the candidate, the corpus, or the relevant chunk ids, so
        the judge cannot be biased toward either arm (ADR-0008 blindness guard).
        """
        client = self._ensure_client()
        prompt = _build_judge_prompt(query, reference_answer, candidate_answer)
        logger.info("judging answer correctness: model=%s", self.model)
        response = client.messages.parse(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=_JUDGE_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": _EFFORT},
            output_format=CorrectnessVerdict,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = response.parsed_output
        if parsed is None:
            # Fail-CLOSED: an unparseable judge response must never silently pass-mark a
            # candidate as correct. Score it incorrect with an explicit reason.
            logger.warning("judge parsed_output is None; failing closed (incorrect)")
            return CorrectnessVerdict(
                correct=False, score=0.0, reason="judge returned no parseable verdict"
            )
        return parsed


class FakeAnswerCorrectnessJudge:
    """Deterministic, dependency-free, key-free ``AnswerCorrectnessJudge`` for tests/offline runs.

    Scores :func:`lexical_f1` of the reference against the candidate and thresholds it at
    :data:`_FAKE_CORRECT_THRESHOLD` -> ``correct``. It deliberately ignores ``query`` (correctness
    here is reference-vs-candidate), so it is blind to the arm by construction, and being a pure
    function of the two strings it is byte-stable across runs.
    """

    #: No real model backs the fake; exposed (like the real judge's ``.model``) for provenance.
    model = _FAKE_JUDGE_MODEL

    def __init__(self, *, threshold: float = _FAKE_CORRECT_THRESHOLD) -> None:
        """Store the correctness threshold (defaults to the balanced 0.5 harmonic-mean point)."""
        self._threshold = threshold

    def judge(self, query: str, reference_answer: str, candidate_answer: str) -> CorrectnessVerdict:
        """Return a deterministic verdict from the reference-vs-candidate token-F1."""
        score = lexical_f1(reference_answer, candidate_answer)
        return CorrectnessVerdict(
            correct=score >= self._threshold,
            score=score,
            reason=f"lexical token-F1={score:.3f} vs threshold {self._threshold:.2f}",
        )


def _build_judge_prompt(query: str, reference_answer: str, candidate_answer: str) -> str:
    """Pure prompt builder: a function of the three blind inputs only (no arm/corpus/ids)."""
    return "\n".join(
        [
            "You are a strict answer-correctness judge. Decide whether the CANDIDATE ANSWER "
            "correctly answers the QUESTION, using the REFERENCE ANSWER as the ground truth.",
            "A candidate is correct iff it conveys the same key facts as the reference "
            "(paraphrase or extra correct detail is fine); it is incorrect if it contradicts, "
            "omits, or is not supported by the reference.",
            "",
            f"QUESTION: {query}",
            f"REFERENCE ANSWER: {reference_answer}",
            f"CANDIDATE ANSWER: {candidate_answer}",
            "",
            "Return correct (bool), score (0..1 confidence in correctness), and a one-sentence "
            "reason. Judge ONLY factual correctness against the reference.",
        ]
    )
