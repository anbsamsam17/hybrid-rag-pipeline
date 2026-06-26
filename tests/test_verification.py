"""Tests for the deterministic lexical citation verifier.

Covers the full matrix: grounded substring, grounded by token-overlap, fabricated quote,
hallucinated chunk id, attribution_rate math, the explicit 0-citation rule, determinism,
unicode-safe / locale-independent matching, and the empty-quote edge case. No LLM, no
network — every assertion is a real computed number.
"""

from __future__ import annotations

from rag.generation.models import Answer, Citation
from rag.retrieval.models import RetrievalResult
from rag.verification.citations import OVERLAP_THRESHOLD, verify_answer


def _ctx(chunk_id: str, text: str, rel_path: str = "doc.md") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=1.0,
        rank=1,
        text=text,
        rel_path=rel_path,
        heading_path=[],
        metadata={},
        sources=["dense"],
    )


def _cite(chunk_id: str, quote: str, rel_path: str = "doc.md") -> Citation:
    return Citation(chunk_id=chunk_id, rel_path=rel_path, supporting_quote=quote)


# --- grounded paths ------------------------------------------------------------------------


def test_quote_in_cited_chunk_is_grounded() -> None:
    contexts = [_ctx("c1", "The mitochondria is the powerhouse of the cell.")]
    answer = Answer(text="x [1]", citations=[_cite("c1", "powerhouse of the cell")])
    report = verify_answer(answer, contexts)
    assert report.attribution_rate == 1.0
    assert report.checks[0].grounded is True
    assert report.checks[0].method == "normalized_substring"
    assert report.unsupported == []


def test_high_token_overlap_is_grounded_without_exact_substring() -> None:
    # Quote reorders words / adds a typo'd connector so it's NOT an exact substring, but
    # >= 0.6 of its distinct tokens are present in the chunk -> grounded by overlap.
    contexts = [_ctx("c1", "alpha beta gamma delta epsilon")]
    answer = Answer(text="x", citations=[_cite("c1", "gamma alpha beta delta")])
    report = verify_answer(answer, contexts)
    assert report.checks[0].grounded is True
    assert report.checks[0].method == "token_overlap"
    assert report.attribution_rate == 1.0


# --- ungrounded paths ----------------------------------------------------------------------


def test_fabricated_quote_is_not_grounded() -> None:
    contexts = [_ctx("c1", "The mitochondria is the powerhouse of the cell.")]
    answer = Answer(text="x", citations=[_cite("c1", "zzqxv totally invented unsupported wzzqxv")])
    report = verify_answer(answer, contexts)
    assert report.attribution_rate == 0.0
    assert report.checks[0].grounded is False
    assert report.checks[0].method == "no_overlap"
    assert report.unsupported == ["c1"]


def test_citation_to_chunk_not_in_context_is_not_grounded() -> None:
    # Bug #1: a hallucinated chunk id must be ungrounded regardless of the quote, even when
    # the quote text happens to appear in a DIFFERENT context.
    contexts = [_ctx("c1", "The mitochondria is the powerhouse of the cell.")]
    answer = Answer(text="x", citations=[_cite("c99", "powerhouse of the cell")])
    report = verify_answer(answer, contexts)
    assert report.attribution_rate == 0.0
    assert report.checks[0].grounded is False
    assert report.checks[0].method == "chunk_not_in_context"
    assert report.checks[0].reason == "cited chunk not in context"
    assert report.unsupported == ["c99"]


def test_quote_in_other_chunk_but_cited_id_absent_is_not_grounded() -> None:
    # Bug #1 variant: the supporting_quote IS a real substring of a DIFFERENT context chunk
    # (c2), but the citation names an ABSENT chunk id (c99). Grounding is keyed to the CITED
    # chunk's text only, never to "the quote appears somewhere in the corpus" — so the
    # chunk-id guard fires FIRST and the citation is ungrounded, with no quote check run.
    contexts = [
        _ctx("c1", "alpha beta gamma"),
        _ctx("c2", "the secret answer is 42"),
    ]
    answer = Answer(text="x", citations=[_cite("c99", "the secret answer is 42")])
    report = verify_answer(answer, contexts)
    assert report.attribution_rate == 0.0
    assert report.checks[0].grounded is False
    assert report.checks[0].method == "chunk_not_in_context"
    assert report.checks[0].reason == "cited chunk not in context"
    assert report.unsupported == ["c99"]


def test_quote_in_other_chunk_but_cites_wrong_present_id_is_not_grounded() -> None:
    # Same gaming vector with a PRESENT-but-wrong id: the quote is a real substring of c2 but
    # the citation names c1 (which is in context, so the id guard does NOT fire). The quote
    # must be checked against c1's text only and fail — proving grounding never leaks across
    # chunks even when the cited id resolves.
    contexts = [
        _ctx("c1", "alpha beta gamma"),
        _ctx("c2", "the secret answer is 42"),
    ]
    answer = Answer(text="x", citations=[_cite("c1", "the secret answer is 42")])
    report = verify_answer(answer, contexts)
    assert report.attribution_rate == 0.0
    assert report.checks[0].grounded is False
    assert report.checks[0].method == "no_overlap"
    assert report.unsupported == ["c1"]


def test_stopword_padded_fabrication_is_not_grounded() -> None:
    # Anti-gaming (tightened overlap fallback): a FABRICATED quote padded with the cited
    # chunk's common/stop-words drives the distinct-token overlap RATIO over the bar
    # (here 5/6 == 0.83 >= 0.6), so the OLD ratio-only check would have marked it grounded.
    # But the quote shares NO contiguous run of distinctive CONTENT tokens with the chunk:
    # chunk content bigrams are (mitochondria, powerhouse) and (powerhouse, cell); the
    # quote's content bigrams are (cell, powerhouse) and (powerhouse, antimatter) — no
    # overlap. The hardened check therefore (correctly) rejects it.
    #
    # Discriminating-ness confirmed: with the contiguous-run guard removed (i.e. the pre-fix
    # ratio-only logic), `_grounding_verdict` returns grounded=True for this exact pair
    # because the ratio is 0.83 >= OVERLAP_THRESHOLD; this test would FAIL pre-fix and PASS
    # post-fix. The reason string also reports the >= ratio so the failure is auditable.
    chunk = "The mitochondria is the powerhouse of the cell."
    fabricated = "the cell is the powerhouse of antimatter"
    contexts = [_ctx("c1", chunk)]
    answer = Answer(text="x", citations=[_cite("c1", fabricated)])
    report = verify_answer(answer, contexts)
    assert report.attribution_rate == 0.0
    assert report.checks[0].grounded is False
    assert report.checks[0].method == "no_overlap"
    assert ">= " in report.checks[0].reason  # ratio cleared the bar; the run guard rejected it
    assert report.unsupported == ["c1"]


# --- attribution_rate math -----------------------------------------------------------------


def test_attribution_rate_two_of_three() -> None:
    contexts = [
        _ctx("c1", "alpha beta gamma delta epsilon"),
        _ctx("c2", "the quick brown fox jumps over the lazy dog"),
        _ctx("c3", "lorem ipsum dolor sit amet consectetur"),
    ]
    answer = Answer(
        text="x",
        citations=[
            _cite("c1", "alpha beta gamma"),  # grounded (substring)
            _cite("c2", "quick brown fox"),  # grounded (substring)
            _cite("c3", "wholly unrelated zzqxv fabricated text"),  # NOT grounded
        ],
    )
    report = verify_answer(answer, contexts)
    assert report.n_citations == 3
    assert report.attribution_rate == 2 / 3  # ~0.667, a real computed number
    assert report.unsupported == ["c3"]
    assert [c.grounded for c in report.checks] == [True, True, False]


# --- 0-citation rule (bug #2: no divide-by-zero) -------------------------------------------


def test_zero_citation_answer_rate_is_zero_not_error() -> None:
    contexts = [_ctx("c1", "some text")]
    answer = Answer(text="The context does not answer this.", citations=[])
    report = verify_answer(answer, contexts)
    assert report.n_citations == 0
    assert report.attribution_rate == 0.0  # documented rule; never divides by zero
    assert report.checks == []
    assert report.unsupported == []


# --- determinism ---------------------------------------------------------------------------


def test_verification_is_deterministic() -> None:
    contexts = [_ctx("c1", "alpha beta gamma delta")]
    answer = Answer(text="x", citations=[_cite("c1", "beta gamma")])
    assert verify_answer(answer, contexts) == verify_answer(answer, contexts)


# --- unicode / locale safety (bug #3) ------------------------------------------------------


def test_unicode_nfc_vs_nfd_substring_matches() -> None:
    # Chunk stores a decomposed "é" (e + combining acute); quote uses composed "é".
    # NFC normalization on both sides must make these match.
    chunk_text = "Café culture in Montréal"  # NFD
    quote = "Café culture"  # NFC composed
    contexts = [_ctx("c1", chunk_text)]
    answer = Answer(text="x", citations=[_cite("c1", quote)])
    report = verify_answer(answer, contexts)
    assert report.checks[0].grounded is True
    assert report.attribution_rate == 1.0


def test_case_folding_is_locale_independent() -> None:
    # Casefold (not locale lower) must match regardless of casing; covers the Turkish-I
    # class of locale bugs deterministically.
    contexts = [_ctx("c1", "The TITLE Case Heading")]
    answer = Answer(text="x", citations=[_cite("c1", "title case heading")])
    report = verify_answer(answer, contexts)
    assert report.checks[0].grounded is True


# --- edge cases ----------------------------------------------------------------------------


def test_empty_quote_is_not_grounded() -> None:
    # An empty string is a trivial substring of everything; it must NOT count as grounded.
    contexts = [_ctx("c1", "some real text")]
    answer = Answer(text="x", citations=[_cite("c1", "")])
    report = verify_answer(answer, contexts)
    assert report.checks[0].grounded is False
    assert report.attribution_rate == 0.0


def test_threshold_boundary_is_inclusive() -> None:
    # Exactly OVERLAP_THRESHOLD of the quote's distinct tokens present -> grounded (>=).
    # 3 distinct quote tokens, want ratio == 0.6 -> not cleanly achievable; use 5 tokens,
    # 3 present = 0.6 exactly.
    assert OVERLAP_THRESHOLD == 0.6
    contexts = [_ctx("c1", "alpha beta gamma")]  # present: alpha, beta, gamma
    # quote distinct tokens: alpha beta gamma delta epsilon (5); 3 present -> 0.6
    answer = Answer(text="x", citations=[_cite("c1", "alpha beta gamma delta epsilon")])
    report = verify_answer(answer, contexts)
    assert report.checks[0].grounded is True
    assert report.checks[0].method == "token_overlap"
