"""Deterministic, lexical citation verification — the anti-hallucination attribution check.

:func:`verify_answer` is the measured core of the pipeline's headline claim. For each
citation in an :class:`~rag.generation.models.Answer` it answers two questions, in order:

1. **Is the cited chunk real?** The cited ``chunk_id`` MUST appear among the provided
   contexts. If not, the citation is ungrounded with reason "cited chunk not in context".
   This is the guard against a model *fabricating a chunk id* to look attributed — counting
   a hallucinated chunk-id citation as grounded is bug #1 this module exists to prevent.

2. **Is the quote actually in that chunk?** The ``supporting_quote`` must be GROUNDED in the
   cited chunk's text by a deterministic LEXICAL check:
   * exact **normalized substring** match (case-folded, whitespace-collapsed, Unicode-NFC), OR
   * **token-overlap fallback** — BOTH a distinct-token overlap ratio ≥
     :data:`OVERLAP_THRESHOLD` AND a **contiguous content-token run** (a shared n-gram of at
     least :data:`MIN_CONTIGUOUS_CONTENT_TOKENS` *content* tokens, stop-words dropped) present
     in the chunk. The contiguous-run requirement is the anti-gaming guard: a fabricated quote
     padded with the chunk's common/stop-words can drive the bare ratio over the bar, but it
     cannot manufacture a contiguous run of distinctive content tokens it never copied — so a
     quote whose distinctive content is NOT in the chunk fails even at high stop-word overlap.

Everything is pure, deterministic, and dependency-free: no LLM, no network, no locale.
Normalization is explicitly Unicode-NFC + ASCII ``casefold`` over a fixed regex, NOT
``str.lower()`` / ``locale``-sensitive comparison — locale- or platform-dependent matching
is bug #3 this module exists to prevent (e.g. Turkish-I, NFC vs NFD "é").

The aggregate ``attribution_rate`` is ``grounded / total``. The **0-citation rule** is
defined explicitly: an answer with no citations has ``attribution_rate = 0.0`` (nothing was
attributed, so no attribution could be verified — we never divide by zero, which is bug #2).
This is a deliberate, documented choice: a "the context doesn't answer this" response is
*correctly* unattributed and scores 0.0; callers that want to treat a no-claim answer as
vacuously fine should branch on ``n_citations == 0`` rather than reading the rate.

Extension point: :func:`verify_answer` takes the lexical check as the only method today. A
future optional NLI / LLM-judge method (run on ``claude-opus-4-8`` / ``claude-sonnet-4-6``
per ``CLAUDE.md``) would slot in as an alternative ``method`` per citation WITHOUT changing
this module's deterministic default — see the ``method`` field on
:class:`~rag.verification.models.CitationCheck`. It is intentionally NOT implemented here.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from rag.generation.models import Answer
from rag.retrieval.models import RetrievalResult
from rag.verification.models import CitationCheck, VerificationReport

logger = logging.getLogger(__name__)

# Fraction of a quote's DISTINCT tokens that must be present in the chunk for the quote to
# count as lexically grounded by overlap (when it is not an exact normalized substring).
# 0.6 tolerates minor model paraphrase/typos while rejecting a fabricated quote that shares
# only stop-words with the source. Tuned conservatively: this is an anti-gaming check, so it
# errs toward NOT crediting a weak match. NOTE: this ratio alone is gameable (a fake quote
# padded with the chunk's frequent words can clear it), so the overlap fallback ALSO requires
# a contiguous content-token run (see MIN_CONTIGUOUS_CONTENT_TOKENS) — both must hold.
OVERLAP_THRESHOLD = 0.6

# Minimum length, in CONTENT tokens (stop-words dropped), of a contiguous run that the quote
# and the chunk must share for the overlap fallback to ground the quote. A real (even
# paraphrased/reordered) quote reuses at least one short contiguous span of distinctive words
# from its source; a fabricated quote padded with common words does not. 2 is the smallest
# value that still rejects single-content-word coincidences while accepting genuine short
# spans. This is the load-bearing anti-gaming guard on top of OVERLAP_THRESHOLD.
MIN_CONTIGUOUS_CONTENT_TOKENS = 2

# Deterministic, locale-independent English stop-word set, applied (after NFC + casefold) only
# to the CONTIGUOUS-RUN guard so common function words cannot, on their own, manufacture a
# "distinctive" shared span. Kept small and explicit (no nltk/spacy dep): these are the high
# frequency tokens an attacker would copy to pad a fake quote. The bare overlap RATIO still
# counts every token — stop-words are removed only from the contiguous-span check.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)

# Token splitter: runs of non-alphanumeric (Unicode-aware via \W is locale-dependent, so we
# split on an explicit ASCII-alnum complement after casefolding + NFC, keeping it stable
# across platforms/locales). Mirrors the package tokenizer's intent without importing it,
# because attribution must remain a pure function of the strings, not of BM25 config.
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Deterministic, locale-independent normalization for substring comparison.

    Unicode-NFC (so "é" composed == "é" decomposed), ``casefold`` (locale-independent,
    unlike ``str.lower``), and whitespace collapsed to single spaces. NEVER uses
    ``locale``-sensitive comparison — that is the locale-dependent-matching bug this guards.
    """
    nfc = unicodedata.normalize("NFC", text)
    folded = nfc.casefold()
    return _WHITESPACE_RE.sub(" ", folded).strip()


def _tokens(text: str) -> list[str]:
    """Tokenize for overlap: NFC + casefold, then split on non-ASCII-alnum runs.

    Deterministic and locale-independent (casefold, not lower; fixed regex, not ``\\w``).
    """
    nfc = unicodedata.normalize("NFC", text)
    folded = nfc.casefold()
    return [tok for tok in _TOKEN_SPLIT_RE.split(folded) if tok]


def _content_shingles(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """Return the set of contiguous ``n``-token shingles over CONTENT tokens only.

    Stop-words are dropped *before* shingling so that a shared span must consist of
    distinctive words to count — adjacency through common function words alone cannot
    manufacture a shingle. Order is preserved (shingles are contiguous over the filtered
    sequence). Deterministic and pure: a function of the token list only.
    """
    content = [tok for tok in tokens if tok not in _STOPWORDS]
    if n <= 0 or len(content) < n:
        return set()
    return {tuple(content[i : i + n]) for i in range(len(content) - n + 1)}


def _has_contiguous_content_run(quote_tokens: list[str], chunk_tokens: list[str]) -> bool:
    """True iff quote and chunk share a contiguous run of >= the min content-token length.

    This is the anti-gaming guard on the overlap fallback: a fabricated quote padded with the
    chunk's common words can clear the distinct-token ratio, but it cannot share a contiguous
    run of distinctive content tokens it never copied from the source.
    """
    quote_shingles = _content_shingles(quote_tokens, MIN_CONTIGUOUS_CONTENT_TOKENS)
    if not quote_shingles:
        return False
    chunk_shingles = _content_shingles(chunk_tokens, MIN_CONTIGUOUS_CONTENT_TOKENS)
    return bool(quote_shingles & chunk_shingles)


def _grounding_verdict(quote: str, chunk_text: str) -> tuple[bool, str, str]:
    """Decide whether ``quote`` is lexically grounded in ``chunk_text``.

    Returns ``(grounded, method, reason)``. An empty quote can never be grounded — an empty
    string is a trivial substring of everything, so we reject it explicitly rather than let
    it pass as a vacuous match.

    The grounding has two tiers:

    1. **Exact normalized substring** — the strong signal; grounds immediately.
    2. **Overlap fallback** — tolerates reordering/paraphrase/typos but is hardened against
       attribution gaming: it requires BOTH a distinct-token overlap ratio
       >= :data:`OVERLAP_THRESHOLD` AND a shared contiguous run of
       >= :data:`MIN_CONTIGUOUS_CONTENT_TOKENS` content tokens. The ratio alone is gameable
       (pad a fake quote with the chunk's frequent/stop-words); requiring a contiguous run of
       distinctive content tokens closes that hole, so a quote whose distinctive content is
       NOT in the chunk fails even at high stop-word overlap.
    """
    norm_quote = _normalize(quote)
    if not norm_quote:
        return False, "no_overlap", "empty supporting quote"

    norm_chunk = _normalize(chunk_text)
    if norm_quote in norm_chunk:
        return True, "normalized_substring", "quote is a normalized substring of the chunk"

    quote_token_list = _tokens(quote)
    quote_tokens = set(quote_token_list)
    if not quote_tokens:
        return False, "no_overlap", "quote has no comparable tokens"

    chunk_token_list = _tokens(chunk_text)
    chunk_tokens = set(chunk_token_list)
    present = len(quote_tokens & chunk_tokens)
    ratio = present / len(quote_tokens)
    overlap = f"{ratio:.2f} ({present}/{len(quote_tokens)})"

    if ratio < OVERLAP_THRESHOLD:
        return False, "no_overlap", f"token overlap {ratio:.2f} < {OVERLAP_THRESHOLD} ({overlap})"

    # Ratio cleared — but require a contiguous content-token run so stop-word padding alone
    # cannot ground a fabricated quote. This is the anti-gaming guard.
    if not _has_contiguous_content_run(quote_token_list, chunk_token_list):
        return (
            False,
            "no_overlap",
            f"token overlap {overlap} >= {OVERLAP_THRESHOLD} but no shared contiguous run of "
            f"{MIN_CONTIGUOUS_CONTENT_TOKENS}+ content tokens (likely stop-word padding)",
        )
    return (
        True,
        "token_overlap",
        f"token overlap {overlap} >= {OVERLAP_THRESHOLD} with a shared contiguous content run",
    )


def verify_answer(answer: Answer, contexts: list[RetrievalResult]) -> VerificationReport:
    """Measure how well ``answer``'s citations are grounded in ``contexts``.

    For each citation:
      (a) the cited ``chunk_id`` must be among ``contexts`` (else ungrounded, reason
          "cited chunk not in context" — catches a fabricated/hallucinated chunk id), then
      (b) the ``supporting_quote`` must be lexically grounded in that chunk's text via a
          normalized substring OR the hardened overlap fallback (ratio >=
          :data:`OVERLAP_THRESHOLD` AND a shared contiguous content-token run).

    ``attribution_rate`` is ``grounded / total``. The 0-citation rule: an answer with no
    citations returns ``attribution_rate = 0.0`` (documented choice; never divides by zero).

    Pure and deterministic: no LLM, no network, no locale-sensitive comparison. The result
    order follows the answer's citation order.
    """
    # Index chunk_id -> text once. dict preserves insertion (deterministic) and gives O(1)
    # membership; we only need the text for grounding.
    text_by_id = {ctx.chunk_id: ctx.text for ctx in contexts}

    checks: list[CitationCheck] = []
    unsupported: list[str] = []

    for citation in answer.citations:
        chunk_text = text_by_id.get(citation.chunk_id)
        if chunk_text is None:
            # Bug #1 guard: a cited id not in context is NOT grounded, no matter the quote.
            grounded, method, reason = False, "chunk_not_in_context", "cited chunk not in context"
        else:
            grounded, method, reason = _grounding_verdict(citation.supporting_quote, chunk_text)

        checks.append(
            CitationCheck(
                chunk_id=citation.chunk_id, grounded=grounded, method=method, reason=reason
            )
        )
        if not grounded:
            unsupported.append(citation.chunk_id)

    n_citations = len(checks)
    grounded_count = sum(1 for check in checks if check.grounded)
    # Bug #2 guard: never divide by zero. 0 citations -> 0.0 by the documented rule.
    attribution_rate = grounded_count / n_citations if n_citations else 0.0

    logger.info(
        "verified answer: %d/%d grounded, attribution_rate=%.3f",
        grounded_count,
        n_citations,
        attribution_rate,
    )
    return VerificationReport(
        attribution_rate=attribution_rate,
        checks=checks,
        unsupported=unsupported,
        n_citations=n_citations,
    )
