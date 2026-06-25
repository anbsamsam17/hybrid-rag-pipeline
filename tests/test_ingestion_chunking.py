"""Tests for the hand-written chunking strategies.

These use only :class:`Settings` overrides and synthetic text — no external corpus, no
optional dependencies. They pin the three correctness invariants the design calls out:
size bound, exact overlap arithmetic, and full coverage with no dropped tail; plus
determinism, the id formula, and edge cases (empty, shorter-than-window, unicode,
oversize-unsplittable, factory errors).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rag.config import Settings
from rag.ingestion.chunking import (
    FixedChunker,
    HeaderAwareChunker,
    RecursiveChunker,
    chunk_document,
    get_chunker,
)
from rag.ingestion.models import Document, make_doc_id

STRATEGIES = ["fixed", "recursive", "semantic"]


def make_doc(text: str, rel_path: str = "t.md") -> Document:
    """Build a Document directly (no loader) for chunker tests."""
    return Document(
        doc_id=make_doc_id(rel_path),
        source_path=f"/x/{rel_path}",
        rel_path=rel_path,
        title="t",
        text=text,
        metadata={"source_format": "text"},
    )


def make_settings(strategy: str, size: int, overlap: int) -> Settings:
    """Settings with chunking overrides and dummy paths (no env, no external corpus)."""
    return Settings(
        chunk_strategy=strategy,
        chunk_size=size,
        chunk_overlap=overlap,
        corpus_dir=Path("."),
        sample_dir=Path("."),
    )


# A multi-paragraph body of small paragraphs (each well under chunk_size).
PARAGRAPHS = "\n\n".join(f"Paragraph number {i} has some words in it." for i in range(20))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_chunk_within_size(strategy: str) -> None:
    text = PARAGRAPHS + "\n\n" + ("x" * 2000)
    settings = make_settings(strategy, size=128, overlap=24)
    chunks = chunk_document(make_doc(text), settings)
    assert chunks
    assert all(len(c.text) <= settings.chunk_size for c in chunks)


def test_fixed_overlap_exact() -> None:
    text = "a" * 1000
    settings = make_settings("fixed", size=200, overlap=50)
    chunks = chunk_document(make_doc(text), settings)
    stride = settings.chunk_size - settings.chunk_overlap
    # Non-final windows advance by exactly `stride`.
    for i in range(1, len(chunks)):
        if chunks[i].end != len(text):
            assert chunks[i].start - chunks[i - 1].start == stride
    # Consecutive windows overlap by exactly `chunk_overlap` while interior.
    for i in range(1, len(chunks)):
        overlap_len = chunks[i - 1].end - chunks[i].start
        assert overlap_len >= 1
        if chunks[i].end != len(text) or len(chunks[i].text) == settings.chunk_size:
            assert overlap_len == settings.chunk_overlap


def _covers(text: str, chunks: list) -> bool:
    """True iff the union of [start, end) spans covers [0, len(text)) with no gaps."""
    spans = sorted((c.start, c.end) for c in chunks)
    cursor = 0
    for start, end in spans:
        if start > cursor:
            return False  # gap
        cursor = max(cursor, end)
    return cursor == len(text)


def test_full_coverage_fixed() -> None:
    text = "abcdefghij" * 137  # 1370 chars, not a multiple of the window
    settings = make_settings("fixed", size=256, overlap=64)
    chunks = chunk_document(make_doc(text), settings)
    assert _covers(text, chunks)
    assert chunks[-1].end == len(text)  # tail present


def test_recursive_full_coverage() -> None:
    text = PARAGRAPHS + "\n\nFinal trailing paragraph with a distinct tail marker ZZZ."
    settings = make_settings("recursive", size=120, overlap=20)
    chunks = chunk_document(make_doc(text), settings)
    assert _covers(text, chunks)
    assert chunks[-1].end == len(text)
    assert "ZZZ" in chunks[-1].text  # the tail is not dropped


def test_recursive_prefers_paragraph() -> None:
    # Paragraphs each fit under chunk_size; boundaries should land on \n\n joins.
    settings = make_settings("recursive", size=120, overlap=20)
    chunks = chunk_document(make_doc(PARAGRAPHS), settings)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        # A non-final chunk ends right after a paragraph separator (kept-left "\n\n").
        assert chunk.text.endswith("\n\n") or chunk.text.endswith("\n")
    # No paragraph that fits is split across a boundary mid-word.
    assert _covers(PARAGRAPHS, chunks)


def test_recursive_prefers_line_then_sentence() -> None:
    # No blank lines, so the splitter must fall back to "\n", then ". ".
    lines = "\n".join(f"Sentence {i}. Another clause {i}." for i in range(40))
    settings = make_settings("recursive", size=80, overlap=10)
    chunks = chunk_document(make_doc(lines), settings)
    assert _covers(lines, chunks)
    # Interior boundaries fall on a line or sentence separator, never mid-token.
    for chunk in chunks[:-1]:
        tail = chunk.text
        assert tail.endswith("\n") or tail.endswith(". ") or tail.endswith(".")


@pytest.mark.parametrize("strategy", ["recursive", "semantic"])
def test_overlap_invariant_recursive_semantic(strategy: str) -> None:
    """Interior consecutive chunks must genuinely overlap by ~chunk_overlap.

    This pins the design's "consecutive chunks overlap by ~chunk_overlap" guarantee for the
    recursive AND semantic strategies (it was previously asserted only for ``fixed``). On a
    dense, separator-free body the chunker produces full-window packs; before the budget-aware
    split fix this collapsed to ZERO overlap, so this test fails against the pre-fix code and
    passes after it.
    """
    size, overlap = 120, 30
    text = "a" * 3000  # no separators -> dense, full-window packs
    settings = make_settings(strategy, size=size, overlap=overlap)
    chunks = chunk_document(make_doc(text), settings)
    assert len(chunks) > 2
    interior_overlaps = [
        chunks[i - 1].end - chunks[i].start
        for i in range(1, len(chunks))
        if chunks[i].end != len(text)  # ignore the (possibly short) trailing chunk
    ]
    assert interior_overlaps  # there is at least one interior boundary
    # Every interior boundary overlaps by exactly chunk_overlap (no silent zero-overlap).
    assert all(ov == overlap for ov in interior_overlaps)
    # And the realized overlap is strictly positive everywhere (the core regression guard).
    assert all((chunks[i - 1].end - chunks[i].start) >= 1 for i in range(1, len(chunks)))


@pytest.mark.parametrize("strategy", ["recursive", "semantic"])
def test_no_chunk_is_strict_superset(strategy: str) -> None:
    """No emitted chunk fully contains another, even with overlap > a short leading pack.

    Guards the "redundant subset chunk" wart: a large overlap relative to a tiny leading
    section used to emit a chunk that strictly contained its predecessor.
    """
    # A tiny first paragraph, then larger ones, with a large overlap relative to it.
    text = "Hi.\n\n" + "\n\n".join("word " * 30 for _ in range(6))
    settings = make_settings(strategy, size=120, overlap=60)
    chunks = chunk_document(make_doc(text), settings)
    spans = [(c.start, c.end) for c in chunks]
    for i, (si, ei) in enumerate(spans):
        for j, (sj, ej) in enumerate(spans):
            if i == j:
                continue
            assert not (
                si <= sj and ei >= ej and (si < sj or ei > ej)
            ), f"chunk {i} {spans[i]} is a strict superset of chunk {j} {spans[j]}"


def test_header_aware_heading_path() -> None:
    text = "# A\n" "Intro line under A.\n\n" "## B\n" "Content under B that is short.\n"
    settings = make_settings("semantic", size=200, overlap=20)
    chunks = chunk_document(make_doc(text), settings)
    paths = [c.heading_path for c in chunks]
    assert ["A"] in paths
    assert ["A", "B"] in paths
    # The last chunk comes from under "## B" nested below "# A".
    assert chunks[-1].heading_path == ["A", "B"]


def test_header_aware_pre_heading_empty_path() -> None:
    text = "Preamble before any heading.\n\n# Title\nBody under title.\n"
    settings = make_settings("semantic", size=200, overlap=20)
    chunks = chunk_document(make_doc(text), settings)
    assert chunks[0].heading_path == []
    assert chunks[0].text.startswith("Preamble")
    assert any(c.heading_path == ["Title"] for c in chunks)


def test_header_aware_indented_heading_not_a_heading() -> None:
    """A 4-space-indented '    ## x' is an indented code block, not a heading (CommonMark).

    A heading with <=3 leading spaces still counts, so we assert both: the real ``# Top``
    is picked up, but the indented ``    ## Fake`` does not start a section (no chunk carries
    it in its heading path).
    """
    text = (
        "# Top\n"
        "Body under top.\n"
        "    ## Fake heading (indented code, 4 spaces)\n"
        "More body still under top.\n"
    )
    settings = make_settings("semantic", size=200, overlap=20)
    chunks = chunk_document(make_doc(text), settings)
    paths = [c.heading_path for c in chunks]
    assert ["Top"] in paths
    # The indented line must NOT have created a ["Top", "Fake ..."] section.
    assert all("Fake heading" not in part for path in paths for part in path)


def test_header_aware_three_space_indent_is_heading() -> None:
    """A heading indented by <=3 spaces is still a heading (CommonMark tolerance)."""
    text = "# Top\nIntro.\n\n   ## Sub\nContent under sub.\n"
    settings = make_settings("semantic", size=200, overlap=20)
    chunks = chunk_document(make_doc(text), settings)
    assert any(c.heading_path == ["Top", "Sub"] for c in chunks)


def test_header_aware_no_headings_like_recursive() -> None:
    text = PARAGRAPHS
    settings = make_settings("semantic", size=120, overlap=20)
    chunks = chunk_document(make_doc(text), settings)
    assert _covers(text, chunks)
    assert all(c.heading_path == [] for c in chunks)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_determinism_ids(strategy: str) -> None:
    text = PARAGRAPHS + "\n\n## Section\n" + ("y" * 500)
    settings = make_settings(strategy, size=128, overlap=24)
    doc = make_doc(text)
    first = chunk_document(doc, settings)
    second = chunk_document(doc, settings)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]
    assert [c.start for c in first] == [c.start for c in second]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chunk_id_formula(strategy: str) -> None:
    text = PARAGRAPHS
    settings = make_settings(strategy, size=130, overlap=30)
    doc = make_doc(text, rel_path="folder/note.md")
    for chunk in chunk_document(doc, settings):
        expected = hashlib.sha256(f"{doc.rel_path}:{chunk.start}:{chunk.end}".encode()).hexdigest()[
            :16
        ]
        assert chunk.chunk_id == expected


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic"])
def test_text_equals_slice(strategy: str) -> None:
    text = PARAGRAPHS + "\n\n# H\nmore"
    settings = make_settings(strategy, size=100, overlap=20)
    doc = make_doc(text)
    for chunk in chunk_document(doc, settings):
        assert chunk.text == doc.text[chunk.start : chunk.end]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("text", ["", "   \n\t  ", "\n\n\n"])
def test_empty_doc(strategy: str, text: str) -> None:
    settings = make_settings(strategy, size=100, overlap=20)
    assert chunk_document(make_doc(text), settings) == []


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_shorter_than_chunk(strategy: str) -> None:
    text = "short body"
    settings = make_settings(strategy, size=100, overlap=20)
    chunks = chunk_document(make_doc(text), settings)
    assert len(chunks) == 1
    assert chunks[0].start == 0
    assert chunks[0].end == len(text)
    assert chunks[0].text == text


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_unicode_safe(strategy: str) -> None:
    unit = "café 漢字 emoji 🚀 résumé naïve. "
    text = unit * 60
    settings = make_settings(strategy, size=90, overlap=20)
    doc = make_doc(text)
    chunks = chunk_document(doc, settings)
    assert all(len(c.text) <= settings.chunk_size for c in chunks)
    assert _covers(text, chunks)
    for chunk in chunks:
        assert chunk.text == doc.text[chunk.start : chunk.end]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_overlap_ge_size_raises(strategy: str) -> None:
    chunker = get_chunker(strategy)
    doc = make_doc("a" * 500)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunker.chunk(doc, chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunker.chunk(doc, chunk_size=100, chunk_overlap=150)


@pytest.mark.parametrize("strategy", ["recursive", "semantic"])
def test_oversize_unsplittable_piece(strategy: str) -> None:
    # A single 3000-char token with no separators must still be hard-split <= size.
    text = "Z" * 3000
    settings = make_settings(strategy, size=512, overlap=64)
    doc = make_doc(text)
    chunks = chunk_document(doc, settings)
    assert all(len(c.text) <= settings.chunk_size for c in chunks)
    assert _covers(text, chunks)


def test_get_chunker_factory() -> None:
    assert isinstance(get_chunker("fixed"), FixedChunker)
    assert isinstance(get_chunker("recursive"), RecursiveChunker)
    assert isinstance(get_chunker("semantic"), HeaderAwareChunker)
    with pytest.raises(ValueError, match="unknown chunk_strategy"):
        get_chunker("nope")


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_ordinals_contiguous(strategy: str) -> None:
    text = PARAGRAPHS + "\n\n## S\n" + ("w" * 400)
    settings = make_settings(strategy, size=128, overlap=24)
    chunks = chunk_document(make_doc(text), settings)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["chunk_index_in_doc"] == c.ordinal for c in chunks)
    assert all(c.metadata["strategy"] == strategy for c in chunks)
