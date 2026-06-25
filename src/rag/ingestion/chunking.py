"""Hand-written, deterministic chunking strategies (no LangChain text splitters).

Implementing the splitters by hand is the whole point: it shows the chunking math is
understood and under our control, mirroring the hand-written RRF elsewhere in the repo.

Three strategies sit behind a common :class:`Chunker` ABC and a :func:`get_chunker`
factory:

* ``fixed``     — fixed-size sliding window with overlap over the raw text.
* ``recursive`` — classic recursive character splitter: split on a priority list of
  separators, pack pieces up to ``chunk_size``, then apply overlap. All by hand.
* ``semantic``  — Markdown **header-aware** splitting. Sections are split on
  ``#``/``##``/``###`` ATX headings, each carrying its heading path; oversized sections
  are sub-chunked with the recursive packer. This is a *structural* approximation, **not**
  embedding-based semantic chunking (documented limitation).

Guarantees (enforced and tested):

* every chunk's length ``<= chunk_size``;
* consecutive chunks overlap by ``~chunk_overlap`` chars (except possibly the last);
* ``chunk_overlap < chunk_size`` (else ``ValueError``);
* chunks fully cover the source text — no dropped content;
* identical input -> identical chunks **and** identical chunk ids (pure arithmetic and
  ordered traversal only; no randomness, time, or set/dict-iteration leaking into output).

``chunk_size`` / ``chunk_overlap`` are interpreted as **character** counts; no external
tokenizer is used in the hot path or in tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.config import Settings
from rag.ingestion.models import Chunk, Document, make_chunk_id

# Recursive splitter separators, highest priority first; then a character-level fallback.
SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")

Span = tuple[int, int]


def _emit(
    doc: Document,
    spans: list[Span],
    *,
    strategy: str,
    heading_paths: list[list[str]] | None = None,
) -> list[Chunk]:
    """Construct chunks from validated, document-ordered ``spans``.

    This is the **only** place chunks are built, guaranteeing uniform id/ordinal/text
    construction across all strategies (they cannot drift on the id formula or break
    invariant T, since ``text`` is always ``doc.text[start:end]``).
    """
    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(spans):
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc.rel_path, start, end),
                doc_id=doc.doc_id,
                source_path=doc.source_path,
                rel_path=doc.rel_path,
                ordinal=index,
                text=doc.text[start:end],
                start=start,
                end=end,
                heading_path=(heading_paths[index] if heading_paths is not None else []),
                metadata={**doc.metadata, "strategy": strategy, "chunk_index_in_doc": index},
            )
        )
    return chunks


def _validate(chunk_size: int, chunk_overlap: int) -> None:
    """Reject configurations that would break the overlap math or loop termination."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")


def _trivial_spans(text: str, chunk_size: int) -> list[Span] | None:
    """Return spans for the empty/whitespace and shorter-than-window cases, else ``None``.

    * empty or whitespace-only -> ``[]`` (nothing to index);
    * ``len(text) <= chunk_size`` -> a single full-coverage chunk ``(0, len(text))``.
    """
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [(0, len(text))]
    return None


class Chunker(ABC):
    """Common interface for all chunking strategies.

    A chunker is **config-stateless**: ``chunk_size`` / ``chunk_overlap`` are passed at
    call time from :class:`~rag.config.Settings`, so a single instance is reusable.
    """

    #: Strategy label stamped into chunk metadata; set by each concrete subclass.
    strategy: str = ""

    @abstractmethod
    def chunk(self, doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
        """Return the document's chunks in stable document order."""
        raise NotImplementedError


class FixedChunker(Chunker):
    """Fixed-size sliding window with overlap over ``doc.text``.

    ``stride = chunk_size - chunk_overlap`` (``>= 1`` because overlap < size), so window
    *k* spans ``[k*stride, k*stride+size)`` and consecutive windows overlap by exactly
    ``size - stride == chunk_overlap``. The loop appends the final window **before**
    breaking, so the trailing characters are never dropped.
    """

    strategy = "fixed"

    def chunk(self, doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
        _validate(chunk_size, chunk_overlap)
        text = doc.text
        trivial = _trivial_spans(text, chunk_size)
        if trivial is not None:
            return _emit(doc, trivial, strategy=self.strategy)

        stride = chunk_size - chunk_overlap
        n = len(text)
        spans: list[Span] = []
        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            spans.append((start, end))
            if end == n:  # final window reached the end -> stop (tail already captured)
                break
            start += stride
        return _emit(doc, spans, strategy=self.strategy)


def _split_recursive(
    text: str, base: int, separators: tuple[str, ...], split_target: int
) -> list[Span]:
    """Recursively split ``text`` into atomic pieces, preserving absolute offsets.

    Returns contiguous, non-overlapping ``(start, end)`` spans (offsets into the parent
    document, via ``base``) that cover ``[base, base+len(text))`` exactly, each of length
    ``<= split_target``.

    ``split_target`` is the **soft pack budget** ``chunk_size - chunk_overlap``, *not*
    ``chunk_size``. Splitting to the budget (rather than to the full window) is what leaves
    headroom for :func:`_apply_overlap` to back-extend each pack by ``chunk_overlap`` without
    exceeding ``chunk_size`` — the fix for the zero-overlap defect where dense, exactly
    full-window packs left no room to overlap.

    The separator is kept attached to the **left** fragment so concatenating all fragments
    reproduces the original text (no characters lost). Fragments still too long are split
    on the next-priority separator; a fragment with no separators left that is still over
    ``split_target`` is hard-sliced at ``split_target`` (the character-level fallback that
    guarantees every piece — hence every pack before overlap — stays ``<= split_target``).
    """
    if len(text) <= split_target:
        return [(base, base + len(text))]
    if not separators:
        # No separators left: hard character split into <= split_target slices.
        spans: list[Span] = []
        offset = 0
        while offset < len(text):
            piece_end = min(offset + split_target, len(text))
            spans.append((base + offset, base + piece_end))
            offset = piece_end
        return spans

    sep, rest = separators[0], separators[1:]
    fragments = _split_keep_sep(text, sep)
    spans = []
    cursor = 0
    for fragment in fragments:
        frag_base = base + cursor
        if len(fragment) <= split_target:
            spans.append((frag_base, frag_base + len(fragment)))
        else:
            spans.extend(_split_recursive(fragment, frag_base, rest, split_target))
        cursor += len(fragment)
    return spans


def _split_keep_sep(text: str, sep: str) -> list[str]:
    """Split ``text`` on ``sep`` keeping each separator attached to its left fragment.

    ``"".join(result) == text`` always holds, so offsets stay exact and no content is
    lost. Empty trailing fragments are dropped (they carry no characters).
    """
    fragments: list[str] = []
    start = 0
    while True:
        hit = text.find(sep, start)
        if hit == -1:
            break
        cut = hit + len(sep)
        fragments.append(text[start:cut])
        start = cut
    if start < len(text):
        fragments.append(text[start:])
    return fragments


def _pack(pieces: list[Span], target: int) -> list[Span]:
    """Greedily pack contiguous ``pieces`` into packs no larger than ``target``.

    ``target`` is the soft budget ``chunk_size - chunk_overlap`` (``>= 1``), leaving room
    for :func:`_apply_overlap` to back-extend without breaching ``chunk_size``. An empty
    accumulator always admits the next piece, so a single piece that is itself larger than
    ``target`` (it can never exceed ``split_target`` after :func:`_split_recursive`, which
    equals ``target``) still lands in its own pack rather than being dropped.

    Pieces are contiguous (``pieces[k].end == pieces[k+1].start``), so packs are
    contiguous and cover the whole range. The final accumulator is flushed **after** the
    loop — the classic "last chunk silently lost" bug is avoided here.
    """
    if not pieces:
        return []
    packs: list[Span] = []
    cur_start = pieces[0][0]
    cur_end = pieces[0][0]
    for start, end in pieces:
        # Extend the current pack while it fits the budget; an empty accumulator
        # (cur_end == cur_start) always admits the next piece so a single over-budget
        # piece still gets its own pack instead of being dropped.
        if cur_end == cur_start or (end - cur_start) <= target:
            cur_end = end
        else:
            packs.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    packs.append((cur_start, cur_end))  # flush last pack — never forget this line
    return packs


def _apply_overlap(packs: list[Span], chunk_size: int, chunk_overlap: int) -> list[Span]:
    """Back-extend each pack's start to overlap its predecessor by ``~chunk_overlap`` chars.

    Overlap is applied as a left-extension so each chunk stays a contiguous slice of
    ``doc.text`` (invariant T holds and ids stay clean). The back-step is

        ``min(chunk_overlap, start - prev_start - 1, chunk_size - (end - start))``

    which simultaneously guarantees three things:

    * **Genuine overlap** — because packs are sized to the budget ``chunk_size -
      chunk_overlap``, ``chunk_size - (end - start) >= chunk_overlap`` for budget-sized
      packs, so the back-step is the full ``chunk_overlap`` (no silent collapse to zero).
    * **No redundant subset chunk** — ``start - prev_start - 1`` keeps the new start
      strictly to the right of the previous pack's start, so a chunk can never fully
      contain its neighbour even when ``chunk_overlap`` exceeds a short leading pack.
    * **Size bound** — ``chunk_size - (end - start)`` caps the back-step so the overlapped
      span never exceeds ``chunk_size`` (the last term only binds for a pack that is itself
      already larger than the budget).

    Coverage is preserved because every original character remains within at least its home
    pack (overlap only adds characters to the left; it never removes any).
    """
    if not packs:
        return []
    spans: list[Span] = [packs[0]]
    for start, end in packs[1:]:
        prev_start = spans[-1][0]
        backstep = min(chunk_overlap, start - prev_start - 1, chunk_size - (end - start))
        if backstep < 0:
            backstep = 0
        spans.append((start - backstep, end))
    return spans


def _recursive_spans(text: str, base: int, chunk_size: int, chunk_overlap: int) -> list[Span]:
    """Full recursive pipeline (split -> pack -> overlap) over ``text[base:...]``.

    ``base`` is the absolute offset of ``text`` within the document, so the returned spans
    index directly into ``doc.text``. Used both by :class:`RecursiveChunker` (base 0, whole
    doc) and by :class:`HeaderAwareChunker` (base = section start) to size-limit a section.

    Splitting and packing both use the soft budget ``chunk_size - chunk_overlap`` (at least
    ``1``) so that :func:`_apply_overlap` has room to create the advertised overlap while
    keeping every emitted span ``<= chunk_size``.
    """
    target = max(1, chunk_size - chunk_overlap)
    pieces = _split_recursive(text, base, SEPARATORS, target)
    packs = _pack(pieces, target)
    return _apply_overlap(packs, chunk_size, chunk_overlap)


class RecursiveChunker(Chunker):
    """Hand-written recursive character splitter + greedy packer + overlap.

    Splitting on ``\\n\\n`` first and only descending to finer separators for fragments
    that are still too big means pack boundaries fall on paragraph joins whenever
    paragraphs fit under ``chunk_size`` — i.e. it *prefers* paragraph, then line, then
    sentence boundaries.
    """

    strategy = "recursive"

    def chunk(self, doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
        _validate(chunk_size, chunk_overlap)
        text = doc.text
        trivial = _trivial_spans(text, chunk_size)
        if trivial is not None:
            return _emit(doc, trivial, strategy=self.strategy)

        spans = _recursive_spans(text, 0, chunk_size, chunk_overlap)
        return _emit(doc, spans, strategy=self.strategy)


def _parse_sections(text: str) -> list[tuple[int, int, list[str]]]:
    """Split ``text`` into Markdown sections, returning ``(start, end, heading_path)``.

    A line is a heading iff it matches ``^(#{1,6})\\s+...`` **and is not inside a fenced
    code block** (toggled by lines starting with ``` or ~~~). A section is a heading line
    plus all content up to (but excluding) the next heading of any level. Content before
    the first heading is a section with an empty heading path. Sections tile ``[0, n)``
    contiguously, so coverage and determinism are preserved.
    """
    n = len(text)
    if n == 0:
        return []

    heading_starts: list[tuple[int, int, str]] = []  # (line_start, level, title)
    in_fence = False
    pos = 0
    while pos < n:
        newline = text.find("\n", pos)
        line_end = n if newline == -1 else newline
        line = text[pos:line_end]
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
        elif not in_fence:
            level = _heading_level(line)
            if level:
                title = line.lstrip()[level:].strip()
                heading_starts.append((pos, level, title))
        pos = n if newline == -1 else newline + 1

    if not heading_starts:
        return [(0, n, [])]

    sections: list[tuple[int, int, list[str]]] = []
    # Leading content before the first heading.
    first_start = heading_starts[0][0]
    if first_start > 0:
        sections.append((0, first_start, []))

    stack: list[tuple[int, str]] = []  # (level, title)
    for i, (start, level, title) in enumerate(heading_starts):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = [t for _, t in stack]
        end = heading_starts[i + 1][0] if i + 1 < len(heading_starts) else n
        sections.append((start, end, heading_path))
    return sections


def _heading_level(line: str) -> int:
    """Return the ATX heading level (1-6) for ``line``, or 0 if it is not a heading.

    Per CommonMark, an ATX heading may have **up to 3** leading spaces of indentation; 4+
    leading spaces make the line an indented code block, not a heading. So ``"   ## x"``
    (3 spaces) is a level-2 heading, but ``"    ## x"`` (4 spaces) is not a heading.
    """
    indent = len(line) - len(line.lstrip(" "))
    if indent >= 4:  # 4+ leading spaces -> indented code block, not a heading
        return 0
    stripped = line.lstrip(" ")
    hashes = 0
    for char in stripped:
        if char == "#":
            hashes += 1
        else:
            break
    if 1 <= hashes <= 6 and len(stripped) > hashes and stripped[hashes] in (" ", "\t"):
        return hashes
    return 0


class HeaderAwareChunker(Chunker):
    """Structural, Markdown-header-aware splitting — the ``semantic`` strategy.

    NOT embedding-based semantic chunking. Sections are split on ``#``/``##``/``###`` ATX
    headings; each section carries its heading path; oversized sections are sub-chunked
    with the :class:`RecursiveChunker` packer and every resulting sub-chunk is stamped with
    the section's heading path. A document with no headings yields one section with an empty
    heading path, behaving like the recursive chunker.
    """

    strategy = "semantic"

    def chunk(self, doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
        _validate(chunk_size, chunk_overlap)
        text = doc.text
        if not text.strip():  # empty / whitespace-only -> nothing to index
            return []
        # NOTE: unlike fixed/recursive we do not short-circuit on len(text) <= chunk_size,
        # because even a short doc may contain headings whose paths must be attached.

        spans: list[Span] = []
        heading_paths: list[list[str]] = []
        for start, end, heading_path in _parse_sections(text):
            section = text[start:end]
            if len(section) <= chunk_size:
                spans.append((start, end))
                heading_paths.append(heading_path)
            else:
                for sub_start, sub_end in _recursive_spans(
                    section, start, chunk_size, chunk_overlap
                ):
                    spans.append((sub_start, sub_end))
                    heading_paths.append(heading_path)
        return _emit(doc, spans, strategy=self.strategy, heading_paths=heading_paths)


_CHUNKERS: dict[str, type[Chunker]] = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "semantic": HeaderAwareChunker,
}


def get_chunker(strategy: str) -> Chunker:
    """Return a chunker instance for ``strategy`` (``fixed``/``recursive``/``semantic``).

    Raises ``ValueError`` for an unknown strategy. Only the strategy string is read; size
    and overlap are supplied later at :meth:`Chunker.chunk` time.
    """
    try:
        return _CHUNKERS[strategy]()
    except KeyError:
        raise ValueError(f"unknown chunk_strategy: {strategy!r}") from None


def chunk_document(doc: Document, settings: Settings) -> list[Chunk]:
    """Chunk a single document using the strategy/size/overlap from ``settings``."""
    chunker = get_chunker(settings.chunk_strategy)
    return chunker.chunk(doc, settings.chunk_size, settings.chunk_overlap)


def chunk_corpus(docs: list[Document], settings: Settings) -> list[Chunk]:
    """Chunk every document in ``docs`` (assumed already in deterministic loader order)."""
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, settings))
    return out
