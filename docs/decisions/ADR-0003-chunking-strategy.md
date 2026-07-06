# ADR-0003 — Chunking strategy: recursive character splitter, 512 / 64 (chars)

- Status: Accepted (retrospective — recorded 2026-07-02)
- Date: 2026-07-02 (decision made and implemented earlier; recorded here retrospectively)
- Deciders: rag-architect (implemented by retrieval-engineer)
- Scope: `src/rag/ingestion/chunking.py` (the three chunkers + factory) and the pinned
  parameters in `src/rag/config.py` (`chunk_strategy`, `chunk_size`, `chunk_overlap`). The
  chunk-id formula that couples this choice to the golden set lives in
  `src/rag/ingestion/models.py` (`make_chunk_id`).

## Context

Chunking decides what a "retrievable unit" is; it sets the granularity both indices
([ADR-0001](ADR-0001-hybrid-retrieval.md)) operate on. Three strategies are implemented by
hand (no LangChain splitters, deliberately, to keep the math under our control and
deterministic), all behind a common `Chunker` ABC and a `get_chunker` factory in
`src/rag/ingestion/chunking.py`:

- **`fixed`** — fixed-size sliding window with overlap over raw text (`FixedChunker`).
- **`recursive`** — recursive character splitter: split on a priority separator list
  (`"\n\n"`, `"\n"`, `". "`, `" "`), greedily pack pieces to a soft budget, then back-extend
  each pack to create overlap (`RecursiveChunker`).
- **`semantic`** — Markdown header-aware structural splitting: sections split on ATX
  headings, each carrying its heading path, oversized sections sub-chunked with the recursive
  packer (`HeaderAwareChunker`). Explicitly **not** embedding-based semantic chunking
  (documented limitation in the module docstring).

Every chunk's id is `sha256(f"{rel_path}:{start}:{end}")[:16]` (`make_chunk_id`), i.e. it is
**derived from the character span** the chunker chose. This makes chunk identity a pure
function of `(rel_path, start, end)` — which is what couples chunking to the golden set below.

## Decision drivers

- **Boundary quality** — chunks should prefer natural boundaries (paragraph → line → sentence)
  over blind mid-word cuts, so a retrieved chunk is a coherent, citable span.
- **Determinism** — identical input must yield identical chunks *and* identical ids (pure
  arithmetic + ordered traversal only), because ids feed `meta.json`, both indices, and the
  golden set.
- **Coverage** — chunks must tile the document with no dropped content (a lost tail is a
  silent recall hole).
- **Golden-set stability** — because ids are span-derived, the chosen strategy/size/overlap
  effectively becomes frozen once a golden set is minted against it (see Consequences).

## Options considered

**(a) `fixed` (sliding window).** Simplest and most predictable size distribution. Cons: cuts
mid-sentence and mid-word at arbitrary offsets, so chunk boundaries ignore document structure;
a retrieved chunk often starts/ends mid-thought, hurting both rerank and citation quality.

**(b) `recursive` character splitter (chosen).** Splits on paragraph, then line, then
sentence, then word separators, descending only for fragments still over budget, then packs
and overlaps. Pros: pack boundaries fall on paragraph/line/sentence joins whenever they fit
under `chunk_size`, giving coherent spans while still guaranteeing every chunk `<= chunk_size`
and `~chunk_overlap` overlap; fully deterministic; language-agnostic (no tokenizer in the hot
path — sizes are **character** counts). Cons: character budget is a coarse proxy for token
budget; a purely structural split can still break a long paragraph mid-sentence at the
character fallback.

**(c) `semantic` (header-aware).** Best structural fidelity on well-headed Markdown (each
chunk carries its heading breadcrumb). Cons: only as good as the document's heading structure;
a long unheaded section degrades to the recursive packer anyway; and it is **not** true
embedding-based semantic chunking, so the name oversells unless caveated.

## Decision

Pin **`recursive`** with **`chunk_size=512`** and **`chunk_overlap=64`** (characters) as the
defaults in `src/rag/config.py`. The golden set (`data/eval/golden.jsonl`) was minted against
exactly this config (recursive / 512 / 64), and its `relevant_chunk_ids` are the span-derived
ids those parameters produce. `fixed` and `semantic` remain implemented and tested but are not
the shipped default. The winning params are recorded in `meta.json` (`chunking.strategy`,
`chunk_size`, `chunk_overlap`) on every build so a run is reconstructible.

## Consequences

Positive:
- Chunk boundaries prefer paragraph/line/sentence joins → coherent, citable spans that help
  both the cross-encoder rerank and citation verification.
- Deterministic and coverage-complete → ids are stable across rebuilds and platforms
  (`rel_path` is POSIX-normalized), so `meta.json` and both indices agree bit-for-bit.
- All three strategies exist behind one seam, so switching is a config change, not a rewrite.

Negative / accepted — **the load-bearing coupling**:
- **Any chunking change invalidates the golden set.** `relevant_chunk_ids` are
  `sha256(rel_path:start:end)[:16]`, valid only for the exact spans recursive/512/64 emits.
  Change the strategy, size, or overlap and every golden id points at a span that no longer
  exists, so the id is absent from the rebuilt index. The eval harness's coverage guard
  (`_check_golden_coverage` in `src/rag/eval/harness.py`) then **hard-fails** with
  `GoldenCoverageError` ("golden id not in index — chunking config likely drifted from the
  config that minted golden.jsonl, or wrong corpus indexed"), listing the missing ids. This is
  by design: a warning would let a corrupted run emit authoritative-looking numbers.
- **Chunking is therefore effectively frozen per golden-set generation.** Re-tuning the
  strategy/size/overlap requires **re-minting** `data/eval/golden.jsonl` against the new spans
  before the eval can pass again — a deliberate, tracked operation, not a casual edit.
- **Character budget ≠ token budget.** 512 chars is a coarse proxy; if a token-accurate budget
  is ever needed, it is a chunking change and triggers the re-mint above.

## Evidence

`config.py` defaults: `chunk_strategy="recursive"`, `chunk_size=512` (`gt=0`),
`chunk_overlap=64` (`ge=0`). The committed n=50 eval (README "Evaluation results"; reproducible
via `make eval`, git 7e8ccb3) was run over `data/sample` chunked with this config: **12
documents → 39 chunks**, corpus SHA-256 `beb2701a`. [ADR-0005](ADR-0005-retrieval-eval-harness.md)
records that the golden set was minted "under recursive / 512 / 64" and that its ids are
span-derived, and documents the coverage guard as the enforcement mechanism.

Honest limitation: there is **no committed eval sweep** comparing `fixed` / `recursive` /
`semantic` or alternative sizes/overlaps head-to-head — the project heuristic is that chunking
*should* be an eval-measured choice, but in practice recursive/512/64 was selected on the
boundary-quality/determinism reasoning above and then frozen by the golden-set coupling. A
proper strategy/size sweep would require minting a **separate golden set per candidate config**
(since ids are span-derived) and comparing them as paired configs in the harness; that sweep is
not in the repo today and is the honest next step before any of these numbers is presented as
"recursive beats fixed by X".

## Cross-links

Feeds both indices in [ADR-0001](ADR-0001-hybrid-retrieval.md); the id formula and the golden
coupling are enforced by the harness in [ADR-0005](ADR-0005-retrieval-eval-harness.md). Linked
from `docs/architecture.md` (Decisions).
