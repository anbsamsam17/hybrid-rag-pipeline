"""Validate a golden-set JSONL against BOTH the schema and the corpus.

A golden label that points at a chunk id which does not exist in the corpus
silently scores ~0 recall for that query and quietly drags down every headline
number — so this check refuses to let a malformed or dangling golden set pass.

Per line it verifies:
  * the line parses as JSON and constructs a :class:`rag.eval.GoldenItem`
    (schema rules: non-empty query_id/query, non-empty relevant_chunk_ids,
    no duplicate relevant ids);
  * every ``relevant_chunk_id`` exists among the chunk ids the corpus produces.
Across the file it also enforces unique ``query_id`` values.

Usage (from the repo root)::

    python scripts/validate_golden.py                         # golden_path + sample_dir from settings
    python scripts/validate_golden.py data/eval/golden.jsonl data/sample

Exit code 0 = valid; non-zero = at least one problem (all problems are listed).
The golden file is only READ here; it is never written or indexed.

NOTE — golden.jsonl is COUPLED to the chunking config. ``relevant_chunk_ids`` are
``sha256(f"{rel_path}:{start}:{end}")[:16]``, so they are only valid for the chunk spans the
*current* chunking config (``chunk_strategy`` / ``chunk_size`` / ``chunk_overlap``) produces —
the golden set was minted under recursive / 512 / 64. This script and ``rag.eval.harness`` both
read the same :class:`Settings` (no hard pin), so they agree by construction; but a deliberate
change to the chunking config will move the spans and require **re-minting golden.jsonl** (use
``scripts/dump_chunks.py`` to pick the new ids). The harness enforces this at runtime: its
golden-coverage guard hard-fails if any golden id is absent from the freshly built index.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `import rag` work whether or not the package is installed editable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import ValidationError  # noqa: E402

from rag.config import get_settings  # noqa: E402
from rag.eval import GoldenItem  # noqa: E402
from rag.ingestion import chunk_corpus, load_corpus  # noqa: E402


def _corpus_chunk_ids(corpus_dir: Path) -> set[str]:
    """Return the set of chunk ids the corpus currently produces (the valid label space)."""
    docs = load_corpus(corpus_dir)
    return {chunk.chunk_id for chunk in chunk_corpus(docs, get_settings())}


def main(argv: list[str]) -> int:
    """Validate the golden file; print every problem found. Returns a process exit code."""
    settings = get_settings()
    golden_path = Path(argv[1]) if len(argv) > 1 else settings.golden_path
    corpus_dir = Path(argv[2]) if len(argv) > 2 else settings.sample_dir

    if not golden_path.exists():
        print(f"golden file not found: {golden_path}", file=sys.stderr)
        return 1
    if not corpus_dir.exists():
        print(f"corpus dir not found: {corpus_dir}", file=sys.stderr)
        return 1

    valid_ids = _corpus_chunk_ids(corpus_dir)
    problems: list[str] = []
    seen_query_ids: set[str] = set()
    n_items = 0

    for lineno, raw in enumerate(golden_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue  # tolerate blank lines
        n_items += 1
        try:
            item = GoldenItem.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            problems.append(f"line {lineno}: invalid record — {exc}")
            continue

        if item.query_id in seen_query_ids:
            problems.append(f"line {lineno}: duplicate query_id {item.query_id!r}")
        seen_query_ids.add(item.query_id)

        dangling = [cid for cid in item.relevant_chunk_ids if cid not in valid_ids]
        if dangling:
            problems.append(
                f"line {lineno} ({item.query_id}): relevant_chunk_ids not in corpus: {dangling}"
            )

    if problems:
        print(f"INVALID — {len(problems)} problem(s) across {n_items} record(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"OK — {n_items} golden record(s) valid; all relevant_chunk_ids exist in the corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
