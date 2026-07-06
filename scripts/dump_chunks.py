"""Dev tool — list every chunk the ingestion pipeline produces for a corpus.

This is the catalogue you use to build the evaluation golden set: it prints one
row per chunk so you can pick the ``chunk_id`` that actually contains a given
question's answer. Chunk ids are deterministic — ``sha256(f"{rel_path}:{start}:{end}")[:16]``
— so the ids printed here are exactly the ids a golden label must reference.

Usage (from the repo root)::

    python scripts/dump_chunks.py            # uses settings.sample_dir
    python scripts/dump_chunks.py data/sample

Output columns (tab-separated): chunk_id, rel_path, #ordinal, text preview.
Nothing is written or indexed; the script is read-only over the corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import rag` work whether or not the package is installed editable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings  # noqa: E402
from rag.ingestion import chunk_corpus, load_corpus  # noqa: E402

_PREVIEW_CHARS = 110


def main(argv: list[str]) -> int:
    """Load + chunk the corpus and print one row per chunk. Returns a process exit code."""
    settings = get_settings()
    corpus_dir = Path(argv[1]) if len(argv) > 1 else settings.sample_dir

    if not corpus_dir.exists():
        print(f"corpus dir not found: {corpus_dir}", file=sys.stderr)
        return 1

    docs = load_corpus(corpus_dir)
    chunks = chunk_corpus(docs, settings)

    print(
        f"# corpus_dir={corpus_dir}  docs={len(docs)}  chunks={len(chunks)}  "
        f"strategy={settings.chunk_strategy}  size={settings.chunk_size}  "
        f"overlap={settings.chunk_overlap}"
    )
    print("# chunk_id\trel_path\t#ordinal\tpreview")
    for chunk in chunks:
        preview = " ".join(chunk.text.split())[:_PREVIEW_CHARS]
        print(f"{chunk.chunk_id}\t{chunk.rel_path}\t#{chunk.ordinal}\t{preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
