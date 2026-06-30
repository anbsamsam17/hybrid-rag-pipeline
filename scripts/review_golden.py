"""Human-review aid — print each golden question next to the FULL text of the
chunk(s) it marks as relevant, so a human can confirm the label is correct.

`validate_golden.py` proves the file is well-formed and that every chunk id
exists. It CANNOT judge whether the cited passage actually answers the question
— that is a human call. This script lays the two side by side for that review.

Usage (from the repo root)::

    python scripts/review_golden.py                          # settings defaults
    python scripts/review_golden.py data/eval/golden.jsonl data/sample

Read-only: nothing is written or indexed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `import rag` work whether or not the package is installed editable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings  # noqa: E402
from rag.eval import GoldenItem  # noqa: E402
from rag.ingestion import chunk_corpus, load_corpus  # noqa: E402


def main(argv: list[str]) -> int:
    """Print question + reference answer + each relevant chunk's text. Returns exit code."""
    settings = get_settings()
    golden_path = Path(argv[1]) if len(argv) > 1 else settings.golden_path
    corpus_dir = Path(argv[2]) if len(argv) > 2 else settings.sample_dir

    if not golden_path.exists():
        print(f"golden file not found: {golden_path}", file=sys.stderr)
        return 1

    docs = load_corpus(corpus_dir)
    text_by_id = {c.chunk_id: (c.rel_path, c.text) for c in chunk_corpus(docs, get_settings())}

    for raw in golden_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        item = GoldenItem.model_validate(json.loads(line))
        print("=" * 90)
        print(f"[{item.query_id}] {item.query}")
        if item.reference_answer:
            print(f"  reference_answer: {item.reference_answer}")
        for cid in item.relevant_chunk_ids:
            found = text_by_id.get(cid)
            if found is None:
                print(f"  !! {cid}: NOT FOUND IN CORPUS")
                continue
            rel_path, text = found
            snippet = " ".join(text.split())
            print(f"  -> {cid} ({rel_path}):")
            print(f"       {snippet}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
