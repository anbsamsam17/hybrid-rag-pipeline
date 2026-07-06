"""Load the labeled evaluation golden set from JSONL (ADR-0005).

:func:`load_golden` reads one :class:`~rag.eval.models.GoldenItem` per line **in file order**
and returns them in that order. The order is *load-bearing*: it is the axis along which the
paired bootstrap aligns the per-query metric vectors of two configs (query at golden index
``i`` is the same query in both bootstrap arms), so it must be stable and never reshuffled.

The loader is strict on purpose — a malformed or duplicate-keyed golden row silently corrupts
the golden<->results join and quietly drags every headline number off, so each failure raises
with the offending line number rather than being skipped:

* a line that is not valid JSON or violates the :class:`GoldenItem` contract (empty
  ``query_id`` / ``query``, empty or duplicated ``relevant_chunk_ids``) raises ``ValueError``;
* a duplicate ``query_id`` across the file raises (ids key the join, so collisions are fatal);
* a missing file raises ``FileNotFoundError`` with an actionable message;
* blank lines are tolerated (skipped); an all-blank/empty file raises.

The golden file is only ever **read** here — it is never written, mutated, or indexed.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from rag.eval.models import GoldenItem


def load_golden(path: Path) -> list[GoldenItem]:
    """Load the golden set at ``path`` into :class:`GoldenItem` objects, in file order.

    Args:
        path: Path to the JSONL golden set (one ``GoldenItem`` record per line).

    Returns:
        The golden items in the **exact order they appear in the file** — the frozen ordering
        the paired bootstrap relies on to align the two configs' per-query vectors.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If any line is not valid JSON / violates the ``GoldenItem`` contract, if a
            ``query_id`` is duplicated across the file, or if the file holds no records.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"golden set not found at {path} — point settings.golden_path at the labeled "
            "JSONL (one GoldenItem per line). It is never created by the harness."
        )

    items: list[GoldenItem] = []
    seen_query_ids: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue  # tolerate blank lines
        try:
            item = GoldenItem.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"golden set line {lineno} is invalid: {exc}") from exc
        if item.query_id in seen_query_ids:
            raise ValueError(
                f"golden set line {lineno}: duplicate query_id {item.query_id!r} — query ids "
                "must be unique (they key the golden<->results join)."
            )
        seen_query_ids.add(item.query_id)
        items.append(item)

    if not items:
        raise ValueError(f"golden set at {path} contains no records")
    return items
