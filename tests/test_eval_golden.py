"""Tests for the golden-set loader (ADR-0005).

The golden file order is the paired-bootstrap pairing axis, so the load order is load-bearing
and these tests pin it. They also pin the strict failure modes: a malformed line, a duplicate
``query_id`` (the join key), and a missing file must raise loudly with an actionable message,
never be skipped — a silently dropped/duplicated label corrupts every downstream metric.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.config import get_settings
from rag.eval.golden import load_golden
from rag.eval.models import GoldenItem


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_load_golden_preserves_file_order(tmp_path: Path) -> None:
    # Deliberately out of query_id order: the loader must return FILE order, not a sorted one.
    rows = [
        {"query_id": "q03", "query": "third?", "relevant_chunk_ids": ["c3"]},
        {"query_id": "q01", "query": "first?", "relevant_chunk_ids": ["c1"]},
        {"query_id": "q02", "query": "second?", "relevant_chunk_ids": ["c2"]},
    ]
    path = tmp_path / "golden.jsonl"
    _write(path, rows)

    items = load_golden(path)

    assert [item.query_id for item in items] == ["q03", "q01", "q02"]
    assert all(isinstance(item, GoldenItem) for item in items)
    assert items[0].relevant_chunk_ids == ("c3",)


def test_load_golden_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps({"query_id": "q1", "query": "a?", "relevant_chunk_ids": ["c1"]})
        + "\n\n   \n"
        + json.dumps({"query_id": "q2", "query": "b?", "relevant_chunk_ids": ["c2"]})
        + "\n",
        encoding="utf-8",
    )
    assert [item.query_id for item in load_golden(path)] == ["q1", "q2"]


def test_load_golden_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="golden set not found"):
        load_golden(tmp_path / "does-not-exist.jsonl")


def test_load_golden_malformed_line_raises_with_lineno(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps({"query_id": "q1", "query": "a?", "relevant_chunk_ids": ["c1"]})
        + "\n{ this is not json }\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 2"):
        load_golden(path)


def test_load_golden_empty_relevant_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    _write(path, [{"query_id": "q1", "query": "a?", "relevant_chunk_ids": []}])
    with pytest.raises(ValueError, match="line 1"):
        load_golden(path)


def test_load_golden_duplicate_query_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    _write(
        path,
        [
            {"query_id": "q1", "query": "a?", "relevant_chunk_ids": ["c1"]},
            {"query_id": "q1", "query": "b?", "relevant_chunk_ids": ["c2"]},
        ],
    )
    with pytest.raises(ValueError, match="duplicate query_id"):
        load_golden(path)


def test_load_golden_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        load_golden(path)


def test_load_golden_reads_committed_golden_set() -> None:
    # The committed golden set must load and have unique query ids (the join invariant).
    items = load_golden(get_settings().golden_path)
    assert len(items) >= 2
    assert len({item.query_id for item in items}) == len(items)
    assert all(item.relevant_chunk_ids for item in items)
