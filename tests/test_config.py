"""Smoke tests for the configuration layer — no API keys or services required."""
from __future__ import annotations

from rag.config import Settings, get_settings


def test_settings_defaults() -> None:
    s = Settings()
    assert s.rrf_k == 60  # RRF constant from the founding paper
    assert s.chunk_size > 0
    assert s.chunk_overlap < s.chunk_size
    assert s.top_k_rerank <= s.top_k_dense
    assert s.chunk_strategy in {"fixed", "recursive", "semantic"}


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_paths_are_under_project_root() -> None:
    s = Settings()
    # corpus (private) and golden set resolve to concrete paths we can build on later.
    assert s.corpus_dir.name == "corpus"
    assert s.golden_path.name == "golden.jsonl"
