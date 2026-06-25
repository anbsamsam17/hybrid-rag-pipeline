"""Single source of configuration truth for the hybrid RAG pipeline.

Everything tunable lives here as a typed Pydantic ``Settings`` object, populated from
environment variables / a local ``.env`` file (never hard-coded, never read ad-hoc deep
in the code). This mirrors the reproducibility discipline of the wider project: the exact
config that produced an index/eval run can be serialized into ``meta.json`` alongside the
git SHA and corpus hash.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (src/rag/config.py -> repo/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed configuration, read from the environment / ``.env`` (case-insensitive)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM providers (at least one key required to run generation) ---
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    llm_model: str = "claude-sonnet-4-6"
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # local default -> 0 API cost for eval
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Vector store ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "hybrid_rag"

    # --- Paths ---
    # PRIVATE local corpus (e.g. proprietary business docs) — gitignored, never published.
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    # PUBLIC demo corpus shipped with the repo so the eval is reproducible by anyone.
    sample_dir: Path = PROJECT_ROOT / "data" / "sample"
    # Labeled evaluation golden set (questions -> relevant chunk ids).
    golden_path: Path = PROJECT_ROOT / "data" / "eval" / "golden.jsonl"
    # Built artifacts (index dumps, meta.json, sqlite logs) — gitignored.
    storage_dir: Path = PROJECT_ROOT / "storage"

    # --- Chunking ---
    chunk_strategy: str = "recursive"  # fixed | recursive | semantic
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)

    # --- Retrieval ---
    top_k_dense: int = Field(default=20, gt=0)
    top_k_sparse: int = Field(default=20, gt=0)
    rrf_k: int = Field(default=60, gt=0)  # Reciprocal Rank Fusion constant (Cormack et al. 2009)
    use_reranker: bool = True
    top_k_rerank: int = Field(default=5, gt=0)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance."""
    return Settings()
