"""Tests for the dense embedders.

These cover only the deterministic, dependency-free :class:`HashingEmbedder` and the
``get_embedder(..., fake=True)`` factory path — no model is loaded, no network is touched,
and ``sentence_transformers`` is never imported (the suite passes with it absent).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from rag.config import Settings
from rag.indexing.embeddings import HashingEmbedder, get_embedder


def make_settings() -> Settings:
    """Settings with dummy paths (no env, no external corpus)."""
    return Settings(corpus_dir=Path("."), sample_dir=Path("."))


def test_dim_property() -> None:
    embedder = HashingEmbedder(dim=128)
    assert embedder.dim == 128
    vectors = embedder.embed_texts(["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 128


def test_default_dim_is_256() -> None:
    assert HashingEmbedder().dim == 256


def test_determinism_same_text_same_vector() -> None:
    embedder = HashingEmbedder()
    text = "Reproducibility is the whole point of this pipeline."
    first = embedder.embed_texts([text])[0]
    second = embedder.embed_texts([text])[0]
    assert first == second  # exact equality, not approximate


def test_determinism_across_instances() -> None:
    text = "deterministic feature hashing"
    a = HashingEmbedder(dim=64).embed_texts([text])[0]
    b = HashingEmbedder(dim=64).embed_texts([text])[0]
    assert a == b


def test_l2_normalized_for_nonempty() -> None:
    embedder = HashingEmbedder()
    vec = embedder.embed_texts(["some tokens here to hash"])[0]
    norm = math.sqrt(sum(value * value for value in vec))
    assert abs(norm - 1.0) < 1e-9


@pytest.mark.parametrize("text", ["", "   \n\t  ", "!!! ??? ...", "----"])
def test_token_less_text_is_all_zeros(text: str) -> None:
    embedder = HashingEmbedder(dim=32)
    vec = embedder.embed_texts([text])[0]
    assert len(vec) == 32
    assert all(value == 0.0 for value in vec)  # no ZeroDivisionError, no NaN


def test_distinct_texts_differ() -> None:
    embedder = HashingEmbedder()
    a = embedder.embed_texts(["alpha beta gamma"])[0]
    b = embedder.embed_texts(["delta epsilon zeta"])[0]
    assert a != b


def test_order_preserved_and_len_matches() -> None:
    embedder = HashingEmbedder(dim=48)
    texts = ["one", "two", "three"]
    vectors = embedder.embed_texts(texts)
    assert len(vectors) == len(texts)
    # Independently embedding "two" matches the batched position.
    assert vectors[1] == embedder.embed_texts(["two"])[0]


def test_invalid_dim_raises() -> None:
    with pytest.raises(ValueError, match="dim"):
        HashingEmbedder(dim=0)


def test_get_embedder_fake_returns_hashing() -> None:
    embedder = get_embedder(make_settings(), fake=True)
    assert isinstance(embedder, HashingEmbedder)
