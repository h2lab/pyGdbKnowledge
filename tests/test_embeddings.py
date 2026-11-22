# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for embedders."""

import math
import sys
import types
from collections.abc import Sequence
from typing import Any

import pytest

from pyGdbKnowledge.config import EmbeddingSettings
from pyGdbKnowledge.retrieval.embeddings import (
    EmbeddingError,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    """Vectors are reproducible, unit length, and of the declared dimension."""
    embedder = HashingEmbedder(32)
    first, second = embedder.embed(["USART overrun flag"]), embedder.embed(["USART overrun flag"])
    assert first == second and len(first[0]) == embedder.dimension == 32
    assert math.isclose(_cosine(first[0], first[0]), 1.0)
    assert embedder.embed([]) == []


def test_hashing_embedder_ranks_shared_words_closer() -> None:
    """Texts with common words are closer than unrelated ones."""
    embedder = HashingEmbedder(256)
    query, near, far = embedder.embed(
        ["usart dma reception overrun", "dma reception of usart data", "clock tree pll divider"]
    )
    assert _cosine(query, near) > _cosine(query, far)


def test_hashing_embedder_handles_text_without_words() -> None:
    """Text without tokens gives a zero vector instead of failing."""
    assert HashingEmbedder(8).embed(["!!!"]) == [[0.0] * 8]
    with pytest.raises(EmbeddingError):
        HashingEmbedder(0)


class _FakeModel:
    max_seq_length = 8192

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name, self.kwargs = name, kwargs
        self.encode_kwargs: dict[str, Any] = {}

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.encode_kwargs = kwargs
        return [[0.6, 0.8, 0.0] for _ in texts]


def _install_fake_module(monkeypatch: pytest.MonkeyPatch) -> list[_FakeModel]:
    created: list[_FakeModel] = []

    def factory(name: str, **kwargs: Any) -> _FakeModel:
        created.append(_FakeModel(name, **kwargs))
        return created[-1]

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return created


def test_sentence_transformer_is_offline_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model loads from local files only and sequence length is capped."""
    created = _install_fake_module(monkeypatch)
    embedder = create_embedder(EmbeddingSettings(model="local-model", max_seq_length=256))
    assert created[0].kwargs == {"local_files_only": True}
    assert created[0].max_seq_length == 256
    assert (embedder.name, embedder.dimension) == ("local-model", 3)
    assert embedder.embed(["a", "b"]) == [[0.6, 0.8, 0.0]] * 2
    assert created[0].encode_kwargs["normalize_embeddings"] is True
    assert embedder.embed([]) == []


def test_sentence_transformer_download_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Downloads happen only when explicitly allowed."""
    created = _install_fake_module(monkeypatch)
    SentenceTransformerEmbedder(EmbeddingSettings(allow_download=True))
    assert created[0].kwargs == {"local_files_only": False}


def test_missing_library_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing extra produces an actionable error."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EmbeddingError, match="vector"):
        create_embedder(EmbeddingSettings())


def test_model_load_failure_is_translated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Library errors on load become EmbeddingError."""
    module = types.ModuleType("sentence_transformers")

    def broken(name: str, **kwargs: Any) -> None:
        raise OSError("not in cache")

    module.SentenceTransformer = broken  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    with pytest.raises(EmbeddingError, match="not in cache"):
        create_embedder(EmbeddingSettings())
