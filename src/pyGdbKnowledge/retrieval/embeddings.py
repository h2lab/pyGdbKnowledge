# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Local embedding models behind a small interface."""

import hashlib
import importlib
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol

from ..config import EmbeddingSettings

_TOKEN_RE = re.compile(r"\w+")


class EmbeddingError(Exception):
    """Raised when an embedding model cannot be loaded or used."""


class Embedder(Protocol):
    """Turns texts into L2-normalized vectors of a fixed dimension."""

    @property
    def name(self) -> str:
        """Identifier of the model, recorded with the vectors it produced."""
        ...

    @property
    def dimension(self) -> int:
        """Length of the produced vectors."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts, preserving order; an empty input yields an empty list."""
        ...


class HashingEmbedder:
    """Deterministic bag-of-words embedder, for tests and offline development only.

    Texts sharing words get similar vectors, but there is no semantic understanding.

    Parameters
    ----------
    dimension : int
        Vector length.
    """

    def __init__(self, dimension: int = 64) -> None:
        if dimension < 1:
            raise EmbeddingError("dimension must be positive")
        self._dimension = dimension

    @property
    def name(self) -> str:
        """Identifier of the model."""
        return f"hashing-{self._dimension}"

    @property
    def dimension(self) -> int:
        """Length of the produced vectors."""
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts by signed feature hashing of their lowercase words."""
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return vector if norm == 0.0 else [v / norm for v in vector]


class SentenceTransformerEmbedder:
    """Embedder backed by a local sentence-transformers model.

    The model is loaded from a local path or the local cache only, unless ``allow_download``
    is set: the daemon makes no outbound connection by default.

    Parameters
    ----------
    settings : EmbeddingSettings
        Model name or path, download policy, batch size and maximum sequence length.

    Raises
    ------
    EmbeddingError
        If sentence-transformers is not installed or the model cannot be loaded.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed; install the 'vector' extra"
            ) from exc
        try:
            model: Any = module.SentenceTransformer(
                settings.model, local_files_only=not settings.allow_download
            )
        except Exception as exc:  # the library raises many unrelated types on load failure
            raise EmbeddingError(f"cannot load embedding model {settings.model!r}: {exc}") from exc
        model.max_seq_length = min(
            int(model.max_seq_length or settings.max_seq_length), settings.max_seq_length
        )
        self._model = model
        self._batch_size = settings.batch_size
        self._name = settings.model
        self._dimension = len(self._encode(["dimension probe"])[0])

    @property
    def name(self) -> str:
        """Identifier of the model."""
        return self._name

    @property
    def dimension(self) -> int:
        """Length of the produced vectors."""
        return self._dimension

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            vectors = self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # the library raises many unrelated types at inference
            raise EmbeddingError(f"embedding failed: {exc}") from exc
        return [[float(value) for value in vector] for vector in vectors]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts with normalized vectors."""
        return self._encode(texts) if texts else []


def create_embedder(settings: EmbeddingSettings) -> Embedder:
    """Create the configured embedder.

    Parameters
    ----------
    settings : EmbeddingSettings
        Embedding configuration.

    Returns
    -------
    Embedder
        A sentence-transformers backed embedder.

    Raises
    ------
    EmbeddingError
        If the model cannot be loaded.
    """
    return SentenceTransformerEmbedder(settings)
