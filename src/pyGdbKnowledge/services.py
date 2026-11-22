# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Wiring of the optional semantic components from the settings."""

from .config import Settings
from .retrieval.embeddings import Embedder, EmbeddingError, create_embedder
from .retrieval.hybrid import HybridSearcher
from .retrieval.vector_store import QdrantVectorStore, VectorStore, VectorStoreError
from .storage.catalog import Catalog

__all__ = ["EmbeddingError", "VectorStoreError", "build_searcher", "build_vector_components"]


def build_vector_components(settings: Settings) -> tuple[Embedder, VectorStore] | None:
    """Create the embedder and the vector store, or None when semantic search is disabled.

    Raises
    ------
    EmbeddingError
        If the embedding model cannot be loaded.
    VectorStoreError
        If Qdrant is unavailable.
    """
    if not settings.embeddings.enabled:
        return None
    embedder = create_embedder(settings.embeddings)
    return embedder, QdrantVectorStore(settings.qdrant, embedder.dimension)


def build_searcher(settings: Settings, catalog: Catalog) -> HybridSearcher:
    """Create the searcher: hybrid when semantic search is enabled, lexical otherwise."""
    components = build_vector_components(settings)
    if components is None:
        return HybridSearcher(catalog)
    embedder, store = components
    return HybridSearcher(catalog, embedder, store)
