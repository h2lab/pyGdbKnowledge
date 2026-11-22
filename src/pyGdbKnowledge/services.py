# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Wiring of the optional semantic components from the settings."""

from .config import Settings
from .ingestion.pipeline import Indexer
from .retrieval.embeddings import Embedder, EmbeddingError, create_embedder
from .retrieval.hybrid import HybridSearcher
from .retrieval.vector_store import QdrantVectorStore, VectorStore, VectorStoreError
from .storage.catalog import Catalog
from .workspace import Workspace

__all__ = ["EmbeddingError", "VectorStoreError", "build_runtime", "build_vector_components"]


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


def build_runtime(
    settings: Settings,
    workspace: Workspace,
    catalog: Catalog,
    searcher: HybridSearcher | None = None,
    indexer: Indexer | None = None,
) -> tuple[HybridSearcher, Indexer]:
    """Create the searcher and the indexer, sharing one embedder and vector store.

    Components that are already provided are kept; the semantic ones (model, Qdrant) are
    only created when something is missing.
    """
    if searcher is not None and indexer is not None:
        return searcher, indexer
    components = build_vector_components(settings)
    embedder, store = components if components else (None, None)
    return (
        searcher or HybridSearcher(catalog, embedder, store),
        indexer or Indexer(settings, workspace, catalog, embedder, store),
    )
