# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Wiring of the optional semantic components from the settings."""

from .config import Settings
from .ingestion.pipeline import Indexer
from .retrieval.concepts import ConceptError, ConceptRegistry, load_concepts
from .retrieval.embeddings import Embedder, EmbeddingError, create_embedder
from .retrieval.hybrid import HybridSearcher
from .retrieval.vector_store import QdrantVectorStore, VectorStore, VectorStoreError
from .storage.catalog import Catalog
from .workspace import Workspace

__all__ = [
    "ConceptError",
    "EmbeddingError",
    "VectorStoreError",
    "build_runtime",
    "build_vector_components",
    "load_concept_registry",
]


def load_concept_registry(settings: Settings) -> ConceptRegistry | None:
    """Load the concept registry, or None when query expansion is not configured.

    Raises
    ------
    ConceptError
        If the configured concept file is unreadable or invalid.
    """
    path = settings.retrieval.concepts_path
    if path is None:
        return None
    return load_concepts(path)


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
        searcher
        or HybridSearcher(catalog, embedder, store, concepts=load_concept_registry(settings)),
        indexer or Indexer(settings, workspace, catalog, embedder, store),
    )
