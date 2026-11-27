# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vector store implementations."""

from collections.abc import Callable

import pytest
from qdrant_client import QdrantClient

from silicore_knowledge.config import QdrantSettings
from silicore_knowledge.retrieval.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorRecord,
    VectorStore,
    VectorStoreError,
)
from silicore_knowledge.storage.models import SearchFilters

Factory = Callable[[int], VectorStore]


def _memory(dimension: int) -> VectorStore:
    return InMemoryVectorStore(dimension)


def _qdrant(dimension: int) -> VectorStore:
    return QdrantVectorStore(QdrantSettings(), dimension, client=QdrantClient(":memory:"))


@pytest.fixture(params=[_memory, _qdrant], ids=["memory", "qdrant"])
def factory(request: pytest.FixtureRequest) -> Factory:
    """Build each store implementation."""
    factory: Factory = request.param
    return factory


def _records() -> list[VectorRecord]:
    return [
        VectorRecord(
            "a:1", "doc-a", (1.0, 0.0), "ST", "STM32U5", "STM32U595", "USART", "reference_manual"
        ),
        VectorRecord("a:2", "doc-a", (0.0, 1.0), "ST", "STM32U5", "STM32U595", "SPI", "datasheet"),
        VectorRecord("b:1", "doc-b", (0.9, 0.1), "NXP", "LPC", "LPC55", "USART", "datasheet"),
    ]


def test_search_ranks_by_similarity(factory: Factory) -> None:
    """The closest vector comes first and scores are descending."""
    store = factory(2)
    store.upsert(_records())
    hits = store.search((1.0, 0.0), SearchFilters(), 3)
    assert [h.chunk_id for h in hits] == ["a:1", "b:1", "a:2"]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_search_limit(factory: Factory) -> None:
    """The limit caps the number of hits."""
    store = factory(2)
    store.upsert(_records())
    assert len(store.search((1.0, 0.0), SearchFilters(), 2)) == 2
    assert store.search((1.0, 0.0), SearchFilters(), 0) == []


def test_filters_are_case_insensitive_and_combined(factory: Factory) -> None:
    """Filters narrow results inside the store."""
    store = factory(2)
    store.upsert(_records())
    only_st = store.search((1.0, 0.0), SearchFilters(manufacturer="st"), 5)
    assert {h.chunk_id for h in only_st} == {"a:1", "a:2"}
    usart_ds = store.search(
        (1.0, 0.0), SearchFilters(peripheral="usart", document_type=("Datasheet",)), 5
    )
    assert [h.chunk_id for h in usart_ds] == ["b:1"]


def test_upsert_replaces_by_chunk_id(factory: Factory) -> None:
    """Re-inserting a chunk id replaces the previous vector."""
    store = factory(2)
    store.upsert(_records())
    store.upsert([VectorRecord("a:1", "doc-a", (0.0, 1.0))])
    assert store.count() == 3
    assert store.search((0.0, 1.0), SearchFilters(), 1)[0].score == pytest.approx(1.0)


def test_delete_document(factory: Factory) -> None:
    """Deleting a document removes all its vectors and nothing else."""
    store = factory(2)
    store.upsert(_records())
    store.delete_document("doc-a")
    store.delete_document("unknown")
    assert store.count() == 1
    assert [h.chunk_id for h in store.search((1.0, 0.0), SearchFilters(), 5)] == ["b:1"]


def test_dimension_mismatch_is_rejected(factory: Factory) -> None:
    """Vectors of the wrong size are refused, on write and on query."""
    store = factory(2)
    with pytest.raises(VectorStoreError, match="dimension"):
        store.upsert([VectorRecord("x", "d", (1.0, 2.0, 3.0))])
    with pytest.raises(VectorStoreError, match="dimension"):
        store.search((1.0,), SearchFilters(), 1)
    assert store.count() == 0


def test_invalid_dimension() -> None:
    """A non-positive dimension is rejected."""
    with pytest.raises(VectorStoreError):
        InMemoryVectorStore(0)
    with pytest.raises(VectorStoreError):
        QdrantVectorStore(QdrantSettings(), 0, client=QdrantClient(":memory:"))


def test_qdrant_reuses_existing_collection() -> None:
    """A second store on the same client keeps the indexed data."""
    client = QdrantClient(":memory:")
    first = QdrantVectorStore(QdrantSettings(), 2, client=client)
    first.upsert(_records())
    assert QdrantVectorStore(QdrantSettings(), 2, client=client).count() == 3


def test_qdrant_errors_are_translated() -> None:
    """Client failures surface as VectorStoreError."""
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(QdrantSettings(), 2, client=client)
    client.delete_collection(store._collection)  # noqa: SLF001
    with pytest.raises(VectorStoreError):
        store.count()
