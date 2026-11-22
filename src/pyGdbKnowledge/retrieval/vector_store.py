# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Vector index behind a small interface (Qdrant in production, in memory for tests).

The vector index is derived data: it holds chunk ids, never the chunk text, and can always be
rebuilt from the catalog. Metadata filters are applied inside the store (ANSSI R27).
"""

import importlib
import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import QdrantSettings
from ..storage.models import SearchFilters

_FILTER_FIELDS = ("manufacturer", "device_family", "device", "peripheral")
_NAMESPACE = uuid.UUID("5f0c1c2e-6d1b-4e47-9a53-0c8d7d2f4b11")


class VectorStoreError(Exception):
    """Raised when the vector index cannot be reached or used."""


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One indexed chunk: its vector and the metadata used for filtering."""

    chunk_id: str
    document_id: str
    vector: tuple[float, ...]
    manufacturer: str | None = None
    device_family: str | None = None
    device: str | None = None
    peripheral: str | None = None
    document_type: str | None = None


@dataclass(frozen=True, slots=True)
class VectorHit:
    """A nearest-neighbour result (higher score is better)."""

    chunk_id: str
    score: float


class VectorStore(Protocol):
    """Stores chunk vectors and answers filtered nearest-neighbour queries."""

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Insert or replace records, keyed by chunk id."""
        ...

    def search(
        self, vector: Sequence[float], filters: SearchFilters, limit: int
    ) -> list[VectorHit]:
        """Return at most ``limit`` hits matching the filters, best first."""
        ...

    def delete_document(self, document_id: str) -> None:
        """Remove every vector of a document (unknown documents are ignored)."""
        ...

    def count(self) -> int:
        """Return the number of indexed vectors."""
        ...


def _norm(value: str | None) -> str | None:
    return value.casefold() if value is not None else None


def _matches(record: VectorRecord, filters: SearchFilters) -> bool:
    for name in _FILTER_FIELDS:
        wanted = _norm(getattr(filters, name))
        if wanted is not None and _norm(getattr(record, name)) != wanted:
            return False
    if filters.document_type:
        allowed = {value.casefold() for value in filters.document_type}
        if _norm(record.document_type) not in allowed:
            return False
    return True


class InMemoryVectorStore:
    """Exact cosine search over a dictionary; for tests and tiny corpora."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise VectorStoreError("dimension must be positive")
        self._dimension = dimension
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Insert or replace records, keyed by chunk id."""
        for record in records:
            if len(record.vector) != self._dimension:
                raise VectorStoreError(
                    f"expected a vector of dimension {self._dimension}, got {len(record.vector)}"
                )
        for record in records:
            self._records[record.chunk_id] = record

    def search(
        self, vector: Sequence[float], filters: SearchFilters, limit: int
    ) -> list[VectorHit]:
        """Return at most ``limit`` hits matching the filters, best first."""
        if len(vector) != self._dimension:
            raise VectorStoreError(
                f"expected a vector of dimension {self._dimension}, got {len(vector)}"
            )
        if limit < 1:
            return []
        query_norm = math.sqrt(sum(x * x for x in vector))
        hits: list[VectorHit] = []
        for record in self._records.values():
            if not _matches(record, filters):
                continue
            record_norm = math.sqrt(sum(x * x for x in record.vector))
            if query_norm == 0.0 or record_norm == 0.0:
                continue
            dot = sum(a * b for a, b in zip(vector, record.vector, strict=True))
            hits.append(VectorHit(record.chunk_id, dot / (query_norm * record_norm)))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return hits[:limit]

    def delete_document(self, document_id: str) -> None:
        """Remove every vector of a document (unknown documents are ignored)."""
        self._records = {k: v for k, v in self._records.items() if v.document_id != document_id}

    def count(self) -> int:
        """Return the number of indexed vectors."""
        return len(self._records)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantVectorStore:
    """Qdrant-backed store; the server must only be reachable on the internal network."""

    def __init__(self, settings: QdrantSettings, dimension: int, client: Any = None) -> None:
        if dimension < 1:
            raise VectorStoreError("dimension must be positive")
        self._collection = settings.collection
        self._dimension = dimension
        self._models: Any
        if client is None:
            try:
                qdrant = importlib.import_module("qdrant_client")
            except ImportError as exc:
                raise VectorStoreError(
                    "qdrant-client is not installed; install the 'vector' extra"
                ) from exc
            client = qdrant.QdrantClient(url=settings.url, timeout=10)
        self._client = client
        self._models = importlib.import_module("qdrant_client.models")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        models = self._models
        try:
            if self._client.collection_exists(self._collection):
                return
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._dimension, distance=models.Distance.COSINE
                ),
            )
        except Exception as exc:
            raise VectorStoreError(f"cannot prepare collection: {exc}") from exc

    def _filter(self, filters: SearchFilters) -> Any:
        models = self._models
        must: list[Any] = []
        for name in _FILTER_FIELDS:
            value = _norm(getattr(filters, name))
            if value is not None:
                must.append(models.FieldCondition(key=name, match=models.MatchValue(value=value)))
        if filters.document_type:
            must.append(
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchAny(any=[v.casefold() for v in filters.document_type]),
                )
            )
        return models.Filter(must=must) if must else None

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Insert or replace records, keyed by chunk id."""
        for record in records:
            if len(record.vector) != self._dimension:
                raise VectorStoreError(
                    f"expected a vector of dimension {self._dimension}, got {len(record.vector)}"
                )
        points = [
            self._models.PointStruct(
                id=_point_id(record.chunk_id),
                vector=list(record.vector),
                payload={
                    "chunk_id": record.chunk_id,
                    "document_id": record.document_id,
                    **{name: _norm(getattr(record, name)) for name in _FILTER_FIELDS},
                    "document_type": _norm(record.document_type),
                },
            )
            for record in records
        ]
        if not points:
            return
        try:
            self._client.upsert(collection_name=self._collection, points=points, wait=True)
        except Exception as exc:
            raise VectorStoreError(f"upsert failed: {exc}") from exc

    def search(
        self, vector: Sequence[float], filters: SearchFilters, limit: int
    ) -> list[VectorHit]:
        """Return at most ``limit`` hits matching the filters, best first."""
        if len(vector) != self._dimension:
            raise VectorStoreError(
                f"expected a vector of dimension {self._dimension}, got {len(vector)}"
            )
        if limit < 1:
            return []
        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=list(vector),
                query_filter=self._filter(filters),
                limit=limit,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
        except Exception as exc:
            raise VectorStoreError(f"search failed: {exc}") from exc
        return [
            VectorHit(str(point.payload["chunk_id"]), float(point.score))
            for point in response.points
            if point.payload and "chunk_id" in point.payload
        ]

    def delete_document(self, document_id: str) -> None:
        """Remove every vector of a document (unknown documents are ignored)."""
        models = self._models
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
        )
        try:
            self._client.delete(
                collection_name=self._collection, points_selector=selector, wait=True
            )
        except Exception as exc:
            raise VectorStoreError(f"delete failed: {exc}") from exc

    def count(self) -> int:
        """Return the number of indexed vectors."""
        try:
            return int(self._client.count(collection_name=self._collection, exact=True).count)
        except Exception as exc:
            raise VectorStoreError(f"count failed: {exc}") from exc
