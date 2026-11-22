# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Indexing pipeline: PDF file to sections, chunks, FTS5 entries and vectors."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..config import Settings
from ..retrieval.embeddings import Embedder, EmbeddingError
from ..retrieval.vector_store import VectorRecord, VectorStore, VectorStoreError
from ..security import audit_logger
from ..storage.catalog import Catalog, CatalogError
from ..storage.models import Chunk, Document, DocumentStatus
from ..workspace import Workspace, WorkspaceError
from .chunking import chunk_document
from .extract import ExtractionError, extract_pdf

logger = logging.getLogger("pyGdbKnowledge")


class IndexingError(Exception):
    """Raised when a document cannot be indexed."""


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Outcome of indexing one document."""

    document_id: str
    sections: int
    chunks: int
    vectors: int


@dataclass(slots=True)
class BatchReport:
    """Outcome of indexing several documents."""

    indexed: list[IndexResult] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


class Indexer:
    """Builds and rebuilds the derived indexes from the PDF files in the workspace.

    The catalog (with FTS5) and the vector store are derived data: they can always be rebuilt
    from ``documents/`` (ANSSI R32). Embeddings are computed before anything is modified, so
    an embedding failure leaves the previous index untouched.

    Parameters
    ----------
    settings : Settings
        Service settings.
    workspace : Workspace
        Workspace holding the PDF files.
    catalog : Catalog
        Catalog to fill.
    embedder : Embedder | None
        Embedding model; None disables the vector side.
    vector_store : VectorStore | None
        Vector index; None disables the vector side.
    """

    def __init__(
        self,
        settings: Settings,
        workspace: Workspace,
        catalog: Catalog,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._settings = settings
        self._workspace = workspace
        self._catalog = catalog
        self._embedder = embedder
        self._vector_store = vector_store

    @property
    def semantic(self) -> bool:
        """Whether vectors are produced."""
        return self._embedder is not None and self._vector_store is not None

    def index_document(self, document_id: str) -> IndexResult:
        """(Re)build the index of one document from its stored PDF.

        Parameters
        ----------
        document_id : str
            Catalog identifier.

        Returns
        -------
        IndexResult
            Counts of sections, chunks and vectors written.

        Raises
        ------
        IndexingError
            If the document is unknown, not active, missing on disk, unreadable, or if
            embedding or vector storage fails.
        """
        document = self._catalog.get_document(document_id)
        if document is None:
            raise IndexingError(f"unknown document {document_id!r}")
        if document.status is not DocumentStatus.ACTIVE:
            raise IndexingError(f"document {document_id!r} is not active")
        try:
            path = self._workspace.document_path(document.content_hash)
            if not path.is_file():
                raise IndexingError(f"stored PDF of {document_id!r} is missing")
            extracted = extract_pdf(path, self._settings.ingestion)
        except (WorkspaceError, ExtractionError) as exc:
            raise IndexingError(f"cannot extract {document_id!r}: {exc}") from exc
        chunked = chunk_document(document_id, extracted, self._settings.ingestion)
        vectors = self._embed(document, chunked.chunks)

        try:
            self._catalog.replace_content(document_id, list(chunked.sections), list(chunked.chunks))
            self._catalog.set_page_count(document_id, extracted.page_count)
        except CatalogError as exc:
            raise IndexingError(f"cannot store the index of {document_id!r}: {exc}") from exc
        if self._vector_store is not None:
            try:
                self._vector_store.delete_document(document_id)
                self._vector_store.upsert(vectors)
            except VectorStoreError as exc:
                raise IndexingError(
                    f"lexical index of {document_id!r} is ready but vectors failed "
                    f"(run reindex): {exc}"
                ) from exc
        audit_logger.info(
            "event=index document_id=%s sha256=%s sections=%d chunks=%d vectors=%d",
            document_id,
            document.content_hash,
            len(chunked.sections),
            len(chunked.chunks),
            len(vectors),
        )
        return IndexResult(document_id, len(chunked.sections), len(chunked.chunks), len(vectors))

    def _embed(self, document: Document, chunks: Sequence[Chunk]) -> list[VectorRecord]:
        if self._embedder is None or self._vector_store is None:
            return []
        step = self._settings.embeddings.batch_size
        records: list[VectorRecord] = []
        try:
            for start in range(0, len(chunks), step):
                batch = chunks[start : start + step]
                texts = [f"{chunk.section_path}\n{chunk.text}".strip() for chunk in batch]
                for chunk, vector in zip(batch, self._embedder.embed(texts), strict=True):
                    records.append(
                        VectorRecord(
                            chunk_id=chunk.chunk_id,
                            document_id=document.document_id,
                            vector=tuple(vector),
                            manufacturer=document.manufacturer,
                            device_family=document.device_family,
                            device=document.device,
                            peripheral=chunk.peripheral,
                            document_type=document.document_type,
                        )
                    )
        except EmbeddingError as exc:
            raise IndexingError(f"embedding failed for {document.document_id!r}: {exc}") from exc
        return records

    def index_documents(
        self, document_ids: Sequence[str] | None = None, *, rebuild: bool = False
    ) -> BatchReport:
        """Index several documents, continuing after individual failures.

        Parameters
        ----------
        document_ids : Sequence[str] | None
            Documents to process; None means every active document.
        rebuild : bool
            Re-index documents that already have chunks; otherwise they are skipped.

        Returns
        -------
        BatchReport
            Successes and per-document error messages.
        """
        if document_ids is None:
            targets = [
                d.document_id for d in self._all_documents() if d.status is DocumentStatus.ACTIVE
            ]
        else:
            targets = list(dict.fromkeys(document_ids))
        report = BatchReport()
        for document_id in targets:
            if not rebuild and self._catalog.chunk_count(document_id) > 0:
                continue
            try:
                report.indexed.append(self.index_document(document_id))
            except IndexingError as exc:
                logger.warning("indexing failed: %s", exc)
                report.failed[document_id] = str(exc)
        return report

    def _all_documents(self) -> list[Document]:
        documents: list[Document] = []
        while batch := self._catalog.list_documents(limit=200, offset=len(documents)):
            documents.extend(batch)
        return documents

    def delete_document(self, document_id: str, *, keep_file: bool = False) -> bool:
        """Remove a document and everything derived from it.

        Parameters
        ----------
        document_id : str
            Catalog identifier.
        keep_file : bool
            Keep the PDF in the workspace (the catalog and vector entries are still removed).

        Returns
        -------
        bool
            True if the document existed.
        """
        document = self._catalog.get_document(document_id)
        if document is None:
            return False
        # Vectors first: a failure here must not leave retrievable orphans behind.
        if self._vector_store is not None:
            try:
                self._vector_store.delete_document(document_id)
            except VectorStoreError as exc:
                raise IndexingError(f"cannot delete vectors of {document_id!r}: {exc}") from exc
        self._catalog.delete_document(document_id)
        if not keep_file:
            try:
                self._workspace.document_path(document.content_hash).unlink(missing_ok=True)
            except (WorkspaceError, OSError) as exc:
                logger.warning("cannot remove the PDF of %s: %s", document_id, exc)
        audit_logger.info(
            "event=delete document_id=%s sha256=%s file_removed=%s",
            document_id,
            document.content_hash,
            not keep_file,
        )
        return True
