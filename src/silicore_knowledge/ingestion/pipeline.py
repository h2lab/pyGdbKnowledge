# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Indexing pipeline: PDF or SVD file to sections, chunks, FTS5 entries and vectors."""

import hashlib
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..retrieval.embeddings import Embedder, EmbeddingError
from ..retrieval.vector_store import VectorRecord, VectorStore, VectorStoreError
from ..security import audit_logger
from ..storage.catalog import Catalog, CatalogError
from ..storage.models import Chunk, Document, DocumentStatus
from ..workspace import Area, Workspace, WorkspaceError
from .chunking import ChunkedDocument, chunk_document
from .extract import ExtractedDocument, ExtractionError, extract_pdf
from .ocr import OcrError, needs_ocr, ocr_pdf
from .screening import screen_chunks
from .svd import SvdError, chunk_svd

logger = logging.getLogger("silicore_knowledge")


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

    def index_document(self, document_id: str, *, screen: bool = True) -> IndexResult:
        """(Re)build the index of one document from its stored PDF.

        Parameters
        ----------
        document_id : str
            Catalog identifier.
        screen : bool
            Screen the content for prompt-injection signals and quarantine the document on a
            hit; disabled only when a reviewer releases a document.

        Returns
        -------
        IndexResult
            Counts of sections, chunks and vectors written.

        Raises
        ------
        IndexingError
            If the document is unknown, not active, quarantined by the screening, missing on
            disk, unreadable, or if embedding or vector storage fails.
        """
        document = self._catalog.get_document(document_id)
        if document is None:
            raise IndexingError(f"unknown document {document_id!r}")
        if document.status is not DocumentStatus.ACTIVE:
            raise IndexingError(f"document {document_id!r} is quarantined pending review")
        chunked, page_count = self._chunk(document, Area.DOCUMENTS)
        if screen and self._settings.ingestion.quarantine_suspicious:
            findings = screen_chunks(chunked.chunks)
            if findings:
                rules = sorted({finding.rule for finding in findings})
                self._quarantine(document, f"rules={','.join(rules)} hits={len(findings)}")
                raise IndexingError(
                    f"document {document_id!r} quarantined for review ({', '.join(rules)})"
                )
        vectors = self._embed(document, chunked.chunks)

        try:
            self._catalog.replace_content(document_id, list(chunked.sections), list(chunked.chunks))
            if page_count is not None:
                self._catalog.set_page_count(document_id, page_count)
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
        self._catalog.record_event(
            "indexed",
            document_id,
            content_hash=document.content_hash,
            source_type=document.source_type.value,
            detail=f"sections={len(chunked.sections)} chunks={len(chunked.chunks)} "
            f"vectors={len(vectors)}",
        )
        audit_logger.info(
            "event=index document_id=%s sha256=%s sections=%d chunks=%d vectors=%d",
            document_id,
            document.content_hash,
            len(chunked.sections),
            len(chunked.chunks),
            len(vectors),
        )
        return IndexResult(document_id, len(chunked.sections), len(chunked.chunks), len(vectors))

    def _chunk(self, document: Document, area: Area) -> tuple[ChunkedDocument, int | None]:
        try:
            path = self._workspace.stored_document(document.content_hash, area)
            if path is None:
                raise IndexingError(f"stored file of {document.document_id!r} is missing")
            if path.suffix == ".svd":
                chunked = chunk_svd(document.document_id, path, self._settings.ingestion)
                page_count = None
            else:
                extracted = self._extract(path)
                chunked = chunk_document(document.document_id, extracted, self._settings.ingestion)
                page_count = extracted.page_count
        except (WorkspaceError, ExtractionError, OcrError, SvdError) as exc:
            raise IndexingError(f"cannot extract {document.document_id!r}: {exc}") from exc
        if not chunked.chunks:
            raise IndexingError(
                f"no extractable text in {document.document_id!r} (scanned? enable OCR)"
            )
        return chunked, page_count

    def _extract(self, path: Path) -> ExtractedDocument:
        extracted = extract_pdf(path, self._settings.ingestion)
        if not self._settings.ocr.enabled or not needs_ocr(extracted, self._settings.ocr):
            return extracted
        logger.info("text layer is sparse, running OCR")
        fd, name = tempfile.mkstemp(dir=self._workspace.area(Area.TMP), suffix=".ocr.pdf")
        os.close(fd)
        scratch = Path(name)
        scratch.unlink()  # ocrmypdf refuses to overwrite an existing output
        try:
            ocr_pdf(path, scratch, self._settings.ocr)
            return extract_pdf(scratch, self._settings.ingestion)
        finally:
            scratch.unlink(missing_ok=True)

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
            for area in (Area.DOCUMENTS, Area.QUARANTINE):
                try:
                    stored = self._workspace.stored_document(document.content_hash, area)
                    if stored is not None:
                        stored.unlink(missing_ok=True)
                except (WorkspaceError, OSError) as exc:
                    logger.warning("cannot remove the file of %s: %s", document_id, exc)
        self._catalog.record_event(
            "deleted",
            document_id,
            content_hash=document.content_hash,
            source_type=document.source_type.value,
            source_url=document.source_url,
            detail=f"file_removed={not keep_file}",
        )
        audit_logger.info(
            "event=delete document_id=%s sha256=%s file_removed=%s",
            document_id,
            document.content_hash,
            not keep_file,
        )
        return True

    def delete_documents(
        self, document_ids: Sequence[str], *, keep_file: bool = False
    ) -> BatchReport:
        """Remove several documents, continuing after individual failures.

        Returns
        -------
        BatchReport
            ``failed`` maps each document that could not be removed (or is unknown) to a message.
        """
        report = BatchReport()
        for document_id in dict.fromkeys(document_ids):
            try:
                if not self.delete_document(document_id, keep_file=keep_file):
                    report.failed[document_id] = "unknown document"
            except IndexingError as exc:
                report.failed[document_id] = str(exc)
        return report

    def find_by_source_url(self, prefix: str) -> list[str]:
        """Return the ids of the documents whose provenance URL starts with ``prefix``."""
        return [
            d.document_id
            for d in self._all_documents()
            if d.source_url is not None and d.source_url.startswith(prefix)
        ]

    def inspect(self, document_id: str) -> tuple[str, ...]:
        """Screen a quarantined document without making it retrievable.

        Raises
        ------
        IndexingError
            If the document is not quarantined, missing or cannot be extracted.
        """
        document = self._catalog.get_document(document_id)
        if document is None or document.status is not DocumentStatus.QUARANTINED:
            raise IndexingError(f"document {document_id!r} is not quarantined")
        try:
            path = self._workspace.stored_document(document.content_hash, Area.QUARANTINE)
            if path is None:
                raise IndexingError(f"quarantined file of {document_id!r} is missing")
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
        except (WorkspaceError, OSError) as exc:
            raise IndexingError(f"cannot read quarantined file of {document_id!r}: {exc}") from exc
        if digest != document.content_hash:
            raise IndexingError(f"quarantined file of {document_id!r} has changed")
        chunked, _ = self._chunk(document, Area.QUARANTINE)
        return tuple(sorted({finding.rule for finding in screen_chunks(chunked.chunks)}))

    def release(self, document_id: str) -> IndexResult:
        """Release a quarantined document after human review, then index it.

        Screening is skipped for this indexing: the reviewer has decided.

        Raises
        ------
        IndexingError
            If the document is unknown or not quarantined, or indexing fails (in which case
            it stays active but unindexed; run ``index`` after fixing the cause).
        """
        document = self._catalog.get_document(document_id)
        if document is None or document.status is not DocumentStatus.QUARANTINED:
            raise IndexingError(f"document {document_id!r} is not quarantined")
        findings = self.inspect(document_id)
        try:
            held = self._workspace.stored_document(document.content_hash, Area.QUARANTINE)
            if held is None:
                raise IndexingError(f"quarantined file of {document_id!r} is missing")
            os.replace(
                held,
                self._workspace.document_path(document.content_hash, suffix=held.suffix),
            )
        except (WorkspaceError, OSError) as exc:
            raise IndexingError(f"cannot restore the file of {document_id!r}: {exc}") from exc
        self._catalog.set_status(document_id, DocumentStatus.ACTIVE)
        detail = f"screening_rules={','.join(findings) or 'none'}"
        self._catalog.record_event(
            "released", document_id, content_hash=document.content_hash, detail=detail
        )
        audit_logger.info("event=release document_id=%s %s", document_id, detail)
        return self.index_document(document_id, screen=False)

    def _quarantine(self, document: Document, reason: str) -> None:
        """Make a document unretrievable and move its file to the quarantine area."""
        document_id = document.document_id
        if self._vector_store is not None:
            try:
                self._vector_store.delete_document(document_id)
            except VectorStoreError as exc:
                raise IndexingError(f"cannot delete vectors of {document_id!r}: {exc}") from exc
        self._catalog.replace_content(document_id, [], [])
        self._catalog.set_status(document_id, DocumentStatus.QUARANTINED)
        try:
            live = self._workspace.stored_document(document.content_hash)
            if live is not None:
                os.replace(
                    live,
                    self._workspace.document_path(
                        document.content_hash, Area.QUARANTINE, live.suffix
                    ),
                )
        except (WorkspaceError, OSError) as exc:
            logger.warning("cannot move the file of %s to quarantine: %s", document_id, exc)
        self._catalog.record_event(
            "quarantined", document_id, content_hash=document.content_hash, detail=reason
        )
        audit_logger.warning("event=quarantine document_id=%s reason=%s", document_id, reason)
