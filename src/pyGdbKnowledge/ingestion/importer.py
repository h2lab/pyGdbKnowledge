# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Import of PDF files into the workspace (explicit, local, deduplicated by SHA-256)."""

import hashlib
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import ValidationError

from ..storage.catalog import Catalog, DuplicateDocumentError
from ..storage.models import Document, DocumentStatus, SourceType
from ..workspace import Area, Workspace

PDF_MAGIC = b"%PDF-"
_MAGIC_WINDOW = 1024
_CHUNK = 1024 * 1024
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class IngestError(Exception):
    """Raised when a file cannot be imported."""


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Provenance and classification supplied by the operator; never guessed.

    ``source_url`` records where the file came from. It is provenance only and is never fetched.
    """

    source_type: SourceType
    title: str | None = None
    manufacturer: str | None = None
    device_family: str | None = None
    device: str | None = None
    document_type: str | None = None
    revision: str | None = None
    source_url: str | None = None


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Outcome of an import: the catalog document and whether it was newly added."""

    document: Document
    created: bool


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:80] or "document"


def _check_source_url(url: str | None) -> None:
    if url is None:
        return
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise IngestError("source_url must be an http(s) URL")


def import_pdf(
    workspace: Workspace,
    catalog: Catalog,
    source: Path,
    metadata: DocumentMetadata,
    max_bytes: int,
) -> ImportResult:
    """Copy a PDF into the workspace and register it in the catalog.

    The content is streamed to ``tmp/`` under a size cap while its SHA-256 is computed, checked
    for the PDF signature, then moved to ``documents/<sha256>.pdf`` (mode 0600). A file whose
    hash is already known is not stored again. Text extraction is a separate step.

    Parameters
    ----------
    workspace : Workspace
        Destination workspace.
    catalog : Catalog
        Catalog to register the document in.
    source : Path
        Regular file to import.
    metadata : DocumentMetadata
        Operator-supplied provenance and classification.
    max_bytes : int
        Maximum accepted file size.

    Returns
    -------
    ImportResult
        The document and whether it was created or already present.

    Raises
    ------
    IngestError
        If the source is not a regular PDF within the size limit, or the metadata is invalid.
    """
    _check_source_url(metadata.source_url)
    try:
        info = source.stat()
    except OSError as exc:
        raise IngestError(f"cannot read {source}: {exc.strerror}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise IngestError(f"not a regular file: {source}")
    if info.st_size > max_bytes:
        raise IngestError(f"file exceeds the {max_bytes} byte limit: {source}")

    digest = hashlib.sha256()
    total = 0
    head = b""
    fd, tmp_name = tempfile.mkstemp(dir=workspace.area(Area.TMP), suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as handle:
            while block := handle.read(_CHUNK):
                total += len(block)
                if total > max_bytes:
                    raise IngestError(f"file exceeds the {max_bytes} byte limit: {source}")
                if len(head) < _MAGIC_WINDOW:
                    head = (head + block)[:_MAGIC_WINDOW]
                digest.update(block)
                out.write(block)
        if PDF_MAGIC not in head:
            raise IngestError(f"not a PDF file: {source}")
        sha256 = digest.hexdigest()
        existing = catalog.get_document_by_hash(sha256)
        if existing is not None:
            return ImportResult(existing, created=False)

        title = metadata.title or source.stem
        try:
            document = Document(
                document_id=f"{_slug(title)}-{sha256[:8]}",
                title=title,
                content_hash=sha256,
                source_type=metadata.source_type,
                status=DocumentStatus.ACTIVE,
                manufacturer=metadata.manufacturer,
                device_family=metadata.device_family,
                device=metadata.device,
                document_type=metadata.document_type,
                revision=metadata.revision,
                source_url=metadata.source_url,
            )
        except ValidationError as exc:
            raise IngestError(f"invalid metadata: {exc.errors()[0]['loc']}") from exc
        target = workspace.document_path(sha256)
        tmp_path.chmod(0o600)
        os.replace(tmp_path, target)
        try:
            stored = catalog.add_document(document)
        except DuplicateDocumentError:
            # Lost a race with a concurrent import of the same content: keep their entry.
            winner = catalog.get_document_by_hash(sha256)
            if winner is None:
                target.unlink(missing_ok=True)
                raise
            return ImportResult(winner, created=False)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return ImportResult(stored, created=True)
    finally:
        tmp_path.unlink(missing_ok=True)
