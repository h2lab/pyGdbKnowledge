# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Administration endpoints: document import and indexing task status (admin token only)."""

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path as PathParam, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from ..config import Settings
from ..ingestion.importer import DocumentMetadata, ImportResult, IngestError, import_pdf
from ..ingestion.pipeline import Indexer, IndexingError
from ..ingestion.tasks import TaskCapacityError, TaskInfo, TaskRegistry
from ..security import audit_logger
from ..storage.catalog import Catalog, CatalogError
from ..storage.models import ID_PATTERN, MAX_KEYWORDS, DocumentStatus, Keyword, SourceType
from ..workspace import Area, Workspace
from .deps import get_catalog, get_settings

logger = logging.getLogger("silicore_knowledge")

router = APIRouter(prefix="/admin")

_CHUNK = 1024 * 1024
_Text = Annotated[str | None, Field(default=None, max_length=300)]


class ImportMetadata(BaseModel):
    """Operator-supplied provenance and classification of a document."""

    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    title: _Text = None
    manufacturer: _Text = None
    device_family: _Text = None
    device: _Text = None
    document_type: _Text = None
    revision: _Text = None
    source_url: Annotated[str | None, Field(default=None, max_length=2000)] = None
    keywords: tuple[Keyword, ...] = Field(default=(), max_length=MAX_KEYWORDS)

    def to_metadata(self) -> DocumentMetadata:
        """Convert to the importer's metadata."""
        data = self.model_dump()
        data["keywords"] = tuple(data["keywords"])
        return DocumentMetadata(**data)


class PathImport(ImportMetadata):
    """Import of a file located under one of the configured import directories."""

    path: Annotated[str, Field(min_length=1, max_length=4096)]


class IngestAccepted(BaseModel):
    """Answer of an import request."""

    document_id: str
    created: bool
    status: DocumentStatus
    task: TaskInfo | None


class ScreeningResult(BaseModel):
    """Heuristic findings on a quarantined document."""

    document_id: str
    rules: tuple[str, ...]


def _tasks(request: Request) -> TaskRegistry:
    tasks: TaskRegistry = request.app.state.tasks
    return tasks


def _workspace(request: Request) -> Workspace:
    workspace: Workspace = request.app.state.workspace
    return workspace


def _submit(request: Request, document_id: str, *, release: bool = False) -> TaskInfo:
    try:
        return _tasks(request).submit(document_id, release=release)
    except TaskCapacityError as exc:
        raise HTTPException(status_code=429, detail="indexing task capacity reached") from exc


def _finish(
    request: Request,
    channel: str,
    result: ImportResult,
) -> IngestAccepted:
    document = result.document
    audit_logger.info(
        "event=ingest channel=%s document_id=%s sha256=%s source_type=%s created=%s "
        "request_id=%s",
        channel,
        document.document_id,
        document.content_hash,
        document.source_type.value,
        result.created,
        getattr(request.state, "request_id", "-"),
    )
    task = (
        _submit(request, document.document_id) if document.status is DocumentStatus.ACTIVE else None
    )
    return IngestAccepted(
        document_id=document.document_id, created=result.created, status=document.status, task=task
    )


async def _import(
    workspace: Workspace,
    catalog: Catalog,
    source: Path,
    metadata: ImportMetadata,
    max_bytes: int,
    channel: str,
) -> ImportResult:
    try:
        return await run_in_threadpool(
            import_pdf, workspace, catalog, source, metadata.to_metadata(), max_bytes, channel
        )
    except IngestError as exc:
        logger.warning("import rejected: %s", exc)
        raise HTTPException(status_code=422, detail="file or metadata rejected") from exc
    except CatalogError as exc:
        logger.error("catalog error during import: %s", exc)
        raise HTTPException(status_code=500, detail="internal server error") from exc


@router.post("/ingest", status_code=202, response_model=IngestAccepted)
async def ingest_upload(
    request: Request,
    metadata: Annotated[ImportMetadata, Query()],
    settings: Annotated[Settings, Depends(get_settings)],
    catalog: Annotated[Catalog, Depends(get_catalog)],
) -> IngestAccepted:
    """Import a PDF sent as the raw request body into quarantine.

    Metadata travels in the query string; the body is the PDF itself (streamed to disk and
    bounded by ``limits.max_upload_bytes``).
    """
    limit = settings.limits.max_upload_bytes
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail="upload too large")
    workspace = _workspace(request)
    fd, name = tempfile.mkstemp(dir=workspace.area(Area.TMP), suffix=".upload")
    tmp = Path(name)
    try:
        total = 0
        with os.fdopen(fd, "wb") as out:
            async for block in request.stream():
                total += len(block)
                if total > limit:
                    raise HTTPException(status_code=413, detail="upload too large")
                out.write(block)
        if total == 0:
            raise HTTPException(status_code=422, detail="empty upload")
        result = await _import(workspace, catalog, tmp, metadata, limit, "upload")
    finally:
        tmp.unlink(missing_ok=True)
    return _finish(request, "upload", result)


@router.post("/ingest/path", status_code=202, response_model=IngestAccepted)
async def ingest_path(
    request: Request,
    body: PathImport,
    settings: Annotated[Settings, Depends(get_settings)],
    catalog: Annotated[Catalog, Depends(get_catalog)],
) -> IngestAccepted:
    """Import a server-side PDF located under an allowed import directory."""
    try:
        source = Path(body.path).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="file or metadata rejected") from exc
    allowed = [directory.resolve() for directory in settings.ingestion.import_dirs]
    if not any(source.is_relative_to(directory) for directory in allowed):
        audit_logger.warning(
            "event=ingest_refused reason=path_not_allowed path_sha=%s",
            hashlib.sha256(body.path.encode()).hexdigest()[:12],
        )
        raise HTTPException(status_code=403, detail="path not allowed")
    metadata = ImportMetadata.model_validate(body.model_dump(exclude={"path"}))
    result = await _import(
        _workspace(request), catalog, source, metadata, settings.limits.max_upload_bytes, "path"
    )
    return _finish(request, "path", result)


@router.post("/documents/{document_id}/screen", response_model=ScreeningResult)
async def screen_document(
    request: Request,
    document_id: Annotated[str, PathParam(pattern=ID_PATTERN)],
) -> ScreeningResult:
    """Screen a quarantined document for review without releasing it."""
    indexer: Indexer = request.app.state.indexer
    try:
        rules = await run_in_threadpool(indexer.inspect, document_id)
    except IndexingError as exc:
        logger.warning("screening failed: %s", exc)
        raise HTTPException(status_code=422, detail="document cannot be screened") from exc
    return ScreeningResult(document_id=document_id, rules=rules)


@router.post("/documents/{document_id}/release", status_code=202, response_model=TaskInfo)
def release_document(
    request: Request,
    document_id: Annotated[str, PathParam(pattern=ID_PATTERN)],
) -> TaskInfo:
    """Approve a reviewed quarantined document and queue indexing."""
    catalog: Catalog = request.app.state.catalog
    document = catalog.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    if document.status is not DocumentStatus.QUARANTINED:
        raise HTTPException(status_code=409, detail="document is not quarantined")
    task = _submit(request, document_id, release=True)
    audit_logger.info(
        "event=release_queued request_id=%s task_id=%s document_id=%s",
        getattr(request.state, "request_id", "-"),
        task.task_id,
        document_id,
    )
    return task


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(
    request: Request,
    document_id: Annotated[str, PathParam(pattern=ID_PATTERN)],
) -> None:
    """Remove a document with its file, passages and vectors."""
    indexer: Indexer = request.app.state.indexer
    try:
        removed = indexer.delete_document(document_id)
    except IndexingError as exc:
        logger.error("deletion failed: %s", exc)
        raise HTTPException(status_code=500, detail="deletion failed") from exc
    if not removed:
        raise HTTPException(status_code=404, detail="document not found")


@router.get("/tasks/{task_id}", response_model=TaskInfo)
def task_status(
    request: Request,
    task_id: Annotated[str, PathParam(pattern=r"^[0-9a-f]{32}$")],
) -> TaskInfo:
    """Return the status of an indexing task."""
    info = _tasks(request).get(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail="task not found")
    return info
