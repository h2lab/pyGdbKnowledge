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
from ..ingestion.tasks import TaskInfo, TaskRegistry
from ..security import audit_logger
from ..storage.catalog import Catalog, CatalogError
from ..storage.models import ID_PATTERN, SourceType
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

    def to_metadata(self) -> DocumentMetadata:
        """Convert to the importer's metadata."""
        return DocumentMetadata(**self.model_dump())


class PathImport(ImportMetadata):
    """Import of a file located under one of the configured import directories."""

    path: Annotated[str, Field(min_length=1, max_length=4096)]


class IngestAccepted(BaseModel):
    """Answer of an import request."""

    document_id: str
    created: bool
    task: TaskInfo


def _tasks(request: Request) -> TaskRegistry:
    tasks: TaskRegistry = request.app.state.tasks
    return tasks


def _workspace(request: Request) -> Workspace:
    workspace: Workspace = request.app.state.workspace
    return workspace


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
    task = _tasks(request).submit(document.document_id)
    return IngestAccepted(document_id=document.document_id, created=result.created, task=task)


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
    """Import a PDF sent as the raw request body, then queue its indexing.

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
