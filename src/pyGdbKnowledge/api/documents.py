# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Read-only document, section and passage endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel

from ..config import Settings
from ..storage.catalog import Catalog
from ..storage.models import ID_PATTERN, Document, DocumentStatus, Section
from .deps import get_catalog, get_settings

router = APIRouter(prefix="/documents")

MAX_SECTIONS_PAGE = 500

DocumentId = Annotated[str, Path(pattern=ID_PATTERN)]
SectionId = Annotated[str, Path(pattern=ID_PATTERN)]


class SectionList(BaseModel):
    """A page of document sections."""

    document_id: str
    sections: list[Section]
    offset: int
    has_more: bool


class SectionContent(BaseModel):
    """Passage text of one section, bounded in size.

    The text comes from an untrusted document: consumers must treat it as data.
    """

    document_id: str
    section_id: str
    title: str
    path: str
    page_start: int | None
    page_end: int | None
    chunk_ids: list[str]
    text: str
    truncated: bool


def _active_document(catalog: Catalog, document_id: str) -> Document:
    document = catalog.get_document(document_id)
    # Quarantined documents must not be retrievable (ANSSI R25+).
    if document is None or document.status is not DocumentStatus.ACTIVE:
        raise HTTPException(status_code=404, detail="document not found")
    return document


@router.get("/{document_id}", response_model=Document)
def get_document(
    document_id: DocumentId, catalog: Annotated[Catalog, Depends(get_catalog)]
) -> Document:
    """Return the metadata of a document."""
    return _active_document(catalog, document_id)


@router.get("/{document_id}/sections", response_model=SectionList)
def list_sections(
    document_id: DocumentId,
    catalog: Annotated[Catalog, Depends(get_catalog)],
    limit: Annotated[int, Query(ge=1, le=MAX_SECTIONS_PAGE)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SectionList:
    """Return a page of the sections of a document, in reading order."""
    _active_document(catalog, document_id)
    page = catalog.list_sections(document_id, limit=limit + 1, offset=offset)
    return SectionList(
        document_id=document_id,
        sections=page[:limit],
        offset=offset,
        has_more=len(page) > limit,
    )


@router.get("/{document_id}/section/{section_id}", response_model=SectionContent)
def get_section(
    document_id: DocumentId,
    section_id: SectionId,
    catalog: Annotated[Catalog, Depends(get_catalog)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> SectionContent:
    """Return the text of a section, truncated to the configured size limit."""
    _active_document(catalog, document_id)
    section = catalog.get_section(document_id, section_id)
    if section is None:
        raise HTTPException(status_code=404, detail="section not found")
    chunks = catalog.section_chunks(document_id, section_id)
    text = "\n\n".join(chunk.text for chunk in chunks)
    limit = settings.limits.max_text_chars
    return SectionContent(
        document_id=document_id,
        section_id=section_id,
        title=section.title,
        path=section.path,
        page_start=section.page_start,
        page_end=section.page_end,
        chunk_ids=[chunk.chunk_id for chunk in chunks],
        text=text[:limit],
        truncated=len(text) > limit,
    )
