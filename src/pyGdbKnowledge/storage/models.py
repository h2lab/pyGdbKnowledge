# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Domain models stored in the catalog.

Metadata fields are optional: unknown values stay ``None`` and are never invented.
"""

from datetime import datetime
from enum import StrEnum, auto, unique

from pydantic import BaseModel, ConfigDict, Field

ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,127}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


@unique
class SourceType(StrEnum):
    """Source reliability class: official (A), vendor (B), community (C)."""

    OFFICIAL = auto()
    VENDOR = auto()
    COMMUNITY = auto()


@unique
class DocumentStatus(StrEnum):
    """Lifecycle status; quarantined documents are never retrievable (ANSSI R25+)."""

    ACTIVE = auto()
    QUARANTINED = auto()


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Document(_Model):
    """A source document and its provenance."""

    document_id: str = Field(pattern=ID_PATTERN)
    title: str = Field(min_length=1)
    content_hash: str = Field(pattern=SHA256_PATTERN)
    source_type: SourceType
    status: DocumentStatus = DocumentStatus.ACTIVE
    manufacturer: str | None = None
    device_family: str | None = None
    device: str | None = None
    document_type: str | None = None
    revision: str | None = None
    source_url: str | None = None
    page_count: int | None = Field(default=None, ge=0)
    version: int = Field(default=1, ge=1)
    indexed_at: datetime | None = None


class Section(_Model):
    """A section of a document, identified within it."""

    document_id: str = Field(pattern=ID_PATTERN)
    section_id: str = Field(pattern=ID_PATTERN)
    title: str = Field(min_length=1)
    path: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)


class Chunk(_Model):
    """A retrievable passage with its structural context."""

    chunk_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,160}$")
    document_id: str = Field(pattern=ID_PATTERN)
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    content_hash: str = Field(pattern=SHA256_PATTERN)
    section_id: str | None = Field(default=None, pattern=ID_PATTERN)
    section_path: str = ""
    peripheral: str | None = None
    page: int | None = Field(default=None, ge=1)


class CatalogStats(_Model):
    """Catalog counters."""

    schema_version: int
    documents: int
    sections: int
    chunks: int
