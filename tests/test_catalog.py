# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SQLite catalog."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError

from silicore_knowledge.storage.catalog import (
    _MIGRATIONS,  # noqa: PLC2701
    SCHEMA_VERSION,
    Catalog,
    CatalogError,
    DuplicateDocumentError,
)
from silicore_knowledge.storage.models import (
    Chunk,
    Document,
    SearchFilters,
    Section,
    SourceType,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def _document(document_id: str = "ds-1", content_hash: str = HASH_A) -> Document:
    return Document(
        document_id=document_id,
        title="STM32U5 Datasheet",
        content_hash=content_hash,
        source_type=SourceType.OFFICIAL,
    )


def _content(document_id: str = "ds-1") -> tuple[list[Section], list[Chunk]]:
    section = Section(
        document_id=document_id,
        section_id="usart",
        title="USART",
        path="Peripherals > USART",
        ordinal=0,
        page_start=10,
        page_end=12,
    )
    chunks = [
        Chunk(
            chunk_id=f"{document_id}:{i}",
            document_id=document_id,
            section_id="usart",
            ordinal=i,
            text=text,
            content_hash=HASH_B,
            section_path=section.path,
            peripheral="USART",
            page=10,
        )
        for i, text in enumerate(["USART_ISR holds status flags", "TXE flag is set when empty"])
    ]
    return [section], chunks


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    """Fresh catalog in a temporary directory."""
    return Catalog(tmp_path / "catalog.db")


def _fts_count(catalog: Catalog, query: str) -> int:
    with closing(sqlite3.connect(catalog.path)) as db:
        return int(
            db.execute(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (query,)
            ).fetchone()[0]
        )


def test_new_catalog_is_migrated(catalog: Catalog) -> None:
    """A new database gets the current schema and zero counters."""
    stats = catalog.stats()
    assert (stats.schema_version, stats.documents, stats.sections, stats.chunks) == (
        SCHEMA_VERSION,
        0,
        0,
        0,
    )
    Catalog(catalog.path)  # reopening is idempotent


def test_newer_schema_refused(tmp_path: Path) -> None:
    """A database from a newer version is not opened."""
    path = tmp_path / "catalog.db"
    with closing(sqlite3.connect(path)) as db:
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(CatalogError, match="newer"):
        Catalog(path)


def test_document_roundtrip_keeps_unknown_metadata_null(catalog: Catalog) -> None:
    """Optional metadata stays None; indexed_at is set on insert."""
    stored = catalog.add_document(_document())
    loaded = catalog.get_document("ds-1")
    assert loaded == stored
    assert loaded is not None and loaded.indexed_at is not None
    assert loaded.revision is None and loaded.device is None
    assert catalog.get_document("missing") is None


def test_duplicates_rejected(catalog: Catalog) -> None:
    """The same id or the same content cannot be cataloged twice."""
    catalog.add_document(_document())
    with pytest.raises(DuplicateDocumentError):
        catalog.add_document(_document(content_hash=HASH_B))
    with pytest.raises(DuplicateDocumentError):
        catalog.add_document(_document(document_id="ds-2"))


def test_list_documents_is_paginated(catalog: Catalog) -> None:
    """Listing honours limit and offset."""
    for i in range(3):
        catalog.add_document(_document(f"ds-{i}", f"{i}" * 64))
    assert [d.document_id for d in catalog.list_documents(limit=2)] == ["ds-0", "ds-1"]
    assert [d.document_id for d in catalog.list_documents(limit=2, offset=2)] == ["ds-2"]


def test_content_and_exact_search_index(catalog: Catalog) -> None:
    """Chunks are indexed in FTS5 and register names stay whole tokens."""
    catalog.add_document(_document())
    sections, chunks = _content()
    catalog.replace_content("ds-1", sections, chunks)
    assert catalog.list_sections("ds-1") == sections
    assert catalog.get_section("ds-1", "usart") == sections[0]
    assert catalog.get_section("ds-1", "nope") is None
    assert catalog.section_chunks("ds-1", "usart") == chunks
    assert _fts_count(catalog, "USART_ISR") == 1
    assert _fts_count(catalog, "USART") == 2  # peripheral column matches both chunks
    assert catalog.stats().chunks == 2


def test_replace_content_is_atomic_and_updates_index(catalog: Catalog) -> None:
    """Replacing content drops old FTS entries; a failure changes nothing."""
    catalog.add_document(_document())
    sections, chunks = _content()
    catalog.replace_content("ds-1", sections, chunks)
    bad = chunks[0].model_copy(update={"section_id": "unknown"})
    with pytest.raises(CatalogError):
        catalog.replace_content("ds-1", sections, [bad])
    assert catalog.stats().chunks == 2
    assert _fts_count(catalog, "TXE") == 1
    catalog.replace_content("ds-1", sections, chunks[:1])
    assert _fts_count(catalog, "TXE") == 0


def test_replace_content_rejects_foreign_or_unknown_document(catalog: Catalog) -> None:
    """Content must belong to an existing document."""
    sections, chunks = _content("other")
    with pytest.raises(CatalogError):
        catalog.replace_content("ds-1", sections, chunks)
    with pytest.raises(CatalogError, match="unknown document"):
        catalog.replace_content("ds-1", [], [])


def test_delete_cascades_to_fts(catalog: Catalog) -> None:
    """Deleting a document removes sections, chunks and their FTS entries."""
    catalog.add_document(_document())
    catalog.replace_content("ds-1", *_content())
    assert catalog.delete_document("ds-1") is True
    assert catalog.delete_document("ds-1") is False
    stats = catalog.stats()
    assert (stats.documents, stats.sections, stats.chunks) == (0, 0, 0)
    assert _fts_count(catalog, "USART_ISR") == 0


def test_models_validate_identifiers() -> None:
    """Malformed ids and hashes are rejected at the boundary."""
    with pytest.raises(ValidationError):
        _document(document_id="../etc")
    with pytest.raises(ValidationError):
        _document(content_hash="xyz")


def test_document_keywords_roundtrip_and_reach_the_fts_index(catalog: Catalog) -> None:
    """Keywords are stored with the document and match its chunks in FTS5."""
    document = _document().model_copy(update={"keywords": ("readout protection", "rdp")})
    catalog.add_document(document)
    loaded = catalog.get_document("ds-1")
    assert loaded is not None and loaded.keywords == ("readout protection", "rdp")
    catalog.replace_content("ds-1", *_content())
    assert _fts_count(catalog, '"readout protection"') == 2  # every chunk of the document
    hits = catalog.search_chunks('"rdp"', SearchFilters(), 10)
    assert {hit.chunk.document_id for hit in hits} == {"ds-1"}
    assert hits[0].document.keywords == ("readout protection", "rdp")
    # Chunks themselves stay free of the denormalized column.
    assert catalog.section_chunks("ds-1", "usart") == list(_content()[1])


def test_catalog_without_keywords_is_migrated_in_place(tmp_path: Path) -> None:
    """A pre-keywords database is upgraded and its content stays searchable."""
    path = tmp_path / "catalog.db"
    with closing(sqlite3.connect(path)) as db:
        for script in _MIGRATIONS[:2]:
            db.executescript(script)
        db.execute(
            "INSERT INTO documents (document_id, title, content_hash, source_type, status,"
            " version, indexed_at) VALUES ('d1', 'T', ?, 'official', 'active', 1,"
            " '2026-01-01T00:00:00+00:00')",
            (HASH_A,),
        )
        db.execute(
            "INSERT INTO sections (document_id, section_id, title, path, ordinal)"
            " VALUES ('d1', 's1', 'S', 'S', 0)"
        )
        db.execute(
            "INSERT INTO chunks (chunk_id, document_id, section_id, ordinal, text,"
            " content_hash, section_path) VALUES ('d1:0', 'd1', 's1', 0, 'USART_ISR flags', ?,"
            " 'S')",
            (HASH_B,),
        )
        db.execute("PRAGMA user_version = 2")
        db.commit()
    catalog = Catalog(path)
    assert catalog.stats().schema_version == SCHEMA_VERSION
    document = catalog.get_document("d1")
    assert document is not None and document.keywords == ()
    hits = catalog.search_chunks('"usart_isr"', SearchFilters(), 10)
    assert [hit.chunk.chunk_id for hit in hits] == ["d1:0"]
