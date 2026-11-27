# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the read-only document endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from silicore_knowledge.config import LimitSettings, Settings
from silicore_knowledge.main import create_app
from silicore_knowledge.storage.catalog import Catalog
from silicore_knowledge.storage.models import Chunk, Document, DocumentStatus, Section, SourceType
from silicore_knowledge.workspace import Workspace

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _populate(catalog: Catalog) -> None:
    catalog.add_document(
        Document(
            document_id="ds-1",
            title="STM32U5 Datasheet",
            content_hash=HASH_A,
            source_type=SourceType.OFFICIAL,
            device="STM32U595",
            revision="Rev 3",
        )
    )
    sections = [
        Section(document_id="ds-1", section_id=f"s{i}", title=f"S{i}", path=f"A > S{i}", ordinal=i)
        for i in range(3)
    ]
    chunks = [
        Chunk(
            chunk_id=f"ds-1:{i}",
            document_id="ds-1",
            section_id="s0",
            ordinal=i,
            text=text,
            content_hash=HASH_B,
            page=5,
        )
        for i, text in enumerate(["first passage", "second passage"])
    ]
    catalog.replace_content("ds-1", sections, chunks)
    catalog.add_document(
        Document(
            document_id="evil",
            title="Quarantined",
            content_hash=HASH_C,
            source_type=SourceType.COMMUNITY,
            status=DocumentStatus.QUARANTINED,
        )
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Test client with a small populated catalog and a 100-character text limit."""
    settings = Settings(workspace=tmp_path / "ws", limits=LimitSettings(max_text_chars=100))
    app = create_app(settings, Workspace(settings.workspace))
    _populate(app.state.catalog)
    return TestClient(app)


def test_stats(client: TestClient) -> None:
    """Stats report the schema version and counters."""
    body = client.get("/api/v1/stats").json()
    assert (body["documents"], body["sections"], body["chunks"]) == (2, 3, 2)


def test_get_document(client: TestClient) -> None:
    """Document metadata is returned, unknown values as null."""
    body = client.get("/api/v1/documents/ds-1").json()
    assert body["title"] == "STM32U5 Datasheet"
    assert body["device"] == "STM32U595" and body["manufacturer"] is None


def test_unknown_and_quarantined_documents_are_not_found(client: TestClient) -> None:
    """Quarantined documents are indistinguishable from missing ones."""
    for document_id in ("missing", "evil"):
        for suffix in ("", "/sections", "/section/s0"):
            assert client.get(f"/api/v1/documents/{document_id}{suffix}").status_code == 404


def test_invalid_identifiers_rejected(client: TestClient) -> None:
    """Identifiers must match the allowed pattern."""
    assert client.get("/api/v1/documents/UPPER").status_code == 422
    assert client.get("/api/v1/documents/ds-1/section/BAD").status_code == 422


def test_sections_pagination(client: TestClient) -> None:
    """Sections are paginated and report whether more exist."""
    first = client.get("/api/v1/documents/ds-1/sections", params={"limit": 2}).json()
    assert [s["section_id"] for s in first["sections"]] == ["s0", "s1"]
    assert first["has_more"] is True
    last = client.get("/api/v1/documents/ds-1/sections", params={"limit": 2, "offset": 2}).json()
    assert [s["section_id"] for s in last["sections"]] == ["s2"]
    assert last["has_more"] is False
    assert client.get("/api/v1/documents/ds-1/sections", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/documents/ds-1/sections", params={"limit": 501}).status_code == 422


def test_section_content(client: TestClient) -> None:
    """A section returns its joined passages and chunk ids."""
    body = client.get("/api/v1/documents/ds-1/section/s0").json()
    assert body["text"] == "first passage\n\nsecond passage"
    assert body["chunk_ids"] == ["ds-1:0", "ds-1:1"]
    assert body["truncated"] is False
    assert client.get("/api/v1/documents/ds-1/section/nope").status_code == 404


def test_section_text_is_truncated(tmp_path: Path) -> None:
    """Section text never exceeds the configured limit."""
    settings = Settings(workspace=tmp_path / "ws", limits=LimitSettings(max_text_chars=100))
    app = create_app(settings, Workspace(settings.workspace))
    catalog: Catalog = app.state.catalog
    catalog.add_document(
        Document(document_id="d", title="T", content_hash=HASH_A, source_type=SourceType.OFFICIAL)
    )
    section = Section(document_id="d", section_id="s", title="S", path="S", ordinal=0)
    chunk = Chunk(
        chunk_id="d:0",
        document_id="d",
        section_id="s",
        ordinal=0,
        text="x" * 250,
        content_hash=HASH_B,
    )
    catalog.replace_content("d", [section], [chunk])
    body = TestClient(app).get("/api/v1/documents/d/section/s").json()
    assert len(body["text"]) == 100 and body["truncated"] is True
