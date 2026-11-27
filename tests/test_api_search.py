# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the search endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from silicore_knowledge.config import LimitSettings, Settings
from silicore_knowledge.main import create_app
from silicore_knowledge.storage.models import Chunk, Document, Section, SourceType
from silicore_knowledge.workspace import Workspace

LONG_TEXT = "usart_isr " + "x" * 300


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Client over a one-document catalog with small limits."""
    settings = Settings(
        workspace=tmp_path / "ws",
        limits=LimitSettings(max_results=2, default_results=1, max_text_chars=100),
    )
    app = create_app(settings, Workspace(settings.workspace))
    catalog = app.state.catalog
    catalog.add_document(
        Document(
            document_id="ds-1",
            title="Datasheet",
            content_hash="a" * 64,
            source_type=SourceType.OFFICIAL,
            manufacturer="ST",
            revision="Rev 3",
            source_url="https://example.org/ds.pdf",
        )
    )
    section = Section(document_id="ds-1", section_id="s", title="S", path="P > S", ordinal=0)
    chunks = [
        Chunk(
            chunk_id=f"ds-1:{i}",
            document_id="ds-1",
            section_id="s",
            ordinal=i,
            text=text,
            content_hash="b" * 64,
            section_path="P > S",
            page=4,
        )
        for i, text in enumerate([LONG_TEXT, "usart_isr again", "usart_isr third"])
    ]
    catalog.replace_content("ds-1", [section], chunks)
    return TestClient(app)


def test_search_returns_documented_fields(client: TestClient) -> None:
    """A result carries provenance, score and bounded text."""
    response = client.post("/api/v1/search", json={"query": "usart_isr"})
    assert response.status_code == 200
    (result,) = response.json()["results"]
    assert result["document_id"] == "ds-1"
    assert result["section"] == "P > S"
    assert result["page"] == 4
    assert result["source_type"] == "official"
    assert result["revision"] == "Rev 3"
    assert result["url"] == "https://example.org/ds.pdf"
    assert 0.0 < result["score"] <= 1.0
    assert len(result["text"]) == 100 and result["truncated"] is True
    assert "embedding" not in result and "vector" not in result


def test_limit_is_capped_by_settings(client: TestClient) -> None:
    """Requested limits above max_results are clamped."""
    response = client.post("/api/v1/search", json={"query": "usart_isr", "limit": 50})
    assert len(response.json()["results"]) == 2


def test_filters_are_applied(client: TestClient) -> None:
    """A filter matching nothing yields an empty list."""
    response = client.post(
        "/api/v1/search", json={"query": "usart_isr", "filters": {"manufacturer": "NXP"}}
    )
    assert response.json() == {"results": []}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"query": "x", "limit": 0},
        {"query": "x", "filters": {"status": "quarantined"}},
        {"query": "x", "collection": "other"},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, body: dict[str, object]) -> None:
    """Missing fields, bad bounds and unknown filter/parameter keys give 422."""
    assert client.post("/api/v1/search", json=body).status_code == 422


def test_query_without_terms_returns_nothing(client: TestClient) -> None:
    """A punctuation-only query is safe and empty."""
    assert client.post("/api/v1/search", json={"query": '"* NEAR('}).json() == {"results": []}
