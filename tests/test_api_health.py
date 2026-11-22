# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the application factory and health endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pyGdbKnowledge import __version__
from pyGdbKnowledge.config import Settings
from pyGdbKnowledge.main import REQUEST_ID_HEADER, create_app
from pyGdbKnowledge.workspace import Workspace


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Test client on an app backed by a temporary workspace."""
    settings = Settings(workspace=tmp_path / "ws")
    return TestClient(create_app(settings, Workspace(settings.workspace)))


def test_health(client: TestClient) -> None:
    """The health endpoint reports status and version only."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_request_id_generated_and_echoed(client: TestClient) -> None:
    """Every response carries a correlation id; a valid supplied id is kept."""
    generated = client.get("/api/v1/health").headers[REQUEST_ID_HEADER]
    assert len(generated) == 32
    kept = client.get("/api/v1/health", headers={REQUEST_ID_HEADER: "trace-1.a_b"})
    assert kept.headers[REQUEST_ID_HEADER] == "trace-1.a_b"


def test_unsafe_request_id_replaced(client: TestClient) -> None:
    """Ids that could forge log lines are replaced."""
    response = client.get("/api/v1/health", headers={REQUEST_ID_HEADER: "x y"})
    assert response.headers[REQUEST_ID_HEADER] != "x y"


def test_interactive_docs_disabled(client: TestClient) -> None:
    """Swagger UI and ReDoc are not exposed; the OpenAPI document is versioned."""
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/api/v1/openapi.json").status_code == 200
