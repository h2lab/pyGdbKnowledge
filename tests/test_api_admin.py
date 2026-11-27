# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the admin ingestion API."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pdfmaker import make_pdf
from pydantic import SecretStr

from silicore_knowledge.config import AuthSettings, IngestionSettings, LimitSettings, Settings
from silicore_knowledge.main import create_app
from silicore_knowledge.workspace import Workspace

ADMIN = {"Authorization": "Bearer admin-secret-value"}
READ = {"Authorization": "Bearer read-secret-value"}
QUERY = {"source_type": "official", "manufacturer": "ST", "title": "Sample"}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Client of an app with auth enabled and one allowed import directory."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    settings = Settings(
        workspace=tmp_path / "ws",
        auth=AuthSettings(
            read_token=SecretStr("read-secret-value"), admin_token=SecretStr("admin-secret-value")
        ),
        ingestion=IngestionSettings(import_dirs=[inbox]),
        limits=LimitSettings(max_upload_bytes=200_000),
    )
    return TestClient(create_app(settings, Workspace(settings.workspace)))


def _pdf_bytes(tmp_path: Path, text: str = "USART_ISR flags") -> bytes:
    return make_pdf(tmp_path / "up.pdf", [text], [[1, "1 USART", 1]]).read_bytes()


def _wait(client: TestClient, task_id: str) -> dict[str, object]:
    for _ in range(100):
        body: dict[str, object] = client.get(f"/api/v1/admin/tasks/{task_id}", headers=ADMIN).json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_upload_indexes_and_makes_document_searchable(client: TestClient, tmp_path: Path) -> None:
    """An uploaded PDF is imported, indexed in the background and then searchable."""
    content = _pdf_bytes(tmp_path)
    response = client.post("/api/v1/admin/ingest", params=QUERY, content=content, headers=ADMIN)
    assert response.status_code == 202
    body = response.json()
    assert body["created"] is True
    done = _wait(client, body["task"]["task_id"])
    assert done["status"] == "done" and done["chunks"]
    found = client.post("/api/v1/search", json={"query": "USART_ISR"}, headers=READ)
    assert [h["document_id"] for h in found.json()["results"]] == [body["document_id"]]

    again = client.post("/api/v1/admin/ingest", params=QUERY, content=content, headers=ADMIN)
    assert again.json()["created"] is False


def test_admin_token_is_required(client: TestClient, tmp_path: Path) -> None:
    """Read tokens and anonymous callers cannot use the admin API."""
    for headers in (READ, {}):
        assert client.post("/api/v1/admin/ingest", params=QUERY, headers=headers).status_code == 401
        assert client.get("/api/v1/admin/tasks/" + "0" * 32, headers=headers).status_code == 401


def test_upload_validation(client: TestClient, tmp_path: Path) -> None:
    """Bad content, oversize bodies and invalid metadata are refused."""
    url = "/api/v1/admin/ingest"
    assert client.post(url, params=QUERY, content=b"not a pdf", headers=ADMIN).status_code == 422
    assert client.post(url, params=QUERY, content=b"", headers=ADMIN).status_code == 422
    big = b"%PDF-1.7\n" + b"x" * 300_000
    assert client.post(url, params=QUERY, content=big, headers=ADMIN).status_code == 413
    assert client.post(url, params={}, content=b"%PDF-", headers=ADMIN).status_code == 422
    extra = {**QUERY, "unknown": "x"}
    assert client.post(url, params=extra, content=b"%PDF-", headers=ADMIN).status_code == 422
    assert not list((tmp_path / "ws" / "tmp").iterdir())


def test_path_import_is_restricted_to_allowed_directories(
    client: TestClient, tmp_path: Path
) -> None:
    """Only files under import_dirs can be imported by path; symlinks cannot escape."""
    url = "/api/v1/admin/ingest/path"
    allowed = make_pdf(tmp_path / "inbox" / "ds.pdf", ["SPI_CR1 register"])
    outside = make_pdf(tmp_path / "outside.pdf", ["secret"])
    (tmp_path / "inbox" / "link.pdf").symlink_to(outside)

    ok = client.post(url, json={**QUERY, "path": str(allowed)}, headers=ADMIN)
    assert ok.status_code == 202
    assert _wait(client, ok.json()["task"]["task_id"])["status"] == "done"
    for path in (outside, tmp_path / "inbox" / "link.pdf", Path("/etc/passwd")):
        refused = client.post(url, json={**QUERY, "path": str(path)}, headers=ADMIN)
        assert refused.status_code == 403
    missing = client.post(
        url, json={**QUERY, "path": str(tmp_path / "inbox/no.pdf")}, headers=ADMIN
    )
    assert missing.status_code == 422


def test_failed_indexing_is_reported(client: TestClient, tmp_path: Path) -> None:
    """A PDF that cannot be parsed ends as a failed task, not a server error."""
    fake = b"%PDF-1.7\nthis is not a real document\n"
    response = client.post("/api/v1/admin/ingest", params=QUERY, content=fake, headers=ADMIN)
    assert response.status_code == 202
    done = _wait(client, response.json()["task"]["task_id"])
    assert done["status"] == "failed" and done["error"]


def test_unknown_task(client: TestClient) -> None:
    """Unknown task ids answer 404."""
    assert client.get("/api/v1/admin/tasks/" + "0" * 32, headers=ADMIN).status_code == 404


def test_delete_endpoint_removes_document(client: TestClient, tmp_path: Path) -> None:
    """DELETE removes the document so that it is no longer searchable or readable."""
    created = client.post(
        "/api/v1/admin/ingest", params=QUERY, content=_pdf_bytes(tmp_path), headers=ADMIN
    ).json()
    _wait(client, created["task"]["task_id"])
    url = f"/api/v1/admin/documents/{created['document_id']}"
    assert client.delete(url, headers=READ).status_code == 401
    assert client.delete(url, headers=ADMIN).status_code == 204
    assert client.delete(url, headers=ADMIN).status_code == 404
    found = client.post("/api/v1/search", json={"query": "USART_ISR"}, headers=READ)
    assert found.json()["results"] == []
    assert (
        client.get(f"/api/v1/documents/{created['document_id']}", headers=READ).status_code == 404
    )
