# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for authentication, throttling, timeouts, error shape and audit logging."""

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.types import Message, Receive, Scope, Send

from silicore_knowledge.api.auth import Principal, require_admin
from silicore_knowledge.config import (
    AuthSettings,
    ConfigError,
    LimitSettings,
    ReadPrincipal,
    Settings,
)
from silicore_knowledge.main import create_app
from silicore_knowledge.security import RateLimiter, RequestBodyLimit
from silicore_knowledge.storage.models import Chunk, Document, Section, SourceType
from silicore_knowledge.workspace import Workspace

READ = "read-secret-value"
ADMIN = "admin-secret-value"


def _app(tmp_path: Path, **overrides: object) -> FastAPI:
    overrides.setdefault("limits", LimitSettings(min_score=0.01))
    settings = Settings(
        workspace=tmp_path / "ws",
        auth=AuthSettings(
            principals={"reader": ReadPrincipal(token=SecretStr(READ), allowed_documents=("*",))},
            admin_token=SecretStr(ADMIN),
        ),
        **overrides,  # type: ignore[arg-type]
    )
    app = create_app(settings, Workspace(settings.workspace))

    @app.get("/admin-only")
    def admin_only(principal: Annotated[Principal, Depends(require_admin)]) -> dict[str, str]:
        return {"role": principal.role.value}

    @app.get("/slow")
    async def slow() -> None:
        await asyncio.sleep(2)

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret detail")

    return app


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Client over an app with both tokens configured."""
    return TestClient(_app(tmp_path), raise_server_exceptions=False)


def test_health_is_open(client: TestClient) -> None:
    """The health probe needs no credentials."""
    assert client.get("/api/v1/health").status_code == 200


@pytest.mark.parametrize("path", ["/api/v1/stats", "/api/v1/documents/x"])
def test_read_endpoints_need_a_token(client: TestClient, path: str) -> None:
    """Missing or wrong tokens get one uniform 401."""
    for headers in ({}, _auth("wrong")):
        response = client.get(path, headers=headers)
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert response.json()["detail"] == "authentication required"
    assert client.get(path, headers=_auth(READ)).status_code in (200, 404)


def test_search_requires_token(client: TestClient) -> None:
    """The search endpoint is protected and works with the read token."""
    assert client.post("/api/v1/search", json={"query": "x"}).status_code == 401
    ok = client.post("/api/v1/search", json={"query": "x"}, headers=_auth(READ))
    assert ok.status_code == 200


def test_admin_scope_is_separate(client: TestClient) -> None:
    """The read token never grants admin; the admin token also reads."""
    assert client.get("/admin-only", headers=_auth(READ)).status_code == 401
    assert client.get("/admin-only").status_code == 401
    assert client.get("/admin-only", headers=_auth(ADMIN)).json() == {"role": "admin"}
    assert client.get("/api/v1/stats", headers=_auth(ADMIN)).status_code == 200


def test_unconfigured_loopback_is_not_open(tmp_path: Path) -> None:
    """Unconfigured deployments do not grant anonymous access."""
    settings = Settings(workspace=tmp_path / "ws", limits=LimitSettings(min_score=0.01))
    app = create_app(settings, Workspace(settings.workspace))
    app.add_api_route("/admin-only", lambda: None, dependencies=[Depends(require_admin)])
    client = TestClient(app)
    assert client.get("/api/v1/stats").status_code == 401
    assert client.get("/admin-only").status_code == 401


def test_principal_grants_apply_to_search_documents_and_stats(tmp_path: Path) -> None:
    """Document grants apply before search limiting and to every document endpoint."""
    settings = Settings(
        workspace=tmp_path / "ws",
        limits=LimitSettings(min_score=0.01),
        auth=AuthSettings(
            principals={
                "alice": ReadPrincipal(token=SecretStr("alice-token"), allowed_documents=("a",)),
                "bob": ReadPrincipal(token=SecretStr("bob-token"), allowed_documents=("b",)),
                "empty": ReadPrincipal(token=SecretStr("empty-token")),
            },
            admin_token=SecretStr(ADMIN),
        ),
    )
    app = create_app(settings, Workspace(settings.workspace))
    for doc_id in ("a", "b"):
        app.state.catalog.add_document(
            Document(
                document_id=doc_id,
                title=doc_id,
                content_hash=doc_id * 64,
                source_type=SourceType.OFFICIAL,
            )
        )
        section = Section(document_id=doc_id, section_id="s", title="S", path="S", ordinal=0)
        chunks = [
            Chunk(
                chunk_id=f"{doc_id}:{i}",
                document_id=doc_id,
                section_id="s",
                ordinal=i,
                text="shared register",
                content_hash="c" * 64,
            )
            for i in range(25 if doc_id == "b" else 1)
        ]
        app.state.catalog.replace_content(doc_id, [section], chunks)
    client = TestClient(app)
    alice = _auth("alice-token")
    response = client.post("/api/v1/search", json={"query": "shared", "limit": 1}, headers=alice)
    assert [hit["document_id"] for hit in response.json()["results"]] == ["a"]
    for suffix in ("", "/sections", "/section/s"):
        assert client.get(f"/api/v1/documents/b{suffix}", headers=alice).status_code == 404
        assert client.get(f"/api/v1/documents/a{suffix}", headers=alice).status_code == 200
    assert client.get("/api/v1/stats", headers=alice).json()["documents"] == 1
    assert client.get("/api/v1/stats", headers=alice).json()["chunks"] == 1
    assert client.get("/api/v1/stats", headers=_auth("empty-token")).json()["documents"] == 0
    assert (
        client.post(
            "/api/v1/search", json={"query": "shared"}, headers=_auth("empty-token")
        ).json()["results"]
        == []
    )
    assert client.get("/api/v1/documents/b", headers=_auth(ADMIN)).status_code == 200
    assert (
        client.post(
            "/api/v1/search",
            json={"query": "shared", "filters": {"allowed_document_ids": ["b"]}},
            headers=alice,
        ).status_code
        == 422
    )


def test_rate_limit(tmp_path: Path) -> None:
    """Requests beyond the per-minute budget get 429 with Retry-After; health is exempt."""
    client = TestClient(
        _app(tmp_path, limits=LimitSettings(rate_limit_per_minute=2, min_score=0.01))
    )
    for _ in range(2):
        assert client.get("/api/v1/stats", headers=_auth(READ)).status_code == 200
    blocked = client.get("/api/v1/stats", headers=_auth(READ))
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    assert "X-Request-ID" in blocked.headers
    assert client.get("/api/v1/health").status_code == 200


def test_timeout(tmp_path: Path) -> None:
    """A request exceeding the timeout is answered with 504."""
    client = TestClient(_app(tmp_path, limits=LimitSettings(request_timeout_s=0.1, min_score=0.01)))
    response = client.get("/slow")
    assert response.status_code == 504
    assert response.json()["detail"] == "request timed out"


def test_errors_are_uniform_and_do_not_echo_input(client: TestClient) -> None:
    """Validation and internal errors share a shape and leak nothing."""
    bad = client.post(
        "/api/v1/search", json={"query": "x", "limit": "SENTINEL"}, headers=_auth(READ)
    )
    assert bad.status_code == 422
    assert "SENTINEL" not in bad.text
    assert bad.json()["request_id"] == bad.headers["X-Request-ID"]
    crash = client.get("/boom")
    assert crash.status_code == 500
    assert crash.json()["detail"] == "internal error"
    assert "secret detail" not in crash.text
    oversized = client.post("/api/v1/search", json={"query": "x" * 501}, headers=_auth(READ))
    assert oversized.status_code == 422


def test_large_json_body_is_rejected_before_parsing(client: TestClient) -> None:
    """A large body is rejected before JSON parsing and keeps a safe correlation ID."""
    response = client.post(
        "/api/v1/search",
        content=b"x" * 70_000,
        headers={**_auth(READ), "X-Request-ID": "test-body"},
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "request body too large", "request_id": "test-body"}
    assert response.headers["X-Request-ID"] == "test-body"


def test_streamed_body_limit_does_not_depend_on_content_length() -> None:
    """Multiple request chunks cannot bypass the body limit by omitting Content-Length."""
    messages = iter(
        [
            {"type": "http.request", "body": b"a" * 700, "more_body": True},
            {"type": "http.request", "body": b"b" * 700, "more_body": False},
        ]
    )
    sent: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        raise AssertionError("an oversized request reached the application")

    async def receive() -> Message:
        return next(messages)

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/search",
        "headers": [],
        "state": {"request_id": "stream-test"},
    }
    asyncio.run(RequestBodyLimit(app, 1024)(scope, receive, send))
    assert sent[0]["status"] == 413
    assert b"stream-test" in sent[1]["body"]


def test_audit_log_records_decisions_and_returned_ids(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Auth decisions and search results are audited without secrets or the raw query."""
    with caplog.at_level(logging.INFO, logger="silicore_knowledge.audit"):
        client.get("/api/v1/stats", headers=_auth("wrong-token-value"))
        client.post("/api/v1/search", json={"query": "private words"}, headers=_auth(READ))
    text = caplog.text
    assert "decision=deny" in text and "role=invalid" in text
    assert "decision=allow" in text and "event=search" in text
    for forbidden in ("wrong-token-value", READ, "private words"):
        assert forbidden not in text


def test_rate_limiter_window_and_bound() -> None:
    """Old events expire, and tracked keys are bounded."""
    limiter = RateLimiter(per_minute=1, max_keys=2)
    assert limiter.check("a", now=0.0) is None
    assert limiter.check("a", now=10.0) == 51
    assert limiter.check("a", now=61.0) is None
    limiter.check("b", now=61.0)
    limiter.check("c", now=61.0)
    assert limiter.check("a", now=62.0) is None  # "a" was forgotten, so it starts afresh


def test_server_requires_a_positive_relevance_threshold(tmp_path: Path) -> None:
    """The API cannot silently run with a disabled relevance threshold."""
    settings = Settings(workspace=tmp_path / "ws")
    with pytest.raises(ConfigError, match="min_score must be positive"):
        create_app(settings, Workspace(settings.workspace))
