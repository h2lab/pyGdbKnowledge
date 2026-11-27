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
from pydantic import SecretStr, ValidationError

from silicore_knowledge.api.auth import Role, require_admin
from silicore_knowledge.config import AuthSettings, LimitSettings, ServerSettings, Settings
from silicore_knowledge.main import create_app
from silicore_knowledge.security import RateLimiter
from silicore_knowledge.workspace import Workspace

READ = "read-secret-value"
ADMIN = "admin-secret-value"


def _app(tmp_path: Path, **overrides: object) -> FastAPI:
    settings = Settings(
        workspace=tmp_path / "ws",
        auth=AuthSettings(read_token=SecretStr(READ), admin_token=SecretStr(ADMIN)),
        **overrides,  # type: ignore[arg-type]
    )
    app = create_app(settings, Workspace(settings.workspace))

    @app.get("/admin-only")
    def admin_only(role: Annotated[Role, Depends(require_admin)]) -> dict[str, str]:
        return {"role": role.value}

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


def test_open_mode_without_read_token(tmp_path: Path) -> None:
    """Loopback deployments without tokens stay open, but admin never is."""
    settings = Settings(workspace=tmp_path / "ws")
    app = create_app(settings, Workspace(settings.workspace))
    app.add_api_route("/admin-only", lambda: None, dependencies=[Depends(require_admin)])
    client = TestClient(app)
    assert client.get("/api/v1/stats").status_code == 200
    assert client.get("/admin-only").status_code == 401


def test_remote_listener_requires_read_token(tmp_path: Path) -> None:
    """A non-loopback address without a read token is refused at configuration time."""
    with pytest.raises(ValidationError, match="read_token"):
        Settings(workspace=tmp_path, server=ServerSettings(host="0.0.0.0"))
    Settings(
        workspace=tmp_path,
        server=ServerSettings(host="0.0.0.0"),
        auth=AuthSettings(read_token=SecretStr(READ)),
    )


def test_rate_limit(tmp_path: Path) -> None:
    """Requests beyond the per-minute budget get 429 with Retry-After; health is exempt."""
    client = TestClient(_app(tmp_path, limits=LimitSettings(rate_limit_per_minute=2)))
    for _ in range(2):
        assert client.get("/api/v1/stats", headers=_auth(READ)).status_code == 200
    blocked = client.get("/api/v1/stats", headers=_auth(READ))
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    assert "X-Request-ID" in blocked.headers
    assert client.get("/api/v1/health").status_code == 200


def test_timeout(tmp_path: Path) -> None:
    """A request exceeding the timeout is answered with 504."""
    client = TestClient(_app(tmp_path, limits=LimitSettings(request_timeout_s=0.1)))
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
