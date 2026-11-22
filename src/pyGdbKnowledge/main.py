# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Application factory and logging setup."""

import asyncio
import logging
import re
import sys
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .api import documents, health, search
from .api.auth import require_read
from .config import Settings
from .retrieval.hybrid import HybridSearcher
from .security import RateLimiter
from .storage.catalog import Catalog
from .workspace import Workspace

API_PREFIX = "/api/v1"
REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

logger = logging.getLogger("pyGdbKnowledge")


def configure_logging(level: str) -> None:
    """Send logs to stdout, as expected from a containerized service.

    Parameters
    ----------
    level : str
        Standard logging level name.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def create_app(
    settings: Settings, workspace: Workspace, searcher: HybridSearcher | None = None
) -> FastAPI:
    """Build the FastAPI application.

    Parameters
    ----------
    settings : Settings
        Validated service settings.
    workspace : Workspace
        Opened workspace.
    searcher : HybridSearcher | None
        Retrieval engine; defaults to lexical-only search over the catalog.

    Returns
    -------
    FastAPI
        Application exposing the versioned API under ``/api/v1``.
    """
    app = FastAPI(
        title="pyGdbKnowledge",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=f"{API_PREFIX}/openapi.json",
    )
    app.state.settings = settings
    app.state.workspace = workspace
    app.state.catalog = Catalog(workspace.catalog_path)
    app.state.searcher = searcher or HybridSearcher(app.state.catalog)

    limiter = RateLimiter(settings.limits.rate_limit_per_minute)
    timeout_s = settings.limits.request_timeout_s

    def error(
        status: int, detail: Any, request: Request, headers: dict[str, str] | None = None
    ) -> JSONResponse:
        body = {"detail": detail, "request_id": getattr(request.state, "request_id", None)}
        return JSONResponse(body, status_code=status, headers=headers)

    @app.middleware("http")
    async def correlation_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach a correlation id, throttle clients and bound the processing time."""
        supplied = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = supplied if _REQUEST_ID_RE.fullmatch(supplied) else uuid.uuid4().hex
        request.state.request_id = request_id
        response: Response
        client = request.client.host if request.client else "-"
        retry_after = None if request.url.path == f"{API_PREFIX}/health" else limiter.check(client)
        if retry_after is not None:
            response = error(429, "too many requests", request, {"Retry-After": str(retry_after)})
        else:
            try:
                response = await asyncio.wait_for(call_next(request), timeout_s)
            except TimeoutError:
                response = error(504, "request timed out", request)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%d",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
        )
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        """Answer errors with one JSON shape that never echoes input."""
        return error(exc.status_code, exc.detail, request, dict(exc.headers or {}))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Response:
        """Report where validation failed, without echoing the submitted values."""
        issues = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return error(422, issues, request)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> Response:
        """Hide internal details of unexpected failures."""
        logger.error(
            "request_id=%s unhandled %s",
            getattr(request.state, "request_id", "-"),
            type(exc).__name__,
        )
        return error(500, "internal error", request)

    protected = [Depends(require_read)]
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(health.stats_router, prefix=API_PREFIX, dependencies=protected)
    app.include_router(documents.router, prefix=API_PREFIX, dependencies=protected)
    app.include_router(search.router, prefix=API_PREFIX, dependencies=protected)
    return app
