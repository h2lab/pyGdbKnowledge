# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Application factory and logging setup."""

import logging
import re
import sys
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from . import __version__
from .api import health
from .config import Settings
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


def create_app(settings: Settings, workspace: Workspace) -> FastAPI:
    """Build the FastAPI application.

    Parameters
    ----------
    settings : Settings
        Validated service settings.
    workspace : Workspace
        Opened workspace.

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

    @app.middleware("http")
    async def correlation_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach a correlation id to each request and echo it in the response."""
        supplied = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = supplied if _REQUEST_ID_RE.fullmatch(supplied) else uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%d",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
        )
        return response

    app.include_router(health.router, prefix=API_PREFIX)
    return app
