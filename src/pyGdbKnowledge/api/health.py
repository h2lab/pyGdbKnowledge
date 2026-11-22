# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Health endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

from .. import __version__

router = APIRouter()


class HealthResponse(BaseModel):
    """Service health report."""

    status: str
    version: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the service is up.

    Returns
    -------
    HealthResponse
        Status and service version; no corpus or configuration detail is disclosed.
    """
    return HealthResponse(status="ok", version=__version__)
