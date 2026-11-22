# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Health and statistics endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .. import __version__
from ..storage.catalog import Catalog
from ..storage.models import CatalogStats
from .deps import get_catalog

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


@router.get("/stats", response_model=CatalogStats)
def stats(catalog: Annotated[Catalog, Depends(get_catalog)]) -> CatalogStats:
    """Return catalog counters (schema version, documents, sections, chunks)."""
    return catalog.stats()
