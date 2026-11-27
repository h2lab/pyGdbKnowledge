# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Shared request dependencies."""

from fastapi import Request

from ..config import Settings
from ..storage.catalog import Catalog


def get_catalog(request: Request) -> Catalog:
    """Return the application catalog."""
    catalog: Catalog = request.app.state.catalog
    return catalog


def get_settings(request: Request) -> Settings:
    """Return the application settings."""
    settings: Settings = request.app.state.settings
    return settings
