# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Bearer-token authentication with separate read and admin credentials (ANSSI R20, R21)."""

import hmac
from enum import StrEnum, auto, unique
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import Settings
from ..security import audit_logger
from .deps import get_settings

_bearer = HTTPBearer(auto_error=False)


@unique
class Role(StrEnum):
    """Privilege level of a caller."""

    ANONYMOUS = auto()
    READ = auto()
    ADMIN = auto()


def _matches(supplied: str, expected: str) -> bool:
    return hmac.compare_digest(supplied.encode(), expected.encode())


def _identify(settings: Settings, credentials: HTTPAuthorizationCredentials | None) -> Role | None:
    """Return the caller role, or None if the credentials are invalid."""
    auth = settings.auth
    if credentials is not None:
        if auth.admin_token is not None and _matches(
            credentials.credentials, auth.admin_token.get_secret_value()
        ):
            return Role.ADMIN
        if auth.read_token is not None and _matches(
            credentials.credentials, auth.read_token.get_secret_value()
        ):
            return Role.READ
        return None
    return Role.ANONYMOUS


def _authorize(request: Request, settings: Settings, role: Role | None, needed: Role) -> Role:
    auth = settings.auth
    # Without a read token the API is open (loopback only, enforced by the settings).
    open_read = needed is Role.READ and auth.read_token is None
    allowed = (
        role is Role.ADMIN
        or (role is Role.READ and needed is Role.READ)
        or (role is Role.ANONYMOUS and open_read)
    )
    client = request.client.host if request.client else "-"
    audit_logger.log(
        20 if allowed else 30,
        "event=auth decision=%s needed=%s role=%s request_id=%s client=%s method=%s path=%s",
        "allow" if allowed else "deny",
        needed.value,
        role.value if role else "invalid",
        getattr(request.state, "request_id", "-"),
        client,
        request.method,
        request.url.path,
    )
    if not allowed or role is None:
        # Same answer for a missing, wrong or under-privileged token.
        raise HTTPException(
            status_code=401,
            detail="authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return role


def require_read(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Role:
    """Allow read or admin tokens (or anonymous access when no read token is configured)."""
    return _authorize(request, settings, _identify(settings, credentials), Role.READ)


def require_admin(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Role:
    """Allow the admin token only; never open."""
    return _authorize(request, settings, _identify(settings, credentials), Role.ADMIN)
