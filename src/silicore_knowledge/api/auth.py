# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Bearer-token authentication with separate read and admin credentials (ANSSI R20, R21)."""

import hmac
from dataclasses import dataclass
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

    READ = auto()
    ADMIN = auto()


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated identity and its server-side document grants."""

    name: str
    role: Role
    allowed_document_ids: tuple[str, ...] | None

    def allows(self, document_id: str) -> bool:
        """Return whether this identity can read a document."""
        return self.allowed_document_ids is None or document_id in self.allowed_document_ids


def _matches(supplied: str, expected: str) -> bool:
    return hmac.compare_digest(supplied.encode(), expected.encode())


def _identify(
    settings: Settings, credentials: HTTPAuthorizationCredentials | None
) -> Principal | None:
    """Return the caller identity, or None if credentials are invalid."""
    auth = settings.auth
    if credentials is not None:
        if auth.admin_token is not None and _matches(
            credentials.credentials, auth.admin_token.get_secret_value()
        ):
            return Principal("admin", Role.ADMIN, None)
        for name, principal in auth.principals.items():
            if _matches(credentials.credentials, principal.token.get_secret_value()):
                grants = principal.allowed_documents
                return Principal(name, Role.READ, None if grants == ("*",) else grants)
        return None
    return None


def _authorize(request: Request, principal: Principal | None, needed: Role) -> Principal:
    allowed = principal is not None and (
        principal.role is Role.ADMIN or (principal.role is Role.READ and needed is Role.READ)
    )
    client = request.client.host if request.client else "-"
    audit_logger.log(
        20 if allowed else 30,
        "event=auth decision=%s needed=%s role=%s identity=%s request_id=%s client=%s method=%s path=%s",
        "allow" if allowed else "deny",
        needed.value,
        principal.role.value if principal else "invalid",
        principal.name if principal else "-",
        getattr(request.state, "request_id", "-"),
        client,
        request.method,
        request.url.path,
    )
    if not allowed or principal is None:
        # Same answer for a missing, wrong or under-privileged token.
        raise HTTPException(
            status_code=401,
            detail="authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def require_read(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    """Allow an authenticated read or admin identity."""
    return _authorize(request, _identify(settings, credentials), Role.READ)


def require_admin(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    """Allow the admin token only; never open."""
    return _authorize(request, _identify(settings, credentials), Role.ADMIN)
