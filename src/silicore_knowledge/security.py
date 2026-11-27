# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Request throttling and audit logging helpers (ANSSI R18, R19)."""

import logging
import re
import threading
import time
import uuid
from collections import deque

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

audit_logger = logging.getLogger("silicore_knowledge.audit")
REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def request_id(supplied: str) -> str:
    """Keep a safe caller-provided correlation ID or generate a new one."""
    return supplied if _REQUEST_ID_RE.fullmatch(supplied) else uuid.uuid4().hex


class RequestBodyLimit:
    """Bound non-upload request bodies before the application parses them."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject oversized requests before passing them to the application."""
        if scope["type"] != "http" or (
            scope["method"] == "POST" and scope["path"] == "/api/v1/admin/ingest"
        ):
            await self._app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            block = message.get("body", b"")
            if len(body) + len(block) > self._max_bytes:
                headers = dict(scope.get("headers", []))
                correlation = scope.get("state", {}).get("request_id") or request_id(
                    headers.get(REQUEST_ID_HEADER.lower().encode(), b"").decode("latin-1")
                )
                audit_logger.warning(
                    "event=request_rejected request_id=%s reason=body_too_large", correlation
                )
                response = JSONResponse(
                    {"detail": "request body too large", "request_id": correlation},
                    status_code=413,
                    headers={REQUEST_ID_HEADER: correlation},
                )
                await response(scope, receive, send)
                return
            body.extend(block)
            if not message.get("more_body", False):
                break

        pending = True

        async def replay() -> Message:
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self._app(scope, replay, send)


class RateLimiter:
    """Sliding-window limiter, per client key, kept in memory.

    Parameters
    ----------
    per_minute : int
        Requests allowed per key in any 60-second window.
    max_keys : int
        Bound on tracked keys; the least recently seen keys are forgotten beyond it.
    """

    WINDOW_S = 60.0

    def __init__(self, per_minute: int, max_keys: int = 10_000) -> None:
        self._per_minute = per_minute
        self._max_keys = max_keys
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> int | None:
        """Record a request.

        Parameters
        ----------
        key : str
            Client identifier.
        now : float | None
            Monotonic timestamp, for tests.

        Returns
        -------
        int | None
            None when allowed, otherwise the seconds to wait before retrying.
        """
        moment = time.monotonic() if now is None else now
        with self._lock:
            events = self._events.pop(key, None)
            if events is None:
                events = deque()
                if len(self._events) >= self._max_keys:
                    del self._events[next(iter(self._events))]
            self._events[key] = events
            while events and moment - events[0] >= self.WINDOW_S:
                events.popleft()
            if len(events) >= self._per_minute:
                return max(1, int(self.WINDOW_S - (moment - events[0])) + 1)
            events.append(moment)
            return None
