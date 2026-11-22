# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Request throttling and audit logging helpers (ANSSI R18, R19)."""

import logging
import threading
import time
from collections import deque

audit_logger = logging.getLogger("pyGdbKnowledge.audit")


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
