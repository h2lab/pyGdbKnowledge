# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Restricted HTTP client for corpus downloads.

HTTPS only, hosts restricted to the manifest allow-list on every redirect hop, bounded size
and time, throttled, and with a descriptive User-Agent. It runs only from the operator CLI.
"""

import hashlib
import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from .manifest import CorpusError

USER_AGENT = "pyGdbKnowledge-sync/0.1 (operator-driven corpus sync)"
_CHUNK = 256 * 1024
_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_REDIRECTS = 5


class Fetcher(Protocol):
    """What the synchronizer needs from the network; replaceable in tests."""

    def get_json(self, url: str) -> Any:
        """Return the decoded JSON document at ``url``."""

    def download(self, url: str, destination: Path, max_bytes: int) -> str:
        """Store ``url`` into ``destination`` and return the SHA-256 of the content."""


class _AllowListRedirects(urllib.request.HTTPRedirectHandler):
    max_redirections = _MAX_REDIRECTS

    def __init__(self, allowed: frozenset[str]) -> None:
        self._allowed = allowed

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        check_url(newurl, self._allowed)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_url(url: str, allowed: frozenset[str]) -> None:
    """Refuse anything that is not an HTTPS URL on an allowed host.

    Raises
    ------
    CorpusError
        If the URL is not allowed.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or host not in allowed or parts.username or parts.password:
        raise CorpusError(f"URL not allowed: {url}")


class HttpFetcher:
    """Downloads over HTTPS from an allow-list of hosts.

    Parameters
    ----------
    allowed_domains : Iterable[str]
        Hosts that may be contacted (exact match).
    timeout_s : float
        Socket timeout of each request.
    delay_s : float
        Minimum delay between two requests.
    """

    def __init__(self, allowed_domains: tuple[str, ...], timeout_s: float, delay_s: float) -> None:
        self._allowed = frozenset(domain.lower() for domain in allowed_domains)
        self._timeout = timeout_s
        self._delay = delay_s
        self._last = 0.0
        # HTTPS only: no file:, ftp: or plain-http handlers are installed.
        self._opener = urllib.request.OpenerDirector()
        for handler in (
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
            _AllowListRedirects(self._allowed),
            urllib.request.HTTPDefaultErrorHandler(),
            urllib.request.HTTPErrorProcessor(),
        ):
            self._opener.add_handler(handler)

    def _open(self, url: str, accept: str) -> Any:
        check_url(url, self._allowed)
        wait = self._last + self._delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        request = urllib.request.Request(  # noqa: S310 - scheme checked by check_url
            url, headers={"User-Agent": USER_AGENT, "Accept": accept}
        )
        try:
            return self._opener.open(request, timeout=self._timeout)
        except urllib.error.HTTPError as exc:
            raise CorpusError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, OSError, ssl.SSLError) as exc:
            raise CorpusError(f"cannot fetch {url}: {exc}") from exc

    def get_json(self, url: str) -> Any:
        """Return the decoded JSON document at ``url`` (bounded in size)."""
        with self._open(url, "application/json") as response:
            body = response.read(_MAX_JSON_BYTES + 1)
        if len(body) > _MAX_JSON_BYTES:
            raise CorpusError(f"response too large: {url}")
        try:
            return json.loads(body)
        except ValueError as exc:
            raise CorpusError(f"invalid JSON from {url}") from exc

    def download(self, url: str, destination: Path, max_bytes: int) -> str:
        """Stream ``url`` to ``destination`` under a size cap; return its SHA-256."""
        digest = hashlib.sha256()
        total = 0
        with self._open(url, "*/*") as response, destination.open("wb") as out:
            declared = response.headers.get("Content-Length")
            if declared is not None and declared.isdigit() and int(declared) > max_bytes:
                raise CorpusError(f"{url}: exceeds the {max_bytes} byte limit")
            while block := response.read(_CHUNK):
                total += len(block)
                if total > max_bytes:
                    raise CorpusError(f"{url}: exceeds the {max_bytes} byte limit")
                digest.update(block)
                out.write(block)
        return digest.hexdigest()
