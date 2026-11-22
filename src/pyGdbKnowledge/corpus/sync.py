# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Synchronization of the workspace with the corpus manifest.

Only the operator CLI calls this module; no HTTP route of the service can trigger a download.
Every download is verified against the manifest hash, or against the lock file written on the
first download (trust on first use), before it is handed to the regular import path.
"""

import json
import fnmatch
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlsplit

from ..ingestion.importer import DocumentMetadata, IngestError, import_document
from ..security import audit_logger
from ..storage.catalog import Catalog, CatalogError
from ..workspace import Area, Workspace
from .fetch import Fetcher
from .manifest import (
    GITHUB_API_HOST,
    GITHUB_RAW_HOST,
    CorpusError,
    GithubSvdSource,
    Manifest,
    UrlSource,
)

logger = logging.getLogger("pyGdbKnowledge")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_LOCK_VERSION = 1


@dataclass(frozen=True, slots=True)
class PlannedDownload:
    """One document to fetch, with the metadata it will be registered with."""

    source: str
    url: str
    metadata: DocumentMetadata
    sha256: str | None = None


@dataclass(slots=True)
class SyncReport:
    """Outcome of a synchronization run."""

    imported: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    planned: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


class LockFile:
    """URL to SHA-256 map recorded on first download (``corpus.lock.json``)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._hashes: dict[str, str] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                hashes = data["hashes"]
                if data["version"] != _LOCK_VERSION or not isinstance(hashes, dict):
                    raise ValueError("unsupported layout")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise CorpusError(f"invalid lock file {path}: {exc}") from exc
            self._hashes = {str(url): str(sha) for url, sha in hashes.items()}

    def get(self, url: str) -> str | None:
        """Return the recorded hash of ``url``."""
        return self._hashes.get(url)

    def record(self, url: str, sha256: str) -> None:
        """Remember the hash of ``url`` (in memory until ``save``)."""
        self._hashes[url] = sha256

    def save(self) -> None:
        """Write the lock file atomically, sorted for readable diffs."""
        payload = {"version": _LOCK_VERSION, "hashes": dict(sorted(self._hashes.items()))}
        fd, name = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                json.dump(payload, out, indent=1)
                out.write("\n")
            os.replace(name, self._path)
        finally:
            Path(name).unlink(missing_ok=True)


def _expand_github(
    source: GithubSvdSource, fetcher: Fetcher, max_bytes: int
) -> list[PlannedDownload]:
    api = f"https://{GITHUB_API_HOST}/repos/{source.repo}"
    commit = fetcher.get_json(f"{api}/commits/{quote(source.ref, safe='')}")
    sha = commit.get("sha") if isinstance(commit, dict) else None
    if not isinstance(sha, str) or not _COMMIT_RE.fullmatch(sha):
        raise CorpusError(f"{source.name}: cannot resolve {source.ref!r} to a commit")
    tree = fetcher.get_json(f"{api}/git/trees/{sha}?recursive=1")
    if (
        not isinstance(tree, dict)
        or tree.get("truncated")
        or not isinstance(tree.get("tree"), list)
    ):
        raise CorpusError(f"{source.name}: repository tree is unavailable or truncated")
    prefix = source.path_prefix.split("/")
    plans: list[PlannedDownload] = []
    for node in tree["tree"]:
        if not isinstance(node, dict) or node.get("type") != "blob":
            continue
        path = node.get("path")
        if not isinstance(path, str) or not path.lower().endswith(".svd"):
            continue
        parts = path.split("/")
        if len(parts) != len(prefix) + 2 or parts[: len(prefix)] != prefix:
            continue
        vendor = parts[-2]
        if vendor not in source.vendors:
            continue
        size = node.get("size")
        if isinstance(size, int) and size > max_bytes:
            logger.warning("%s: %s exceeds the size limit, ignored", source.name, path)
            continue
        stem = parts[-1][: -len(".svd")]
        if source.include and not any(
            fnmatch.fnmatch(stem.lower(), pattern.lower()) for pattern in source.include
        ):
            continue
        manufacturer = source.manufacturers.get(vendor, vendor)
        family = None
        if source.family_pattern and (found := re.match(source.family_pattern, stem, re.I)):
            family = (found.group(1) if found.groups() else found.group(0)).upper()
        url = f"https://{GITHUB_RAW_HOST}/{source.repo}/{sha}/{quote(path)}"
        plans.append(
            PlannedDownload(
                source=source.name,
                url=url,
                metadata=DocumentMetadata(
                    source_type=source.source_type,
                    title=f"{stem} SVD",
                    manufacturer=manufacturer,
                    device_family=family,
                    device=stem,
                    document_type="svd",
                    revision=sha[:12],
                    source_url=url,
                ),
            )
        )
    return plans


def _expand_urls(source: UrlSource) -> list[PlannedDownload]:
    plans = []
    for entry in source.documents:
        plans.append(
            PlannedDownload(
                source=source.name,
                url=entry.url,
                sha256=entry.sha256,
                metadata=DocumentMetadata(
                    source_type=source.source_type,
                    title=entry.title or Path(urlsplit(entry.url).path).stem,
                    manufacturer=entry.manufacturer or source.manufacturer,
                    device_family=entry.device_family,
                    device=entry.device,
                    document_type=entry.document_type,
                    revision=entry.revision,
                    source_url=entry.url,
                ),
            )
        )
    return plans


def plan_sync(
    manifest: Manifest, fetcher: Fetcher, max_bytes: int, only: str | None = None
) -> list[PlannedDownload]:
    """List every document the manifest describes (contacts GitHub to expand repositories).

    Raises
    ------
    CorpusError
        If ``only`` names no source, or a repository cannot be listed.
    """
    sources = [s for s in manifest.sources if only is None or s.name == only]
    if only is not None and not sources:
        raise CorpusError(f"unknown source {only!r}")
    plans: list[PlannedDownload] = []
    for source in sources:
        if isinstance(source, GithubSvdSource):
            plans.extend(_expand_github(source, fetcher, max_bytes))
        else:
            plans.extend(_expand_urls(source))
    return plans


def _known_urls(catalog: Catalog) -> set[str]:
    urls: set[str] = set()
    offset = 0
    while batch := catalog.list_documents(limit=200, offset=offset):
        offset += len(batch)
        urls.update(d.source_url for d in batch if d.source_url)
    return urls


def run_sync(
    plans: list[PlannedDownload],
    lock: LockFile,
    workspace: Workspace,
    catalog: Catalog,
    fetcher: Fetcher,
    max_bytes: int,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> SyncReport:
    """Download and import the documents that are not in the catalog yet.

    A document is "present" when its URL is a recorded provenance, or when the lock file knows
    its hash and the catalog holds that content. The lock file is saved after each import, so
    an interrupted run resumes where it stopped.

    Parameters
    ----------
    plans : list[PlannedDownload]
        Documents to consider, from ``plan_sync``.
    lock : LockFile
        Hash records (trust on first use).
    workspace : Workspace
        Destination workspace.
    catalog : Catalog
        Catalog to register the documents in.
    fetcher : Fetcher
        Network access.
    max_bytes : int
        Maximum size of one downloaded file.
    dry_run : bool
        Only report what would be downloaded.
    limit : int | None
        Stop after this many downloads.

    Returns
    -------
    SyncReport
        Imported, skipped, planned (dry run) and failed URLs.
    """
    report = SyncReport()
    known = _known_urls(catalog)
    attempts = 0
    for plan in plans:
        recorded = plan.sha256 or lock.get(plan.url)
        if plan.url in known or (recorded and catalog.get_document_by_hash(recorded)):
            report.skipped.append(plan.url)
            continue
        if limit is not None and attempts >= limit:
            break
        if dry_run:
            report.planned.append(plan.url)
            continue
        attempts += 1
        try:
            _fetch_one(plan, recorded, lock, workspace, catalog, fetcher, max_bytes, report)
        except (CorpusError, IngestError, CatalogError) as exc:
            logger.warning("sync failed for %s: %s", plan.url, exc)
            audit_logger.warning("event=sync_failed host=%s", urlsplit(plan.url).hostname)
            report.failed[plan.url] = str(exc)
    return report


def _fetch_one(
    plan: PlannedDownload,
    expected: str | None,
    lock: LockFile,
    workspace: Workspace,
    catalog: Catalog,
    fetcher: Fetcher,
    max_bytes: int,
    report: SyncReport,
) -> None:
    fd, name = tempfile.mkstemp(dir=workspace.area(Area.TMP), suffix=".download")
    os.close(fd)
    scratch = Path(name)
    try:
        sha256 = fetcher.download(plan.url, scratch, max_bytes)
        if expected is not None and sha256 != expected:
            raise CorpusError(f"hash mismatch (expected {expected[:12]}…, got {sha256[:12]}…)")
        result = import_document(
            workspace, catalog, scratch, plan.metadata, max_bytes, channel="sync"
        )
        if expected is None:
            lock.record(plan.url, sha256)
            lock.save()
        report.imported.append(result.document.document_id)
    finally:
        scratch.unlink(missing_ok=True)
