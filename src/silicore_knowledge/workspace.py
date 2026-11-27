# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Persistent workspace: layout, permissions and safe path resolution."""

import os
import re
import stat
from enum import StrEnum, unique
from pathlib import Path

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DIR_MODE = 0o700
DOCUMENT_SUFFIXES = (".pdf", ".svd")


class WorkspaceError(Exception):
    """Raised when the workspace is unsafe, invalid, or a path escapes it."""


@unique
class Area(StrEnum):
    """Workspace subdirectories; values are the directory names."""

    DOCUMENTS = "documents"
    EXTRACTED = "extracted"
    QUARANTINE = "quarantine"
    QDRANT = "qdrant"
    TMP = "tmp"


def ensure_unprivileged() -> None:
    """Refuse to run with administrative privileges.

    Raises
    ------
    WorkspaceError
        If the effective user is root.
    """
    if os.geteuid() == 0:
        raise WorkspaceError("refusing to run as root; use a dedicated unprivileged user")


def _check_directory(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise WorkspaceError(f"{path} is not a real directory")
    if info.st_uid != os.geteuid():
        raise WorkspaceError(f"{path} is not owned by the current user")
    if info.st_mode & 0o022:
        raise WorkspaceError(f"{path} must not be group- or world-writable")


class Workspace:
    """Validated workspace rooted at a directory owned by the service user.

    Parameters
    ----------
    root : Path
        Workspace root; created with mode 0700 when ``create`` is true.
    create : bool
        Create the root and subdirectories when missing.

    Raises
    ------
    WorkspaceError
        If running as root, or the root or a subdirectory is unsafe or missing.
    """

    def __init__(self, root: Path, create: bool = True) -> None:
        ensure_unprivileged()
        if not root.is_absolute():
            raise WorkspaceError(f"workspace path must be absolute: {root}")
        try:
            if create:
                root.mkdir(mode=_DIR_MODE, parents=True, exist_ok=True)
                for area in Area:
                    (root / area).mkdir(mode=_DIR_MODE, exist_ok=True)
            _check_directory(root)
            for area in Area:
                _check_directory(root / area)
        except OSError as exc:
            raise WorkspaceError(f"cannot prepare workspace {root}: {exc}") from exc
        self.root = root.resolve()

    @property
    def catalog_path(self) -> Path:
        """Path of the SQLite catalog file."""
        return self.root / "catalog.db"

    def area(self, area: Area) -> Path:
        """Return the directory of a workspace area."""
        return self.root / area

    def resolve(self, area: Area, relative: str) -> Path:
        """Resolve a user-supplied relative path inside an area.

        Parameters
        ----------
        area : Area
            Area the path must stay in.
        relative : str
            Relative path; absolute paths, ``..`` components and NUL bytes are rejected.

        Returns
        -------
        Path
            Resolved path, guaranteed to be inside the area (symlinks included).

        Raises
        ------
        WorkspaceError
            If the path is absolute, malformed, or escapes the area.
        """
        if not relative or "\x00" in relative or Path(relative).is_absolute():
            raise WorkspaceError("invalid relative path")
        if ".." in Path(relative).parts:
            raise WorkspaceError("path must not contain '..'")
        base = self.area(area).resolve()
        target = (base / relative).resolve()
        if not target.is_relative_to(base):
            raise WorkspaceError("path escapes the workspace area")
        return target

    def document_path(self, sha256: str, area: Area = Area.DOCUMENTS, suffix: str = ".pdf") -> Path:
        """Return the storage path of a document named by its SHA-256.

        Parameters
        ----------
        sha256 : str
            Lowercase hexadecimal SHA-256 of the file content.
        area : Area
            ``DOCUMENTS`` or ``QUARANTINE``.
        suffix : str
            One of ``.pdf`` and ``.svd``.

        Returns
        -------
        Path
            ``<area>/<sha256><suffix>``.

        Raises
        ------
        WorkspaceError
            If the digest is malformed or the area cannot store documents.
        """
        if area not in (Area.DOCUMENTS, Area.QUARANTINE):
            raise WorkspaceError(f"area {area.value!r} does not store documents")
        if not _SHA256_RE.fullmatch(sha256):
            raise WorkspaceError("invalid SHA-256 digest")
        if suffix not in DOCUMENT_SUFFIXES:
            raise WorkspaceError(f"unsupported document suffix {suffix!r}")
        return self.area(area) / f"{sha256}{suffix}"

    def stored_document(self, sha256: str, area: Area = Area.DOCUMENTS) -> Path | None:
        """Return the stored file of a document whatever its format, or None if absent."""
        for suffix in DOCUMENT_SUFFIXES:
            path = self.document_path(sha256, area, suffix)
            if path.is_file():
                return path
        return None
