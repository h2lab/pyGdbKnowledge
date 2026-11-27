# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the workspace."""

import os
import stat
from pathlib import Path

import pytest

from silicore_knowledge.workspace import Area, Workspace, WorkspaceError

SHA = "a" * 64


def test_creates_layout_with_private_permissions(tmp_path: Path) -> None:
    """Missing root and areas are created with mode 0700."""
    root = tmp_path / "ws"
    workspace = Workspace(root)
    for area in Area:
        path = workspace.area(area)
        assert path.is_dir()
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert workspace.catalog_path == root.resolve() / "catalog.db"


def test_reopen_existing_workspace(tmp_path: Path) -> None:
    """An existing valid workspace opens without create."""
    Workspace(tmp_path / "ws")
    Workspace(tmp_path / "ws", create=False)


def test_missing_workspace_without_create(tmp_path: Path) -> None:
    """Opening a missing workspace with create=False fails."""
    with pytest.raises(WorkspaceError):
        Workspace(tmp_path / "absent", create=False)


def test_refuses_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The service never runs as root."""
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    with pytest.raises(WorkspaceError, match="root"):
        Workspace(tmp_path / "ws")


def test_rejects_relative_root() -> None:
    """The root must be absolute."""
    with pytest.raises(WorkspaceError, match="absolute"):
        Workspace(Path("relative"))


def test_rejects_group_writable_root(tmp_path: Path) -> None:
    """Overly permissive existing directories are refused."""
    root = tmp_path / "ws"
    Workspace(root)
    root.chmod(0o770)
    with pytest.raises(WorkspaceError, match="writable"):
        Workspace(root)


def test_rejects_symlinked_area(tmp_path: Path) -> None:
    """An area replaced by a symlink is refused."""
    root = tmp_path / "ws"
    Workspace(root)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    (root / "tmp").rmdir()
    (root / "tmp").symlink_to(outside)
    with pytest.raises(WorkspaceError, match="real directory"):
        Workspace(root)


@pytest.mark.parametrize("bad", ["", "/etc/passwd", "../x", "a/../../x", "a\x00b"])
def test_resolve_rejects_unsafe_paths(tmp_path: Path, bad: str) -> None:
    """Absolute, parent-traversing and malformed paths are rejected."""
    workspace = Workspace(tmp_path / "ws")
    with pytest.raises(WorkspaceError):
        workspace.resolve(Area.TMP, bad)


def test_resolve_rejects_symlink_escape(tmp_path: Path) -> None:
    """A symlink inside an area cannot lead outside it."""
    workspace = Workspace(tmp_path / "ws")
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace.area(Area.TMP) / "link").symlink_to(outside)
    with pytest.raises(WorkspaceError, match="escapes"):
        workspace.resolve(Area.TMP, "link/file")


def test_resolve_accepts_nested_path(tmp_path: Path) -> None:
    """A normal nested path resolves inside the area."""
    workspace = Workspace(tmp_path / "ws")
    path = workspace.resolve(Area.EXTRACTED, "doc/page1.txt")
    assert path == workspace.area(Area.EXTRACTED).resolve() / "doc" / "page1.txt"


def test_document_path(tmp_path: Path) -> None:
    """Documents are named by digest; bad digests and areas are refused."""
    workspace = Workspace(tmp_path / "ws")
    assert workspace.document_path(SHA).name == f"{SHA}.pdf"
    assert workspace.document_path(SHA, Area.QUARANTINE).parent.name == "quarantine"
    for bad in ("", "../" + SHA, SHA.upper(), "a" * 63):
        with pytest.raises(WorkspaceError):
            workspace.document_path(bad)
    with pytest.raises(WorkspaceError):
        workspace.document_path(SHA, Area.TMP)
