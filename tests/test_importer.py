# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for PDF import into the workspace."""

import hashlib
import stat
from pathlib import Path

import pytest

from pyGdbKnowledge import cli
from pyGdbKnowledge.ingestion.importer import DocumentMetadata, IngestError, import_pdf
from pyGdbKnowledge.storage.catalog import Catalog
from pyGdbKnowledge.storage.models import SourceType
from pyGdbKnowledge.workspace import Area, Workspace

PDF = b"%PDF-1.7\n% fake content for tests\n"
META = DocumentMetadata(source_type=SourceType.OFFICIAL, manufacturer="ST")


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Fresh workspace."""
    return Workspace(tmp_path / "ws")


@pytest.fixture
def catalog(workspace: Workspace) -> Catalog:
    """Catalog of the workspace."""
    return Catalog(workspace.catalog_path)


def _pdf(tmp_path: Path, name: str = "STM32 Datasheet.pdf", data: bytes = PDF) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_import_stores_file_and_registers_document(
    tmp_path: Path, workspace: Workspace, catalog: Catalog
) -> None:
    """The PDF is copied under its SHA-256 and registered with its metadata."""
    result = import_pdf(workspace, catalog, _pdf(tmp_path), META, 10_000)
    sha = hashlib.sha256(PDF).hexdigest()
    stored = workspace.document_path(sha)
    assert result.created and stored.read_bytes() == PDF
    assert stat.S_IMODE(stored.stat().st_mode) == 0o600
    assert result.document.document_id == f"stm32-datasheet-{sha[:8]}"
    assert result.document.title == "STM32 Datasheet"
    assert result.document.manufacturer == "ST"
    assert catalog.get_document_by_hash(sha) == result.document
    assert list(workspace.area(Area.TMP).iterdir()) == []


def test_duplicate_content_is_not_stored_twice(
    tmp_path: Path, workspace: Workspace, catalog: Catalog
) -> None:
    """The same bytes under another name resolve to the existing document."""
    first = import_pdf(workspace, catalog, _pdf(tmp_path), META, 10_000)
    again = import_pdf(workspace, catalog, _pdf(tmp_path, "copy.pdf"), META, 10_000)
    assert not again.created and again.document.document_id == first.document.document_id
    assert len(list(workspace.area(Area.DOCUMENTS).iterdir())) == 1
    assert list(workspace.area(Area.TMP).iterdir()) == []


@pytest.mark.parametrize(
    ("data", "limit", "message"),
    [(b"MZ not a pdf", 10_000, "unsupported file type"), (PDF, 5, "exceeds")],
)
def test_invalid_files_are_rejected(
    tmp_path: Path,
    workspace: Workspace,
    catalog: Catalog,
    data: bytes,
    limit: int,
    message: str,
) -> None:
    """Wrong content or oversized files leave nothing behind."""
    with pytest.raises(IngestError, match=message):
        import_pdf(workspace, catalog, _pdf(tmp_path, data=data), META, limit)
    assert list(workspace.area(Area.DOCUMENTS).iterdir()) == []
    assert list(workspace.area(Area.TMP).iterdir()) == []
    assert catalog.stats().documents == 0


def test_non_regular_and_missing_sources_are_rejected(
    tmp_path: Path, workspace: Workspace, catalog: Catalog
) -> None:
    """Directories and missing paths are refused."""
    with pytest.raises(IngestError, match="not a regular file"):
        import_pdf(workspace, catalog, tmp_path, META, 10_000)
    with pytest.raises(IngestError, match="cannot read"):
        import_pdf(workspace, catalog, tmp_path / "nope.pdf", META, 10_000)


def test_metadata_is_validated(tmp_path: Path, workspace: Workspace, catalog: Catalog) -> None:
    """A non-http provenance URL is refused and nothing is stored."""
    meta = DocumentMetadata(source_type=SourceType.VENDOR, source_url="file:///etc/passwd")
    with pytest.raises(IngestError, match="http"):
        import_pdf(workspace, catalog, _pdf(tmp_path), meta, 10_000)
    assert catalog.stats().documents == 0


def test_cli_ingest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The ingest command imports files, reports duplicates and failures."""
    config = tmp_path / "s.yaml"
    config.write_text(f"workspace: {tmp_path / 'ws'}\n", encoding="utf-8")
    good = _pdf(tmp_path)
    base = ["ingest", "--config", str(config), "--source-type", "official"]
    assert cli.main([*base, "--manufacturer", "ST", str(good)]) == 0
    assert "imported:" in capsys.readouterr().out
    assert cli.main([*base, str(good)]) == 0
    assert "already present" in capsys.readouterr().out
    bad = _pdf(tmp_path, "bad.pdf", b"nope")
    assert cli.main([*base, str(bad)]) == 1
    assert "unsupported file type" in capsys.readouterr().err
    assert cli.main([*base, "--title", "T", str(good), str(bad)]) == 2
