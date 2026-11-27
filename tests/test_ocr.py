# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional OCR step."""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pdfmaker import make_pdf

from silicore_knowledge.config import OcrSettings, Settings
from silicore_knowledge.ingestion import ocr as ocr_module
from silicore_knowledge.ingestion import pipeline
from silicore_knowledge.ingestion.extract import ExtractedDocument, ExtractedPage
from silicore_knowledge.ingestion.importer import DocumentMetadata, import_pdf
from silicore_knowledge.ingestion.ocr import OcrError, needs_ocr, ocr_pdf
from silicore_knowledge.ingestion.pipeline import Indexer, IndexingError
from silicore_knowledge.storage.catalog import Catalog
from silicore_knowledge.storage.models import SourceType
from silicore_knowledge.workspace import Workspace

META = DocumentMetadata(source_type=SourceType.OFFICIAL)


def test_needs_ocr_depends_on_average_text_per_page() -> None:
    """Sparse text layers trigger OCR, dense ones do not."""
    settings = OcrSettings(min_chars_per_page=40)

    def doc(text: str) -> ExtractedDocument:
        pages = (ExtractedPage(1, text), ExtractedPage(2, ""))
        return ExtractedDocument(None, 2, pages, ())

    assert needs_ocr(doc(""), settings)
    assert needs_ocr(doc("x" * 79), settings)
    assert not needs_ocr(doc("x" * 80), settings)


def test_ocr_pdf_runs_ocrmypdf_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command is a fixed argument list with a timeout and no stdin."""
    seen: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen.update(command=command, **kwargs)
        Path(command[-1]).write_bytes(b"%PDF-")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ocrmypdf")
    monkeypatch.setattr(ocr_module.subprocess, "run", fake_run)
    out = tmp_path / "out.pdf"
    ocr_pdf(tmp_path / "in.pdf", out, OcrSettings(language="eng+fra", timeout_s=42))
    assert seen["command"][-4:] == ["-l", "eng+fra", str(tmp_path / "in.pdf"), str(out)]
    assert seen["timeout"] == 42 and seen["stdin"] == subprocess.DEVNULL
    assert "shell" not in seen and out.is_file()


def test_ocr_failures_are_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing binary, non-zero exit and timeout raise OcrError and leave no output."""
    out = tmp_path / "out.pdf"
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(OcrError, match="not installed"):
        ocr_pdf(tmp_path / "in.pdf", out, OcrSettings())

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ocrmypdf")

    def failing(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        out.write_bytes(b"partial")
        return subprocess.CompletedProcess(command, 6, b"", b"")

    monkeypatch.setattr(ocr_module.subprocess, "run", failing)
    with pytest.raises(OcrError, match="exit code 6"):
        ocr_pdf(tmp_path / "in.pdf", out, OcrSettings())
    assert not out.exists()

    def slow(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(ocr_module.subprocess, "run", slow)
    with pytest.raises(OcrError, match="exceeded"):
        ocr_pdf(tmp_path / "in.pdf", out, OcrSettings())


def _indexer(tmp_path: Path, pdf: Path, *, enabled: bool) -> tuple[Indexer, str, Workspace]:
    settings = Settings(workspace=tmp_path / "ws", ocr=OcrSettings(enabled=enabled))
    workspace = Workspace(settings.workspace)
    catalog = Catalog(workspace.catalog_path)
    doc_id = import_pdf(workspace, catalog, pdf, META, 10_000_000).document.document_id
    return Indexer(settings, workspace, catalog), doc_id, workspace


def test_scanned_pdf_is_ocred_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A PDF without text layer is indexed from the OCR output; scratch files are removed."""
    scanned = make_pdf(tmp_path / "scan.pdf", [""])
    readable = make_pdf(tmp_path / "text.pdf", ["USART_ISR register description"], [[1, "A", 1]])
    calls: list[Path] = []

    def fake_ocr(source: Path, destination: Path, settings: OcrSettings) -> None:
        calls.append(destination)
        shutil.copy(readable, destination)

    monkeypatch.setattr(pipeline, "ocr_pdf", fake_ocr)
    indexer, doc_id, workspace = _indexer(tmp_path, scanned, enabled=True)
    assert indexer.release(doc_id).chunks >= 1
    assert len(calls) == 2
    assert not list(workspace.area(pipeline.Area.TMP).iterdir())


def test_scanned_pdf_without_ocr_is_an_error(tmp_path: Path) -> None:
    """Without OCR, a document with no text is reported instead of silently empty."""
    indexer, doc_id, _ = _indexer(tmp_path, make_pdf(tmp_path / "scan.pdf", [""]), enabled=False)
    with pytest.raises(IndexingError, match="no extractable text"):
        indexer.release(doc_id)


def test_text_pdf_skips_ocr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OCR is not run when the text layer is sufficient."""

    def boom(*args: object) -> None:
        raise AssertionError("OCR must not run")

    monkeypatch.setattr(pipeline, "ocr_pdf", boom)
    text = "SPI_CR1 register description with plenty of readable words " * 2
    pdf = make_pdf(tmp_path / "t.pdf", [text], [[1, "A", 1]])
    indexer, doc_id, _ = _indexer(tmp_path, pdf, enabled=True)
    assert indexer.release(doc_id).chunks >= 1
