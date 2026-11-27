# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for PDF text and structure extraction."""

from pathlib import Path

import pytest
from pdfmaker import make_pdf

from silicore_knowledge.config import IngestionSettings
from silicore_knowledge.ingestion.extract import (
    ExtractionError,
    build_sections,
    clean_text,
    extract_pdf,
    strip_running_headers,
)

SETTINGS = IngestionSettings()


def _make_pdf(
    path: Path,
    bodies: list[str],
    toc: list[list[object]] | None = None,
    *,
    running_header: bool = False,
) -> Path:
    return make_pdf(path, bodies, toc, running_header=running_header)


def test_extracts_pages_title_and_outline(tmp_path: Path) -> None:
    """Page text, document title and outline sections are returned."""
    path = make_pdf(
        tmp_path / "a.pdf",
        ["USART_ISR register", "SPI chapter", "end"],
        [[1, "1 USART", 1], [2, "1.1 Registers", 1], [1, "2 SPI", 2]],
        title="Sample datasheet",
    )
    result = extract_pdf(path, SETTINGS)
    assert result.title == "Sample datasheet"
    assert result.page_count == 3
    assert [p.number for p in result.pages] == [1, 2, 3]
    assert "USART_ISR register" in result.pages[0].text
    assert [(s.path, s.page_start, s.page_end) for s in result.sections] == [
        ("1 USART", 1, 1),
        ("1 USART > 1.1 Registers", 1, 2),
        ("2 SPI", 2, 3),
    ]


def test_running_headers_and_footers_are_removed(tmp_path: Path) -> None:
    """Lines repeated on many pages (page counters included) are dropped."""
    words = [[f"{chr(97 + i)}{chr(97 + k)}" * 3 for k in range(6)] for i in range(6)]
    bodies = ["\n".join(row) for row in words]
    path = _make_pdf(tmp_path / "h.pdf", bodies, running_header=True)
    result = extract_pdf(path, SETTINGS)
    for index, page in enumerate(result.pages):
        assert all(word in page.text for word in words[index])
        assert "DS999" not in page.text and "Confidential" not in page.text


def test_pdf_without_outline_is_one_section(tmp_path: Path) -> None:
    """A missing outline falls back to a single document section."""
    result = extract_pdf(_make_pdf(tmp_path / "n.pdf", ["a", "b"]), SETTINGS)
    assert [(s.title, s.page_start, s.page_end) for s in result.sections] == [("Document", 1, 2)]


def test_page_and_size_bounds(tmp_path: Path) -> None:
    """PDF files beyond the configured bounds are refused."""
    path = _make_pdf(tmp_path / "b.pdf", ["\n".join(["x" * 40] * 30), "y", "z"])
    with pytest.raises(ExtractionError, match="pages"):
        extract_pdf(path, IngestionSettings(max_pages=2))
    with pytest.raises(ExtractionError, match="size limit"):
        extract_pdf(path, IngestionSettings(max_chars=1000))


def test_malformed_and_encrypted_files_are_refused(tmp_path: Path) -> None:
    """Garbage and password-protected PDF files raise ExtractionError."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7\nnot really a pdf")
    with pytest.raises(ExtractionError):
        extract_pdf(broken, SETTINGS)
    locked = make_pdf(tmp_path / "locked.pdf", ["secret"], password="u")
    with pytest.raises(ExtractionError, match="encrypted"):
        extract_pdf(locked, SETTINGS)


def test_clean_text() -> None:
    """Control characters and blank-line runs are removed."""
    assert clean_text("a\x00b  \n\n\n\n\x07c\n") == "ab\n\nc"


def test_build_sections_skips_captions_and_clamps_pages() -> None:
    """Table and figure bookmarks are not sections; pages stay within the document."""
    sections = build_sections(
        [(1, "1 A", 1), (2, "Table 1. Foo", 1), (2, "Figure 2: Bar", 2), (1, "2 B", 99)], 5
    )
    assert [(s.title, s.page_start, s.page_end) for s in sections] == [
        ("1 A", 1, 5),
        ("2 B", 5, 5),
    ]


def test_strip_running_headers_needs_enough_pages() -> None:
    """Too few pages give no reliable repetition signal: nothing is removed."""
    pages = [["Header 1", "body"], ["Header 2", "body2"]]
    assert strip_running_headers(pages) == pages
