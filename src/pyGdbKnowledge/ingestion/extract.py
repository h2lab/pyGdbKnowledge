# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Text and structure extraction from PDF files (PyMuPDF, behind the ``pdf`` extra).

PDF files are untrusted input: parsing is bounded in pages and characters, encrypted files
are refused, and control characters are removed from the extracted text.
"""

import importlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import IngestionSettings

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BLANKS_RE = re.compile(r"\n{3,}")
_DIGITS_RE = re.compile(r"\d+")
_CAPTION_RE = re.compile(r"^(table|figure)\s+\d+[.:]", re.IGNORECASE)
_HEADER_LINES = 4
_FOOTER_LINES = 3
_MIN_PAGES_FOR_HEADERS = 4
_HEADER_SHARE = 0.3


class ExtractionError(Exception):
    """Raised when a PDF cannot be parsed or exceeds the configured bounds."""


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Cleaned text of one page (numbered from 1)."""

    number: int
    text: str


@dataclass(frozen=True, slots=True)
class ExtractedSection:
    """A section derived from the PDF outline; ``path`` joins the titles of its ancestors."""

    ordinal: int
    level: int
    title: str
    path: str
    page_start: int
    page_end: int


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Result of a PDF extraction."""

    title: str | None
    page_count: int
    pages: tuple[ExtractedPage, ...]
    sections: tuple[ExtractedSection, ...]


def clean_text(text: str) -> str:
    """Remove control characters, trailing blanks and runs of empty lines."""
    text = _CONTROL_RE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return _BLANKS_RE.sub("\n\n", text).strip()


def _shape(line: str) -> str:
    return _DIGITS_RE.sub("#", line.strip())


def strip_running_headers(pages: list[list[str]]) -> list[list[str]]:
    """Drop page headers and footers repeated on many pages.

    Lines at the top or bottom of a page are compared with digits masked (so page numbers
    and revision counters match); a shape found on at least 30% of the pages is removed.

    Parameters
    ----------
    pages : list[list[str]]
        Lines of each page.

    Returns
    -------
    list[list[str]]
        Lines of each page without the running headers and footers.
    """
    if len(pages) < _MIN_PAGES_FOR_HEADERS:
        return pages
    counts: Counter[str] = Counter()
    for lines in pages:
        edge = lines[:_HEADER_LINES] + lines[-_FOOTER_LINES:]
        counts.update({_shape(line) for line in edge if line.strip()})
    threshold = max(2, int(len(pages) * _HEADER_SHARE))
    repeated = {shape for shape, seen in counts.items() if seen >= threshold}
    result: list[list[str]] = []
    for lines in pages:
        last = len(lines) - _FOOTER_LINES
        result.append(
            [
                line
                for index, line in enumerate(lines)
                if not ((index < _HEADER_LINES or index >= last) and _shape(line) in repeated)
            ]
        )
    return result


def build_sections(outline: list[tuple[int, str, int]], page_count: int) -> list[ExtractedSection]:
    """Turn a PDF outline into sections with hierarchical paths and page ranges.

    Table and figure bookmarks are not sections and are skipped. A section ends on the page
    where the next section starts. Without any usable outline, one section covers the document.

    Parameters
    ----------
    outline : list[tuple[int, str, int]]
        ``(level, title, page)`` entries in reading order, levels starting at 1.
    page_count : int
        Number of pages of the document.

    Returns
    -------
    list[ExtractedSection]
        Sections in reading order.
    """
    entries = [
        (level, " ".join(title.split()), min(max(page, 1), page_count))
        for level, title, page in outline
        if title.strip() and not _CAPTION_RE.match(title.strip()) and page >= 1
    ]
    if not entries:
        return [ExtractedSection(0, 1, "Document", "Document", 1, page_count)]
    sections: list[ExtractedSection] = []
    ancestors: list[tuple[int, str]] = []
    for ordinal, (level, title, page) in enumerate(entries):
        while ancestors and ancestors[-1][0] >= level:
            ancestors.pop()
        ancestors.append((level, title))
        following = entries[ordinal + 1][2] if ordinal + 1 < len(entries) else page_count
        sections.append(
            ExtractedSection(
                ordinal=ordinal,
                level=level,
                title=title,
                path=" > ".join(name for _, name in ancestors),
                page_start=page,
                page_end=max(page, following),
            )
        )
    return sections


def extract_pdf(path: Path, settings: IngestionSettings) -> ExtractedDocument:
    """Extract page text and outline sections from a PDF.

    Parameters
    ----------
    path : Path
        PDF file stored in the workspace.
    settings : IngestionSettings
        Page and character bounds.

    Returns
    -------
    ExtractedDocument
        Cleaned page texts and outline sections.

    Raises
    ------
    ExtractionError
        If PyMuPDF is missing, the file is unreadable, encrypted or beyond the bounds.
    """
    try:
        pymupdf: Any = importlib.import_module("pymupdf")
    except ImportError as exc:
        raise ExtractionError("PyMuPDF is not installed; install the 'pdf' extra") from exc
    try:
        document = pymupdf.open(path)
    except Exception as exc:  # the library raises many unrelated types on malformed input
        raise ExtractionError(f"cannot open PDF: {exc}") from exc
    try:
        if document.needs_pass or document.is_encrypted:
            raise ExtractionError("encrypted PDF files are not supported")
        page_count = int(document.page_count)
        if page_count < 1:
            raise ExtractionError("the PDF has no pages")
        if page_count > settings.max_pages:
            raise ExtractionError(f"the PDF has more than {settings.max_pages} pages")
        raw_pages: list[list[str]] = []
        total = 0
        for index in range(page_count):
            try:
                text = clean_text(document[index].get_text())
            except Exception as exc:
                raise ExtractionError(f"cannot read page {index + 1}: {exc}") from exc
            total += len(text)
            if total > settings.max_chars:
                raise ExtractionError("the extracted text exceeds the configured size limit")
            raw_pages.append(text.splitlines())
        try:
            outline = [(int(lv), str(title), int(pg)) for lv, title, pg in document.get_toc()]
        except Exception:
            outline = []
        title = (document.metadata or {}).get("title") or None
    finally:
        document.close()
    pages = tuple(
        ExtractedPage(number, clean_text("\n".join(lines)))
        for number, lines in enumerate(strip_running_headers(raw_pages), start=1)
    )
    return ExtractedDocument(
        title=clean_text(title) if title else None,
        page_count=page_count,
        pages=pages,
        sections=tuple(build_sections(outline, page_count)),
    )
