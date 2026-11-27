# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Split extracted documents into sections and retrievable chunks.

Chunks never cross a section or page boundary and carry their section path and page. Bit-field
descriptions (``Bits 7:4 NAME: ...``) and table-like runs of short lines are kept whole when
they fit on a page, because splitting them makes register documentation unusable.
"""

import hashlib
import re
from dataclasses import dataclass
from itertools import groupby

from ..config import IngestionSettings
from ..storage.models import Chunk, Section
from .extract import ExtractedDocument, ExtractedSection

_BITFIELD_RE = re.compile(r"^Bits?\s+\d+(:\d+)?\b")
_REGISTER_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,11})_[A-Z0-9_]+\b")
_ACRONYM_RE = re.compile(r"\(([A-Z][A-Z0-9]{1,11})\)")
_SOFTWARE_BEFORE_RE = re.compile(r"\b(?:macro|pragma)\s+$", re.IGNORECASE)
_SOFTWARE_AFTER_RE = re.compile(r"\s+(?:macro|pragma)\b", re.IGNORECASE)
_SHORT_LINE = 30
_MIN_TABLE_LINES = 6
_MIN_PARTIAL_HEADING = 8


@dataclass(frozen=True, slots=True)
class ChunkedDocument:
    """Sections and chunks ready for ``Catalog.replace_content``."""

    sections: tuple[Section, ...]
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True, slots=True)
class _Line:
    page: int
    text: str


@dataclass(frozen=True, slots=True)
class _Block:
    lines: tuple[_Line, ...]
    splittable_with_overlap: bool

    @property
    def size(self) -> int:
        return sum(len(line.text) + 1 for line in self.lines)


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _find_heading(
    lines: list[_Line],
    start: int,
    page: int,
    title: str,
    references: dict[tuple[str, str], int],
) -> tuple[int, int] | None:
    """Locate a heading on its start page; return ``(index, line_count)``."""
    wanted = _norm(title)
    for index in range(start, len(lines)):
        if lines[index].page < page:
            continue
        if lines[index].page > page:
            return None
        one = _norm(lines[index].text)
        if one == wanted:
            count = 1
        elif (
            index + 1 < len(lines)
            and lines[index + 1].page == page
            and _norm(f"{lines[index].text} {lines[index + 1].text}") == wanted
        ):
            count = 2
        elif len(one) >= _MIN_PARTIAL_HEADING and wanted.startswith(one) and one != wanted:
            count = 1
        else:
            continue
        prefix = lines[index - 1].text.strip() if index > start else ""
        if (
            index > start
            and lines[index - 1].page == page
            and (index < 2 or lines[index - 2].page != page)
            and (
                prefix.isdecimal()
                or (
                    len(prefix) == 1
                    and "A" <= prefix <= "Z"
                    and (
                        re.match(rf"appendix {prefix.lower()}\b", wanted) is not None
                        or references.get((prefix, wanted), page) < page
                    )
                )
            )
        ):
            return index - 1, count + 1
        return index, count
    return None


def _split_sections(
    document: ExtractedDocument,
) -> list[tuple[ExtractedSection, list[_Line]]]:
    """Assign each line of the document to the section whose heading precedes it."""
    lines = [
        _Line(page.number, text)
        for page in document.pages
        for text in page.text.splitlines()
        if text.strip()
    ]
    titles = {_norm(section.title) for section in document.sections}
    references: dict[tuple[str, str], int] = {}
    for line in lines:
        words = line.text.split()
        if (
            len(words) < 3
            or len(words[0]) != 1
            or not "A" <= words[0] <= "Z"
            or not words[-1].isdecimal()
        ):
            continue
        title_words = words[1:-1]
        while title_words and set(title_words[-1]) == {"."}:
            title_words.pop()
        title = _norm(" ".join(title_words))
        if title in titles:
            key = (words[0], title)
            references[key] = min(line.page, references.get(key, line.page))
    starts: list[tuple[int, int]] = []
    cursor = 0
    for section in document.sections:
        found = _find_heading(lines, cursor, section.page_start, section.title, references)
        if found is not None:
            begin, count = found
            starts.append((begin + count, begin))
            cursor = begin + count
        else:
            first = next(
                (i for i in range(cursor, len(lines)) if lines[i].page >= section.page_start),
                len(lines),
            )
            starts.append((first, first))
            cursor = first
    result: list[tuple[ExtractedSection, list[_Line]]] = []
    for index, section in enumerate(document.sections):
        body_start = starts[index][0]
        body_end = starts[index + 1][1] if index + 1 < len(starts) else len(lines)
        result.append((section, lines[body_start : max(body_start, body_end)]))
    return result


def _blocks(lines: list[_Line]) -> list[_Block]:
    """Group lines into bit-field, table-like and paragraph blocks."""
    short = [len(line.text.strip()) <= _SHORT_LINE for line in lines]
    table = [False] * len(lines)
    start = 0
    while start < len(lines):
        if not short[start]:
            start += 1
            continue
        end = start
        while end < len(lines) and short[end]:
            end += 1
        if end - start >= _MIN_TABLE_LINES:
            table[start:end] = [True] * (end - start)
        start = end

    blocks: list[_Block] = []
    current: list[_Line] = []
    kind = ""

    def flush() -> None:
        if current:
            blocks.append(_Block(tuple(current), kind == "paragraph"))
            current.clear()

    for index, line in enumerate(lines):
        if _BITFIELD_RE.match(line.text.strip()):
            new_kind, starts = "bitfield", True
        elif table[index]:
            new_kind, starts = "table", kind != "table"
        elif kind == "bitfield":
            # Continuation lines belong to the bit-field description above them.
            new_kind, starts = "bitfield", False
        else:
            new_kind, starts = "paragraph", kind != "paragraph"
        if starts:
            flush()
            kind = new_kind
        current.append(line)
    flush()
    return blocks


def _pack(blocks: list[_Block], max_chars: int, overlap: int) -> list[list[_Line]]:
    """Pack blocks into chunks of at most ``max_chars``; split oversize blocks by line."""
    chunks: list[list[_Line]] = []
    current: list[_Line] = []
    size = 0
    fresh = False  # whether ``current`` holds lines beyond the carried-over overlap
    overlap_ok = False

    for block in blocks:
        for part in _split_block(block, max_chars, overlap):
            part_size = _size(part)
            if fresh and size + part_size > max_chars:
                chunks.append(current)
                current = _tail(current, overlap) if overlap_ok else []
                size = _size(current)
                fresh = False
                if size + part_size > max_chars:
                    current, size = [], 0
            current.extend(part)
            size += part_size
            fresh = True
            overlap_ok = block.splittable_with_overlap
    if fresh:
        chunks.append(current)
    return chunks


def _size(lines: list[_Line]) -> int:
    return sum(len(line.text) + 1 for line in lines)


def _tail(lines: list[_Line], budget: int) -> list[_Line]:
    """Return the last lines fitting in ``budget`` characters."""
    kept: list[_Line] = []
    total = 0
    for line in reversed(lines):
        total += len(line.text) + 1
        if total > budget:
            break
        kept.append(line)
    return kept[::-1]


def _split_block(block: _Block, max_chars: int, overlap: int) -> list[list[_Line]]:
    """Split an oversize block by line, and exceptionally long lines by character.

    Parts leave room for the overlap carried over from the previous chunk.
    """
    if block.size <= max_chars:
        return [list(block.lines)]
    limit = max_chars - overlap
    parts: list[list[_Line]] = []
    piece: list[_Line] = []
    lines: list[_Line] = []
    for line in block.lines:
        if len(line.text) <= max_chars:
            lines.append(line)
            continue
        start = 0
        while start < len(line.text):
            end = min(start + limit - 1, len(line.text))
            if end < len(line.text):
                space = line.text.rfind(" ", start + 1, end)
                if space != -1:
                    end = space + 1
            text = line.text[start:end]
            if text.strip():
                lines.append(_Line(line.page, text))
            start = end
    for line in lines:
        if piece and _size(piece) + len(line.text) + 1 > limit:
            parts.append(piece)
            piece = []
        piece.append(line)
    if piece:
        parts.append(piece)
    return parts


def infer_peripheral(path: str) -> str | None:
    """Infer a peripheral name from a section path, or None when it is not evident.

    A register name such as ``USART_CR1`` gives ``USART``, except when directly preceded or
    followed by ``macro`` or ``pragma``; otherwise a parenthesized acronym such as ``(ICACHE)``
    is used. The deepest section wins.

    Parameters
    ----------
    path : str
        Section path, titles joined by `` > ``.

    Returns
    -------
    str | None
        Peripheral name.
    """
    titles = path.split(" > ")
    for title in reversed(titles):
        for match in _REGISTER_RE.finditer(title):
            if _SOFTWARE_BEFORE_RE.search(title[: match.start()]) or _SOFTWARE_AFTER_RE.match(
                title[match.end() :]
            ):
                continue
            return match.group(1)
    for title in reversed(titles):
        acronym_match = _ACRONYM_RE.search(title)
        if acronym_match:
            return acronym_match.group(1)
    return None


def chunk_document(
    document_id: str, document: ExtractedDocument, settings: IngestionSettings
) -> ChunkedDocument:
    """Build sections and chunks with stable identifiers.

    Section ids are ``s<ordinal>``; chunk ids are ``<document>:<section>:c<index>`` and chunk
    hashes are the SHA-256 of their text, so re-ingesting the same file yields the same ids.
    Text before the first outline entry (cover, table of contents) is not indexed.

    Parameters
    ----------
    document_id : str
        Catalog identifier of the document.
    document : ExtractedDocument
        Extraction result.
    settings : IngestionSettings
        Chunk size and overlap.

    Returns
    -------
    ChunkedDocument
        Sections and chunks in reading order.
    """
    sections: list[Section] = []
    chunks: list[Chunk] = []
    for extracted, lines in _split_sections(document):
        section_id = f"s{extracted.ordinal:04d}"
        sections.append(
            Section(
                document_id=document_id,
                section_id=section_id,
                title=extracted.title,
                path=extracted.path,
                ordinal=extracted.ordinal,
                page_start=extracted.page_start,
                page_end=extracted.page_end,
            )
        )
        peripheral = infer_peripheral(extracted.path)
        packed = [
            part
            for _, page_lines in groupby(lines, key=lambda line: line.page)
            for part in _pack(
                _blocks(list(page_lines)), settings.chunk_max_chars, settings.chunk_overlap_chars
            )
        ]
        for index, chunk_lines in enumerate(packed):
            text = "\n".join(line.text.strip() for line in chunk_lines)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:{section_id}:c{index:03d}",
                    document_id=document_id,
                    ordinal=len(chunks),
                    text=text,
                    content_hash=hashlib.sha256(text.encode()).hexdigest(),
                    section_id=section_id,
                    section_path=extracted.path,
                    peripheral=peripheral,
                    page=chunk_lines[0].page,
                )
            )
    return ChunkedDocument(tuple(sections), tuple(chunks))
