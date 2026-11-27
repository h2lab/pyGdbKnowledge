# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for technical chunking."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from silicore_knowledge.config import IngestionSettings
from silicore_knowledge.ingestion.chunking import chunk_document, infer_peripheral
from silicore_knowledge.ingestion.extract import (
    ExtractedDocument,
    ExtractedPage,
    ExtractedSection,
)
from silicore_knowledge.storage.catalog import Catalog
from silicore_knowledge.storage.models import Document, SourceType

SETTINGS = IngestionSettings(chunk_max_chars=300, chunk_overlap_chars=60)


def _doc(pages: list[str], sections: list[tuple[int, str, str, int, int]]) -> ExtractedDocument:
    return ExtractedDocument(
        title=None,
        page_count=len(pages),
        pages=tuple(ExtractedPage(n, text) for n, text in enumerate(pages, start=1)),
        sections=tuple(
            ExtractedSection(i, level, title, path, start, end)
            for i, (level, title, path, start, end) in enumerate(sections)
        ),
    )


def test_sections_split_at_headings_within_a_page() -> None:
    """Two sections starting on the same page share it without duplicating text."""
    document = _doc(
        ["Cover\n1 First\nalpha text\n2 Second\nbeta text", "more beta"],
        [(1, "1 First", "1 First", 1, 1), (1, "2 Second", "2 Second", 1, 2)],
    )
    result = chunk_document("ds", document, SETTINGS)
    texts = {c.section_id: c.text for c in result.chunks}
    assert texts == {"s0000": "alpha text", "s0001": "beta text\nmore beta"}
    assert "Cover" not in "".join(texts.values())
    assert [c.page for c in result.chunks] == [1, 1]


def test_heading_split_over_two_lines_is_found() -> None:
    """A number and a title on separate lines still match the outline entry."""
    document = _doc(
        ["3.2.1\nInstruction cache\nbody line"],
        [(1, "3.2.1 Instruction cache", "3.2.1 Instruction cache", 1, 1)],
    )
    (chunk,) = chunk_document("ds", document, SETTINGS).chunks
    assert chunk.text == "body line"


def test_missing_heading_falls_back_to_page_start() -> None:
    """When the heading is not in the text, the section starts at its page."""
    document = _doc(["p1 text", "p2 text"], [(1, "Ghost", "Ghost", 2, 2)])
    (chunk,) = chunk_document("ds", document, SETTINGS).chunks
    assert chunk.text == "p2 text" and chunk.page == 2


def test_bit_fields_stay_whole() -> None:
    """Each bit-field description is kept in one chunk when it fits."""
    fields = [
        f"Bit {i} TXE{i}: transmit flag number {i}\nSet by hardware when ready {'x' * 60}"
        for i in range(8)
    ]
    document = _doc(["\n".join(fields)], [(1, "R", "R", 1, 1)])
    chunks = chunk_document("ds", document, SETTINGS).chunks
    assert len(chunks) > 1
    for chunk in chunks:
        for line in chunk.text.splitlines():
            if line.startswith("Set by hardware"):
                previous = chunk.text.splitlines()[chunk.text.splitlines().index(line) - 1]
                assert previous.startswith("Bit ")


def test_table_like_runs_are_not_split_when_they_fit() -> None:
    """A run of short lines (a flattened table) stays in a single chunk."""
    table = "\n".join(f"R{i}" for i in range(20))
    prose = "\n".join(
        f"This is a long explanatory sentence number {i} about timing." for i in range(5)
    )
    document = _doc([f"{prose}\n{table}"], [(1, "T", "T", 1, 1)])
    chunks = chunk_document("ds", document, SETTINGS).chunks
    holding = [c for c in chunks if "R0" in c.text]
    assert len(holding) == 1 and "R19" in holding[0].text


def test_chunks_respect_size_and_overlap_for_prose() -> None:
    """Prose is cut under the limit with a bounded overlap between neighbours."""
    prose = "\n".join(f"Sentence {i:02d} " + "w" * 40 for i in range(30))
    document = _doc([prose], [(1, "P", "P", 1, 1)])
    chunks = chunk_document("ds", document, SETTINGS).chunks
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)
    first, second = chunks[0].text.splitlines(), chunks[1].text.splitlines()
    assert first[-1] == second[0]
    assert len(first[-1]) <= 60


def test_ids_are_stable_and_valid_for_the_catalog(tmp_path: Path) -> None:
    """Re-chunking gives identical ids and hashes, and the catalog accepts the result."""
    document = _doc(
        ["1 A\nalpha register USART_CR1", "2 B\nbeta"],
        [(1, "1 A", "1 A", 1, 1), (1, "2 B", "2 B", 2, 2)],
    )
    first = chunk_document("ds-1", document, SETTINGS)
    second = chunk_document("ds-1", document, SETTINGS)
    assert first == second
    assert [c.chunk_id for c in first.chunks] == ["ds-1:s0000:c000", "ds-1:s0001:c000"]
    assert [c.ordinal for c in first.chunks] == [0, 1]
    catalog = Catalog(tmp_path / "c.db")
    catalog.add_document(
        Document(
            document_id="ds-1", title="T", content_hash="a" * 64, source_type=SourceType.OFFICIAL
        )
    )
    catalog.replace_content("ds-1", list(first.sections), list(first.chunks))
    assert catalog.stats().chunks == 2


@pytest.mark.parametrize(
    ("path", "peripheral"),
    [
        ("30 USART > 30.7 Registers > 30.7.1 Control register 1 (USART_CR1)", "USART"),
        ("3 Overview > 3.2.1 Instruction cache (ICACHE)", "ICACHE"),
        ("1 Introduction", None),
        ("Notes:", None),
    ],
)
def test_infer_peripheral(path: str, peripheral: str | None) -> None:
    """Peripheral names come from register names or parenthesized acronyms only."""
    assert infer_peripheral(path) == peripheral


def test_overlap_setting_is_bounded() -> None:
    """An overlap above half the chunk size is rejected."""
    with pytest.raises(ValidationError, match="overlap"):
        IngestionSettings(chunk_max_chars=300, chunk_overlap_chars=200)
