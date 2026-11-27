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
from silicore_knowledge.storage.models import Document, SearchFilters, SourceType

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
    assert [(c.section_id, c.text, c.page) for c in result.chunks] == [
        ("s0000", "alpha text", 1),
        ("s0001", "beta text", 1),
        ("s0001", "more beta", 2),
    ]


def test_heading_split_over_two_lines_is_found() -> None:
    """A number and a title on separate lines still match the outline entry."""
    document = _doc(
        ["3.2.1\nInstruction cache\nbody line"],
        [(1, "3.2.1 Instruction cache", "3.2.1 Instruction cache", 1, 1)],
    )
    (chunk,) = chunk_document("ds", document, SETTINGS).chunks
    assert chunk.text == "body line"


def test_heading_number_at_page_start_is_not_assigned_to_previous_section() -> None:
    """Keep a page-leading chapter number with the heading that follows it."""
    document = _doc(
        [
            "Previous heading\nPrevious body",
            "5\nSpecific constraints of MCUs\nwithout memory abstraction\nNew body",
        ],
        [
            (1, "Previous heading", "Previous heading", 1, 2),
            (
                1,
                "Specific constraints of MCUs without memory abstraction",
                "Specific constraints of MCUs without memory abstraction",
                2,
                2,
            ),
        ],
    )
    chunks = chunk_document("ds", document, SETTINGS).chunks
    assert [(chunk.section_id, chunk.text, chunk.page) for chunk in chunks] == [
        ("s0000", "Previous body", 1),
        ("s0001", "New body", 2),
    ]


def test_number_before_heading_inside_page_remains_previous_content() -> None:
    """Do not treat ordinary numeric content inside a page as a heading number."""
    document = _doc(
        ["Previous heading\nPrevious body\n5\nNew heading\nNew body"],
        [
            (1, "Previous heading", "Previous heading", 1, 1),
            (1, "New heading", "New heading", 1, 1),
        ],
    )
    chunks = chunk_document("ds", document, SETTINGS).chunks
    assert [(chunk.section_id, chunk.text) for chunk in chunks] == [
        ("s0000", "Previous body\n5"),
        ("s0001", "New body"),
    ]


@pytest.mark.parametrize(("label", "is_heading"), [("A", True), ("B", False)])
def test_page_leading_appendix_label_matches_outline(label: str, is_heading: bool) -> None:
    """Treat an appendix letter as a heading only when it matches the outline."""
    document = _doc(
        ["Previous heading\nPrevious body", f"{label}\nAppendix A - Details\nNew body"],
        [
            (1, "Previous heading", "Previous heading", 1, 2),
            (1, "Appendix A - Details", "Appendix A - Details", 2, 2),
        ],
    )
    chunks = chunk_document("ds", document, SETTINGS).chunks
    expected = [("s0000", "Previous body", 1)]
    if not is_heading:
        expected.append(("s0000", label, 2))
    expected.append(("s0001", "New body", 2))
    assert [(chunk.section_id, chunk.text, chunk.page) for chunk in chunks] == expected


@pytest.mark.parametrize(
    ("reference", "is_heading"),
    [
        ("A Comparative table of vendor JTAG implementations 24", True),
        ("A Comparative table of vendor JTAG implementations . . . 24", True),
        ("B Comparative table of vendor JTAG implementations 24", False),
        ("A Another heading 24", False),
        ("A Comparative table of vendor JTAG implementations", False),
    ],
)
def test_page_leading_label_matches_earlier_heading_reference(
    reference: str, is_heading: bool
) -> None:
    """A heading reference confirms a page-leading letter omitted by the PDF outline."""
    title = "Comparative table of vendor JTAG implementations"
    document = _doc(
        [
            f"Contents\n{reference}",
            "Previous heading\nPrevious body",
            "A\nComparative table of vendor\nJTAG implementations\nNew body",
        ],
        [(1, "Previous heading", "Previous heading", 2, 3), (1, title, title, 3, 3)],
    )
    chunks = chunk_document("ds", document, SETTINGS).chunks
    expected = [("s0000", "Previous body", 2)]
    if not is_heading:
        expected.append(("s0000", "A", 3))
    expected.append(("s0001", "New body", 3))
    assert [(chunk.section_id, chunk.text, chunk.page) for chunk in chunks] == expected


def test_later_heading_reference_does_not_consume_previous_content() -> None:
    """A reference after the heading cannot confirm its preceding letter."""
    title = "Example heading"
    document = _doc(
        [
            "Previous heading\nPrevious body",
            f"A\n{title}\nNew body",
            f"Index\nA {title} 5",
        ],
        [(1, "Previous heading", "Previous heading", 1, 2), (1, title, title, 2, 3)],
    )
    chunks = chunk_document("ds", document, SETTINGS).chunks
    assert [(chunk.section_id, chunk.text, chunk.page) for chunk in chunks[:3]] == [
        ("s0000", "Previous body", 1),
        ("s0000", "A", 2),
        ("s0001", "New body", 2),
    ]


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


@pytest.mark.parametrize(
    "line",
    [
        " ".join(f"word{i:03}" for i in range(100)),
        "x" * 800,
        "Bit 7 FIELD: " + "y" * 800,
    ],
)
def test_oversize_line_is_split_without_losing_text(line: str) -> None:
    """A single extracted line cannot bypass the configured chunk size."""
    settings = IngestionSettings(chunk_max_chars=300, chunk_overlap_chars=0)
    document = _doc([line], [(1, "P", "P", 1, 1)])
    chunks = chunk_document("ds", document, settings).chunks
    assert len(chunks) > 1
    assert all(len(chunk.text) <= settings.chunk_max_chars for chunk in chunks)
    assert "".join(chunk.text.replace(" ", "") for chunk in chunks) == line.replace(" ", "")
    assert all(chunk.page == 1 and chunk.section_path == "P" for chunk in chunks)


def test_oversize_line_prefers_word_boundaries_with_overlap() -> None:
    """Splitting long prose keeps words intact and works with preceding and following lines."""
    words = [f"word{i:03}" for i in range(100)]
    document = _doc(
        ["before\n" + " ".join(words) + "\nafter"],
        [(1, "P", "P", 1, 1)],
    )
    chunks = chunk_document("ds", document, SETTINGS).chunks
    assert all(len(chunk.text) <= SETTINGS.chunk_max_chars for chunk in chunks)
    assert set(words).issubset({word for chunk in chunks for word in chunk.text.split()})
    assert chunks[0].text.startswith("before")
    assert chunks[-1].text.endswith("after")


def test_line_at_size_limit_remains_intact() -> None:
    """A line that already fits is not split."""
    document = _doc(["x" * 300], [(1, "P", "P", 1, 1)])
    (chunk,) = chunk_document("ds", document, SETTINGS).chunks
    assert chunk.text == "x" * 300


def test_chunks_preserve_pages_and_do_not_overlap_across_page_boundaries(tmp_path: Path) -> None:
    """A search hit cites the page that actually contains its text."""
    first_page = "\n".join(f"First page line {i:02} " + "x" * 40 for i in range(8))
    second_page = "\n".join(f"Second page line {i:02} " + "y" * 40 for i in range(5))
    document = _doc([first_page, second_page], [(1, "P", "P", 1, 2)])
    result = chunk_document("ds", document, SETTINGS)
    first = [chunk for chunk in result.chunks if chunk.page == 1]
    second = [chunk for chunk in result.chunks if chunk.page == 2]
    assert len(first) > 1 and second
    assert first[0].text.splitlines()[-1] == first[1].text.splitlines()[0]
    assert all("Second page" not in chunk.text for chunk in first)
    assert all("First page" not in chunk.text for chunk in second)

    catalog = Catalog(tmp_path / "c.db")
    catalog.add_document(
        Document(
            document_id="ds", title="T", content_hash="a" * 64, source_type=SourceType.OFFICIAL
        )
    )
    catalog.replace_content("ds", list(result.sections), list(result.chunks))
    hits = catalog.search_chunks('"second"', SearchFilters(), 5)
    assert hits and all(hit.chunk.page == 2 for hit in hits)


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
        ("7.6 Floating-point environment <fenv.h> > 7.6.1 The FENV_ACCESS pragma", None),
        ("7.17 Atomics <stdatomic.h> > 7.17.2.1 The ATOMIC_VAR_INIT macro", None),
        ("Macro ATOMIC_VAR_INIT", None),
        ("PRAGMA FENV_ACCESS", None),
        ("Control register (USART_CR1) > The ATOMIC_VAR_INIT macro", "USART"),
        ("ATOMIC_VAR_INIT macro controlling register USART_CR1", "USART"),
        ("1 Introduction", None),
        ("Notes:", None),
    ],
)
def test_infer_peripheral(path: str, peripheral: str | None) -> None:
    """Peripheral names come from register names or parenthesized acronyms only."""
    assert infer_peripheral(path) == peripheral


def test_software_macro_is_not_indexed_as_a_peripheral(tmp_path: Path) -> None:
    """A C macro remains searchable without becoming a hardware filter value."""
    title = "The ATOMIC_VAR_INIT macro"
    document = _doc(
        [f"{title}\nATOMIC_VAR_INIT initializes an atomic object."], [(1, title, title, 1, 1)]
    )
    result = chunk_document("c11", document, SETTINGS)
    assert result.chunks[0].peripheral is None

    catalog = Catalog(tmp_path / "c.db")
    catalog.add_document(
        Document(
            document_id="c11", title="C11", content_hash="a" * 64, source_type=SourceType.OFFICIAL
        )
    )
    catalog.replace_content("c11", list(result.sections), list(result.chunks))
    assert catalog.search_chunks('"ATOMIC_VAR_INIT"', SearchFilters(), 5)
    assert not catalog.search_chunks('"ATOMIC_VAR_INIT"', SearchFilters(peripheral="ATOMIC"), 5)


def test_overlap_setting_is_bounded() -> None:
    """An overlap above half the chunk size is rejected."""
    with pytest.raises(ValidationError, match="overlap"):
        IngestionSettings(chunk_max_chars=300, chunk_overlap_chars=200)
