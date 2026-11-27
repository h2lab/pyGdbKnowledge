# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for SVD parsing, chunking, import and indexing."""

from pathlib import Path

import pytest

from silicore_knowledge.config import IngestionSettings, Settings
from silicore_knowledge.ingestion.importer import (
    DocumentMetadata,
    IngestError,
    import_document,
    import_pdf,
)
from silicore_knowledge.ingestion.pipeline import Indexer
from silicore_knowledge.ingestion.svd import SvdError, chunk_svd
from silicore_knowledge.retrieval.hybrid import HybridSearcher
from silicore_knowledge.storage.catalog import Catalog
from silicore_knowledge.storage.models import Document, SearchFilters, SourceType
from silicore_knowledge.workspace import Area, Workspace

SVD = """<?xml version="1.0" encoding="utf-8"?>
<device schemaVersion="1.3">
  <name>STM32TEST</name>
  <size>32</size>
  <access>read-write</access>
  <peripherals>
    <peripheral>
      <name>USART1</name>
      <description>Universal   synchronous
        receiver</description>
      <baseAddress>0x40011000</baseAddress>
      <interrupt><name>USART1</name><value>37</value></interrupt>
      <registers>
        <register>
          <name>CR1</name>
          <description>Control register 1</description>
          <addressOffset>0x0</addressOffset>
          <resetValue>0x0</resetValue>
          <fields>
            <field>
              <name>UE</name><description>USART enable</description>
              <bitOffset>0</bitOffset><bitWidth>1</bitWidth>
            </field>
            <field>
              <name>M</name><description>Word length</description>
              <bitRange>[13:12]</bitRange>
              <enumeratedValues>
                <enumeratedValue><name>Bits8</name><value>0</value></enumeratedValue>
              </enumeratedValues>
            </field>
          </fields>
        </register>
        <cluster>
          <name>CH</name><addressOffset>0x100</addressOffset>
          <register><name>CTL</name><addressOffset>0x4</addressOffset></register>
        </cluster>
      </registers>
    </peripheral>
    <peripheral derivedFrom="USART1">
      <name>USART2</name>
      <baseAddress>0x40004400</baseAddress>
    </peripheral>
  </peripherals>
</device>
"""
ENTITY = """<?xml version="1.0"?>
<!DOCTYPE device [<!ENTITY a "aaaa">]>
<device><name>X</name></device>
"""


@pytest.fixture
def svd_file(tmp_path: Path) -> Path:
    """Write a small SVD file."""
    path = tmp_path / "STM32TEST.svd"
    path.write_text(SVD, encoding="utf-8")
    return path


def test_chunking_describes_registers_and_fields(svd_file: Path) -> None:
    """Each peripheral is a section; registers carry addresses, fields and reset values."""
    chunked = chunk_svd("dev-1", svd_file, IngestionSettings())
    assert [s.title for s in chunked.sections] == ["USART1", "USART2"]
    first = chunked.chunks[0]
    assert first.peripheral == "USART1"
    assert first.section_path == "STM32TEST > USART1"
    assert "base address 0x40011000" in first.text
    assert "Universal synchronous receiver" in first.text
    assert "Interrupt USART1 = 37" in first.text
    assert "Register USART1_CR1 (offset 0x0, address 0x40011000, size 32" in first.text
    assert "reset 0x00000000" in first.text
    assert "Bits 0 UE (read-write): USART enable" in first.text
    assert "Bits 13:12 M (read-write): Word length [values: 0=Bits8]" in first.text
    assert "Register USART1_CH.CTL (offset 0x104, address 0x40011104" in first.text


def test_derived_peripheral_reuses_base_registers(svd_file: Path) -> None:
    """A derivedFrom peripheral gets the base registers at its own address."""
    chunked = chunk_svd("dev-1", svd_file, IngestionSettings())
    text = "\n".join(c.text for c in chunked.chunks if c.peripheral == "USART2")
    assert "Same registers as USART1" in text
    assert "Register USART2_CR1 (offset 0x0, address 0x40004400" in text


def test_large_peripheral_is_split_within_the_chunk_limit(tmp_path: Path) -> None:
    """Registers are packed into chunks no larger than the configured size."""
    registers = "".join(
        f"<register><name>R{i}</name><description>{'word ' * 20}</description>"
        f"<addressOffset>{i * 4}</addressOffset></register>"
        for i in range(60)
    )
    path = tmp_path / "big.svd"
    path.write_text(
        "<device><name>D</name><peripherals><peripheral><name>P</name>"
        f"<baseAddress>0</baseAddress><registers>{registers}</registers></peripheral>"
        "</peripherals></device>",
        encoding="utf-8",
    )
    settings = IngestionSettings(chunk_max_chars=800, chunk_overlap_chars=0)
    chunks = chunk_svd("big-1", path, settings).chunks
    assert len(chunks) > 3
    assert all(len(c.text) <= settings.chunk_max_chars for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


@pytest.mark.parametrize("limit", [200, 1500])
def test_oversized_bitfield_keeps_register_context_and_all_words(
    tmp_path: Path, limit: int
) -> None:
    """Long field descriptions are bounded without losing words or their register context."""
    words = [f"word{i:04}" for i in range(400)]
    field = (
        "<field><name>TCEM</name><bitOffset>30</bitOffset><bitWidth>2</bitWidth>"
        f"<description>{' '.join(words)}</description></field>"
    )
    path = tmp_path / "long-field.svd"
    path.write_text(SVD.replace("<fields>", f"<fields>{field}", 1), encoding="utf-8")
    settings = IngestionSettings(chunk_max_chars=limit, chunk_overlap_chars=0)
    chunked = chunk_svd("dev-1", path, settings)
    assert all(len(chunk.text) <= limit for chunk in chunked.chunks)
    matching = [
        chunk for chunk in chunked.chunks if "word" in chunk.text and chunk.peripheral == "USART1"
    ]
    assert len(matching) > 1
    assert all(
        "Register USART1_CR1" in chunk.text and "Bits 31:30 TCEM" in chunk.text
        for chunk in matching
    )
    assert [word for chunk in matching for word in chunk.text.split() if word in words] == words
    assert chunk_svd("dev-1", path, settings) == chunked
    catalog = Catalog(tmp_path / "c.db")
    catalog.add_document(
        Document(
            document_id="dev-1",
            title="STM32TEST",
            content_hash="a" * 64,
            source_type=SourceType.VENDOR,
        )
    )
    catalog.replace_content("dev-1", list(chunked.sections), list(chunked.chunks))
    hits = HybridSearcher(catalog).search(
        "USART1_CR1 word0399", SearchFilters(peripheral="USART1"), 5
    )
    assert any("word0399" in hit.chunk.text and "Bits 31:30 TCEM" in hit.chunk.text for hit in hits)


@pytest.mark.parametrize("element", ["peripheral", "register", "field"])
def test_oversized_svd_line_keeps_unbroken_text(tmp_path: Path, element: str) -> None:
    """Very long descriptions are split even when they have no word boundaries."""
    marker = "Z" * 800
    snippets = {
        "peripheral": "<description>Universal   synchronous\n        receiver</description>",
        "register": "<description>Control register 1</description>",
        "field": "<description>USART enable</description>",
    }
    content = SVD.replace(snippets[element], f"<description>{marker}</description>", 1)
    path = tmp_path / f"long-{element}.svd"
    path.write_text(content, encoding="utf-8")
    settings = IngestionSettings(chunk_max_chars=200, chunk_overlap_chars=0)
    chunks = chunk_svd("dev-1", path, settings).chunks
    assert all(len(chunk.text) <= settings.chunk_max_chars for chunk in chunks)
    # USART2 inherits the USART1 description or registers.
    for peripheral in ("USART1", "USART2"):
        assert (
            sum(chunk.text.count("Z") for chunk in chunks if chunk.peripheral == peripheral) == 800
        )
        matching = [
            chunk for chunk in chunks if chunk.peripheral == peripheral and "Z" in chunk.text
        ]
        context = "Peripheral" if element == "peripheral" else f"Register {peripheral}_CR1"
        assert all(context in chunk.text for chunk in matching)


def test_oversized_register_name_does_not_prevent_bounded_chunks(tmp_path: Path) -> None:
    """An identifier too large to repeat still remains in the bounded output."""
    path = tmp_path / "long-register-name.svd"
    path.write_text(
        SVD.replace("<name>CR1</name>", f"<name>{'Z' * 800}</name>", 1), encoding="utf-8"
    )
    settings = IngestionSettings(chunk_max_chars=200, chunk_overlap_chars=0)
    chunks = chunk_svd("dev-1", path, settings).chunks
    assert all(len(chunk.text) <= settings.chunk_max_chars for chunk in chunks)
    for peripheral in ("USART1", "USART2"):
        assert (
            sum(chunk.text.count("Z") for chunk in chunks if chunk.peripheral == peripheral) == 800
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [(ENTITY, "DOCTYPE"), ("<device><name>", "invalid XML"), ("<other/>", "not an SVD")],
)
def test_unsafe_or_malformed_svd_is_refused(tmp_path: Path, content: str, message: str) -> None:
    """DTDs, broken XML and foreign documents are rejected."""
    path = tmp_path / "bad.svd"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SvdError, match=message):
        chunk_svd("bad-1", path, IngestionSettings())


def test_import_and_index_svd(svd_file: Path, tmp_path: Path) -> None:
    """An SVD file is stored as .svd, indexed, searchable by peripheral, and deletable."""
    settings = Settings(workspace=tmp_path / "ws")
    workspace = Workspace(settings.workspace)
    catalog = Catalog(workspace.catalog_path)
    meta = DocumentMetadata(
        source_type=SourceType.VENDOR, manufacturer="STMicroelectronics", document_type="svd"
    )
    document = import_document(workspace, catalog, svd_file, meta, 10_000_000).document
    stored = workspace.stored_document(document.content_hash)
    assert stored is not None and stored.suffix == ".svd"
    indexer = Indexer(settings, workspace, catalog)
    result = indexer.index_document(document.document_id)
    assert result.sections == 2 and result.chunks >= 2
    hits = HybridSearcher(catalog).search(
        "USART1_CR1 enable", SearchFilters(peripheral="usart1"), 5
    )
    assert hits and hits[0].chunk.peripheral == "USART1"
    assert indexer.delete_document(document.document_id)
    assert workspace.stored_document(document.content_hash) is None
    assert not workspace.document_path(document.content_hash, Area.QUARANTINE).exists()


def test_pdf_only_import_refuses_svd(svd_file: Path, tmp_path: Path) -> None:
    """The upload path (PDF only) does not accept SVD files."""
    workspace = Workspace(tmp_path / "ws")
    catalog = Catalog(workspace.catalog_path)
    meta = DocumentMetadata(source_type=SourceType.VENDOR)
    with pytest.raises(IngestError, match="unsupported file type"):
        import_pdf(workspace, catalog, svd_file, meta, 10_000_000)
