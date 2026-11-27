# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the indexing pipeline and the indexing commands."""

from pathlib import Path

import pytest
from pdfmaker import make_pdf

from silicore_knowledge import cli
from silicore_knowledge.config import Settings
from silicore_knowledge.ingestion.importer import DocumentMetadata, import_pdf
from silicore_knowledge.ingestion.pipeline import Indexer, IndexingError
from silicore_knowledge.retrieval.embeddings import EmbeddingError, HashingEmbedder
from silicore_knowledge.retrieval.vector_store import InMemoryVectorStore, VectorStoreError
from silicore_knowledge.storage.catalog import Catalog
from silicore_knowledge.storage.models import SearchFilters, SourceType
from silicore_knowledge.workspace import Workspace

META = DocumentMetadata(source_type=SourceType.OFFICIAL, manufacturer="ST", device="STM32U5")


class _FailingEmbedder(HashingEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("boom")


class _FailingStore(InMemoryVectorStore):
    def upsert(self, records: object) -> None:
        raise VectorStoreError("down")


def _pdf(path: Path, text: str = "USART_ISR flags and baud rate") -> Path:
    return make_pdf(path, [text], [[1, "1 USART", 1]])


@pytest.fixture
def env(tmp_path: Path) -> tuple[Settings, Workspace, Catalog, str]:
    """Provide settings, workspace, catalog and the id of one imported PDF."""
    settings = Settings(workspace=tmp_path / "ws")
    workspace = Workspace(settings.workspace)
    catalog = Catalog(workspace.catalog_path)
    result = import_pdf(workspace, catalog, _pdf(tmp_path / "a.pdf"), META, 10_000_000)
    return settings, workspace, catalog, result.document.document_id


def test_index_fills_catalog_and_vectors(
    env: tuple[Settings, Workspace, Catalog, str],
) -> None:
    """Chunks are searchable lexically and stored as vectors."""
    settings, workspace, catalog, doc_id = env
    store = InMemoryVectorStore(32)
    indexer = Indexer(settings, workspace, catalog, HashingEmbedder(32), store)
    result = indexer.index_document(doc_id)
    assert result.chunks == result.vectors == store.count() >= 1
    assert catalog.chunk_count(doc_id) == result.chunks
    document = catalog.get_document(doc_id)
    assert document is not None and document.page_count == 1
    assert catalog.search_chunks("USART_ISR", SearchFilters(), 5)


def test_reindex_is_idempotent_and_batch_skips_indexed(
    env: tuple[Settings, Workspace, Catalog, str],
) -> None:
    """Re-indexing does not duplicate; non-rebuild batches skip indexed documents."""
    settings, workspace, catalog, doc_id = env
    store = InMemoryVectorStore(32)
    indexer = Indexer(settings, workspace, catalog, HashingEmbedder(32), store)
    first = indexer.index_document(doc_id)
    indexer.index_document(doc_id)
    assert store.count() == first.chunks
    assert indexer.index_documents().indexed == []
    assert len(indexer.index_documents(rebuild=True).indexed) == 1


def test_lexical_only_indexing(env: tuple[Settings, Workspace, Catalog, str]) -> None:
    """Without embedder no vector is produced."""
    settings, workspace, catalog, doc_id = env
    indexer = Indexer(settings, workspace, catalog)
    assert not indexer.semantic
    result = indexer.index_document(doc_id)
    assert result.chunks >= 1 and result.vectors == 0


def test_embedding_failure_keeps_previous_index(
    env: tuple[Settings, Workspace, Catalog, str],
) -> None:
    """A failing embedder aborts before the catalog is touched."""
    settings, workspace, catalog, doc_id = env
    indexer = Indexer(settings, workspace, catalog, _FailingEmbedder(32), InMemoryVectorStore(32))
    report = indexer.index_documents([doc_id])
    assert "embedding failed" in report.failed[doc_id]
    assert catalog.chunk_count(doc_id) == 0


def test_vector_failure_is_reported(env: tuple[Settings, Workspace, Catalog, str]) -> None:
    """A vector store failure after the catalog write points to reindex."""
    settings, workspace, catalog, doc_id = env
    indexer = Indexer(settings, workspace, catalog, HashingEmbedder(32), _FailingStore(32))
    with pytest.raises(IndexingError, match="reindex"):
        indexer.index_document(doc_id)


def test_unknown_or_missing_file(env: tuple[Settings, Workspace, Catalog, str]) -> None:
    """Unknown documents and vanished PDFs are refused."""
    settings, workspace, catalog, doc_id = env
    indexer = Indexer(settings, workspace, catalog)
    with pytest.raises(IndexingError, match="unknown"):
        indexer.index_document("nope-00000000")
    document = catalog.get_document(doc_id)
    assert document is not None
    workspace.document_path(document.content_hash).unlink()
    with pytest.raises(IndexingError, match="missing"):
        indexer.index_document(doc_id)


def test_delete_removes_everything(env: tuple[Settings, Workspace, Catalog, str]) -> None:
    """Deletion clears catalog, vectors and the PDF unless asked otherwise."""
    settings, workspace, catalog, doc_id = env
    store = InMemoryVectorStore(32)
    indexer = Indexer(settings, workspace, catalog, HashingEmbedder(32), store)
    indexer.index_document(doc_id)
    document = catalog.get_document(doc_id)
    assert document is not None
    pdf = workspace.document_path(document.content_hash)
    assert indexer.delete_document(doc_id)
    assert store.count() == 0 and catalog.get_document(doc_id) is None and not pdf.exists()
    assert not indexer.delete_document(doc_id)


def test_cli_index_reindex_delete(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The commands ingest with --index, reindex and delete."""
    config = tmp_path / "s.yaml"
    config.write_text(f"workspace: {tmp_path / 'ws'}\n", encoding="utf-8")
    pdf = _pdf(tmp_path / "a.pdf")
    base = ["--config", str(config)]
    assert cli.main(["ingest", *base, "--source-type", "official", "--index", str(pdf)]) == 0
    out = capsys.readouterr().out
    assert "indexed:" in out
    doc_id = out.split("imported: ")[1].split()[0]
    assert cli.main(["index", *base]) == 0
    assert "indexed:" not in capsys.readouterr().out
    assert cli.main(["reindex", *base, doc_id]) == 0
    assert "indexed:" in capsys.readouterr().out
    assert cli.main(["delete", *base, doc_id]) == 0
    assert cli.main(["delete", *base, doc_id]) == 1
    assert "unknown document" in capsys.readouterr().err
