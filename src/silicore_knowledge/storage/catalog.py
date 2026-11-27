# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""SQLite catalog with an FTS5 index for exact-match search."""

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import (
    AuditEvent,
    CatalogStats,
    Chunk,
    Document,
    DocumentStatus,
    ScopedSearchFilters,
    SearchFilters,
    SearchHit,
    Section,
    SourceType,
)


class CatalogError(Exception):
    """Raised when the catalog is unusable (unsupported schema, missing FTS5)."""


class DuplicateDocumentError(CatalogError):
    """Raised when a document id or content hash is already cataloged."""


# Each entry migrates the schema from version index to index + 1.
_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE documents (
        document_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        content_hash TEXT NOT NULL UNIQUE,
        source_type TEXT NOT NULL,
        status TEXT NOT NULL,
        manufacturer TEXT,
        device_family TEXT,
        device TEXT,
        document_type TEXT,
        revision TEXT,
        source_url TEXT,
        page_count INTEGER,
        version INTEGER NOT NULL,
        indexed_at TEXT NOT NULL
    );
    CREATE INDEX documents_device ON documents (manufacturer, device);
    CREATE TABLE sections (
        document_id TEXT NOT NULL REFERENCES documents (document_id) ON DELETE CASCADE,
        section_id TEXT NOT NULL,
        title TEXT NOT NULL,
        path TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        page_start INTEGER,
        page_end INTEGER,
        PRIMARY KEY (document_id, section_id)
    );
    CREATE TABLE chunks (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents (document_id) ON DELETE CASCADE,
        section_id TEXT,
        ordinal INTEGER NOT NULL,
        text TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        section_path TEXT NOT NULL,
        peripheral TEXT,
        page INTEGER,
        FOREIGN KEY (document_id, section_id) REFERENCES sections (document_id, section_id)
            ON DELETE CASCADE
    );
    CREATE INDEX chunks_document ON chunks (document_id, ordinal);
    CREATE VIRTUAL TABLE chunks_fts USING fts5 (
        text, section_path, peripheral,
        content='chunks', content_rowid='rowid',
        tokenize="unicode61 tokenchars '_'"
    );
    CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts (rowid, text, section_path, peripheral)
        VALUES (new.rowid, new.text, new.section_path, new.peripheral);
    END;
    CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts (chunks_fts, rowid, text, section_path, peripheral)
        VALUES ('delete', old.rowid, old.text, old.section_path, old.peripheral);
    END;
    CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts (chunks_fts, rowid, text, section_path, peripheral)
        VALUES ('delete', old.rowid, old.text, old.section_path, old.peripheral);
        INSERT INTO chunks_fts (rowid, text, section_path, peripheral)
        VALUES (new.rowid, new.text, new.section_path, new.peripheral);
    END;
    """,
    """
    CREATE TABLE audit_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        occurred_at TEXT NOT NULL,
        event TEXT NOT NULL,
        document_id TEXT NOT NULL,
        content_hash TEXT,
        source_type TEXT,
        source_url TEXT,
        revision TEXT,
        channel TEXT,
        detail TEXT
    );
    CREATE INDEX audit_events_document ON audit_events (document_id, event_id);
    """,
    # Document keywords (generic retrieval terms): stored as JSON on the document and
    # denormalized on its chunks so the FTS index can match them.
    """
    ALTER TABLE documents ADD COLUMN keywords TEXT NOT NULL DEFAULT '';
    ALTER TABLE chunks ADD COLUMN doc_keywords TEXT NOT NULL DEFAULT '';
    DROP TRIGGER chunks_ai;
    DROP TRIGGER chunks_ad;
    DROP TRIGGER chunks_au;
    DROP TABLE chunks_fts;
    CREATE VIRTUAL TABLE chunks_fts USING fts5 (
        text, section_path, peripheral, doc_keywords,
        content='chunks', content_rowid='rowid',
        tokenize="unicode61 tokenchars '_'"
    );
    CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts (rowid, text, section_path, peripheral, doc_keywords)
        VALUES (new.rowid, new.text, new.section_path, new.peripheral, new.doc_keywords);
    END;
    CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts (chunks_fts, rowid, text, section_path, peripheral, doc_keywords)
        VALUES ('delete', old.rowid, old.text, old.section_path, old.peripheral,
                old.doc_keywords);
    END;
    CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts (chunks_fts, rowid, text, section_path, peripheral, doc_keywords)
        VALUES ('delete', old.rowid, old.text, old.section_path, old.peripheral,
                old.doc_keywords);
        INSERT INTO chunks_fts (rowid, text, section_path, peripheral, doc_keywords)
        VALUES (new.rowid, new.text, new.section_path, new.peripheral, new.doc_keywords);
    END;
    INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild');
    """,
)

SCHEMA_VERSION = len(_MIGRATIONS)

# BM25 weights for (text, section_path, peripheral, doc_keywords); tuned during corpus
# evaluation. Keywords are curated document-level signals: they help a query reach the
# document without overwhelming in-text matches.
_BM25_WEIGHTS = "1.0, 1.5, 0.5, 0.8"
_CHUNK_COLUMNS = (
    "chunk_id",
    "document_id",
    "section_id",
    "ordinal",
    "text",
    "content_hash",
    "section_path",
    "peripheral",
    "page",
)
_DOCUMENT_COLUMNS = (
    "source_type",
    "status",
    "manufacturer",
    "device_family",
    "device",
    "document_type",
    "revision",
    "source_url",
    "keywords",
    "page_count",
    "version",
    "indexed_at",
)


_SQL_BATCH = 500
_HIT_SELECT = (
    "SELECT c.chunk_id, c.document_id, c.section_id, c.ordinal, c.text,"
    " c.content_hash, c.section_path, c.peripheral, c.page,"
    " d.title AS d_title, d.content_hash AS d_content_hash, d.source_type,"
    " d.status, d.manufacturer, d.device_family, d.device, d.document_type,"
    " d.revision, d.source_url, d.keywords, d.page_count, d.version, d.indexed_at"
)


def _hit(data: dict[str, Any], score: float) -> SearchHit:
    document = _document(
        {
            **{k: data[k] for k in _DOCUMENT_COLUMNS},
            "document_id": data["document_id"],
            "title": data["d_title"],
            "content_hash": data["d_content_hash"],
        }
    )
    chunk = Chunk(**{k: data[k] for k in _CHUNK_COLUMNS})
    return SearchHit(chunk=chunk, document=document, score=score)


class Catalog:
    """Document catalog stored in a SQLite file.

    A connection is opened per operation, so the catalog is safe to share between threads.
    All statements are parameterized.

    Parameters
    ----------
    path : Path
        Database file, created and migrated when needed.

    Raises
    ------
    CatalogError
        If the database schema is newer than supported, or SQLite lacks FTS5.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=10.0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection

    def _migrate(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            current = int(db.execute("PRAGMA user_version").fetchone()[0])
            if current > SCHEMA_VERSION:
                raise CatalogError(
                    f"catalog schema {current} is newer than supported {SCHEMA_VERSION}"
                )
            for version in range(current, SCHEMA_VERSION):
                try:
                    db.executescript(f"BEGIN;{_MIGRATIONS[version]}")
                except sqlite3.OperationalError as exc:
                    db.rollback()
                    raise CatalogError(f"catalog migration {version + 1} failed: {exc}") from exc
                db.execute(f"PRAGMA user_version = {version + 1}")
                db.commit()

    def add_document(self, document: Document) -> Document:
        """Add a document; ``indexed_at`` is set to the current UTC time.

        Parameters
        ----------
        document : Document
            Document to add.

        Returns
        -------
        Document
            The stored document, with ``indexed_at`` filled in.

        Raises
        ------
        DuplicateDocumentError
            If the id or the content hash already exists.
        """
        stored = document.model_copy(update={"indexed_at": datetime.now(UTC)})
        assert stored.indexed_at is not None
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO documents (document_id, title, content_hash, source_type,"
                    " status, manufacturer, device_family, device, document_type, revision,"
                    " source_url, keywords, page_count, version, indexed_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        stored.document_id,
                        stored.title,
                        stored.content_hash,
                        stored.source_type.value,
                        stored.status.value,
                        stored.manufacturer,
                        stored.device_family,
                        stored.device,
                        stored.document_type,
                        stored.revision,
                        stored.source_url,
                        _dump_keywords(stored.keywords),
                        stored.page_count,
                        stored.version,
                        stored.indexed_at.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateDocumentError(
                f"document {document.document_id!r} or its content is already cataloged"
            ) from exc
        return stored

    def get_document(self, document_id: str) -> Document | None:
        """Return a document, or None when it does not exist."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return None if row is None else _document(row)

    def get_document_by_hash(self, content_hash: str) -> Document | None:
        """Return the document with this SHA-256 content hash, or None."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return None if row is None else _document(row)

    def chunk_count(self, document_id: str) -> int:
        """Return the number of chunks stored for a document."""
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchone()
        return int(row[0])

    def set_page_count(self, document_id: str, page_count: int) -> None:
        """Record the page count discovered at extraction time."""
        with self._connect() as db:
            db.execute(
                "UPDATE documents SET page_count = ? WHERE document_id = ?",
                (page_count, document_id),
            )

    def list_documents(self, limit: int = 50, offset: int = 0) -> list[Document]:
        """Return documents ordered by id, using bounded pagination."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM documents ORDER BY document_id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_document(row) for row in rows]

    def set_status(self, document_id: str, status: DocumentStatus) -> bool:
        """Change the lifecycle status of a document; return False if it is unknown."""
        with self._connect() as db:
            return (
                db.execute(
                    "UPDATE documents SET status = ? WHERE document_id = ?",
                    (status.value, document_id),
                ).rowcount
                == 1
            )

    def record_event(
        self,
        event: str,
        document_id: str,
        *,
        content_hash: str | None = None,
        source_type: str | None = None,
        source_url: str | None = None,
        revision: str | None = None,
        channel: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Append a lifecycle event to the audit trail (never holds document content)."""
        with self._connect() as db:
            db.execute(
                "INSERT INTO audit_events (occurred_at, event, document_id, content_hash,"
                " source_type, source_url, revision, channel, detail)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    event,
                    document_id,
                    content_hash,
                    source_type,
                    source_url,
                    revision,
                    channel,
                    detail,
                ),
            )

    def list_events(self, document_id: str | None = None, limit: int = 100) -> list[AuditEvent]:
        """Return the most recent audit events, oldest first, optionally for one document."""
        query = "SELECT * FROM audit_events"
        params: tuple[Any, ...] = ()
        if document_id is not None:
            query += " WHERE document_id = ?"
            params = (document_id,)
        with self._connect() as db:
            rows = db.execute(
                f"SELECT * FROM ({query} ORDER BY event_id DESC LIMIT ?) ORDER BY event_id",
                (*params, limit),
            ).fetchall()
        return [AuditEvent(**dict(row)) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        """Delete a document with its sections, chunks and FTS entries.

        Returns
        -------
        bool
            True if the document existed.
        """
        with self._connect() as db:
            return (
                db.execute("DELETE FROM documents WHERE document_id = ?", (document_id,)).rowcount
                == 1
            )

    def replace_content(
        self, document_id: str, sections: Sequence[Section], chunks: Sequence[Chunk]
    ) -> None:
        """Atomically replace all sections and chunks of a document.

        Parameters
        ----------
        document_id : str
            Existing document.
        sections : Sequence[Section]
            New sections.
        chunks : Sequence[Chunk]
            New chunks; their ``section_id`` must refer to ``sections`` or be None.

        Raises
        ------
        CatalogError
            If the document does not exist or the content is inconsistent; nothing changes.
        """
        if any(s.document_id != document_id for s in sections) or any(
            c.document_id != document_id for c in chunks
        ):
            raise CatalogError("section or chunk belongs to a different document")
        try:
            with self._connect() as db:
                row = db.execute(
                    "SELECT keywords FROM documents WHERE document_id = ?", (document_id,)
                ).fetchone()
                if row is None:
                    raise CatalogError(f"unknown document {document_id!r}")
                # Document keywords are denormalized on the chunks so FTS5 can match them.
                doc_keywords = " ".join(_load_keywords(row["keywords"]))
                db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                db.execute("DELETE FROM sections WHERE document_id = ?", (document_id,))
                db.executemany(
                    "INSERT INTO sections (document_id, section_id, title, path, ordinal,"
                    " page_start, page_end) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            s.document_id,
                            s.section_id,
                            s.title,
                            s.path,
                            s.ordinal,
                            s.page_start,
                            s.page_end,
                        )
                        for s in sections
                    ],
                )
                db.executemany(
                    "INSERT INTO chunks (chunk_id, document_id, section_id, ordinal, text,"
                    " content_hash, section_path, peripheral, page, doc_keywords)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            c.chunk_id,
                            c.document_id,
                            c.section_id,
                            c.ordinal,
                            c.text,
                            c.content_hash,
                            c.section_path,
                            c.peripheral,
                            c.page,
                            doc_keywords,
                        )
                        for c in chunks
                    ],
                )
        except sqlite3.IntegrityError as exc:
            raise CatalogError(f"inconsistent content for {document_id!r}: {exc}") from exc

    def list_sections(
        self, document_id: str, limit: int | None = None, offset: int = 0
    ) -> list[Section]:
        """Return the sections of a document in reading order, optionally paginated."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM sections WHERE document_id = ? ORDER BY ordinal LIMIT ? OFFSET ?",
                (document_id, -1 if limit is None else limit, offset),
            ).fetchall()
        return [Section(**dict(row)) for row in rows]

    def get_section(self, document_id: str, section_id: str) -> Section | None:
        """Return one section, or None when it does not exist."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM sections WHERE document_id = ? AND section_id = ?",
                (document_id, section_id),
            ).fetchone()
        return None if row is None else Section(**dict(row))

    def section_chunks(self, document_id: str, section_id: str) -> list[Chunk]:
        """Return the chunks of a section in reading order."""
        columns = ", ".join(_CHUNK_COLUMNS)
        with self._connect() as db:
            rows = db.execute(
                f"SELECT {columns} FROM chunks"
                " WHERE document_id = ? AND section_id = ? ORDER BY ordinal",
                (document_id, section_id),
            ).fetchall()
        return [Chunk(**dict(row)) for row in rows]

    def search_chunks(self, match: str, filters: SearchFilters, limit: int) -> list[SearchHit]:
        """Search chunks with an FTS5 expression, restricted to active documents.

        Parameters
        ----------
        match : str
            Trusted FTS5 expression, built by ``build_match_query``.
        filters : SearchFilters
            Metadata filters; values are bound as parameters.
        limit : int
            Maximum number of hits.

        Returns
        -------
        list[SearchHit]
            Hits ordered by decreasing relevance (negated BM25).
        """
        clauses = ["chunks_fts MATCH ?", "d.status = ?"]
        params: list[str | int] = [match, DocumentStatus.ACTIVE.value]
        if isinstance(filters, ScopedSearchFilters) and filters.allowed_document_ids is not None:
            if not filters.allowed_document_ids:
                return []
            marks = ", ".join("?" for _ in filters.allowed_document_ids)
            clauses.append(f"d.document_id IN ({marks})")
            params.extend(filters.allowed_document_ids)
        for column, value in (
            ("d.manufacturer", filters.manufacturer),
            ("d.device_family", filters.device_family),
            ("d.device", filters.device),
            ("c.peripheral", filters.peripheral),
        ):
            if value is not None:
                clauses.append(f"{column} = ? COLLATE NOCASE")
                params.append(value)
        if filters.document_type:
            marks = ", ".join("?" for _ in filters.document_type)
            clauses.append(f"LOWER(d.document_type) IN ({marks})")
            params.extend(value.lower() for value in filters.document_type)
        params.append(limit)
        sql = (
            f"{_HIT_SELECT}, bm25(chunks_fts, {_BM25_WEIGHTS}) AS rank"
            " FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid"
            " JOIN documents d ON d.document_id = c.document_id"
            f" WHERE {' AND '.join(clauses)} ORDER BY rank LIMIT ?"
        )
        with self._connect() as db:
            rows = db.execute(sql, params).fetchall()
        return [_hit(dict(row), -float(row["rank"])) for row in rows]

    def get_hits(
        self, chunk_ids: Sequence[str], filters: SearchFilters | None = None
    ) -> dict[str, SearchHit]:
        """Load chunks with their documents, keeping only active documents.

        Used to re-validate candidates coming from the vector index, which may be stale.

        Parameters
        ----------
        chunk_ids : Sequence[str]
            Chunk identifiers.

        Returns
        -------
        dict[str, SearchHit]
            Hits (score 0) by chunk id; unknown or quarantined chunks are absent.
        """
        unique = list(dict.fromkeys(chunk_ids))
        found: dict[str, SearchHit] = {}
        for start in range(0, len(unique), _SQL_BATCH):
            batch = unique[start : start + _SQL_BATCH]
            marks = ", ".join("?" for _ in batch)
            sql = (
                f"{_HIT_SELECT} FROM chunks c JOIN documents d ON d.document_id = c.document_id"
                f" WHERE d.status = ? AND c.chunk_id IN ({marks})"
            )
            with self._connect() as db:
                rows = db.execute(sql, [DocumentStatus.ACTIVE.value, *batch]).fetchall()
            for row in rows:
                hit = _hit(dict(row), 0.0)
                if (
                    isinstance(filters, ScopedSearchFilters)
                    and filters.allowed_document_ids is not None
                    and hit.document.document_id not in filters.allowed_document_ids
                ):
                    continue
                found[hit.chunk.chunk_id] = hit
        return found

    def stats(
        self, allowed_document_ids: tuple[str, ...] | None = None, *, active_only: bool = False
    ) -> CatalogStats:
        """Return catalog counters."""
        with self._connect() as db:
            if allowed_document_ids is None and not active_only:
                counts = {
                    table: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                    for table in ("documents", "sections", "chunks")
                }
            elif allowed_document_ids == ():
                counts = dict.fromkeys(("documents", "sections", "chunks"), 0)
            else:
                marks = ", ".join("?" for _ in allowed_document_ids or ())
                scope = (
                    f" AND d.document_id IN ({marks})" if allowed_document_ids is not None else ""
                )
                counts = {
                    table: int(
                        db.execute(
                            "SELECT COUNT(*) FROM "
                            + (
                                "documents d"
                                if table == "documents"
                                else f"{table} t JOIN documents d ON t.document_id = d.document_id"
                            )
                            + f" WHERE d.status = ?{scope}",
                            [DocumentStatus.ACTIVE.value, *(allowed_document_ids or ())],
                        ).fetchone()[0]
                    )
                    for table in ("documents", "sections", "chunks")
                }
        return CatalogStats(schema_version=SCHEMA_VERSION, **counts)


def _dump_keywords(keywords: tuple[str, ...]) -> str:
    return json.dumps(list(keywords)) if keywords else ""


def _load_keywords(stored: str | None) -> tuple[str, ...]:
    if not stored:
        return ()
    loaded = json.loads(stored)
    return tuple(str(keyword) for keyword in loaded)


def _document(row: sqlite3.Row | dict[str, Any]) -> Document:
    data = dict(row)
    data["source_type"] = SourceType(data["source_type"])
    data["status"] = DocumentStatus(data["status"])
    data["keywords"] = _load_keywords(data.get("keywords"))
    return Document(**data)
