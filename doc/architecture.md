<!--
SPDX-FileCopyrightText: 2026 H2Lab Development Team
SPDX-License-Identifier: Apache-2.0
-->

# Architecture

pyGdbKnowledge is a standalone knowledge server (hybrid RAG) queried by pyGdbToolkit over
HTTP. It owns its documents, ingestion pipeline, indexes and API. The agent in pyGdbToolkit
only calls it through read-only tools. Ollama (or any LLM backend) stays a separate service
used for inference by the agent, not by this server.

## Responsibilities

- Serve documentary search results (datasheets, reference manuals, errata, application notes,
  SDK/HAL guides) with full provenance.
- Ingest and index documents through an administrative channel, separate from the read API.
- Never access GDB, OpenOCD, target hardware or approval decisions.

## Components

```text
agent tools (pyGdbToolkit) --HTTP/JSON--> Knowledge API (FastAPI)
                                              |
                       +----------------------+----------------------+
                       |                      |                      |
                 SQLite + FTS5             Qdrant             workspace files
              catalog, exact search     vector search        original PDFs, cache
```

| Component | Choice | Role |
| --- | --- | --- |
| API | FastAPI + Uvicorn | Versioned REST API (`/api/v1`) |
| Catalog / exact search | SQLite + FTS5 | Metadata, register names, addresses, error codes |
| Vector search | Qdrant (separate container) | Semantic similarity with metadata filters |
| Embeddings | Local model (sentence-transformers or Ollama) | Passage and query vectors |
| PDF | pypdf (BSD, no AGPL dependency); optional OCR and advanced layout parsing | Text, pages, sections, tables |

No Kubernetes, Kafka, Elasticsearch or PostgreSQL: SQLite/FTS5 is enough for the catalog and
exact search.

## Daemon and workspace

The service runs as an unprivileged daemon with a persistent workspace, separate from code and
configuration. Original PDFs are plain files; SQLite and Qdrant are derived indexes that can
be rebuilt from them.

```text
<workspace>/
├── documents/      original PDFs (named by hash)
├── extracted/      extracted text and pages (cache)
├── quarantine/     untrusted documents awaiting validation
├── qdrant/         Qdrant storage if embedded
├── catalog.db      SQLite: catalog + FTS5
└── tmp/            uploads in progress
```

The service refuses to run as root, writes only inside the workspace, and validates every
user-supplied path against it. It is meant to run in a container with the workspace as a
volume, configuration from file and environment, and logs on stdout.

## Ingestion pipeline

```text
local file -> validation + SHA-256 + duplicate/version detection
           -> extraction (text, pages, titles, tables; OCR if text is insufficient)
           -> cleaning, normalization, content checks, metadata
           -> technical chunking
           -> local embeddings
           -> Qdrant (vectors) + SQLite/FTS5 (exact text + catalog)
```

- Documents are loaded explicitly (CLI or admin endpoint), never fetched by agent tools.
  `source_url` is provenance metadata only.
- Two formats are accepted: PDF and CMSIS-SVD (XML, stored as `<sha256>.svd`). SVD files are
  parsed defensively (DOCTYPE and entities refused, bounded size and depth) and chunked into one
  section per peripheral with registers, addresses, reset values and bit fields, using the same
  `Bits 7:4 NAME: ...` notation as datasheets. The PDF-only upload endpoint refuses SVD.
- `pyGdbKnowledge sync` (operator CLI only) reads `config/corpus.yaml`: sources are either an
  explicit URL list (ST documents) or a GitHub repository of SVD files, pinned to a commit.
  Downloads are HTTPS-only, limited to the manifest's allowed domains on every redirect hop,
  size-capped, throttled and verified against the manifest SHA-256 or, failing that, the hash
  recorded in `corpus.lock.json` on first download (trust on first use). Files then follow the
  normal import path (screening, audit trail with channel `sync`).
- Chunking follows document structure (sections, registers and bit fields, tables), keeps
  section path, page and peripheral context, limits overlap, and uses stable chunk ids.
- Unknown metadata stays null; values are never invented.

Minimal metadata: `document_id`, `title`, `manufacturer`, `device_family`, `device`,
`document_type`, `revision`, `peripheral`, `section`, `page`, `source_type`, `source_url`,
`indexed_at`, `content_hash`.

## Hybrid search

Vector search (natural-language questions), FTS5 (exact names such as `USART_ISR`, `TXE`,
addresses), and metadata filters (manufacturer, family, device, peripheral, document type,
revision) are fused (reciprocal rank fusion) and ranked, with a bonus by source reliability.
No LLM reranker until retrieval quality has been measured.

Source reliability: A official documentation, B official vendor support/forum, C community
(GitHub, third-party forums), D agent hypothesis (never produced by this server).

## API (V1)

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/v1/health` | Service health |
| GET | `/api/v1/stats` | Documents, chunks, indexes |
| POST | `/api/v1/search` | Hybrid search |
| GET | `/api/v1/documents/{id}` | Document metadata |
| GET | `/api/v1/documents/{id}/sections` | Available sections |
| GET | `/api/v1/documents/{id}/section/{section_id}` | Detailed section |
| POST | `/api/v1/admin/ingest` | Ingestion (admin only) |
| POST | `/api/v1/web/search` | Web search (later phase) |

Search request: `query`, optional `filters` (`manufacturer`, `device`, `peripheral`,
`document_type`), `limit`. Search result: `chunk_id`, `document_id`, `title`, `section`,
`page`, `score`, `source_type`, `text`, `url`, plus revision and indexing date when known.

## Security

Requirements follow the ANSSI guide on securing RAG systems (see
[AGENTS.md](../AGENTS.md)). Main points:

- PDF and web content is untrusted data, never instructions. Results are returned as data with
  provenance; the consumer must delimit them.
- Read-only API for agent tools; ingestion and administration use a separate credential.
- Server-side control of collections and filters; strict parameter validation; limits on
  result count and size; minimum relevance threshold.
- Rate limits, quotas, timeouts.
- Untrusted sources go to quarantine before indexing; every document lifecycle event is
  traced; documents can be removed quickly together with all derived data.
- Embeddings are never returned or logged by default.
- Qdrant is never exposed to clients or the public network; the service stays on the internal
  network. Token authentication, TLS when traversing an untrusted network.
- Correlation id per request and ingestion job; logs treated as sensitive.
