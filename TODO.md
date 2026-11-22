<!--
SPDX-FileCopyrightText: 2026 H2Lab Development Team
SPDX-License-Identifier: Apache-2.0
-->

# TODO

Each step is one commit and must leave `tox` green (tests, lint, typecheck).
Reference: [doc/architecture.md](doc/architecture.md).
Development rules: [AGENTS.md](AGENTS.md), including the ANSSI RAG security guide (R1-R35). Everything in this repository is in English.

Scope: the server only. The pyGdbToolkit integration (`KnowledgeClient`, `--knowledge-url`,
tools, prompt) belongs to the other repository, once the API is stable.

## Operating principles

- The service is a daemon: it owns a **workspace** (persistent data directory) separate from
  code and configuration.
- Original PDFs are stored **as plain files** in the workspace, not in a database. SQLite
  (catalog + FTS5) and Qdrant are derived indexes that can be rebuilt from the files.
- Reduced privileges: dedicated unprivileged user, write access limited to the workspace, no
  access to GDB, OpenOCD, hardware or approvals. User-supplied paths are validated (no escape
  from the workspace). Qdrant is never exposed outside the internal network.
- Container-ready (Docker): configuration through file + environment variables, workspace
  mounted as a volume, no hard-coded paths, logs on stdout.
- Documents are data, not configuration: they go through neither `settings.yaml` nor the agent
  tools.

Workspace layout:

```text
<workspace>/
├── documents/      # original PDFs (named by hash)
├── extracted/      # extracted text/pages (cache)
├── quarantine/     # untrusted documents awaiting validation
├── qdrant/         # Qdrant storage if embedded
├── catalog.db      # SQLite: catalog + FTS5
└── tmp/            # uploads in progress
```

## Foundation

- [ ] **1. Layout and dependencies**: `api/`, `ingestion/`, `retrieval/`, `storage/` tree
  . Dependencies: FastAPI, Uvicorn, Pydantic, PyYAML (httpx for dev). Optional
  extras `[vector]`, `[pdf]`, `[ocr]` for heavy dependencies.
- [ ] **2. Configuration**: `config/settings.yaml` + environment variable overrides
  (`config.py`). Workspace, Qdrant URL, embedding model, read/admin tokens, limits.
- [ ] **3. Workspace**: creation/validation of the workspace and its subdirectories,
  restrictive permissions, safe path resolution (anti path traversal), refuse to run as root.
- [ ] **4. FastAPI app, `/health`, `serve` CLI**: application factory, `main.py`,
  `pyGdbKnowledge serve` (uvicorn), logs on stdout, `TestClient` tests.

## Phase 1: base server

- [ ] **5. SQLite catalog**: `storage/models.py`, `storage/catalog.py`. Tables `documents`,
  `sections`, `chunks`, FTS5 table, versioned schema with migrations. Optional metadata
  (never invent values).
- [ ] **6. `/stats` and document endpoints**: `GET /documents/{id}`, `/sections`,
  `/section/{section_id}`. Read-only, bounded response size.
- [ ] **7. FTS5 lexical search**: metadata filters, exact register-name search test.
- [ ] **8. Embeddings abstraction**: `Embedder` interface, local implementation (to decide:
  sentence-transformers or Ollama embeddings), deterministic fake embedder for tests.
- [ ] **9. Qdrant vector store**: configured collection, upsert, filtered search, in-memory
  implementation for tests.
- [ ] **10. Hybrid search and `POST /search`**: RRF fusion of lexical + vector results,
  ranking with reliability per `source_type`, response schema, result limits.
- [ ] **11. API security** (ANSSI R18, R19, R27, R29, R30, R31): separate read/admin
  credentials, server-side filter/collection control, parameter validation, rate limits,
  timeouts, result count/size limits, relevance threshold, correlation id, audit logging of
  auth decisions and returned ids, no embeddings in responses or logs, uniform errors.

## Phase 2: ingestion (local import)

- [ ] **12. Basic ingestion**: validation (PDF type, size), SHA-256, deduplication and
  versions, copy into `workspace/documents/`, explicitly provided metadata.
  CLI `pyGdbKnowledge ingest file.pdf --manufacturer ... --device ... --document-type ...
  --revision ... --source-url ...` (`--source-url` is provenance only, no download).
- [ ] **13. PDF extraction**: pypdf page by page, titles and sections, tests on a generated
  PDF.
- [ ] **14. Technical chunking**: per section with path and page, registers/bit fields,
  preserved tables, limited overlap, stable chunk/document ids + hash + version.
- [ ] **15. Indexing pipeline**: extraction, chunks, embeddings, Qdrant + FTS5. Commands
  `reindex` (rebuild from the files) and controlled deletion.
- [ ] **16. `POST /admin/ingest`**: upload or path under an allowed directory, admin token,
  asynchronous task with status tracking.
- [ ] **17. Optional OCR**: enabled when extracted text is insufficient (`ocr.py`).
- [ ] **18. Security hardening of ingestion** (ANSSI R22, R25, R25+, R26): content/metadata
  checks (hidden text, injection patterns), quarantine area, lifecycle audit trail (id,
  source, date, version, SHA-256, channel, validation status, chunk/vector ids), fast removal
  of a document/batch/source with all derived data, retrievability test after removal.
- [ ] **19. Corpus manifest and SVD support**: `config/corpus.yaml` (allowed domains, sources,
  optional SHA-256, metadata) and `pyGdbKnowledge sync` (`--dry-run`, `--source`, `--limit`,
  `--index`): downloads what is missing, verifies the hash (manifest, else lock file written on
  first download), imports. HTTPS only, allow-listed hosts including redirects, never triggered
  by the API. Two initial corpora: `cmsis-svd/cmsis-svd-data` (SVD files, vendor, pinned commit)
  and `www.st.com` (explicit list of official documents). SVD import, parsing and chunking
  (one section per peripheral, registers and bit fields).

## Quality and operations

- [ ] **20. Pilot corpus and evaluation**: 10-30 documents, questions (register, bit field,
  timing, errata, DMA), page/section verification, Recall@k, chunking/scoring tuning.
  Started: pilot corpus (21 STM32 F4/U5 SVD files, 8 ST PDFs), `config/eval/questions.yaml`,
  `pyGdbKnowledge evaluate` (Recall@k, MRR). Remaining: errata, embeddings run, tuning.
- [ ] **21. Docker and Compose**: `Dockerfile` (non-root user, read-only filesystem outside
  workspace/tmp), `docker-compose.yml` with Qdrant, workspace volume, `.env.example`,
  healthchecks.
- [ ] **22. Documentation**: installation, operations, backup (`documents/` + `catalog.db` +
  config), index rebuild, embedding versions.

## Later (V2)

- [ ] `POST /web/search`: prioritized official domains, trust level, source and access dates.
- [ ] Selective community indexing (forums, GitHub), duplicate detection and freshness.
- [ ] RAM/CPU/disk monitoring, upgrade and restore tests.

## Open decisions

- Local embeddings: sentence-transformers or Ollama embeddings (before step 8).
- Qdrant: separate container (recommended) or embedded mode.
