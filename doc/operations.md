<!--
SPDX-FileCopyrightText: 2026 H2Lab Development Team
SPDX-License-Identifier: Apache-2.0
-->

# Operations guide

See [architecture.md](architecture.md) for the design and container hardening.

## Installation

Python 3.12 or newer.

```sh
pip install .[pdf]            # lexical search (SQLite/FTS5), PDF and SVD import
pip install .[pdf,vector]     # adds semantic search (Qdrant client, sentence-transformers)
pip install .[ocr]            # optional OCR of scanned PDFs (needs the tesseract binary)
```

Container: `cp .env.example .env`, set `KNOWLEDGE_READ_TOKEN` and `KNOWLEDGE_ADMIN_TOKEN`
(distinct values, e.g. `openssl rand -hex 32`), then `docker compose up -d`.

## Configuration

[config/settings.yaml](../config/settings.yaml) documents every key. Pass it with `--config`
or `SILICORE_KNOWLEDGE_CONFIG`; any key can be overridden by `SILICORE_KNOWLEDGE__<SECTION>__<KEY>`.

- Tokens belong in the environment, never in the file; see the "Tokens" section of the
  [README](../README.md) for creation, use and rotation. Listening on a non-loopback address
  requires `auth.read_token`.
- The read token allows search and document reading; the admin token is needed for ingestion
  and deletion.
- Without `embeddings.enabled`, search is lexical only. With it, results fuse lexical, identifier
  and vector rankings (RRF).

## Running

```sh
silicore-knowledge serve --config settings.yaml
curl http://127.0.0.1:8080/api/v1/health
curl -H "Authorization: Bearer $READ_TOKEN" -X POST http://127.0.0.1:8080/api/v1/search \
     -H 'Content-Type: application/json' -d '{"query": "FLASH_ACR LATENCY wait states"}'
```

API routes (prefix `/api/v1`): `GET /health`, `GET /stats`, `POST /search`,
`GET /documents/{id}`, `GET /documents/{id}/sections`, `GET /documents/{id}/section/{section}`,
and with the admin token `POST /admin/ingest`, `POST /admin/ingest/path`,
`DELETE /admin/documents/{id}`, `GET /admin/tasks/{id}`.

## Loading documents

The service never fetches anything by itself. Documents enter through one of:

1. `silicore-knowledge ingest FILE... --source-type official|vendor|community [--index]`
   for local PDF or SVD files (metadata options: `--device-family`, `--device`,
   `--document-type`, `--keyword` (repeatable), ...).
2. `silicore-knowledge sync [--dry-run] [--index]`: downloads the documents listed in
   [config/corpus.yaml](../config/corpus.yaml) (allow-listed HTTPS hosts, size cap, throttling)
   and records SHA-256 digests in `config/corpus.lock.json` (trust on first use; a changed file
   is refused unless the manifest pins another hash). Re-running skips what is present.
3. The admin API (`/admin/ingest`).

Documents flagged by content screening go to quarantine: review, then
`silicore-knowledge release DOCUMENT_ID`. Every lifecycle event is kept in the audit trail
(`silicore-knowledge audit [DOCUMENT_ID]`).

To add a corpus entry, append the URL (a direct PDF link; HTML pages cannot be ingested) to the
manifest with its title, `document_type` and, when applicable, `device_family`, `device` and
`keywords`, then run `sync`. New hosts must be added to `allowed_domains`. Document types
should come from [config/taxonomy.yaml](../config/taxonomy.yaml): `sync` warns about unknown
types (add the type to the taxonomy first). `keywords` are indexed with the document's
passages and help queries that use another vocabulary reach it.

## Query expansion concepts

[config/concepts.yaml](../config/concepts.yaml) (configured through `retrieval.concepts_path`)
declares concepts with their aliases, acronyms and related concepts. When a query names a
concept, the sibling and related terms form one extra lexical ranking, so an intent such as
"prevent firmware extraction" can reach passages about readout protection, RDP, JTAG or SWD.
Review this file like code: it directly shapes retrieval. After editing it, run the evaluation
(the `intent` category measures expansion) — no reindexing is needed. A narrowly triggered
boot-integrity concept maps queries about modified firmware to image-integrity and signature
verification vocabulary without broadening other lifecycle-security queries.

## Index maintenance

- `silicore-knowledge index [ID...]` indexes documents that have no index yet.
- `silicore-knowledge reindex [ID...]` rebuilds chunks, FTS5 rows and vectors from the stored files.
- `silicore-knowledge delete ID... [--source-url-prefix P]` removes files, index and vectors.

SQLite and Qdrant are derived data: both can be rebuilt from `documents/` alone.
PDF chunks now stay within individual pages, so older indexes should be rebuilt with
`reindex` to correct their page citations. Reindexing also applies the size limit to previously
oversized single-line PDF passages and refreshes inferred peripheral metadata, including
removing false matches for identifiers immediately adjacent to `macro` or `pragma`. Reindex
SVD documents as well to apply the chunk size limit to long descriptions. Reindexing can change
chunk IDs and search rankings.

Identifier queries with additional terms also receive a lexical bonus for passages matching all
retained terms. This ranking change does not require reindexing.

## Embedding model versions

Vectors are not tagged with the model that produced them. After changing
`embeddings.model` (or its `max_seq_length`), run `silicore-knowledge reindex` with embeddings
enabled, otherwise old and new vectors are mixed. If the vector dimension changes, delete the
Qdrant collection first. The default model `BAAI/bge-m3` is large; on a CPU-only machine a
small model (for example `BAAI/bge-small-en-v1.5`) is far faster, with a lower quality ceiling.
Models are read locally; set `embeddings.allow_download: true` only for the one-off fetch.

## Backup and restore

Back up, while the service is stopped or idle:

- `<workspace>/documents/` (original files, named by hash) and `catalog.db`;
- the configuration (without tokens, which live in your secret store);
- `config/corpus.yaml` and `config/corpus.lock.json`.

Restore: put them back, then run `silicore-knowledge reindex` to rebuild vectors (Qdrant storage
can also be backed up but is not required). With compose, the data lives in the `workspace`
and `qdrant` volumes.

## Quality evaluation

`silicore-knowledge evaluate config/eval/questions.yaml [--verbose] [--min-recall 0.9]` reports
Recall@k per question category and overall MRR on the indexed corpus. Run it after any change to
chunking, ranking or the embedding model, and add questions when the corpus grows.
Recall@k is the fraction of questions with at least one matching expected passage in the top k;
each `expect` entry is an alternative, not a separate required passage.

Use `--min-category-recall 0.8` to fail when any category falls below that Recall@5 threshold;
it can be combined with `--min-recall`, which checks the overall score. An empty question set
fails the category gate. Interpret category scores in light of their question counts and ensure
the benchmark corpus contains the documents required by the questions.

To compare lexical and hybrid retrieval, first reindex the corpus with embeddings enabled.
Then run the same questions with two configurations pointing to the same workspace and concept
registry: one with embeddings enabled and one with them disabled. Use `--verbose` to compare
individual ranks as well as overall and category metrics. Confirm that vectors cover the indexed
chunks and check for semantic-search fallback warnings: a fallback measures lexical search, not
hybrid retrieval. Model, corpus and filter choices can improve some queries while degrading others.
An expected hit matches the listed criteria, not every relevant passage; inspect the results
before interpreting a changed rank as a change in answer quality. When expectation criteria
change, evaluate both versions on the same corpus: score changes may reflect corrected evidence
rather than a change in the retriever.

## Security notes

Treat retrieved text as untrusted data. Keep Qdrant off any public network, terminate TLS in
front of the API for any remote use, and treat logs as sensitive (they can contain queries).
