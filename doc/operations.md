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

For containers, follow the [Compose workflow](#docker-compose-document-and-corpus-workflow)
before starting the API.

## Configuration

[config/settings.yaml](../config/settings.yaml) documents every key. Pass it with `--config`
or `SILICORE_KNOWLEDGE_CONFIG`; any key can be overridden by `SILICORE_KNOWLEDGE__<SECTION>__<KEY>`.

- Tokens belong in the environment, never in the file; see the "Tokens" section of the
  [README](../README.md) for creation, migration, use and rotation. Each read identity has a
  distinct token and an explicit list of allowed document IDs; an empty list denies all.
- Anonymous reads are disabled, including on loopback. The admin token is needed for ingestion,
  release and deletion. CLI operations require filesystem access to the workspace.
- `limits.min_score` defaults to zero for offline evaluation. The API refuses to start until it
  is configured to a positive value. Calibrate it with positive and misleading-passage
  benchmarks for the deployed corpus and model; RRF scores are relative ranks, not
  probabilities of relevance. Compose requires this value as `KNOWLEDGE_MIN_SCORE`.
- Non-upload request bodies are bounded before JSON parsing by `limits.max_api_body_bytes`;
  raw PDF uploads use the separate streamed `limits.max_upload_bytes` limit.
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
`POST /admin/documents/{id}/screen`, `POST /admin/documents/{id}/release`,
`DELETE /admin/documents/{id}`, `GET /admin/tasks/{id}`.

## Loading documents

The service never fetches anything by itself. Documents enter through one of:

1. `silicore-knowledge ingest FILE... --source-type official|vendor|community`
   for local PDF or SVD files (metadata options: `--device-family`, `--device`,
   `--document-type`, `--keyword` (repeatable), ...).
2. `silicore-knowledge sync [--dry-run]`: downloads the documents listed in
   [config/corpus.yaml](../config/corpus.yaml) (allow-listed HTTPS hosts, size cap, throttling)
   and records SHA-256 digests in the selected lock file (trust on first use if no hash is
   pinned). Re-running skips catalogued URLs even if their manifest hash changes.
3. The admin API (`/admin/ingest`).

All newly imported documents go to quarantine, regardless of source class. Find the document
ID and hash with `silicore-knowledge audit`, inspect the original file under
`<workspace>/quarantine/`, and run `silicore-knowledge screen DOCUMENT_ID`. Screening is a
limited heuristic and does not establish safety. After human review, use
`silicore-knowledge release DOCUMENT_ID` to approve and index the document. The admin API
offers the same screen and release operations; release returns an indexing task ID. A changed
quarantined file is rejected. Existing active documents are not re-quarantined or retroactively
validated by this migration. Every lifecycle event is kept in the audit trail.

The legacy `ingest --index` and `sync --index` flags do not bypass review: quarantined documents
produce an indexing error and a nonzero exit status. Re-importing the same bytes does not release
a quarantined document. Update each reader's document grants after approval.

To add a corpus entry, append the URL (a direct PDF link; HTML pages cannot be ingested) to the
manifest with its title, `document_type` and, when applicable, `device_family`, `device` and
`keywords`, then run `sync`. New hosts must be added to `allowed_domains`. Document types
should come from [config/taxonomy.yaml](../config/taxonomy.yaml): `sync` warns about unknown
types (add the type to the taxonomy first). `keywords` are indexed with the document's
passages and help queries that use another vocabulary reach it.

The bundled corpus includes NIST SP 800-218 SSDF v1.1 as a high-level secure development
best-practice guide and a CISA/FBI alert on preventing buffer overflows. Neither is a
language-specific coding reference.

## Docker Compose document and corpus workflow

Run these commands from the repository root. `knowledge` runs as UID 10001, `/corpus` is the
read-only `config/` bind mount, and `/var/lib/silicore-knowledge` is the persistent workspace
volume. Its `documents/`, `quarantine/` and `catalog.db` survive `docker compose down`.
`docker compose down -v` deletes the volumes and their contents.

### 1. Prepare the deployment and the corpus lock

Copy `.env.example` to `.env`, restrict its permissions (`chmod 600 .env`), and set distinct
`KNOWLEDGE_READ_TOKEN` and `KNOWLEDGE_ADMIN_TOKEN` secrets. Keep
`KNOWLEDGE_ALLOWED_DOCUMENTS='[]'` until document IDs have been approved. Start with
`KNOWLEDGE_EMBEDDINGS=false`; semantic indexing additionally needs the model and Qdrant.
Never commit `.env`. The examples below use `KNOWLEDGE_MIN_SCORE=0` only for one-off CLI
commands; no API server is started with that value.

```sh
KNOWLEDGE_MIN_SCORE=0 docker compose build
```

The container user must be able to traverse and read the files it uses in `config/`. If they
are not already accessible, Linux ACLs can grant the minimum access:

```sh
setfacl -m u:10001:--x config config/eval
setfacl -m u:10001:r-- config/corpus.yaml config/corpus.lock.json \
  config/concepts.yaml config/taxonomy.yaml config/eval/questions.yaml
```

ACL support and `setfacl` are required for this option. Parent directories must also be
traversable as needed by the bind mount. Repeat the relevant file ACL when a file is replaced.
Do not make secret files readable to the container user this way.

The checked-in lock is mounted read-only. Initialize a separate writable lock in the workspace
**once**, before the first sync, so that the shipped hashes are checked rather than silently
using trust on first use:

```sh
KNOWLEDGE_MIN_SCORE=0 docker compose run --rm --no-deps --entrypoint /bin/sh knowledge -c \
  'if [ -e /var/lib/silicore-knowledge/corpus.lock.json ]; then
     echo "Workspace lock already exists; compare it before changing it."
   else
     cp /corpus/corpus.lock.json /var/lib/silicore-knowledge/corpus.lock.json
   fi'
```

Do not overwrite an existing workspace lock blindly: it may contain hashes recorded by earlier
syncs. Compare it with the reviewed repository lock and reconcile differences deliberately.

### 2. Import into quarantine

Preview the manifest, then synchronize it. Both commands use the persistent workspace lock:

```sh
KNOWLEDGE_MIN_SCORE=0 docker compose run --rm --no-deps knowledge sync \
  --manifest /corpus/corpus.yaml --lock /var/lib/silicore-knowledge/corpus.lock.json --dry-run
KNOWLEDGE_MIN_SCORE=0 docker compose run --rm --no-deps knowledge sync \
  --manifest /corpus/corpus.yaml --lock /var/lib/silicore-knowledge/corpus.lock.json
KNOWLEDGE_MIN_SCORE=0 docker compose run --rm --no-deps knowledge audit --limit 500
```

`sync` can contact the configured HTTPS hosts; it is an operator command, not an agent API.
Even `--dry-run` may contact GitHub to expand an SVD source. Add `--source SOURCE_NAME` to
either command to restrict it to one manifest source, or `--limit N` to cap downloads.
New files are imported as `quarantined`, even for official or vendor sources. The audit output
gives document IDs and SHA-256 hashes for the most recent events; increase `--limit` if needed.
A first-seen URL without a manifest hash is recorded in the
workspace lock on first download: this detects later changes but does **not** authenticate the
first download. Prefer an independently verified manifest SHA-256. When a download is attempted,
a mismatched hash is reported and the file is not imported.

For a local PDF or SVD instead, place it in an operator-controlled `inbox/` and grant UID 10001
traversal and read access if needed. The mount is read-only and the import still enters
quarantine:

```sh
KNOWLEDGE_MIN_SCORE=0 docker compose run --rm --no-deps -v "$PWD/inbox:/in:ro" knowledge \
  ingest /in/manual.pdf --source-type vendor --title "Manual" --device STM32U585
```

Use the corresponding filename for SVD input. The admin upload API accepts PDF only. The CLI
may import either PDF or SVD. Do not use `--index` to bypass review: it fails for quarantined
documents.

### 3. Review, release and authorize

For each document, run the heuristic screening and independently review the original:

```sh
KNOWLEDGE_MIN_SCORE=0 docker compose run --rm --no-deps knowledge screen DOCUMENT_ID
```

Before the API is running, review an independently obtained local copy of the exact source and
compare `sha256sum reviewed-file.pdf` (or `.svd`) with the SHA-256 from the audit output. After
the service is running, an operator can also copy the quarantined original for review:

```sh
mkdir -p review && chmod 700 review
test ! -e ./review/SHA256.pdf && \
  docker compose cp knowledge:/var/lib/silicore-knowledge/quarantine/SHA256.pdf \
    ./review/SHA256.pdf
sha256sum ./review/SHA256.pdf
```

Replace `SHA256` with the complete audited digest and use `.svd` for SVD files. Treat the
review copy as untrusted; keep it private and do not commit it. The screening rules are
heuristics, not proof that the content is safe. A suspicious finding requires a human decision.
After reviewing the exact bytes, explicitly approve and index each document:

```sh
KNOWLEDGE_MIN_SCORE=0 docker compose run --rm --no-deps knowledge release DOCUMENT_ID
KNOWLEDGE_MIN_SCORE=0 docker compose run --rm --no-deps knowledge audit DOCUMENT_ID
```

Release verifies the quarantined bytes against the recorded hash; a changed file is refused.
If indexing fails after release, the document can remain active but unindexed: resolve the
reported error and run `index DOCUMENT_ID`; normal screening on retry may quarantine it again.
Existing active documents remain active during migration; they are not retroactively reviewed.
Re-importing duplicate bytes does not release a quarantined file.

To discard a rejected document, run the command below with its exact ID. Remove it from the
manifest as appropriate, otherwise a later sync can import it again. With existing vector data,
enable embeddings and ensure Qdrant is available when deleting, so its vectors are removed too;
check logs for any file-removal failure.

```sh
KNOWLEDGE_MIN_SCORE=0 docker compose run --rm --no-deps knowledge delete DOCUMENT_ID
```

With the API running, the administrator can instead upload a PDF through `/admin/ingest`,
screen through `/admin/documents/{id}/screen` and explicitly release through
`/admin/documents/{id}/release`; the release returns a task ID for `/admin/tasks/{id}`.
Review the original and hash independently before calling release. SVD ingestion remains
available through the CLI only.

### 4. Evaluate and start the API

The benchmark needs its expected documents to have been reviewed, released and indexed.
Start with lexical evaluation at zero to see the baseline; repeat with candidate positive values
and inspect relevant and forbidden passages before selecting a deployment threshold:

```sh
KNOWLEDGE_MIN_SCORE=0 docker compose run --rm --no-deps knowledge \
  evaluate /corpus/eval/questions.yaml --verbose
```

Replace zero with each candidate threshold for comparison. An incomplete corpus or the
current limited set of forbidden criteria cannot establish a safe general threshold. Verbose
evaluation reports ranks, not passage text. Inspect actual passages through the search API
using an admin token and a candidate positive threshold in an isolated validation deployment,
not an agent-facing production service. Review both apparent hits and misleading results. The
offline benchmark searches all active documents, not a particular reader's grants; also test
each intended reader scope through the API, since its rankings can differ.

Once calibrated for the intended corpus and model, set a **positive** `KNOWLEDGE_MIN_SCORE` in
`.env`, and set `KNOWLEDGE_ALLOWED_DOCUMENTS` to a JSON list of the approved document IDs.
An empty list grants no documents; `["*"]` explicitly grants all active documents, including
future ones. Restart after changing the environment:

```sh
docker compose up -d
docker compose ps
curl -s http://127.0.0.1:8080/api/v1/health
```

Use the read token to test `/api/v1/search`. The API refuses to start with a zero threshold,
and anonymous access is disabled. Authentication alone does not grant a document outside the
identity's configured list.

### 5. Update the corpus and maintain the lock

Add reviewed manifest entries to `config/corpus.yaml`, including source provenance, metadata,
keywords and an independently verified SHA-256 where available. Add a new allowed HTTPS host
only after reviewing it. Review taxonomy and concept changes separately. Repeat the dry-run,
sync, screening, human review, release and grant steps above; then rerun the benchmark.

The writable workspace lock and the reviewed `config/corpus.lock.json` are separate files.
When sync records first-use hashes, inspect the workspace lock before incorporating those
hashes into the repository lock. Copy it to a **new** review path and compare it, even before
the API is running:

```sh
mkdir -p review && chmod 700 review
( set -C; KNOWLEDGE_MIN_SCORE=0 docker compose run -T --rm --no-deps \
  --entrypoint /bin/sh knowledge -c 'cat /var/lib/silicore-knowledge/corpus.lock.json' \
  > ./review/corpus.lock.container.json )
diff -u config/corpus.lock.json review/corpus.lock.container.json
```

The no-clobber redirection refuses to overwrite an existing review copy; choose a fresh path for
another comparison. Reconcile independently verified changes in `config/corpus.lock.json`,
preserving all reviewed entries from both files, and version the repository manifest and lock
together. After review, and with no sync running, deliberately replace the workspace lock with
the reconciled file:

```sh
KNOWLEDGE_MIN_SCORE=0 docker compose run --rm --no-deps --entrypoint /bin/sh knowledge -c \
  'set -C; cat /corpus/corpus.lock.json > /var/lib/silicore-knowledge/corpus.lock.new.json &&
   mv /var/lib/silicore-knowledge/corpus.lock.new.json /var/lib/silicore-knowledge/corpus.lock.json'
```

If the temporary lock already exists, investigate it rather than overwriting it. A hash
mismatch must be investigated, not silently accepted. A URL already present in the catalog is
skipped by `sync`, even if the remote bytes change: changing a lock or manifest hash alone does
not update that document. For a new revision at the same URL, obtain the file independently and
import it explicitly with `ingest ... --revision REVISION --source-url URL`; it receives a
separate ID when its bytes differ. Review and release it as above. Sync continues to skip that
URL while it is catalogued. Do not delete the old document until its replacement has been
reviewed and authorized.

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

FTS rows and vectors can be rebuilt from the active documents and their catalog metadata.
The catalog also holds document IDs, metadata, status and audit history; it cannot be recreated
from `documents/` alone. Preserve the catalog, quarantine and lock as well as active files.
PDF chunks now stay within individual pages, so older indexes should be rebuilt with
`reindex` to correct their page citations. Reindexing also applies the size limit to previously
oversized single-line PDF passages and refreshes inferred peripheral metadata, including
removing false matches for identifiers immediately adjacent to `macro` or `pragma`. Reindex
SVD documents as well to apply the chunk size limit to long descriptions. Reindexing can change
chunk IDs and search rankings.

Identifier queries with additional terms also receive a lexical bonus for passages matching all
retained terms. Queries with numbered identifiers and an adjacent contextual phrase can also
match suffixed identifiers, while distinguishing adjacent numbers. These ranking changes do not
require reindexing.

## Embedding model versions

Vectors are not tagged with the model that produced them. After changing
`embeddings.model` (or its `max_seq_length`), run `silicore-knowledge reindex` with embeddings
enabled, otherwise old and new vectors are mixed. If the vector dimension changes, delete the
Qdrant collection first. The default model `BAAI/bge-m3` is large; on a CPU-only machine a
small model (for example `BAAI/bge-small-en-v1.5`) is far faster, with a lower quality ceiling.
Models are read locally; set `embeddings.allow_download: true` only for the one-off fetch.
For Compose, prepare a Hugging Face cache in `KNOWLEDGE_MODELS_DIR` readable by UID 10001, use
an image built with the vector extra and set `KNOWLEDGE_EMBEDDINGS=true`. Stop the API and start
Qdrant before running one-off indexing or evaluation commands with `--no-deps`:

```sh
KNOWLEDGE_MIN_SCORE=0 docker compose stop knowledge
KNOWLEDGE_MIN_SCORE=0 docker compose up -d qdrant
KNOWLEDGE_MIN_SCORE=0 docker compose run --rm --no-deps knowledge reindex
KNOWLEDGE_MIN_SCORE=0 docker compose run --rm --no-deps knowledge \
  evaluate /corpus/eval/questions.yaml --verbose
```

Reindexing screens content again and may quarantine previously active documents. Compare lexical
and hybrid retrieval and calibrate the threshold again before serving agents; enabling vectors
is not a quality guarantee.

## Backup and restore

Stop the service and all one-off CLI jobs before taking a consistent backup:

- the whole workspace volume, including `documents/`, `quarantine/`, `catalog.db`, any SQLite
  WAL files and the workspace lock;
- the configuration (without tokens, which live in your secret store);
- `config/corpus.yaml` and `config/corpus.lock.json`.

Restore: put them back, then run `silicore-knowledge reindex` to rebuild vectors (Qdrant storage
can also be backed up but is not required). Reindexing applies content screening again and can
change document status. With Compose, the data lives in the `workspace` and `qdrant` volumes.
Test the restoration before relying on a backup.

## Quality evaluation

`silicore-knowledge evaluate config/eval/questions.yaml [--verbose] [--min-recall 0.9]` reports
Recall@k per question category and overall MRR on the indexed corpus. Run it after any change to
chunking, ranking or the embedding model, and add questions when the corpus grows.
Recall@k is the fraction of questions with at least one matching expected passage in the top k;
each `expect` entry is an alternative, not a separate required passage.
The evaluation uses `limits.min_score` from the configuration. An optional `forbid` list
identifies explicitly misleading passages; `forbidden@5` counts questions returning at least one
such passage in the top five. Forbidden hits cannot satisfy an `expect` entry, and
`--max-forbidden 0` fails when any appears. `not_contains` excludes passages containing a term
from an expectation; the gate refuses a benchmark with no forbidden criteria. These manually
curated criteria are not an exhaustive relevance judgment.

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
