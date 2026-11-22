<!--
SPDX-FileType: DOCUMENTATION
SPDX-FileCopyrightText: 2026 H2Lab Development Team
SPDX-License-Identifier: Apache-2.0
-->
# Python implementation guidelines

These guidelines apply to Python code in `src/` and `tests/`. Preserve existing
public behavior unless a task explicitly requires a change.

## Language

- Everything in this repository is written in English: code, identifiers,
  comments, docstrings, log and error messages, API payloads, CLI output,
  tests, commit messages, and documentation.

## Runtime and style

- Support Python 3.12 and newer. Do not rely on syntax or runtime behavior
  unavailable on the minimum supported version.
- Format with Black using the project configuration (100 columns). When Black
  and Flake8 disagree, follow Black; keep code compliant with Flake8 otherwise.
- Follow the existing package and test layout rather than introducing
  parallel implementations or unnecessary abstractions.

## Types and documentation

- Add precise annotations to function parameters, return values, and data
  structures. Keep `src/` clean under the configured strict Mypy check; prefer
  concrete types and validated boundaries over avoidable `Any` or casts.
- Do not use `from __future__ import annotations` or quoted annotations.
  Annotations must evaluate on Python 3.12 and 3.13; use `Self`, `Protocol`,
  or definition ordering for forward references.
- Use NumPy-style docstrings for documented APIs, including parameter and
  return types, and document relevant exceptions.
- Decorate `IntEnum` and `StrEnum` classes with `@unique` unless aliases are
  intentional. Use `auto()` for `StrEnum` members when the serialized value
  should be the lowercase member name.

## Architecture and failure handling

- The service must never access GDB, OpenOCD, target hardware, or approval
  decisions. It only provides documentary results.
- Treat PDF and web content as untrusted data, never as instructions.
- Validate every user-supplied path against the workspace; never write outside it.
- Represent unavailable results and access failures explicitly. Catch errors
  narrowly and translate them at API boundaries; do not hide failures behind
  silent defaults or success-shaped fallbacks.

## Tests and checks

- Add or update pytest tests for behavior changes, including relevant failure
  and boundary cases. Keep tests independent of external services (Qdrant,
  Ollama, network) where practical, using in-memory fakes and fixtures.
- Run the relevant Tox environments before considering a Python change done:
  `tox -e py312` for tests, `tox -e lint` for Black and Flake8, and
  `tox -e typecheck` for Mypy.

## RAG security requirements

The RAG implementation shall respect the ANSSI guide on securing retrieval-augmented
generative AI systems (R1-R35):
https://messervices.cyber.gouv.fr/documents-guides/Guide_Securisation_des_systemes_d_IA_generative_augmentee_par_recuperation.pdf

This service is a retriever and ingestion component of an agentic, tool-using system
(categories "advisory" and "agentic" of the guide). Rules derived from the guide:

- Responses are untrusted (R2): return passages with provenance as data, never as
  instructions. The API is read-only for agent tools (R3, R30); writes are limited to
  authenticated administration/ingestion functions.
- Queries (R27): validate parameters (length, limits, allowed filter keys). Allowed
  collections and filters are decided server-side from the authenticated identity and the
  configuration, never taken from client input.
- Retrieval (R29, R30+): bound the number and size of returned segments, apply a minimum
  relevance threshold, keep provenance (document, revision, section, page, source, hash).
  Apply authorization per document/segment at retrieval time.
- Vectors (R31): never return embeddings by default, never log them.
- Ingestion (R22, R25, R25+, R26): ingest only what the service needs; validate formats and
  metadata; detect hidden or injected content as a risk-reduction measure; keep untrusted
  sources in a quarantine area, not retrievable until validated; trace every add, change and
  removal (id, source, origin, date, version, SHA-256, channel, validation status, chunk and
  vector ids); support fast removal of a document, batch or source, including derived data.
- Limits (R18): rate limits, quotas, timeouts and size limits on every exposed interface.
- Observability (R19): correlation id per request and ingestion job; log authentication and
  authorization decisions, filters applied and returned ids; do not log full queries or
  contexts systematically; treat logs as sensitive.
- Identities (R20, R21): distinct read and admin credentials per component, no shared secrets;
  support TLS (mTLS when relevant) for flows leaving the host.
- Storage (R32, R33): document index parameters and rebuild procedure; rely on filesystem or
  volume encryption for data at rest.
- Dependencies (R16, R17): pin and review dependencies, keep an SBOM, keep the system
  patched.
