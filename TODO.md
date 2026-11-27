<!--
SPDX-FileCopyrightText: 2026 H2Lab Development Team
SPDX-License-Identifier: Apache-2.0
-->
## Enhancing V1


Task: Generalize Silicore Knowledge into a generic technical knowledge RAG

### Context

Silicore Knowledge is an existing hybrid RAG system designed
initially for embedded systems and hardware documentation.

The current corpus is defined in config/corpus.yaml.
The system already supports document synchronization,
ingestion, indexing and retrieval.

The objective is to evolve the system into a generic technical
knowledge engine capable of indexing heterogeneous technical
documents and retrieving them efficiently for LLM usage.

Supported document categories should include, but not be limited to:

- Hardware and software architecture guides
- Development guidelines
- Secure coding guides
- Programming language standards
- Formal specification languages
- RFCs and technical specifications
- Datasheets
- Reference manuals
- Application notes
- Security guides
- Assembly language specifications
- Protocol specifications
- Hardware components and peripherals

Examples of target technologies:

- C, C++, Python, Rust, Assembly
- ACSL
- STM32, FPGA, SoC
- SPI, I2C, UART, USB
- JTAG, SWD
- ARM Cortex-M, Thumb-2

### Main objective

The system must be able to retrieve relevant documents based on
technical intent, even when the query does not use the exact
vocabulary of the source documents.

For example, a query about preventing firmware extraction
should be able to retrieve documentation about JTAG, SWD,
debug access, readout protection, lifecycle security,
debug authentication and firmware protection.

### Architectural principles

1. Preserve existing functionality.
2. Avoid hardware-specific assumptions in generic components.
3. Make document types extensible.
4. Keep metadata optional where not applicable.
5. Separate source configuration, taxonomy, concepts and retrieval.
6. Prefer declarative configuration over hardcoded logic.
7. Make ingestion steps independently testable and replayable.
8. Preserve source references, sections and page numbers.
9. Do not introduce unnecessary infrastructure.
10. Measure retrieval quality through automated benchmarks.

### Target workstreams

1. Generic metadata schema
2. Generic document taxonomy
3. Concept registry with aliases and acronyms
4. Semantic document enrichment
5. Adaptive extraction and chunking
6. Query understanding
7. Controlled query expansion
8. Hybrid retrieval
9. Reranking
10. Context building
11. Retrieval benchmark and regression testing

### Instructions

Do not implement the entire roadmap at once.

First inspect the repository and report:

- Current architecture
- Existing ingestion pipeline
- Existing document and chunk models
- Existing metadata
- Existing lexical and vector retrieval
- Existing API
- Current tests
- Technical debt and limitations
- Breaking changes that might be required

Then propose:

- A target architecture
- A migration strategy
- An incremental implementation plan
- The smallest useful first milestone

Wait for approval before implementing architectural changes.

For each implementation task:

- Keep the scope limited
- Preserve backward compatibility where possible
- Add tests
- Update documentation
- Avoid unnecessary dependencies
- Report changed files
- Report test results
- Identify remaining limitations

## Later (V2)

- [ ] `POST /web/search`: prioritized official domains, trust level, source and access dates.
- [ ] Selective community indexing (forums, GitHub), duplicate detection and freshness.
- [ ] RAM/CPU/disk monitoring, upgrade and restore tests.

## Open decisions

- Local embeddings: sentence-transformers or Ollama embeddings (before step 8).
- Qdrant: separate container (recommended) or embedded mode.
