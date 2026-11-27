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

### V1 status and remaining work

The current implementation includes generic metadata and taxonomy, controlled concept
expansion, adaptive PDF/SVD chunking, hybrid retrieval, document-scoped read identities,
quarantine and explicit release, and benchmark gates for recall and annotated forbidden
passages. Existing active documents were deliberately left active. These measures are not
evidence of complete ANSSI compliance or of benchmark quality on the full deployed corpus.

Prioritized work remaining, excluding V2:

1. **Validate RAG security and operations**
   - [ ] Audit ANSSI R1-R35 against the actual deployment and document evidence, gaps and
     residual risks; remediate the gaps identified. Review existing active documents without
     automatically deactivating them.
   - [ ] Assess identity and credential separation, TLS/mTLS, authorization policy, audit
     coverage, quarantine/release traceability, deletion of derived data, and rate/size limits
     in the intended deployment, including concurrent or multiple service instances.
   - [ ] Pin and review dependencies and images, produce and maintain an SBOM, and establish
     vulnerability and patch management.
2. **Strengthen and validate retrieval evaluation**
   - [ ] Run the complete benchmark on the reviewed target corpus; record reproducible corpus,
     lock, configuration and model versions. Earlier lexical pilot scores predate the latest
     security and benchmark changes.
   - [ ] Expand representative positive, negative and misleading-passage cases across document
     types, intents and filters; review relevance and provenance manually. The current forbidden
     annotations are limited and do not exhaustively detect false positives.
   - [ ] Calibrate a positive minimum score on the intended corpus and model, measure its impact
     on both false positives and missed answers, and define justified regression gates.
3. **Validate hybrid retrieval**
   - [ ] Select and validate the embedding model and deployment approach; compare lexical and
     hybrid results on the same corpus and questions, and investigate semantic fallbacks.
   - [ ] Track vector/model provenance and compatibility; define and test safe model changes,
     collection migration and reindexing without mixing incompatible vectors.
   - [ ] Decide from measured results whether semantic enrichment, improved query understanding,
     reranking or context building are needed; implement only the justified V1 improvements.
4. **Finish delivery validation and corpus management**
   - [ ] Run the test suite under Python 3.12 and validate a real Docker Compose deployment,
     including permissions, quarantine, approval, corpus update, backup and restore.
   - [ ] Validate the read-only integration with pyGdbToolkit, including scoped identities and
     rejection of inaccessible documents, and document the deployment-specific configuration.
   - [ ] Define a safe workflow for revised content served at an already-catalogued URL;
     synchronization currently skips that URL. Complete traceability and removal for document
     versions, batches, sources and derived data as required by the security audit.
   - [ ] Close the remaining target-domain and document coverage gaps with reviewed, authorized
     sources and add corresponding benchmark questions.

## Later (V2)

- [ ] `POST /web/search`: prioritized official domains, trust level, source and access dates.
- [ ] Selective community indexing (forums, GitHub), duplicate detection and freshness.
- [ ] RAM/CPU/disk monitoring, upgrade and restore tests.
