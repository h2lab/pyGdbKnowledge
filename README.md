<!--
SPDX-FileCopyrightText: 2026 H2Lab Development Team
SPDX-License-Identifier: Apache-2.0
-->

# pyGdbKnowledge

Standalone knowledge server (hybrid RAG) used by pyGdbToolkit over HTTP.
Target architecture: see [doc/architecture.md](doc/architecture.md).

Status: project skeleton only, the service is not implemented yet.

## Development

```sh
pip install -e .[dev]
tox            # py312, lint, typecheck
```
