# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for pyGdbKnowledge."""

import argparse
from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="pyGdbKnowledge",
        description="Knowledge server (hybrid RAG) for pyGdbToolkit.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    _parser().parse_args(argv)
    print("pyGdbKnowledge: service not implemented yet")
    return 0
