# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for pyGdbKnowledge."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from .config import ConfigError, load_settings
from .main import configure_logging, create_app
from .storage.catalog import CatalogError
from .workspace import Workspace, WorkspaceError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyGdbKnowledge",
        description="Knowledge server (hybrid RAG) for pyGdbToolkit.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the HTTP API")
    serve.add_argument(
        "--config",
        type=Path,
        help="YAML configuration file (default: $PYGDBKNOWLEDGE_CONFIG, else built-in defaults)",
    )
    return parser


def _serve(config: Path | None) -> int:
    try:
        settings = load_settings(config)
        workspace = Workspace(settings.workspace)
        app = create_app(settings, workspace)
    except (ConfigError, WorkspaceError, CatalogError) as exc:
        print(f"pyGdbKnowledge: {exc}", file=sys.stderr)
        return 1
    configure_logging(settings.log_level)
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_config=None,
        server_header=False,
    )
    return 0


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
    args = _parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args.config)
    raise AssertionError(f"unhandled command {args.command!r}")
