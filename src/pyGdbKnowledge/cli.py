# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for pyGdbKnowledge."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from .config import ConfigError, load_settings
from .ingestion.importer import DocumentMetadata, IngestError, import_pdf
from .main import configure_logging, create_app
from .storage.catalog import Catalog, CatalogError
from .storage.models import SourceType
from .workspace import Workspace, WorkspaceError

_CONFIG_HELP = "YAML configuration file (default: $PYGDBKNOWLEDGE_CONFIG, else built-in defaults)"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyGdbKnowledge",
        description="Knowledge server (hybrid RAG) for pyGdbToolkit.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--config", type=Path, help=_CONFIG_HELP)
    ingest = commands.add_parser("ingest", help="import local PDF files into the workspace")
    ingest.add_argument("files", nargs="+", type=Path, metavar="FILE", help="PDF files to import")
    ingest.add_argument("--config", type=Path, help=_CONFIG_HELP)
    ingest.add_argument(
        "--source-type",
        required=True,
        choices=[t.value for t in SourceType],
        help="reliability class of the source: official, vendor or community",
    )
    ingest.add_argument("--title", help="document title (single file only; default: file name)")
    ingest.add_argument("--manufacturer")
    ingest.add_argument("--device-family")
    ingest.add_argument("--device")
    ingest.add_argument("--document-type", help="e.g. datasheet, reference_manual")
    ingest.add_argument("--revision")
    ingest.add_argument("--source-url", help="provenance only; the URL is never fetched")
    return parser


def _ingest(args: argparse.Namespace) -> int:
    if args.title and len(args.files) > 1:
        print("pyGdbKnowledge: --title requires a single file", file=sys.stderr)
        return 2
    try:
        settings = load_settings(args.config)
        workspace = Workspace(settings.workspace)
        catalog = Catalog(workspace.catalog_path)
    except (ConfigError, WorkspaceError, CatalogError) as exc:
        print(f"pyGdbKnowledge: {exc}", file=sys.stderr)
        return 1
    metadata = DocumentMetadata(
        source_type=SourceType(args.source_type),
        title=args.title,
        manufacturer=args.manufacturer,
        device_family=args.device_family,
        device=args.device,
        document_type=args.document_type,
        revision=args.revision,
        source_url=args.source_url,
    )
    failures = 0
    for path in args.files:
        try:
            result = import_pdf(
                workspace, catalog, path, metadata, settings.limits.max_upload_bytes
            )
        except (IngestError, CatalogError) as exc:
            print(f"pyGdbKnowledge: {exc}", file=sys.stderr)
            failures += 1
            continue
        state = "imported" if result.created else "already present"
        print(f"{state}: {result.document.document_id} ({path.name})")
    return 1 if failures else 0


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
    if args.command == "ingest":
        return _ingest(args)
    raise AssertionError(f"unhandled command {args.command!r}")
