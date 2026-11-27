# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for silicore-knowledge."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from .config import ConfigError, Settings, load_settings
from .corpus.fetch import Fetcher, HttpFetcher
from .corpus.manifest import CorpusError, Manifest, load_manifest
from .corpus.sync import LockFile, plan_sync, run_sync
from .corpus.taxonomy import TaxonomyError, load_taxonomy
from .evaluation import EvaluationError, evaluate, load_questions
from .ingestion.importer import DocumentMetadata, IngestError, import_document
from .ingestion.pipeline import BatchReport, Indexer, IndexingError
from .main import configure_logging, create_app
from .retrieval.concepts import ConceptError
from .retrieval.embeddings import EmbeddingError
from .retrieval.hybrid import HybridSearcher
from .retrieval.vector_store import VectorStoreError
from .services import build_vector_components, load_concept_registry
from .storage.catalog import Catalog, CatalogError
from .storage.models import SourceType
from .workspace import Workspace, WorkspaceError

_CONFIG_HELP = (
    "YAML configuration file (default: $SILICORE_KNOWLEDGE_CONFIG, else built-in defaults)"
)


def _recall_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "recall threshold must be a number between 0 and 1"
        ) from None
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("recall threshold must be a number between 0 and 1")
    return threshold


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="silicore-knowledge",
        description="Knowledge server (hybrid RAG) for pyGdbToolkit.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--config", type=Path, help=_CONFIG_HELP)
    ingest = commands.add_parser("ingest", help="import local PDF or SVD files")
    ingest.add_argument("files", nargs="+", type=Path, metavar="FILE", help="PDF or SVD files")
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
    ingest.add_argument(
        "--keyword",
        action="append",
        dest="keywords",
        metavar="KEYWORD",
        help="retrieval keyword indexed with the document (repeatable)",
    )
    ingest.add_argument("--index", action="store_true", help="index the documents after import")
    for name, help_text in (
        ("index", "index documents that have no index yet (all active ones by default)"),
        ("reindex", "rebuild the index of the given documents (all by default)"),
    ):
        indexing = commands.add_parser(name, help=help_text)
        indexing.add_argument("documents", nargs="*", metavar="DOCUMENT_ID")
        indexing.add_argument("--config", type=Path, help=_CONFIG_HELP)
    delete = commands.add_parser(
        "delete", help="remove documents and everything derived from them (files, index, vectors)"
    )
    delete.add_argument("documents", nargs="*", metavar="DOCUMENT_ID")
    delete.add_argument(
        "--source-url-prefix", help="also remove every document whose source URL starts with this"
    )
    delete.add_argument("--keep-file", action="store_true", help="keep the PDF in the workspace")
    delete.add_argument("--config", type=Path, help=_CONFIG_HELP)
    release = commands.add_parser(
        "release", help="release a quarantined document after review, then index it"
    )
    release.add_argument("document", metavar="DOCUMENT_ID")
    release.add_argument("--config", type=Path, help=_CONFIG_HELP)
    audit = commands.add_parser("audit", help="print the document lifecycle audit trail")
    audit.add_argument("document", nargs="?", metavar="DOCUMENT_ID")
    audit.add_argument("--limit", type=int, default=100)
    audit.add_argument("--config", type=Path, help=_CONFIG_HELP)
    sync = commands.add_parser(
        "sync", help="download the documents listed in the corpus manifest and import them"
    )
    sync.add_argument("--manifest", type=Path, default=Path("config/corpus.yaml"))
    sync.add_argument("--lock", type=Path, help="lock file (default: corpus.lock.json beside it)")
    sync.add_argument(
        "--taxonomy",
        type=Path,
        help="document type taxonomy (default: taxonomy.yaml beside the manifest, when present)",
    )
    sync.add_argument("--source", help="only synchronize this source of the manifest")
    sync.add_argument("--dry-run", action="store_true", help="list what would be downloaded")
    sync.add_argument("--limit", type=int, help="stop after this many downloads")
    sync.add_argument("--index", action="store_true", help="index the imported documents")
    sync.add_argument("--config", type=Path, help=_CONFIG_HELP)
    evaluation = commands.add_parser(
        "evaluate", help="measure retrieval quality (Recall@k, MRR) on a question file"
    )
    evaluation.add_argument("questions", type=Path, metavar="QUESTIONS_YAML")
    evaluation.add_argument("--min-recall", type=float, help="exit 1 below this Recall@5 (0-1)")
    evaluation.add_argument(
        "--min-category-recall",
        type=_recall_threshold,
        metavar="THRESHOLD",
        help="exit 1 if any category's Recall@5 is below this threshold (0-1)",
    )
    evaluation.add_argument("--verbose", action="store_true", help="list every question")
    evaluation.add_argument("--config", type=Path, help=_CONFIG_HELP)
    return parser


def _open(config: Path | None) -> tuple[Settings, Workspace, Catalog] | None:
    try:
        settings = load_settings(config)
        workspace = Workspace(settings.workspace)
        return settings, workspace, Catalog(workspace.catalog_path)
    except (ConfigError, WorkspaceError, CatalogError) as exc:
        print(f"silicore-knowledge: {exc}", file=sys.stderr)
        return None


def _indexer(settings: Settings, workspace: Workspace, catalog: Catalog) -> Indexer | None:
    try:
        components = build_vector_components(settings)
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"silicore-knowledge: {exc}", file=sys.stderr)
        return None
    embedder, store = components if components else (None, None)
    return Indexer(settings, workspace, catalog, embedder, store)


def _print_report(report: BatchReport) -> int:
    for result in report.indexed:
        print(
            f"indexed: {result.document_id} "
            f"({result.sections} sections, {result.chunks} chunks, {result.vectors} vectors)"
        )
    for document_id, message in report.failed.items():
        print(f"silicore-knowledge: {document_id}: {message}", file=sys.stderr)
    return 1 if report.failed else 0


def _index(args: argparse.Namespace, *, rebuild: bool) -> int:
    opened = _open(args.config)
    if opened is None:
        return 1
    indexer = _indexer(*opened)
    if indexer is None:
        return 1
    return _print_report(indexer.index_documents(args.documents or None, rebuild=rebuild))


def _delete(args: argparse.Namespace) -> int:
    if not args.documents and not args.source_url_prefix:
        print("silicore-knowledge: give document ids or --source-url-prefix", file=sys.stderr)
        return 2
    opened = _open(args.config)
    if opened is None:
        return 1
    indexer = _indexer(*opened)
    if indexer is None:
        return 1
    targets = list(args.documents)
    if args.source_url_prefix:
        targets += indexer.find_by_source_url(args.source_url_prefix)
    report = indexer.delete_documents(targets, keep_file=args.keep_file)
    for document_id in dict.fromkeys(targets):
        if document_id not in report.failed:
            print(f"deleted: {document_id}")
    for document_id, message in report.failed.items():
        print(f"silicore-knowledge: {document_id}: {message}", file=sys.stderr)
    return 1 if report.failed else 0


def _release(args: argparse.Namespace) -> int:
    opened = _open(args.config)
    if opened is None:
        return 1
    indexer = _indexer(*opened)
    if indexer is None:
        return 1
    try:
        result = indexer.release(args.document)
    except IndexingError as exc:
        print(f"silicore-knowledge: {exc}", file=sys.stderr)
        return 1
    print(f"released: {result.document_id} ({result.chunks} chunks)")
    return 0


def _audit(args: argparse.Namespace) -> int:
    opened = _open(args.config)
    if opened is None:
        return 1
    for event in opened[2].list_events(args.document, args.limit):
        print(
            f"{event.occurred_at.isoformat()} {event.event} {event.document_id}"
            f" sha256={event.content_hash or '-'} channel={event.channel or '-'}"
            f" {event.detail or ''}".rstrip()
        )
    return 0


def _ingest(args: argparse.Namespace) -> int:
    if args.title and len(args.files) > 1:
        print("silicore-knowledge: --title requires a single file", file=sys.stderr)
        return 2
    opened = _open(args.config)
    if opened is None:
        return 1
    settings, workspace, catalog = opened
    metadata = DocumentMetadata(
        source_type=SourceType(args.source_type),
        title=args.title,
        manufacturer=args.manufacturer,
        device_family=args.device_family,
        device=args.device,
        document_type=args.document_type,
        revision=args.revision,
        source_url=args.source_url,
        keywords=tuple(args.keywords or ()),
    )
    failures = 0
    imported: list[str] = []
    for path in args.files:
        try:
            result = import_document(
                workspace, catalog, path, metadata, settings.limits.max_upload_bytes
            )
        except (IngestError, CatalogError) as exc:
            print(f"silicore-knowledge: {exc}", file=sys.stderr)
            failures += 1
            continue
        state = "imported" if result.created else "already present"
        print(f"{state}: {result.document.document_id} ({path.name})")
        imported.append(result.document.document_id)
    if args.index and imported:
        indexer = _indexer(settings, workspace, catalog)
        if indexer is None:
            return 1
        failures += _print_report(indexer.index_documents(imported))
    return 1 if failures else 0


def _check_taxonomy(manifest: Manifest, manifest_path: Path, taxonomy_path: Path | None) -> int:
    """Warn about manifest document types absent from the taxonomy; never reject them.

    Returns
    -------
    int
        0, or 1 when an explicitly given taxonomy file is unreadable or invalid.
    """
    path = taxonomy_path or manifest_path.with_name("taxonomy.yaml")
    if taxonomy_path is None and not path.exists():
        return 0
    try:
        taxonomy = load_taxonomy(path)
    except TaxonomyError as exc:
        print(f"silicore-knowledge: {exc}", file=sys.stderr)
        return 1
    for name in taxonomy.unknown_document_types(manifest):
        print(
            f"silicore-knowledge: warning: document_type {name!r} is not in the taxonomy {path}",
            file=sys.stderr,
        )
    return 0


def _sync(args: argparse.Namespace, fetcher: Fetcher | None = None) -> int:
    opened = _open(args.config)
    if opened is None:
        return 1
    settings, workspace, catalog = opened
    max_bytes = settings.limits.max_upload_bytes
    try:
        manifest = load_manifest(args.manifest)
        if _check_taxonomy(manifest, args.manifest, args.taxonomy):
            return 1
        lock = LockFile(args.lock or args.manifest.with_name("corpus.lock.json"))
        client = fetcher or HttpFetcher(
            manifest.allowed_domains, manifest.timeout_s, manifest.request_delay_s
        )
        plans = plan_sync(manifest, client, max_bytes, args.source)
        report = run_sync(
            plans,
            lock,
            workspace,
            catalog,
            client,
            max_bytes,
            dry_run=args.dry_run,
            limit=args.limit,
        )
    except CorpusError as exc:
        print(f"silicore-knowledge: {exc}", file=sys.stderr)
        return 1
    for url in report.planned:
        print(f"would download: {url}")
    for url, message in report.failed.items():
        print(f"silicore-knowledge: {url}: {message}", file=sys.stderr)
    print(
        f"sync: {len(report.imported)} imported, {len(report.skipped)} already present, "
        f"{len(report.planned)} planned, {len(report.failed)} failed"
    )
    failures = len(report.failed)
    if args.index and report.imported:
        indexer = _indexer(settings, workspace, catalog)
        if indexer is None:
            return 1
        failures += _print_report(indexer.index_documents(report.imported))
    return 1 if failures else 0


def _evaluate(args: argparse.Namespace) -> int:
    opened = _open(args.config)
    if opened is None:
        return 1
    settings, _, catalog = opened
    try:
        questions = load_questions(args.questions)
        components = build_vector_components(settings)
        concepts = load_concept_registry(settings)
    except (EvaluationError, EmbeddingError, VectorStoreError, ConceptError) as exc:
        print(f"silicore-knowledge: {exc}", file=sys.stderr)
        return 1
    embedder, store = components if components else (None, None)
    report = evaluate(HybridSearcher(catalog, embedder, store, concepts=concepts), questions)
    for result in report.results:
        if args.verbose or result.rank is None:
            rank = result.rank or "miss"
            print(f"{result.question.id:<24} {result.question.category:<12} rank={rank}")
    for category in [None, *report.categories]:
        label = category or "all"
        recalls = " ".join(f"R@{k}={report.recall(k, category):.2f}" for k in report.ks)
        print(f"{label:<12} {recalls}")
    print(f"questions={len(report.results)} MRR={report.mrr:.3f}")
    failed = args.min_recall is not None and report.recall(5) < args.min_recall
    if args.min_category_recall is not None:
        if not report.categories:
            print("silicore-knowledge: no categories to evaluate", file=sys.stderr)
            return 1
        for category in report.categories:
            recall = report.recall(5, category)
            if recall < args.min_category_recall:
                print(
                    f"silicore-knowledge: category {category!r}: Recall@5={recall!r} "
                    f"is below {args.min_category_recall!r}",
                    file=sys.stderr,
                )
                failed = True
    return 1 if failed else 0


def _serve(config: Path | None) -> int:
    try:
        settings = load_settings(config)
        workspace = Workspace(settings.workspace)
        app = create_app(settings, workspace)
    except (
        ConfigError,
        WorkspaceError,
        CatalogError,
        EmbeddingError,
        VectorStoreError,
        ConceptError,
    ) as exc:
        print(f"silicore-knowledge: {exc}", file=sys.stderr)
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
    if args.command in ("index", "reindex"):
        return _index(args, rebuild=args.command == "reindex")
    if args.command == "delete":
        return _delete(args)
    if args.command == "release":
        return _release(args)
    if args.command == "audit":
        return _audit(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "sync":
        return _sync(args)
    raise AssertionError(f"unhandled command {args.command!r}")
