# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the corpus manifest, restricted fetching and synchronization."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pdfmaker import make_pdf

from silicore_knowledge import cli
from silicore_knowledge.corpus.fetch import HttpFetcher, check_url
from silicore_knowledge.corpus.manifest import CorpusError, UrlSource, load_manifest
from silicore_knowledge.corpus.taxonomy import load_taxonomy
from silicore_knowledge.corpus.sync import LockFile, plan_sync, run_sync
from silicore_knowledge.storage.catalog import Catalog
from silicore_knowledge.workspace import Workspace

COMMIT = "a" * 40
RAW = f"https://raw.githubusercontent.com/o/r/{COMMIT}/data"
SVD = (
    "<device><name>{}</name><peripherals><peripheral><name>P</name>"
    "<baseAddress>0x1000</baseAddress></peripheral></peripherals></device>"
)
HOSTS = ["api.github.com", "raw.githubusercontent.com", "www.arm.com", "www.st.com"]


class FakeFetcher:
    """In-memory network: JSON documents and file bodies by URL."""

    def __init__(self, files: dict[str, bytes], tree: list[dict[str, Any]] | None = None) -> None:
        self.files = files
        self.tree = tree or []
        self.downloads: list[str] = []

    def get_json(self, url: str) -> Any:
        """Serve the commit and tree documents of the fake repository."""
        if "/commits/" in url:
            return {"sha": COMMIT}
        return {"truncated": False, "tree": self.tree}

    def download(self, url: str, destination: Path, max_bytes: int) -> str:
        """Write the canned body of ``url``."""
        if url not in self.files:
            raise CorpusError(f"HTTP 404 for {url}")
        self.downloads.append(url)
        destination.write_bytes(self.files[url])
        return hashlib.sha256(self.files[url]).hexdigest()


def blob(path: str, size: int = 100) -> dict[str, Any]:
    """Return a git tree blob entry."""
    return {"type": "blob", "path": path, "size": size}


def write_manifest(path: Path, documents: list[dict[str, Any]] | None = None) -> Path:
    """Write a manifest with one SVD source and one URL source."""
    manifest = {
        "allowed_domains": HOSTS,
        "request_delay_s": 0,
        "sources": [
            {
                "name": "svd",
                "kind": "github_svd",
                "source_type": "vendor",
                "repo": "o/r",
                "vendors": ["STMicro"],
                "manufacturers": {"STMicro": "STMicroelectronics"},
            },
            {
                "name": "st",
                "kind": "urls",
                "source_type": "official",
                "manufacturer": "STMicroelectronics",
                "documents": documents or [],
            },
        ],
    }
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def test_manifest_rejects_a_host_outside_the_allow_list(tmp_path: Path) -> None:
    """A listed URL on a host that is not allowed invalidates the manifest."""
    path = write_manifest(tmp_path / "c.yaml", [{"url": "https://evil.example/a.pdf"}])
    with pytest.raises(CorpusError, match="not an allowed domain"):
        load_manifest(path)


def test_manifest_rejects_plain_http_and_unknown_keys(tmp_path: Path) -> None:
    """Only HTTPS URLs and known keys are accepted."""
    with pytest.raises(CorpusError):
        load_manifest(write_manifest(tmp_path / "a.yaml", [{"url": "http://www.st.com/a.pdf"}]))
    path = tmp_path / "b.yaml"
    path.write_text("allowed_domains: [www.st.com]\nsurprise: 1\n", encoding="utf-8")
    with pytest.raises(CorpusError):
        load_manifest(path)


def test_url_check() -> None:
    """HTTPS on an allowed host passes; other schemes, hosts and credentials do not."""
    allowed = frozenset({"www.st.com"})
    check_url("https://www.st.com/a.pdf", allowed)
    for bad in (
        "http://www.st.com/a.pdf",
        "https://evil.example/a.pdf",
        "file:///etc/passwd",
        "https://user:pw@www.st.com/a.pdf",
        "https://www.st.com.evil.example/a.pdf",
    ):
        with pytest.raises(CorpusError):
            check_url(bad, allowed)


def test_fetcher_refuses_disallowed_urls_without_network() -> None:
    """The real fetcher validates the URL before any connection."""
    fetcher = HttpFetcher(("www.st.com",), 5, 0)
    with pytest.raises(CorpusError, match="not allowed"):
        fetcher.get_json("https://evil.example/x.json")


class Sync:
    """Workspace, catalog and manifest of one sync scenario."""

    def __init__(self, root: Path, documents: list[dict[str, Any]], fetcher: FakeFetcher) -> None:
        self.workspace = Workspace(root / "ws")
        self.catalog = Catalog(self.workspace.catalog_path)
        self.manifest = load_manifest(write_manifest(root / "c.yaml", documents))
        self.lock = LockFile(root / "c.lock.json")
        self.lock_path = root / "c.lock.json"
        self.fetcher = fetcher

    def run(self, **kwargs: Any) -> Any:
        """Plan and run a synchronization."""
        plans = plan_sync(self.manifest, self.fetcher, 10_000_000)
        return run_sync(
            plans, self.lock, self.workspace, self.catalog, self.fetcher, 10_000_000, **kwargs
        )


def scenario(tmp_path: Path) -> Sync:
    """One ST datasheet and two STMicro SVD files, plus files that must be ignored."""
    pdf = make_pdf(tmp_path / "ds.pdf", ["USART_ISR text"], [[1, "1 USART", 1]]).read_bytes()
    st_url = "https://www.st.com/resource/en/datasheet/ds.pdf"
    files = {
        st_url: pdf,
        f"{RAW}/STMicro/STM32A.svd": SVD.format("A").encode(),
        f"{RAW}/STMicro/STM32B.svd": SVD.format("B").encode(),
    }
    tree = [
        blob("data/STMicro/STM32A.svd"),
        blob("data/STMicro/STM32B.svd"),
        blob("data/STMicro/Contents.txt"),
        blob("data/NXP/K64.svd"),
        blob("data/STMicro/sub/deep.svd"),
        {"type": "tree", "path": "data/STMicro"},
    ]
    documents = [{"url": st_url, "title": "DS", "document_type": "datasheet"}]
    return Sync(tmp_path, documents, FakeFetcher(files, tree))


def test_sync_downloads_verifies_and_records_metadata(tmp_path: Path) -> None:
    """Only the listed vendor's top-level SVD files and listed URLs are fetched."""
    env = scenario(tmp_path)
    report = env.run()
    assert len(report.imported) == 3 and not report.failed
    assert sorted(env.fetcher.downloads) == sorted(env.fetcher.files)
    documents = env.catalog.list_documents()
    by_type = {d.document_type: d for d in documents}
    assert by_type["datasheet"].source_type.value == "official"
    assert by_type["datasheet"].manufacturer == "STMicroelectronics"
    svd = next(d for d in documents if d.device == "STM32A")
    assert svd.source_type.value == "vendor" and svd.revision == COMMIT[:12]
    assert svd.source_url == f"{RAW}/STMicro/STM32A.svd"
    assert {e.channel for e in env.catalog.list_events(None, 10) if e.event == "imported"} == {
        "sync"
    }


def test_sync_is_incremental_and_writes_the_lock(tmp_path: Path) -> None:
    """A second run downloads nothing; hashes are recorded for trust on first use."""
    env = scenario(tmp_path)
    env.run()
    count = len(env.fetcher.downloads)
    again = env.run()
    assert again.imported == [] and len(again.skipped) == 3
    assert len(env.fetcher.downloads) == count
    assert len(json.loads(env.lock_path.read_text())["hashes"]) == 3


def test_dry_run_and_limit(tmp_path: Path) -> None:
    """A dry run downloads nothing; a limit caps the number of downloads."""
    env = scenario(tmp_path)
    dry = env.run(dry_run=True)
    assert len(dry.planned) == 3 and env.fetcher.downloads == []
    limited = env.run(limit=1)
    assert len(limited.imported) == 1


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    """A file whose content differs from the locked or listed hash is not imported."""
    env = scenario(tmp_path)
    env.run()
    url = f"{RAW}/STMicro/STM32A.svd"
    env.catalog.delete_document(
        next(d for d in env.catalog.list_documents() if d.device == "STM32A").document_id
    )
    env.fetcher.files[url] = SVD.format("TAMPERED").encode()
    report = env.run()
    assert "hash mismatch" in report.failed[url]
    assert all(d.device != "STM32A" for d in env.catalog.list_documents())


def test_manifest_hash_is_enforced(tmp_path: Path) -> None:
    """An explicit sha256 in the manifest wins and must match."""
    pdf = make_pdf(tmp_path / "x.pdf", ["x"]).read_bytes()
    url = "https://www.st.com/resource/en/datasheet/x.pdf"
    env = Sync(
        tmp_path,
        [{"url": url, "sha256": "0" * 64}],
        FakeFetcher({url: pdf}),
    )
    report = env.run()
    assert "hash mismatch" in report.failed[url]
    assert env.catalog.stats().documents == 0


def test_failed_download_does_not_stop_the_run(tmp_path: Path) -> None:
    """One missing file is reported and the others are still imported."""
    env = scenario(tmp_path)
    del env.fetcher.files[f"{RAW}/STMicro/STM32B.svd"]
    report = env.run()
    assert len(report.imported) == 2 and len(report.failed) == 1


def test_cli_sync(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command wires settings, manifest, lock and index together."""
    env = scenario(tmp_path)
    manifest = write_manifest(tmp_path / "m.yaml", [])
    config = tmp_path / "settings.yaml"
    config.write_text(yaml.safe_dump({"workspace": str(tmp_path / "cli-ws")}), encoding="utf-8")
    args = cli._parser().parse_args(  # noqa: SLF001
        ["sync", "--manifest", str(manifest), "--config", str(config), "--source", "svd", "--index"]
    )
    assert cli._sync(args, env.fetcher) == 0  # noqa: SLF001
    out = capsys.readouterr().out
    assert "sync: 2 imported" in out
    assert (tmp_path / "corpus.lock.json").exists()
    unknown = cli._parser().parse_args(  # noqa: SLF001
        ["sync", "--manifest", str(manifest), "--config", str(config), "--source", "nope"]
    )
    assert cli._sync(unknown, env.fetcher) == 1  # noqa: SLF001


def test_include_patterns_restrict_the_svd_files(tmp_path: Path) -> None:
    """Only files whose name matches an include pattern are planned."""
    env = scenario(tmp_path)
    source = env.manifest.sources[0].model_copy(update={"include": ("stm32a*",)})
    narrowed = env.manifest.model_copy(update={"sources": (source,)})
    plans = plan_sync(narrowed, env.fetcher, 10_000_000)
    assert [p.metadata.device for p in plans] == ["STM32A"]


def test_family_pattern_sets_the_device_family(tmp_path: Path) -> None:
    """The first group of the pattern becomes the upper-case device family."""
    env = scenario(tmp_path)
    source = env.manifest.sources[0].model_copy(update={"family_pattern": "(stm32[a-z])"})
    narrowed = env.manifest.model_copy(update={"sources": (source,)})
    plans = plan_sync(narrowed, env.fetcher, 10_000_000)
    assert {p.metadata.device_family for p in plans} == {"STM32A", "STM32B"}


def test_document_entries_accept_generic_types_and_keywords(tmp_path: Path) -> None:
    """Every taxonomy-style document type and its keywords validate and are preserved."""
    documents = [
        {
            "url": f"https://www.st.com/resource/en/doc/{name}.pdf",
            "title": name,
            "document_type": name,
            "keywords": ["getting started", name],
        }
        for name in (
            "technical-guide",
            "requirement-document",
            "datasheet",
            "specification",
            "best-practice",
        )
    ]
    manifest = load_manifest(write_manifest(tmp_path / "c.yaml", documents))
    source = manifest.sources[1]
    assert isinstance(source, UrlSource)
    assert {entry.document_type for entry in source.documents} == {
        "technical-guide",
        "requirement-document",
        "datasheet",
        "specification",
        "best-practice",
    }
    for entry in source.documents:
        assert entry.keywords == ("getting started", entry.document_type)


def test_keywords_are_optional_and_bounded(tmp_path: Path) -> None:
    """Missing or empty keywords stay empty; oversize keyword lists are rejected."""
    documents = [
        {"url": "https://www.st.com/a.pdf", "document_type": "datasheet"},
        {"url": "https://www.st.com/b.pdf", "document_type": "datasheet", "keywords": []},
    ]
    manifest = load_manifest(write_manifest(tmp_path / "c.yaml", documents))
    source = manifest.sources[1]
    assert isinstance(source, UrlSource)
    assert all(entry.keywords == () for entry in source.documents)
    too_many = [{"url": "https://www.st.com/c.pdf", "keywords": [f"k{i}" for i in range(17)]}]
    with pytest.raises(CorpusError, match="keywords"):
        load_manifest(write_manifest(tmp_path / "d.yaml", too_many))


def test_sync_passes_keywords_to_the_import_metadata(tmp_path: Path) -> None:
    """Keywords listed in the manifest reach the catalog through the import path."""
    pdf = make_pdf(tmp_path / "ds.pdf", ["USART_ISR text"], [[1, "1 USART", 1]]).read_bytes()
    url = "https://www.st.com/resource/en/datasheet/kw.pdf"
    documents = [
        {
            "url": url,
            "title": "KW",
            "document_type": "datasheet",
            "keywords": ["readout protection", "rdp"],
        }
    ]
    env = Sync(tmp_path, documents, FakeFetcher({url: pdf}))
    plans = plan_sync(env.manifest, env.fetcher, 10_000_000)
    assert plans[0].metadata.keywords == ("readout protection", "rdp")
    report = env.run()
    assert len(report.imported) == 1
    document = env.catalog.get_document(report.imported[0])
    assert document is not None and document.keywords == ("readout protection", "rdp")


def test_repository_corpus_manifest_is_valid() -> None:
    """The manifest shipped in config/ must stay loadable by `sync`."""
    manifest = load_manifest(Path(__file__).parent.parent / "config" / "corpus.yaml")
    assert len(manifest.allowed_domains) == len({d.lower() for d in manifest.allowed_domains})
    assert {source.name for source in manifest.sources} >= {"cmsis-svd", "st-documents"}


def test_repository_taxonomy_covers_the_corpus_manifest() -> None:
    """Every document type used by config/corpus.yaml is declared in config/taxonomy.yaml."""
    config = Path(__file__).parent.parent / "config"
    manifest = load_manifest(config / "corpus.yaml")
    taxonomy = load_taxonomy(config / "taxonomy.yaml")
    assert taxonomy.unknown_document_types(manifest) == []


def test_cli_sync_warns_about_unknown_document_types(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown document_type triggers a warning, not a failure; a bad taxonomy fails."""
    env = scenario(tmp_path)
    manifest = write_manifest(
        tmp_path / "m.yaml",
        [
            {
                "url": "https://www.st.com/resource/en/datasheet/ds.pdf",
                "document_type": "mystery-type",
            }
        ],
    )
    (tmp_path / "taxonomy.yaml").write_text(
        "document_types: [{name: datasheet}]\n", encoding="utf-8"
    )
    config = tmp_path / "settings.yaml"
    config.write_text(yaml.safe_dump({"workspace": str(tmp_path / "cli-ws")}), encoding="utf-8")
    args = cli._parser().parse_args(  # noqa: SLF001
        ["sync", "--manifest", str(manifest), "--config", str(config), "--dry-run"]
    )
    assert cli._sync(args, env.fetcher) == 0  # noqa: SLF001
    assert "document_type 'mystery-type'" in capsys.readouterr().err
    broken = tmp_path / "broken.yaml"
    broken.write_text("document_types: []\n", encoding="utf-8")
    args = cli._parser().parse_args(  # noqa: SLF001
        [
            "sync",
            "--manifest",
            str(manifest),
            "--config",
            str(config),
            "--dry-run",
            "--taxonomy",
            str(broken),
        ]
    )
    assert cli._sync(args, env.fetcher) == 1  # noqa: SLF001
