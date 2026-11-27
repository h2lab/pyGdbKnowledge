# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the declarative document taxonomy."""

from pathlib import Path

import pytest
import yaml

from silicore_knowledge.corpus.manifest import load_manifest
from silicore_knowledge.corpus.taxonomy import TaxonomyError, load_taxonomy


def _write(path: Path, names: list[str]) -> Path:
    path.write_text(
        yaml.safe_dump({"document_types": [{"name": name} for name in names]}), encoding="utf-8"
    )
    return path


def _manifest(path: Path, document_types: list[str]) -> Path:
    manifest = {
        "allowed_domains": ["www.st.com"],
        "sources": [
            {
                "name": "st",
                "kind": "urls",
                "source_type": "official",
                "documents": [
                    {"url": f"https://www.st.com/{i}.pdf", "document_type": document_type}
                    for i, document_type in enumerate(document_types)
                ],
            }
        ],
    }
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def test_taxonomy_roundtrip_and_unknown_types(tmp_path: Path) -> None:
    """Known types are accepted; unknown manifest types are reported sorted."""
    taxonomy = load_taxonomy(_write(tmp_path / "t.yaml", ["datasheet", "errata"]))
    assert taxonomy.names == {"datasheet", "errata"}
    manifest = load_manifest(_manifest(tmp_path / "m.yaml", ["datasheet", "zz-new", "aa-new"]))
    assert taxonomy.unknown_document_types(manifest) == ["aa-new", "zz-new"]
    covered = load_manifest(_manifest(tmp_path / "m2.yaml", ["datasheet"]))
    assert taxonomy.unknown_document_types(covered) == []


def test_invalid_taxonomy_is_rejected(tmp_path: Path) -> None:
    """Duplicates, malformed names, empty sets and unreadable files raise TaxonomyError."""
    with pytest.raises(TaxonomyError, match="unique"):
        load_taxonomy(_write(tmp_path / "dup.yaml", ["datasheet", "datasheet"]))
    with pytest.raises(TaxonomyError, match="pattern"):
        load_taxonomy(_write(tmp_path / "bad.yaml", ["Not Valid"]))
    with pytest.raises(TaxonomyError):
        load_taxonomy(_write(tmp_path / "empty.yaml", []))
    with pytest.raises(TaxonomyError, match="cannot read"):
        load_taxonomy(tmp_path / "missing.yaml")


def test_repository_taxonomy_is_valid() -> None:
    """The taxonomy shipped in config/ must stay loadable."""
    taxonomy = load_taxonomy(Path(__file__).parent.parent / "config" / "taxonomy.yaml")
    assert {"datasheet", "errata", "specification", "svd"} <= taxonomy.names
