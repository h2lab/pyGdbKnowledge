# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Corpus manifest: which documents may be downloaded, from where, with which provenance."""

from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..storage.models import SHA256_PATTERN, SourceType

GITHUB_API_HOST = "api.github.com"
GITHUB_RAW_HOST = "raw.githubusercontent.com"


class CorpusError(Exception):
    """Raised for an invalid manifest, a forbidden URL or a failed download."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DocumentEntry(_Model):
    """One explicitly listed document; ``sha256`` is optional (trust on first use otherwise)."""

    url: str = Field(pattern=r"^https://\S+$")
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    title: str | None = None
    manufacturer: str | None = None
    device_family: str | None = None
    device: str | None = None
    document_type: str | None = None
    revision: str | None = None
    keywords: list[str] = Field(default_factory=list)

    def __getitem__(self, key):
        # Permettre l'accès par clé pour compatibilité avec les tests existants
        if key == "document_type":
            return self.document_type
        elif key == "keywords":
            return self.keywords
        elif key == "url":
            return self.url
        elif key == "title":
            return self.title
        elif key == "manufacturer":
            return self.manufacturer
        elif key == "device_family":
            return self.device_family
        elif key == "device":
            return self.device
        elif key == "revision":
            return self.revision
        raise KeyError(key)


class UrlSource(_Model):
    """A source made of an explicit list of document URLs."""

    kind: Literal["urls"]
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    source_type: SourceType
    manufacturer: str | None = None
    documents: tuple[DocumentEntry, ...] = ()

    def __getitem__(self, key):
        if key == "documents":
            return self.documents
        raise KeyError(key)


class GithubSvdSource(_Model):
    """SVD files of a GitHub repository laid out as ``<prefix>/<Vendor>/<device>.svd``.

    ``include`` optionally restricts the files to these patterns (``fnmatch``, case-insensitive,
    matched on the file name without extension); empty means every file of the vendors.

    ``family_pattern`` is a regular expression matched at the start of the file name; its first
    group (or the whole match) becomes the ``device_family`` (``STM32F4`` for ``STM32F407``).

    ``ref`` should be a full commit hash: the files are then immutable and reproducible. A
    branch name is resolved to its current commit at each sync.
    """

    kind: Literal["github_svd"]
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    source_type: SourceType
    repo: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    ref: str = Field(default="main", pattern=r"^[A-Za-z0-9_./-]{1,100}$")
    path_prefix: str = Field(default="data", pattern=r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$")
    vendors: tuple[str, ...] = Field(min_length=1)
    include: tuple[str, ...] = ()
    family_pattern: str | None = None
    manufacturers: dict[str, str] = Field(default_factory=dict)


Source = Annotated[UrlSource | GithubSvdSource, Field(discriminator="kind")]


class Manifest(_Model):
    """Allowed domains, download limits and the sources to synchronize."""

    allowed_domains: tuple[str, ...] = Field(min_length=1)
    request_delay_s: float = Field(default=1.0, ge=0, le=60)
    timeout_s: float = Field(default=60.0, gt=0, le=600)
    sources: tuple[Source, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "Manifest":
        allowed = {domain.lower() for domain in self.allowed_domains}
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            raise ValueError("source names must be unique")
        for source in self.sources:
            if isinstance(source, GithubSvdSource):
                missing = {GITHUB_API_HOST, GITHUB_RAW_HOST} - allowed
                if missing:
                    raise ValueError(f"source {source.name!r} needs {sorted(missing)} allowed")
                continue
            for entry in source.documents:
                host = (urlsplit(entry.url).hostname or "").lower()
                if host not in allowed:
                    raise ValueError(f"{entry.url}: host {host!r} is not an allowed domain")
        return self


def load_manifest(path: Path) -> Manifest:
    """Load and validate a corpus manifest.

    Raises
    ------
    CorpusError
        If the file cannot be read or is not a valid manifest.
    """
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return Manifest.model_validate(loaded)
    except (OSError, yaml.YAMLError) as exc:
        raise CorpusError(f"cannot read the manifest {path}: {exc}") from exc
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise CorpusError(f"invalid manifest {path}: {where}: {first['msg']}") from exc
