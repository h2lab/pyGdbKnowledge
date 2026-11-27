# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Declarative document taxonomy: the document types the corpus is allowed to use.

The taxonomy is advisory: an unknown ``document_type`` in the corpus manifest is reported as a
warning, never rejected, so existing corpora keep working while the vocabulary converges.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .manifest import Manifest, UrlSource

_TYPE_PATTERN = r"^[a-z0-9][a-z0-9-]{0,47}$"


class TaxonomyError(Exception):
    """Raised when the taxonomy file is unreadable or invalid."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DocumentTypeDef(_Model):
    """One document type of the taxonomy."""

    name: str = Field(pattern=_TYPE_PATTERN)
    description: str | None = Field(default=None, max_length=300)


class Taxonomy(_Model):
    """The set of known document types."""

    document_types: tuple[DocumentTypeDef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> "Taxonomy":
        names = [entry.name for entry in self.document_types]
        if len(set(names)) != len(names):
            raise ValueError("document type names must be unique")
        return self

    @property
    def names(self) -> frozenset[str]:
        """Known document type names."""
        return frozenset(entry.name for entry in self.document_types)

    def unknown_document_types(self, manifest: Manifest) -> list[str]:
        """Return the manifest's document types absent from the taxonomy, sorted.

        Parameters
        ----------
        manifest : Manifest
            Corpus manifest to check; only explicit URL sources carry a ``document_type``.

        Returns
        -------
        list[str]
            Unknown type names; empty when every used type is declared.
        """
        used = {
            entry.document_type
            for source in manifest.sources
            if isinstance(source, UrlSource)
            for entry in source.documents
            if entry.document_type is not None
        }
        return sorted(used - self.names)


def load_taxonomy(path: Path) -> Taxonomy:
    """Load and validate a taxonomy file (a mapping with ``document_types``).

    Raises
    ------
    TaxonomyError
        If the file cannot be read or is not a valid taxonomy.
    """
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return Taxonomy.model_validate(loaded)
    except (OSError, yaml.YAMLError) as exc:
        raise TaxonomyError(f"cannot read the taxonomy {path}: {exc}") from exc
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise TaxonomyError(f"invalid taxonomy {path}: {where}: {first['msg']}") from exc
