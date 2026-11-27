# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Concept registry: declarative aliases, acronyms and related terms for query expansion.

A concept groups the terms a technical intent may hide behind (``JTAG``, ``SWD``, ``readout
protection`` for firmware extraction). Expansion is controlled: it comes from a reviewed
configuration file, never from the query or from a model, and it is bounded in every
dimension (ANSSI R27).
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

MAX_MATCHED_CONCEPTS = 4
MAX_EXPANSION_TERMS = 16

_Term = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class ConceptError(Exception):
    """Raised when the concept file is unreadable or inconsistent."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Concept(_Model):
    """One concept: the terms that name it and the concepts it relates to."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,47}$")
    label: str | None = Field(default=None, max_length=120)
    terms: tuple[_Term, ...] = Field(min_length=1, max_length=16)
    related: tuple[Annotated[str, StringConstraints(min_length=1, max_length=48)], ...] = Field(
        default=(), max_length=8
    )


class _ConceptFile(_Model):
    concepts: tuple[Concept, ...] = Field(min_length=1)


def _term_pattern(term: str) -> re.Pattern[str]:
    # A multi-word term matches across any whitespace; boundaries keep BLE from matching CABLE.
    words = [re.escape(word) for word in term.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)", re.IGNORECASE)


class ConceptRegistry:
    """Validated set of concepts with precompiled term matching.

    Parameters
    ----------
    concepts : Sequence[Concept]
        Concepts to register.

    Raises
    ------
    ConceptError
        If concept ids are not unique or a ``related`` id is unknown.
    """

    def __init__(self, concepts: Sequence[Concept]) -> None:
        by_id: dict[str, Concept] = {}
        for concept in concepts:
            if concept.id in by_id:
                raise ConceptError(f"duplicate concept id {concept.id!r}")
            by_id[concept.id] = concept
        for concept in concepts:
            for related in concept.related:
                if related not in by_id:
                    raise ConceptError(f"concept {concept.id!r} relates to unknown {related!r}")
        self._by_id = by_id
        self._terms: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
            concept.id: tuple((term, _term_pattern(term)) for term in concept.terms)
            for concept in concepts
        }

    def __len__(self) -> int:
        """Return the number of registered concepts."""
        return len(self._by_id)

    def match(self, text: str) -> tuple[Concept, ...]:
        """Return the concepts whose terms appear in ``text``, bounded, in registry order."""
        matched: list[Concept] = []
        for concept in self._by_id.values():
            if any(pattern.search(text) for _, pattern in self._terms[concept.id]):
                matched.append(concept)
                if len(matched) == MAX_MATCHED_CONCEPTS:
                    break
        return tuple(matched)

    def expansion_terms(self, text: str) -> tuple[str, ...]:
        """Return the expansion terms for ``text``: sibling and related terms it does not use.

        Matched concepts contribute their other terms; concepts they declare as ``related``
        contribute theirs (one hop). Terms already present in the text are dropped and the
        result is deduplicated case-insensitively and capped.

        Parameters
        ----------
        text : str
            User query.

        Returns
        -------
        tuple[str, ...]
            Expansion terms; empty when no concept matches.
        """
        scope: dict[str, Concept] = {}
        for concept in self.match(text):
            scope.setdefault(concept.id, concept)
            for related in concept.related:
                scope.setdefault(related, self._by_id[related])
        expansion: dict[str, str] = {}
        for concept in scope.values():
            for term, pattern in self._terms[concept.id]:
                if pattern.search(text):
                    continue
                expansion.setdefault(term.casefold(), term)
                if len(expansion) == MAX_EXPANSION_TERMS:
                    return tuple(expansion.values())
        return tuple(expansion.values())


def load_concepts(path: Path) -> ConceptRegistry:
    """Load and validate a concept file (a mapping with ``concepts``).

    Raises
    ------
    ConceptError
        If the file cannot be read, is invalid, or is inconsistent.
    """
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        parsed = _ConceptFile.model_validate(loaded)
    except (OSError, yaml.YAMLError) as exc:
        raise ConceptError(f"cannot read the concepts {path}: {exc}") from exc
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise ConceptError(f"invalid concepts {path}: {where}: {first['msg']}") from exc
    return ConceptRegistry(parsed.concepts)
