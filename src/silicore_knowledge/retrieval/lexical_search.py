# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Exact-match search over the FTS5 index."""

import re
from collections.abc import Sequence

from ..storage.catalog import Catalog
from ..storage.models import SearchFilters, SearchHit

MAX_TERMS = 16
_TERM_RE = re.compile(r"\w+")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_NUMBERED_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9]{2,}[0-9]", re.IGNORECASE)
# Words present in nearly every chunk (or in no useful way): they only add BM25 noise. A query
# made only of such words falls back to using them all.
_STOP_WORDS = frozenset(
    "a an the of to in on for and or is are be how what which does do can with from by at as "
    "it its this that bit bits register registers value values".split()
)


def build_match_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 expression.

    Only word characters are kept and every term is quoted, so user input can never inject FTS5
    operators, column filters or prefix/proximity syntax. Terms are OR-ed and ranked by BM25,
    which favors rare terms such as register names; common stop words are dropped. The number of terms is capped (ANSSI R18).

    Parameters
    ----------
    text : str
        User query.

    Returns
    -------
    str | None
        FTS5 expression, or None when the text holds no searchable term.
    """
    terms = [term.lower() for term in _TERM_RE.findall(text)]
    kept = [term for term in terms if term not in _STOP_WORDS] or terms
    seen: dict[str, None] = {}
    for term in kept:
        seen.setdefault(term, None)
        if len(seen) == MAX_TERMS:
            break
    if not seen:
        return None
    return " OR ".join(f'"{term}"' for term in seen)


def build_identifier_query(text: str) -> str | None:
    """Build an FTS5 expression from the identifier-like terms only (``FLASH_ACR``).

    A register or macro name is the most specific evidence in a question; ranking by these
    terms alone keeps them from being diluted by the surrounding natural language.

    Returns
    -------
    str | None
        FTS5 expression, or None when the text holds no identifier with an underscore.
    """
    seen: dict[str, None] = {}
    for match in _IDENTIFIER_RE.finditer(text):
        seen.setdefault(match.group(0).lower(), None)
        if len(seen) == MAX_TERMS:
            break
    return " OR ".join(f'"{term}"' for term in seen) or None


def build_conjunctive_identifier_query(text: str) -> str | None:
    """Match all retained query terms when one is an identifier and another adds context.

    Parameters
    ----------
    text : str
        User query.

    Returns
    -------
    str | None
        Bounded FTS5 expression, or None when no identifier and context are both retained.
    """
    match = build_match_query(text)
    identifiers = build_identifier_query(text)
    if match is None or identifiers is None:
        return None
    terms = match.split(" OR ")
    if len(terms) < 2 or not set(terms).intersection(identifiers.split(" OR ")):
        return None
    return " AND ".join(terms)


def build_numbered_identifier_query(text: str) -> str | None:
    """Match numbered identifier prefixes and an adjacent contextual phrase.

    The prefix search is only a candidate generator: callers must check that a longer numeric
    identifier (for example ``USART10``) is not mistaken for the requested ``USART1``.

    Parameters
    ----------
    text : str
        User query.

    Returns
    -------
    str | None
        Bounded FTS5 expression, or None without a numbered identifier and contextual phrase.
    """
    match = build_match_query(text)
    if match is None:
        return None
    terms = match.split(" OR ")
    if len(terms) < 2:
        return None
    identifiers = [term for term in terms if _NUMBERED_IDENTIFIER_RE.fullmatch(term[1:-1])]
    if not identifiers:
        return None
    retained = {term[1:-1] for term in terms}
    words = _TERM_RE.findall(text.lower())
    phrases: dict[str, None] = {}
    for first, second in zip(words, words[1:]):
        if (
            first in retained
            and second in retained
            and first not in _STOP_WORDS
            and second not in _STOP_WORDS
            and not _NUMBERED_IDENTIFIER_RE.fullmatch(first)
            and not _NUMBERED_IDENTIFIER_RE.fullmatch(second)
            and not _IDENTIFIER_RE.fullmatch(first)
            and not _IDENTIFIER_RE.fullmatch(second)
        ):
            phrases.setdefault(f'"{first} {second}"', None)
            if len(phrases) == MAX_TERMS:
                break
    if not phrases:
        return None
    conjunction = " AND ".join(f"{term}*" if term in identifiers else term for term in terms)
    return f"{conjunction} AND ({' OR '.join(phrases)})"


def build_expansion_query(terms: Sequence[str]) -> str | None:
    """Turn controlled expansion terms into a safe FTS5 expression.

    Each term becomes a quoted phrase of its word characters, so multi-word aliases such as
    ``readout protection`` match consecutive tokens and nothing can inject FTS5 syntax.

    Parameters
    ----------
    terms : Sequence[str]
        Expansion terms from the concept registry.

    Returns
    -------
    str | None
        FTS5 expression, or None when no term holds a searchable token.
    """
    phrases: dict[str, None] = {}
    for term in terms:
        tokens = [token.lower() for token in _TERM_RE.findall(term)]
        if not tokens:
            continue
        phrases.setdefault(" ".join(tokens), None)
        if len(phrases) == MAX_TERMS:
            break
    if not phrases:
        return None
    return " OR ".join(f'"{phrase}"' for phrase in phrases)


def expansion_search(
    catalog: Catalog, terms: Sequence[str], filters: SearchFilters | None = None, limit: int = 20
) -> list[SearchHit]:
    """Search chunks by controlled expansion terms only; empty when none is searchable."""
    match = build_expansion_query(terms)
    if match is None:
        return []
    return catalog.search_chunks(match, filters or SearchFilters(), limit)


def identifier_search(
    catalog: Catalog, query: str, filters: SearchFilters | None = None, limit: int = 20
) -> list[SearchHit]:
    """Search by the identifier-like terms of ``query`` only; empty when it has none."""
    match = build_identifier_query(query)
    if match is None:
        return []
    return catalog.search_chunks(match, filters or SearchFilters(), limit)


def conjunctive_identifier_search(
    catalog: Catalog, query: str, filters: SearchFilters | None = None, limit: int = 20
) -> list[SearchHit]:
    """Rank passages containing an identifier and all other retained query terms."""
    match = build_conjunctive_identifier_query(query)
    if match is None:
        return []
    return catalog.search_chunks(match, filters or SearchFilters(), limit)


def numbered_identifier_search(
    catalog: Catalog, query: str, filters: SearchFilters | None = None, limit: int = 20
) -> list[SearchHit]:
    """Find context-matching identifier prefixes without confusing adjacent numbers.

    An exact device-family match can legitimately prefix a numbered device in a section path.

    Parameters
    ----------
    catalog : Catalog
        Catalog holding the FTS5 index.
    query : str
        User query.
    filters : SearchFilters | None
        Metadata filters.
    limit : int
        Maximum number of candidates checked before numeric-boundary validation.

    Returns
    -------
    list[SearchHit]
        Validated candidates ordered by decreasing relevance.
    """
    match = build_numbered_identifier_query(query)
    if match is None:
        return []
    terms = build_match_query(query)
    if terms is None:
        return []
    identifiers = [
        term[1:-1] for term in terms.split(" OR ") if _NUMBERED_IDENTIFIER_RE.fullmatch(term[1:-1])
    ]
    patterns = [
        re.compile(r"(?<!\w)" + re.escape(identifier) + r"(?!\d)", re.IGNORECASE)
        for identifier in identifiers
    ]
    hits = catalog.search_chunks(match, filters or SearchFilters(), limit)
    results = []
    for hit in hits:
        family = (hit.document.device_family or "").lower()
        indexed_text = " ".join(
            (
                hit.chunk.text,
                hit.chunk.section_path,
                hit.chunk.peripheral or "",
                *hit.document.keywords,
            )
        )
        if all(
            identifier == family or pattern.search(indexed_text)
            for identifier, pattern in zip(identifiers, patterns, strict=True)
        ):
            results.append(hit)
    return results


def lexical_search(
    catalog: Catalog, query: str, filters: SearchFilters | None = None, limit: int = 20
) -> list[SearchHit]:
    """Search chunks by exact terms (register names, addresses, error codes).

    Terms match whole tokens: ``USART_ISR`` matches ``USART_ISR`` but not ``USART``.
    Quarantined documents are never returned.

    Parameters
    ----------
    catalog : Catalog
        Catalog to search.
    query : str
        User query.
    filters : SearchFilters | None
        Metadata filters.
    limit : int
        Maximum number of hits.

    Returns
    -------
    list[SearchHit]
        Hits by decreasing relevance; empty when the query has no searchable term.
    """
    match = build_match_query(query)
    if match is None:
        return []
    return catalog.search_chunks(match, filters or SearchFilters(), limit)
