# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Exact-match search over the FTS5 index."""

import re

from ..storage.catalog import Catalog
from ..storage.models import SearchFilters, SearchHit

MAX_TERMS = 16
_TERM_RE = re.compile(r"\w+")


def build_match_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 expression.

    Only word characters are kept and every term is quoted, so user input can never inject FTS5
    operators, column filters or prefix/proximity syntax. Terms are OR-ed and ranked by BM25,
    which favors rare terms such as register names. The number of terms is capped (ANSSI R18).

    Parameters
    ----------
    text : str
        User query.

    Returns
    -------
    str | None
        FTS5 expression, or None when the text holds no searchable term.
    """
    seen: dict[str, None] = {}
    for term in _TERM_RE.findall(text):
        seen.setdefault(term.lower(), None)
        if len(seen) == MAX_TERMS:
            break
    if not seen:
        return None
    return " OR ".join(f'"{term}"' for term in seen)


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
