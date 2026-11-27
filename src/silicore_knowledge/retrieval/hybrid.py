# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Hybrid retrieval: exact (FTS5) and semantic (vector) results fused by rank."""

import logging
from collections.abc import Mapping

from ..storage.catalog import Catalog
from ..storage.models import SearchFilters, SearchHit, SourceType
from .concepts import ConceptRegistry
from .embeddings import Embedder, EmbeddingError
from .lexical_search import expansion_search, identifier_search, lexical_search
from .vector_store import VectorStore, VectorStoreError

logger = logging.getLogger("silicore_knowledge")

RRF_K = 60
CANDIDATE_FACTOR = 5
MAX_CANDIDATES = 200
EXPANSION_RRF_WEIGHT = 0.25

# Multiplier applied per source reliability class; official documents rank first.
DEFAULT_SOURCE_WEIGHTS: Mapping[SourceType, float] = {
    SourceType.OFFICIAL: 1.0,
    SourceType.VENDOR: 0.9,
    SourceType.COMMUNITY: 0.7,
}


class HybridSearcher:
    """Fuse lexical and vector rankings with reciprocal rank fusion (RRF).

    Scores are normalized to ``(0, 1]``: a chunk ranked first by both retrievers from an
    official source scores 1.0, so a single relevance threshold is meaningful. When the vector
    side is absent or failing, the search degrades to lexical results only.

    Parameters
    ----------
    catalog : Catalog
        Catalog holding chunks and the FTS5 index.
    embedder : Embedder | None
        Query embedder; None disables semantic search.
    vector_store : VectorStore | None
        Vector index; None disables semantic search.
    source_weights : Mapping[SourceType, float] | None
        Reliability multiplier per source type.
    concepts : ConceptRegistry | None
        Declarative concept registry; None disables query expansion.
    """

    def __init__(
        self,
        catalog: Catalog,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        source_weights: Mapping[SourceType, float] | None = None,
        concepts: ConceptRegistry | None = None,
    ) -> None:
        self._catalog = catalog
        self._embedder = embedder
        self._vector_store = vector_store
        self._weights = dict(source_weights or DEFAULT_SOURCE_WEIGHTS)
        self._concepts = concepts

    @property
    def semantic(self) -> bool:
        """Whether semantic search is configured."""
        return self._embedder is not None and self._vector_store is not None

    def search(
        self, query: str, filters: SearchFilters, limit: int, min_score: float = 0.0
    ) -> list[SearchHit]:
        """Search and rank chunks.

        Parameters
        ----------
        query : str
            User query.
        filters : SearchFilters
            Metadata filters, enforced by both retrievers.
        limit : int
            Maximum number of hits.
        min_score : float
            Hits scoring below this normalized score are dropped (ANSSI R29).

        Returns
        -------
        list[SearchHit]
            Hits by decreasing score, with normalized scores.
        """
        if limit < 1:
            return []
        candidates = min(max(limit * CANDIDATE_FACTOR, 20), MAX_CANDIDATES)
        rankings: list[tuple[list[SearchHit], float, bool]] = [
            (lexical_search(self._catalog, query, filters, candidates), 1.0, True)
        ]
        identifiers = identifier_search(self._catalog, query, filters, candidates)
        if identifiers:
            rankings.append((identifiers, 1.0, True))
        expansion = self._expansion_ranking(query, filters, candidates)
        if expansion:
            rankings.append((expansion, EXPANSION_RRF_WEIGHT, False))
        vector_ranking = self._vector_ranking(query, filters, candidates)
        if vector_ranking is not None:
            rankings.append((vector_ranking, 1.0, True))

        fused: dict[str, float] = {}
        hits: dict[str, SearchHit] = {}
        for ranking, weight, _ in rankings:
            for rank, hit in enumerate(ranking, start=1):
                chunk_id = hit.chunk.chunk_id
                hits[chunk_id] = hit
                fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (RRF_K + rank)

        # Expansion is a low-confidence bonus, not another full retrieval modality. Excluding
        # it from the normalizer prevents an active concept from lowering existing result scores
        # and unexpectedly filtering them through min_score.
        best_possible = sum(weight for _, weight, normalize in rankings if normalize) / (RRF_K + 1)
        scored = [
            (chunk_id, fused[chunk_id] * self._weights[hits[chunk_id].document.source_type])
            for chunk_id in fused
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        results: list[SearchHit] = []
        for chunk_id, raw in scored:
            score = min(raw / best_possible, 1.0)
            if score < min_score:
                break
            results.append(hits[chunk_id].model_copy(update={"score": score}))
            if len(results) == limit:
                break
        return results

    def _expansion_ranking(
        self, query: str, filters: SearchFilters, candidates: int
    ) -> list[SearchHit]:
        """Rank chunks by the controlled expansion terms of the query's concepts.

        The expansion comes from the declarative registry only; its ranking has a lower RRF
        weight than direct lexical, identifier, or vector matches so it can surface a document
        using another vocabulary without overwhelming stronger evidence.
        """
        if self._concepts is None:
            return []
        terms = self._concepts.expansion_terms(query)
        if not terms:
            return []
        logger.debug("query expanded with %d concept terms", len(terms))
        return expansion_search(self._catalog, terms, filters, candidates)

    def _vector_ranking(
        self, query: str, filters: SearchFilters, candidates: int
    ) -> list[SearchHit] | None:
        if self._embedder is None or self._vector_store is None or not query.strip():
            return None
        try:
            vector = self._embedder.embed([query])[0]
            vector_hits = self._vector_store.search(vector, filters, candidates)
        except (EmbeddingError, VectorStoreError) as exc:
            logger.warning("semantic search unavailable, using lexical only: %s", exc)
            return None
        # The vector index may be stale: only chunks of active documents survive.
        loaded = self._catalog.get_hits([hit.chunk_id for hit in vector_hits])
        return [loaded[hit.chunk_id] for hit in vector_hits if hit.chunk_id in loaded]
