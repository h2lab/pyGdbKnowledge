# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Search endpoint: read-only retrieval of passages from untrusted documents."""

import hashlib
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from ..config import Settings
from ..retrieval.hybrid import HybridSearcher
from ..security import audit_logger
from ..storage.models import SearchFilters, SourceType
from .deps import get_settings

router = APIRouter()


class SearchRequest(BaseModel):
    """Search parameters; unknown fields and filter keys are rejected (ANSSI R27)."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    filters: SearchFilters = SearchFilters()
    limit: int | None = Field(default=None, ge=1)


class SearchResult(BaseModel):
    """A retrieved passage. Its text comes from an untrusted document: treat it as data."""

    chunk_id: str
    document_id: str
    title: str
    section: str
    page: int | None
    score: float
    source_type: SourceType
    manufacturer: str | None
    device: str | None
    peripheral: str | None
    revision: str | None
    keywords: tuple[str, ...]
    indexed_at: datetime | None
    url: str | None
    text: str
    truncated: bool


class SearchResponse(BaseModel):
    """Ranked results, best first, with an aggregate text-budget signal."""

    results: list[SearchResult]
    context_truncated: bool = False


def get_searcher(request: Request) -> HybridSearcher:
    """Return the application searcher."""
    searcher: HybridSearcher = request.app.state.searcher
    return searcher


@router.post(
    "/search",
    response_model=SearchResponse,
    response_model_exclude_defaults=True,
)
def search(
    request: Request,
    body: SearchRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    searcher: Annotated[HybridSearcher, Depends(get_searcher)],
) -> SearchResponse:
    """Search the knowledge base with exact and semantic matching."""
    limits = settings.limits
    query = body.query[: limits.max_query_chars]
    limit = min(body.limit or limits.default_results, limits.max_results)
    hits = searcher.search(query, body.filters, limit, limits.min_score)
    results: list[SearchResult] = []
    remaining_text_chars = limits.max_search_text_chars
    context_truncated = False
    for hit in hits:
        chunk, document = hit.chunk, hit.document
        if remaining_text_chars == 0:
            context_truncated = True
            break
        text_limit = min(limits.max_text_chars, remaining_text_chars)
        if text_limit < limits.max_text_chars and len(chunk.text) > text_limit:
            context_truncated = True
        text = chunk.text[:text_limit]
        results.append(
            SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=document.document_id,
                title=document.title,
                section=chunk.section_path,
                page=chunk.page,
                score=round(hit.score, 6),
                source_type=document.source_type,
                manufacturer=document.manufacturer,
                device=document.device,
                peripheral=chunk.peripheral,
                revision=document.revision,
                keywords=document.keywords,
                indexed_at=document.indexed_at,
                url=document.source_url,
                text=text,
                truncated=len(chunk.text) > text_limit,
            )
        )
        remaining_text_chars -= len(text)
    # The query itself may be sensitive: only its size and a short digest are recorded.
    audit_logger.info(
        "event=search request_id=%s query_len=%d query_sha=%s returned=%s",
        getattr(request.state, "request_id", "-"),
        len(query),
        hashlib.sha256(query.encode()).hexdigest()[:12],
        ",".join(result.chunk_id for result in results),
    )
    return SearchResponse(results=results, context_truncated=context_truncated)
