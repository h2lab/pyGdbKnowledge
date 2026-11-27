# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Retrieval evaluation: questions with expected evidence, Recall@k and MRR.

A question lists the evidence a correct answer needs (document, peripheral, section, page,
text). A hit is relevant when it satisfies every criterion of at least one expectation.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from .retrieval.hybrid import HybridSearcher
from .storage.models import SearchFilters, SearchHit

DEFAULT_KS = (1, 3, 5, 10)
_MAX_K = 100
_Criterion = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]


class EvaluationError(Exception):
    """Raised when the question file is unreadable or invalid."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Expectation(_Model):
    """Evidence criteria; every given criterion must match (case-insensitive substrings)."""

    document: _Criterion | None = None
    peripheral: _Criterion | None = None
    section: _Criterion | None = None
    page: int | None = Field(default=None, ge=1)
    page_tolerance: int = Field(default=0, ge=0)
    contains: tuple[_Criterion, ...] = Field(default=(), max_length=16)
    not_contains: tuple[_Criterion, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def _has_positive_criterion(self) -> Self:
        if not (self.document or self.peripheral or self.section or self.page or self.contains):
            raise ValueError("an expectation must contain a positive criterion")
        return self

    def matches(self, hit: SearchHit) -> bool:
        """Return whether ``hit`` satisfies every criterion."""
        chunk, document = hit.chunk, hit.document
        if self.document is not None:
            names = f"{document.document_id} {document.title} {document.device or ''}".lower()
            if self.document.lower() not in names:
                return False
        if self.peripheral is not None and (chunk.peripheral or "").lower() != (
            self.peripheral.lower()
        ):
            return False
        if self.section is not None and self.section.lower() not in chunk.section_path.lower():
            return False
        if self.page is not None and (
            chunk.page is None or abs(chunk.page - self.page) > self.page_tolerance
        ):
            return False
        text = chunk.text.lower()
        return all(needle.lower() in text for needle in self.contains) and all(
            needle.lower() not in text for needle in self.not_contains
        )


class Question(_Model):
    """One evaluation question."""

    id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    category: str = Field(default="general", max_length=32)
    query: str = Field(min_length=1, max_length=500)
    filters: SearchFilters = SearchFilters()
    expect: tuple[Expectation, ...] = Field(min_length=1)
    forbid: tuple[Expectation, ...] = ()


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """Rank (from 1) of the first relevant hit, or None when no returned hit is relevant."""

    question: Question
    rank: int | None
    forbidden_ranks: tuple[int, ...] = ()


@dataclass(slots=True)
class EvaluationReport:
    """Per-question results and aggregate metrics."""

    results: list[QuestionResult] = field(default_factory=list)
    ks: tuple[int, ...] = DEFAULT_KS

    def recall(self, k: int, category: str | None = None) -> float:
        """Share of questions whose first relevant hit is within the top ``k``."""
        scoped = [r for r in self.results if category is None or r.question.category == category]
        if not scoped:
            return 0.0
        return sum(1 for r in scoped if r.rank is not None and r.rank <= k) / len(scoped)

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank of the first relevant hit."""
        if not self.results:
            return 0.0
        return sum(1 / r.rank for r in self.results if r.rank) / len(self.results)

    @property
    def categories(self) -> list[str]:
        """Question categories, sorted."""
        return sorted({r.question.category for r in self.results})

    def forbidden(self, k: int) -> int:
        """Count questions with at least one explicitly forbidden hit within the top k."""
        return sum(any(rank <= k for rank in result.forbidden_ranks) for result in self.results)


def load_questions(path: Path) -> list[Question]:
    """Load and validate a question file (a YAML list, or a mapping with ``questions``).

    Raises
    ------
    EvaluationError
        If the file cannot be read, is invalid, or has duplicate ids.
    """
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        items = loaded["questions"] if isinstance(loaded, dict) else loaded
        questions = [Question.model_validate(item) for item in items]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise EvaluationError(f"cannot read the questions {path}: {exc}") from exc
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise EvaluationError(f"invalid questions {path}: {where}: {first['msg']}") from exc
    if len({q.id for q in questions}) != len(questions):
        raise EvaluationError("question ids must be unique")
    return questions


def evaluate(
    searcher: HybridSearcher,
    questions: Sequence[Question],
    ks: tuple[int, ...] = DEFAULT_KS,
    min_score: float = 0.0,
) -> EvaluationReport:
    """Run every question and locate the first relevant hit.

    Parameters
    ----------
    searcher : HybridSearcher
        Retriever under test.
    questions : Sequence[Question]
        Questions to run.
    ks : tuple[int, ...]
        Cut-offs reported; the largest one is the number of hits requested.
    min_score : float
        Minimum score applied by the searcher.

    Returns
    -------
    EvaluationReport
        Results and metrics.
    """
    depth = min(max(ks), _MAX_K)
    report = EvaluationReport(ks=ks)
    for question in questions:
        hits = searcher.search(question.query, question.filters, depth, min_score)
        forbidden_ranks = tuple(
            index
            for index, hit in enumerate(hits, start=1)
            if any(expectation.matches(hit) for expectation in question.forbid)
        )
        rank = next(
            (
                index
                for index, hit in enumerate(hits, start=1)
                if index not in forbidden_ranks
                and any(expectation.matches(hit) for expectation in question.expect)
            ),
            None,
        )
        report.results.append(QuestionResult(question, rank, forbidden_ranks))
    return report
