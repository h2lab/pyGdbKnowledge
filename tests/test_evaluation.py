# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the retrieval evaluation (questions, Recall@k, MRR, command)."""

from pathlib import Path

import pytest
import yaml
from test_svd import SVD

from pyGdbKnowledge import cli
from pyGdbKnowledge.config import Settings
from pyGdbKnowledge.evaluation import EvaluationError, evaluate, load_questions
from pyGdbKnowledge.ingestion.importer import DocumentMetadata, import_document
from pyGdbKnowledge.ingestion.pipeline import Indexer
from pyGdbKnowledge.retrieval.hybrid import HybridSearcher
from pyGdbKnowledge.storage.catalog import Catalog
from pyGdbKnowledge.storage.models import SourceType
from pyGdbKnowledge.workspace import Workspace

QUESTIONS = [
    {
        "id": "hit",
        "category": "register",
        "query": "USART1 CR1 control register",
        "expect": [{"peripheral": "USART1", "contains": ["USART1_CR1"]}],
    },
    {
        "id": "wrong-page",
        "query": "USART1 CR1 control register",
        "expect": [{"peripheral": "USART1", "page": 3}],
    },
    {
        "id": "alternative",
        "query": "USART2 derived registers",
        "expect": [{"peripheral": "NOPE"}, {"document": "stm32test", "section": "USART2"}],
    },
]


@pytest.fixture
def env(tmp_path: Path) -> tuple[Settings, Workspace, Catalog]:
    """Workspace with one indexed SVD document."""
    source = tmp_path / "STM32TEST.svd"
    source.write_text(SVD, encoding="utf-8")
    settings = Settings(workspace=tmp_path / "ws")
    workspace = Workspace(settings.workspace)
    catalog = Catalog(workspace.catalog_path)
    meta = DocumentMetadata(source_type=SourceType.VENDOR, title="STM32TEST SVD")
    document = import_document(workspace, catalog, source, meta, 10_000_000).document
    Indexer(settings, workspace, catalog).index_document(document.document_id)
    return settings, workspace, catalog


def write(path: Path, questions: object) -> Path:
    """Write a question file."""
    path.write_text(yaml.safe_dump({"questions": questions}), encoding="utf-8")
    return path


def test_metrics_and_criteria(env: tuple[Settings, Workspace, Catalog], tmp_path: Path) -> None:
    """Criteria are conjunctive within an expectation and alternative across expectations."""
    questions = load_questions(write(tmp_path / "q.yaml", QUESTIONS))
    report = evaluate(HybridSearcher(env[2]), questions)
    ranks = {r.question.id: r.rank for r in report.results}
    assert ranks["hit"] == 1
    assert ranks["wrong-page"] is None  # SVD chunks have no page
    assert ranks["alternative"] is not None
    assert report.recall(1, "register") == 1.0
    assert report.recall(10) == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx((1 + 1 / ranks["alternative"]) / 3)  # type: ignore[operator]


@pytest.mark.parametrize(
    "content",
    [
        [{"id": "a", "query": "x", "expect": []}],
        [{"id": "a", "query": "x", "expect": [{"bogus": 1}]}],
        [{"id": "a", "query": "x", "expect": [{"page": 1}]}] * 2,
    ],
)
def test_invalid_question_files(tmp_path: Path, content: object) -> None:
    """Empty expectations, unknown keys and duplicate ids are refused."""
    with pytest.raises(EvaluationError):
        load_questions(write(tmp_path / "q.yaml", content))


def test_command(
    env: tuple[Settings, Workspace, Catalog],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command prints metrics and fails below the requested recall."""
    config = tmp_path / "s.yaml"
    config.write_text(yaml.safe_dump({"workspace": str(tmp_path / "ws")}), encoding="utf-8")
    questions = write(tmp_path / "q.yaml", QUESTIONS)
    assert cli.main(["evaluate", str(questions), "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "wrong-page" in out and "R@5=" in out and "MRR=" in out
    assert cli.main(["evaluate", str(questions), "--config", str(config), "--min-recall", "1"]) == 1
