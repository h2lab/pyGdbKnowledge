# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the retrieval evaluation (questions, Recall@k, MRR, command)."""

from pathlib import Path

import pytest
import yaml
from test_svd import SVD

from silicore_knowledge import cli
from silicore_knowledge.config import Settings
from silicore_knowledge.evaluation import EvaluationError, Expectation, evaluate, load_questions
from silicore_knowledge.ingestion.importer import DocumentMetadata, import_document
from silicore_knowledge.ingestion.pipeline import Indexer
from silicore_knowledge.retrieval.hybrid import HybridSearcher
from silicore_knowledge.storage.catalog import Catalog
from silicore_knowledge.storage.models import Chunk, Document, SearchHit, SourceType
from silicore_knowledge.workspace import Workspace

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
    Indexer(settings, workspace, catalog).release(document.document_id)
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
    ("title", "peripheral", "text", "relevant"),
    [
        ("STM32U595 SVD", "RCC", "RCC_APB2ENR: USART1EN clock enable", True),
        ("STM32U595 SVD", "SEC_RCC", "RCC_APB2ENR: USART1EN clock enable", True),
        ("STM32U595xx datasheet", None, "RCC_APB2ENR: USART1EN clock enable", True),
        ("STM32U595 SVD", "RCC", "USART1SEL: USART1 clock source selection", False),
        ("STM32U595xx datasheet", None, "General device information", False),
        ("STM32F407 SVD", "RCC", "RCC_APB2ENR: USART1EN clock enable", False),
    ],
)
def test_u5_clock_question_requires_clock_enable_evidence(
    title: str, peripheral: str | None, text: str, relevant: bool
) -> None:
    """Clock selection or an unrelated U5 passage does not prove clock enablement."""
    questions = load_questions(Path(__file__).parent.parent / "config" / "eval" / "questions.yaml")
    question = next(q for q in questions if q.id == "cross-u5-rcc")
    document = Document(
        document_id=title.lower().replace(" ", "-"),
        title=title,
        content_hash="a" * 64,
        source_type=SourceType.VENDOR,
    )
    chunk = Chunk(
        chunk_id="example:c0",
        document_id=document.document_id,
        ordinal=0,
        text=text,
        content_hash="b" * 64,
        peripheral=peripheral,
    )
    hit = SearchHit(chunk=chunk, document=document, score=1.0)
    assert any(expectation.matches(hit) for expectation in question.expect) is relevant
    expected_forbidden = title == "STM32U595 SVD" and "USART1SEL" in text and "USART1EN" not in text
    assert any(expectation.matches(hit) for expectation in question.forbid) is expected_forbidden


@pytest.mark.parametrize(
    ("title", "text", "relevant"),
    [
        ("NIST SP 800-218", "PW.7.1: determine whether code review should be used", True),
        ("NIST SP 800-218", "PW.8.1: test executable code for vulnerabilities", False),
        ("Other development guide", "PW.7.1: determine whether code review is needed", False),
    ],
)
def test_ssdf_question_requires_code_review_evidence(title: str, text: str, relevant: bool) -> None:
    """Unrelated NIST passages or other guides do not satisfy the SSDF question."""
    questions = load_questions(Path(__file__).parent.parent / "config" / "eval" / "questions.yaml")
    question = next(q for q in questions if q.id == "nist-ssdf-code-review")
    document = Document(
        document_id=title.lower().replace(" ", "-"),
        title=title,
        content_hash="c" * 64,
        source_type=SourceType.OFFICIAL,
    )
    chunk = Chunk(
        chunk_id="example:c1",
        document_id=document.document_id,
        ordinal=0,
        text=text,
        content_hash="d" * 64,
    )
    hit = SearchHit(chunk=chunk, document=document, score=1.0)
    assert any(expectation.matches(hit) for expectation in question.expect) is relevant


@pytest.mark.parametrize(
    ("title", "text", "relevant"),
    [
        (
            "CISA Secure by Design Eliminating Buffer Overflow Vulnerabilities",
            "Run unit tests with AddressSanitizer to detect buffer overflows.",
            True,
        ),
        (
            "CISA Secure by Design Eliminating Buffer Overflow Vulnerabilities",
            "Buffer overflows are memory safety vulnerabilities.",
            False,
        ),
        ("Other security guide", "Run unit tests with AddressSanitizer.", False),
    ],
)
def test_buffer_overflow_question_requires_testing_evidence(
    title: str, text: str, relevant: bool
) -> None:
    """A general warning or another document does not satisfy the CISA question."""
    questions = load_questions(Path(__file__).parent.parent / "config" / "eval" / "questions.yaml")
    question = next(q for q in questions if q.id == "cisa-buffer-overflow-tests")
    document = Document(
        document_id=title.lower().replace(" ", "-"),
        title=title,
        content_hash="e" * 64,
        source_type=SourceType.OFFICIAL,
    )
    chunk = Chunk(
        chunk_id="example:c2",
        document_id=document.document_id,
        ordinal=0,
        text=text,
        content_hash="f" * 64,
    )
    hit = SearchHit(chunk=chunk, document=document, score=1.0)
    assert any(expectation.matches(hit) for expectation in question.expect) is relevant


@pytest.mark.parametrize(
    "content",
    [
        [{"id": "a", "query": "x", "expect": []}],
        [{"id": "a", "query": "x", "expect": [{"bogus": 1}]}],
        [{"id": "a", "query": "x", "expect": [{"page": 1}]}] * 2,
        [{"id": "a", "query": "x", "expect": [{}]}],
        [{"id": "a", "query": "x", "expect": [{"contains": [" "]}]}],
        [{"id": "a", "query": "x", "expect": [{"not_contains": ["x"]}]}],
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


def test_forbidden_passage_is_reported_and_cannot_count_as_expected(
    env: tuple[Settings, Workspace, Catalog], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A misleading passage is counted separately and cannot improve the positive rank."""
    data = [
        {
            "id": "misleading",
            "query": "USART1_CR1",
            "expect": [{"peripheral": "USART1"}],
            "forbid": [{"peripheral": "USART1", "contains": ["USART1_CR1"]}],
        }
    ]
    path = write(tmp_path / "q.yaml", data)
    questions = load_questions(path)
    report = evaluate(HybridSearcher(env[2]), questions)
    assert report.results[0].rank is None
    assert report.results[0].forbidden_ranks == (1,)
    assert report.forbidden(1) == 1
    config = tmp_path / "s.yaml"
    config.write_text(yaml.safe_dump({"workspace": str(tmp_path / "ws")}), encoding="utf-8")
    assert cli.main(["evaluate", str(path), "--config", str(config), "--max-forbidden", "0"]) == 1
    captured = capsys.readouterr()
    assert "forbidden@5=1" in captured.out
    assert "exceeds 0" in captured.err


def test_forbidden_gate_requires_curated_negative_criteria(
    env: tuple[Settings, Workspace, Catalog], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A benchmark with no forbidden examples cannot pass the forbidden gate."""
    config = tmp_path / "s.yaml"
    config.write_text(yaml.safe_dump({"workspace": str(tmp_path / "ws")}), encoding="utf-8")
    questions = write(tmp_path / "q.yaml", QUESTIONS)
    assert (
        cli.main(["evaluate", str(questions), "--config", str(config), "--max-forbidden", "0"]) == 1
    )
    assert "requires at least one forbidden criterion" in capsys.readouterr().err


def test_forbidden_negative_criteria_are_conjunctive() -> None:
    """A forbidden clock-selection passage does not exclude one also proving clock enable."""
    expectation = Expectation(contains=("USART1SEL",), not_contains=("USART1EN",))
    document = Document(
        document_id="u5", title="U5", content_hash="a" * 64, source_type=SourceType.VENDOR
    )
    chunk = Chunk(
        chunk_id="u5:0",
        document_id="u5",
        ordinal=0,
        text="USART1SEL USART1EN",
        content_hash="b" * 64,
    )
    assert not expectation.matches(SearchHit(chunk=chunk, document=document, score=1))
    assert expectation.matches(
        SearchHit(chunk=chunk.model_copy(update={"text": "USART1SEL"}), document=document, score=1)
    )


def test_evaluate_command_uses_configured_relevance_threshold(
    env: tuple[Settings, Workspace, Catalog], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Evaluation exercises the same score cutoff as the API."""
    config = tmp_path / "s.yaml"
    config.write_text(
        yaml.safe_dump({"workspace": str(tmp_path / "ws"), "limits": {"min_score": 1.0}}),
        encoding="utf-8",
    )
    questions = write(tmp_path / "q.yaml", [QUESTIONS[0]])
    assert cli.main(["evaluate", str(questions), "--config", str(config)]) == 0
    assert "hit                      register     rank=miss" in capsys.readouterr().out


def test_category_threshold_detects_failure_hidden_by_global_recall(
    env: tuple[Settings, Workspace, Catalog],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A category regression fails even when overall recall meets its threshold."""
    config = tmp_path / "s.yaml"
    config.write_text(yaml.safe_dump({"workspace": str(tmp_path / "ws")}), encoding="utf-8")
    questions = write(tmp_path / "q.yaml", QUESTIONS)
    args = ["evaluate", str(questions), "--config", str(config), "--min-recall", "0.6"]
    assert cli.main([*args, "--min-category-recall", "0.5"]) == 0
    assert cli.main([*args, "--min-category-recall", "0.75"]) == 1
    error = capsys.readouterr().err
    assert "category 'general'" in error
    assert "Recall@5=0.5" in error
    assert "category 'register'" not in error


@pytest.mark.parametrize("threshold", ["-0.1", "1.1", "nan", "inf", "not-a-number"])
def test_invalid_category_threshold_is_rejected(threshold: str) -> None:
    """Category thresholds must be finite values in the unit interval."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["evaluate", "questions.yaml", "--min-category-recall", threshold])
    assert exc.value.code == 2


def test_category_threshold_rejects_empty_question_set(
    env: tuple[Settings, Workspace, Catalog],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No categories cannot satisfy a category-based quality gate."""
    config = tmp_path / "s.yaml"
    config.write_text(yaml.safe_dump({"workspace": str(tmp_path / "ws")}), encoding="utf-8")
    questions = write(tmp_path / "q.yaml", [])
    assert (
        cli.main(
            ["evaluate", str(questions), "--config", str(config), "--min-category-recall", "0"]
        )
        == 1
    )
    assert "no categories" in capsys.readouterr().err
