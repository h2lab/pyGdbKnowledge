# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the command-line entry point."""

from pathlib import Path

import pytest

from pyGdbKnowledge import cli


def test_command_is_required() -> None:
    """Running without a command is a usage error."""
    with pytest.raises(SystemExit) as info:
        cli.main([])
    assert info.value.code == 2


def test_serve_runs_uvicorn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Command serve builds the app from the configuration and starts uvicorn."""
    config = tmp_path / "s.yaml"
    config.write_text(
        f"workspace: {tmp_path / 'ws'}\nserver:\n  host: 127.0.0.1\n  port: 9123\n",
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kw: calls.append(kw))
    assert cli.main(["serve", "--config", str(config)]) == 0
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == 9123
    assert (tmp_path / "ws" / "documents").is_dir()


def test_serve_reports_configuration_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Configuration failures exit with status 1 and a message on stderr."""
    assert cli.main(["serve", "--config", str(tmp_path / "missing.yaml")]) == 1
    assert "cannot read configuration" in capsys.readouterr().err


def test_serve_reports_unsafe_workspace(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A relative workspace path is refused before starting the server."""
    config = tmp_path / "s.yaml"
    config.write_text("workspace: relative/ws\n", encoding="utf-8")
    assert cli.main(["serve", "--config", str(config)]) == 1
    assert "absolute" in capsys.readouterr().err
