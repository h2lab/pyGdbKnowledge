# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for configuration loading."""

from pathlib import Path

import pytest

from pyGdbKnowledge.config import ConfigError, Settings, load_settings

REPO_CONFIG = Path(__file__).parent.parent / "config" / "settings.yaml"


def test_defaults_without_file() -> None:
    """Defaults apply with no file and no overrides."""
    settings = load_settings(environ={})
    assert settings == Settings()
    assert settings.server.host == "127.0.0.1"
    assert settings.auth.read_token is None


def test_shipped_example_is_valid() -> None:
    """The example configuration validates."""
    assert load_settings(REPO_CONFIG, environ={}).qdrant.collection == "knowledge"


def test_environment_overrides_file(tmp_path: Path) -> None:
    """Environment variables win over file values and are type-coerced."""
    path = tmp_path / "s.yaml"
    path.write_text("server:\n  port: 9000\n  host: 0.0.0.0\n", encoding="utf-8")
    settings = load_settings(path, environ={"PYGDBKNOWLEDGE__SERVER__PORT": "9100"})
    assert settings.server.port == 9100
    assert settings.server.host == "0.0.0.0"


def test_config_path_from_environment(tmp_path: Path) -> None:
    """PYGDBKNOWLEDGE_CONFIG selects the file when no path is given."""
    path = tmp_path / "s.yaml"
    path.write_text("log_level: DEBUG\n", encoding="utf-8")
    assert load_settings(environ={"PYGDBKNOWLEDGE_CONFIG": str(path)}).log_level == "DEBUG"


def test_unknown_key_rejected(tmp_path: Path) -> None:
    """Typos in keys fail instead of being silently ignored."""
    path = tmp_path / "s.yaml"
    path.write_text("server:\n  prot: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="server.prot"):
        load_settings(path, environ={})


def test_unknown_environment_override_rejected() -> None:
    """Unknown overrides under the prefix are rejected."""
    with pytest.raises(ConfigError):
        load_settings(environ={"PYGDBKNOWLEDGE__SERVER__PROT": "1"})


def test_missing_file_and_bad_yaml(tmp_path: Path) -> None:
    """Unreadable or malformed files raise ConfigError."""
    with pytest.raises(ConfigError, match="cannot read"):
        load_settings(tmp_path / "missing.yaml", environ={})
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_settings(bad, environ={})


def test_identical_tokens_rejected_without_leaking() -> None:
    """Read and admin tokens must differ; the error never echoes secrets."""
    env = {
        "PYGDBKNOWLEDGE__AUTH__READ_TOKEN": "s3cret-value",
        "PYGDBKNOWLEDGE__AUTH__ADMIN_TOKEN": "s3cret-value",
    }
    with pytest.raises(ConfigError, match="must be different") as info:
        load_settings(environ=env)
    assert "s3cret-value" not in str(info.value)


def test_secrets_are_masked() -> None:
    """Tokens do not appear in repr."""
    settings = load_settings(environ={"PYGDBKNOWLEDGE__AUTH__READ_TOKEN": "abc123"})
    assert "abc123" not in repr(settings)


def test_limits_consistency() -> None:
    """default_results cannot exceed max_results."""
    with pytest.raises(ConfigError):
        load_settings(
            environ={
                "PYGDBKNOWLEDGE__LIMITS__MAX_RESULTS": "5",
                "PYGDBKNOWLEDGE__LIMITS__DEFAULT_RESULTS": "10",
            }
        )
