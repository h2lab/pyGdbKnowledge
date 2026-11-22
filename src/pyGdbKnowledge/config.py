# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Service configuration: YAML file overridden by environment variables."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

CONFIG_PATH_ENV = "PYGDBKNOWLEDGE_CONFIG"
ENV_PREFIX = "PYGDBKNOWLEDGE__"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class ConfigError(Exception):
    """Raised when the configuration cannot be loaded or is invalid."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServerSettings(_Section):
    """HTTP listener settings."""

    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)


class QdrantSettings(_Section):
    """Qdrant vector store settings."""

    url: str = "http://127.0.0.1:6333"
    collection: str = Field(default="knowledge", min_length=1)


class EmbeddingSettings(_Section):
    """Local embedding model settings.

    The model is read from a local path or cache; downloads are refused unless
    ``allow_download`` is set (no outbound connection by default).
    """

    model: str = "BAAI/bge-m3"
    enabled: bool = False
    allow_download: bool = False
    batch_size: int = Field(default=16, ge=1, le=256)
    max_seq_length: int = Field(default=512, ge=16, le=8192)


class AuthSettings(_Section):
    """Bearer tokens; read and admin credentials must differ (ANSSI R21)."""

    read_token: SecretStr | None = None
    admin_token: SecretStr | None = None

    @model_validator(mode="after")
    def _distinct_tokens(self) -> Self:
        if (
            self.read_token is not None
            and self.admin_token is not None
            and self.read_token.get_secret_value() == self.admin_token.get_secret_value()
        ):
            raise ValueError("read_token and admin_token must be different")
        return self


class IngestionSettings(_Section):
    """Bounds applied while parsing untrusted PDF files (ANSSI R18, R22)."""

    max_pages: int = Field(default=5000, ge=1)
    max_chars: int = Field(default=40_000_000, ge=1000)
    chunk_max_chars: int = Field(default=1500, ge=200, le=8000)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    # Directories the admin API may import files from; empty disables server-side paths.
    import_dirs: list[Path] = Field(default_factory=list)

    @model_validator(mode="after")
    def _overlap_below_half(self) -> Self:
        if self.chunk_overlap_chars * 2 > self.chunk_max_chars:
            raise ValueError("chunk_overlap_chars must not exceed half of chunk_max_chars")
        return self


class LimitSettings(_Section):
    """Bounds applied to every exposed interface (ANSSI R18, R29)."""

    max_results: int = Field(default=20, ge=1, le=100)
    default_results: int = Field(default=8, ge=1, le=100)
    max_text_chars: int = Field(default=4000, ge=100)
    max_query_chars: int = Field(default=500, ge=1)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    request_timeout_s: float = Field(default=20.0, gt=0)
    rate_limit_per_minute: int = Field(default=120, ge=1)
    max_upload_bytes: int = Field(default=200 * 1024 * 1024, ge=1)

    @model_validator(mode="after")
    def _default_within_max(self) -> Self:
        if self.default_results > self.max_results:
            raise ValueError("default_results must not exceed max_results")
        return self


class Settings(_Section):
    """Complete service settings."""

    workspace: Path = Path("/var/lib/pygdbknowledge")
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    server: ServerSettings = ServerSettings()
    qdrant: QdrantSettings = QdrantSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    auth: AuthSettings = AuthSettings()
    ingestion: IngestionSettings = IngestionSettings()
    limits: LimitSettings = LimitSettings()

    @model_validator(mode="after")
    def _remote_requires_token(self) -> Self:
        if self.server.host not in LOOPBACK_HOSTS and self.auth.read_token is None:
            raise ValueError("auth.read_token is required when listening on a non-loopback address")
        return self


def _merge(base: dict[str, Any], key_path: list[str], value: str) -> None:
    node = base
    for key in key_path[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ConfigError(f"environment override conflicts with scalar key {key!r}")
        node = child
    node[key_path[-1]] = value


def load_settings(path: Path | None = None, environ: Mapping[str, str] | None = None) -> Settings:
    """Load settings from an optional YAML file and environment overrides.

    Environment variables named ``PYGDBKNOWLEDGE__<SECTION>__<KEY>`` override file values
    (for example ``PYGDBKNOWLEDGE__SERVER__PORT=9000``). If ``path`` is None, the
    ``PYGDBKNOWLEDGE_CONFIG`` variable is used when set; otherwise defaults apply.

    Parameters
    ----------
    path : Path | None
        YAML configuration file.
    environ : Mapping[str, str] | None
        Environment to read; defaults to ``os.environ``.

    Returns
    -------
    Settings
        Validated, immutable settings.

    Raises
    ------
    ConfigError
        If the file is unreadable or malformed, an override is malformed, or validation fails.
    """
    env = os.environ if environ is None else environ
    if path is None and env.get(CONFIG_PATH_ENV):
        path = Path(env[CONFIG_PATH_ENV])

    data: dict[str, Any] = {}
    if path is not None:
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read configuration {path}: {exc}") from exc
        if loaded is not None and not isinstance(loaded, dict):
            raise ConfigError(f"configuration {path} must contain a mapping")
        data = loaded or {}

    for name, value in env.items():
        if name.startswith(ENV_PREFIX):
            key_path = [part.lower() for part in name[len(ENV_PREFIX) :].split("__")]
            if not all(key_path):
                raise ConfigError(f"malformed environment override {name!r}")
            _merge(data, key_path, value)

    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        # Pydantic error text may echo input values, which can be secrets.
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"invalid configuration: {problems}") from None
