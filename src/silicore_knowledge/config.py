# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Service configuration: YAML file overridden by environment variables."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    ValidationError,
    model_validator,
)

CONFIG_PATH_ENV = "SILICORE_KNOWLEDGE_CONFIG"
ENV_PREFIX = "SILICORE_KNOWLEDGE__"


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


class ReadPrincipal(_Section):
    """A read identity and its explicit document grants."""

    token: SecretStr = Field(min_length=1)
    allowed_documents: tuple[
        Annotated[str, StringConstraints(pattern=r"^(\*|[a-z0-9][a-z0-9._-]{0,127})$")], ...
    ] = Field(default=(), max_length=500)

    @model_validator(mode="after")
    def _valid_grants(self) -> Self:
        if "*" in self.allowed_documents and len(self.allowed_documents) != 1:
            raise ValueError("the wildcard must be the only document grant")
        return self


class AuthSettings(_Section):
    """Distinct bearer tokens for named readers and administration."""

    principals: dict[
        Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,63}$")], ReadPrincipal
    ] = Field(default_factory=dict)
    admin_token: SecretStr | None = None

    @model_validator(mode="after")
    def _distinct_tokens(self) -> Self:
        if "admin" in self.principals:
            raise ValueError("the identity name admin is reserved")
        tokens = [principal.token.get_secret_value() for principal in self.principals.values()]
        if self.admin_token is not None:
            tokens.append(self.admin_token.get_secret_value())
        if any(not token for token in tokens) or len(tokens) != len(set(tokens)):
            raise ValueError("authentication tokens must be nonempty and different")
        return self


class IngestionSettings(_Section):
    """Bounds applied while parsing untrusted PDF files (ANSSI R18, R22)."""

    max_pages: int = Field(default=5000, ge=1)
    max_chars: int = Field(default=40_000_000, ge=1000)
    chunk_max_chars: int = Field(default=1500, ge=200, le=8000)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    # Documents whose content looks like prompt injection are quarantined instead of indexed.
    quarantine_suspicious: bool = True
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
    max_search_text_chars: int = Field(default=32_000, ge=1)
    max_query_chars: int = Field(default=500, ge=1)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    request_timeout_s: float = Field(default=20.0, gt=0)
    rate_limit_per_minute: int = Field(default=120, ge=1)
    max_upload_bytes: int = Field(default=200 * 1024 * 1024, ge=1)
    max_api_body_bytes: int = Field(default=65_536, ge=1024)

    @model_validator(mode="after")
    def _default_within_max(self) -> Self:
        if self.default_results > self.max_results:
            raise ValueError("default_results must not exceed max_results")
        return self


class OcrSettings(_Section):
    """Optional OCR of scanned PDF files, run as an external ``ocrmypdf`` process."""

    enabled: bool = False
    language: str = Field(default="eng", pattern=r"^[a-z_]{3,12}(\+[a-z_]{3,12})*$")
    timeout_s: int = Field(default=1800, ge=10)
    # OCR runs when the average extracted text per page is below this value.
    min_chars_per_page: int = Field(default=40, ge=1)


class RetrievalSettings(_Section):
    """Retrieval-side options.

    ``concepts_path`` points to the declarative concept registry used for controlled query
    expansion; None disables expansion.
    """

    concepts_path: Path | None = None


class Settings(_Section):
    """Complete service settings."""

    workspace: Path = Path("/var/lib/silicore-knowledge")
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    server: ServerSettings = ServerSettings()
    qdrant: QdrantSettings = QdrantSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    auth: AuthSettings = AuthSettings()
    ingestion: IngestionSettings = IngestionSettings()
    ocr: OcrSettings = OcrSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    limits: LimitSettings = LimitSettings()


def _merge(base: dict[str, Any], key_path: list[str], value: str) -> None:
    node = base
    for key in key_path[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ConfigError(f"environment override conflicts with scalar key {key!r}")
        node = child
    if (
        len(key_path) == 4
        and key_path[:2] == ["auth", "principals"]
        and key_path[-1] == "allowed_documents"
    ):
        try:
            node[key_path[-1]] = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ConfigError("invalid allowed_documents environment override") from exc
    else:
        node[key_path[-1]] = value


def load_settings(path: Path | None = None, environ: Mapping[str, str] | None = None) -> Settings:
    """Load settings from an optional YAML file and environment overrides.

    Environment variables named ``SILICORE_KNOWLEDGE__<SECTION>__<KEY>`` override file values
    (for example ``SILICORE_KNOWLEDGE__SERVER__PORT=9000``). If ``path`` is None, the
    ``SILICORE_KNOWLEDGE_CONFIG`` variable is used when set; otherwise defaults apply.

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
