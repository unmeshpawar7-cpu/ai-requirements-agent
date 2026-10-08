"""Application settings loaded from environment variables (and an optional `.env` file).

Every tunable that influences AI behaviour (chunk sizes, retrieval depth, token budgets,
retry bounds, prices) lives here so that it is visible, documented and overridable per
environment without code changes.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_DEV_JWT_SECRET = "dev-only-insecure-secret-change-me-0123456789"  # noqa: S105 - documented dev default

# USD per 1K tokens. These are *estimates* used for cost reporting only; check the provider's
# current price list before relying on them. Override with AI_PRICE_TABLE (JSON).
DEFAULT_PRICE_TABLE: dict[str, dict[str, float]] = {
    "openai:gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
    "openai:gpt-4o": {"input_per_1k": 0.0025, "output_per_1k": 0.01},
    "openai:text-embedding-3-small": {"input_per_1k": 0.00002, "output_per_1k": 0.0},
    "anthropic:claude-sonnet-4-5": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "offline:heuristic-v1": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "hashing:hashing-v1": {"input_per_1k": 0.0, "output_per_1k": 0.0},
}


class Settings(BaseSettings):
    """Typed configuration. Field names map 1:1 to upper-case environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- application -------------------------------------------------------------------
    app_name: str = "AI Requirements Agent"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = True
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- database ----------------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./requirements_agent.db"
    database_auto_create: bool = Field(
        default=False,
        description="Create tables with metadata.create_all at startup (dev/test only; "
        "production uses Alembic migrations).",
    )
    database_echo: bool = False

    # --- auth --------------------------------------------------------------------------
    jwt_secret: SecretStr = SecretStr(_DEV_JWT_SECRET)
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    jwt_issuer: str = "ai-requirements-agent"
    jwt_audience: str = "ai-requirements-agent-api"
    jwt_expires_minutes: int = Field(default=60, ge=1, le=24 * 60)
    bcrypt_rounds: int = Field(default=12, ge=4, le=16)
    admin_email: str | None = None
    admin_password: SecretStr | None = None
    default_signup_role: Literal["viewer", "analyst"] = "viewer"

    # --- AI providers ------------------------------------------------------------------
    llm_provider: Literal["offline", "openai", "anthropic"] = "offline"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-4-5"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    llm_max_output_tokens: int = Field(default=4096, ge=256, le=32_000)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_transport_retries: int = Field(default=3, ge=1, le=6)

    embedding_provider: Literal["hashing", "openai"] = "hashing"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=384, ge=32, le=3072)

    vector_backend: Literal["auto", "pgvector", "memory"] = "auto"
    vector_rebuild_on_startup: bool = Field(
        default=True,
        description="When the in-memory vector store is used, re-embed persisted chunks at "
        "startup so search survives restarts.",
    )

    # --- ingestion ---------------------------------------------------------------------
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    chunk_max_tokens: int = Field(default=350, ge=50, le=4000)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=1000)

    # --- retrieval / analysis ----------------------------------------------------------
    search_default_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_fetch_multiplier: int = Field(default=4, ge=1, le=20)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    analysis_top_k_per_query: int = Field(default=8, ge=1, le=50)
    analysis_max_evidence: int = Field(default=24, ge=1, le=100)
    analysis_min_score: float = Field(default=0.02, ge=0.0, le=1.0)
    analysis_context_tokens: int = Field(default=6000, ge=500, le=100_000)
    structured_max_repairs: int = Field(default=2, ge=0, le=5)

    # --- jobs / rate limiting ----------------------------------------------------------
    job_concurrency: int = Field(default=2, ge=1, le=32)
    rate_limit_ai_capacity: int = Field(default=10, ge=1)
    rate_limit_ai_refill_per_minute: float = Field(default=10.0, gt=0)

    # --- cost --------------------------------------------------------------------------
    ai_price_table: Annotated[dict[str, dict[str, float]], NoDecode] = Field(
        default_factory=lambda: dict(DEFAULT_PRICE_TABLE)
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                return json.loads(value)
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("ai_price_table", mode="before")
    @classmethod
    def _parse_price_table(cls, value: Any) -> Any:
        if isinstance(value, str):
            parsed = json.loads(value)
            merged = dict(DEFAULT_PRICE_TABLE)
            merged.update(parsed)
            return merged
        return value

    @model_validator(mode="after")
    def _validate_security(self) -> Settings:
        secret = self.jwt_secret.get_secret_value()
        if len(secret) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters long")
        if self.environment == "production" and secret == _DEV_JWT_SECRET:
            raise ValueError("JWT_SECRET must be set explicitly in production")
        if self.chunk_overlap_tokens >= self.chunk_max_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_MAX_TOKENS")
        if (self.admin_email is None) != (self.admin_password is None):
            raise ValueError("ADMIN_EMAIL and ADMIN_PASSWORD must be provided together")
        return self

    # --- derived -----------------------------------------------------------------------
    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def resolved_vector_backend(self) -> Literal["pgvector", "memory"]:
        if self.vector_backend != "auto":
            return self.vector_backend
        return "pgvector" if self.database_url.startswith("postgresql") else "memory"

    @property
    def uses_dev_jwt_secret(self) -> bool:
        return self.jwt_secret.get_secret_value() == _DEV_JWT_SECRET


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (tests construct `Settings` directly instead)."""
    return Settings()
