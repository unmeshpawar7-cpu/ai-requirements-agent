from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt
import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from requirements_agent.ai.prompts.templates import (
    PromptTemplateError,
    load_seed_templates,
    parse_seed_file,
    render,
    split_template,
    validate_template,
)
from requirements_agent.api.deps import page_params, require_role
from requirements_agent.core.config import Settings
from requirements_agent.core.errors import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    UnprocessableError,
)
from requirements_agent.core.rate_limit import TokenBucketRateLimiter
from requirements_agent.core.security import InvalidTokenError, PasswordHasher, TokenService
from requirements_agent.db.models import User
from requirements_agent.db.session import Database
from requirements_agent.domain.enums import Role
from requirements_agent.services.prompts import PromptService
from tests.conftest import make_settings

VALID_TEMPLATE = "<<<SYSTEM>>>\nBe precise.\n$schema\n<<<USER>>>\n$sections\n$evidence\n"


# ----------------------------------------------------------------------------- security
def test_password_hashing_round_trip_and_limits() -> None:
    hasher = PasswordHasher(rounds=4)
    hashed = hasher.hash("correct-horse-battery")
    assert hashed.startswith("$2")
    assert hasher.verify("correct-horse-battery", hashed)
    assert not hasher.verify("wrong-password", hashed)
    assert not hasher.verify("anything", None)  # unknown user path still runs bcrypt
    assert not hasher.verify("x" * 100, hashed)
    assert not hasher.verify("pw", "not-a-bcrypt-hash")
    with pytest.raises(ValueError, match="72-byte"):
        hasher.hash("é" * 40)


def test_token_create_and_verify(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, jwt_expires_minutes=5)
    tokens = TokenService(settings)
    user_id = uuid.uuid4()
    claims = tokens.decode(tokens.create_access_token(user_id, Role.ANALYST))
    assert claims.subject == user_id
    assert claims.role is Role.ANALYST
    assert claims.token_id
    assert tokens.ttl_seconds == 300


def test_token_rejects_expired_tampered_and_foreign_tokens(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    tokens = TokenService(settings)
    expired = tokens.create_access_token(
        uuid.uuid4(), Role.VIEWER, now=datetime.now(UTC) - timedelta(hours=3)
    )
    with pytest.raises(InvalidTokenError):
        tokens.decode(expired)
    valid = tokens.create_access_token(uuid.uuid4(), Role.VIEWER)
    with pytest.raises(InvalidTokenError):
        tokens.decode(valid[:-4] + "AAAA")
    foreign = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "role": "admin",
            "exp": datetime.now(UTC) + timedelta(hours=1),
            "iat": datetime.now(UTC),
            "aud": "other-api",
            "iss": settings.jwt_issuer,
        },
        settings.jwt_secret.get_secret_value(),
        algorithm="HS256",
    )
    with pytest.raises(InvalidTokenError):
        tokens.decode(foreign)
    other_secret = TokenService(make_settings(tmp_path, jwt_secret="x" * 40))
    with pytest.raises(InvalidTokenError):
        other_secret.decode(valid)


async def test_role_hierarchy() -> None:
    viewer = User(email="v@x.io", full_name="v", password_hash="x", role=Role.VIEWER)
    admin = User(email="a@x.io", full_name="a", password_hash="x", role=Role.ADMIN)
    analyst_only = require_role(Role.ANALYST)
    assert await analyst_only(admin) is admin
    with pytest.raises(ForbiddenError):
        await analyst_only(viewer)
    assert await require_role(Role.VIEWER)(viewer) is viewer


def test_page_params_cap() -> None:
    assert page_params(page=2, page_size=500).page_size == 100
    assert page_params(page=3, page_size=10).offset == 20


def test_settings_validation(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="at least 32"):
        make_settings(tmp_path, jwt_secret="short")
    with pytest.raises(ValidationError, match="production"):
        Settings(_env_file=None, environment="production")  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make_settings(tmp_path, chunk_max_tokens=100, chunk_overlap_tokens=100)
    with pytest.raises(ValidationError, match="together"):
        make_settings(tmp_path, admin_email="a@b.c")
    settings = make_settings(
        tmp_path,
        cors_origins="https://a.example, https://b.example",
        ai_price_table='{"openai:gpt-x": {"input_per_1k": 1.0, "output_per_1k": 2.0}}',
    )
    assert settings.cors_origins == ["https://a.example", "https://b.example"]
    assert settings.ai_price_table["openai:gpt-x"]["output_per_1k"] == 2.0
    assert "openai:gpt-4o-mini" in settings.ai_price_table  # defaults are kept
    assert settings.resolved_vector_backend == "memory"
    pg = make_settings(tmp_path, database_url="postgresql+asyncpg://u:p@h/db")
    assert pg.resolved_vector_backend == "pgvector"


# ------------------------------------------------------------------------- rate limiter
class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_token_bucket_consumes_refills_and_reports_retry_after() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(capacity=2, refill_per_second=0.5, clock=clock)
    assert limiter.consume("u1").allowed
    assert limiter.consume("u1").allowed
    denied = limiter.consume("u1")
    assert not denied.allowed
    assert denied.retry_after_seconds == pytest.approx(2.0)
    assert limiter.consume("u2").allowed  # buckets are per key
    clock.now = 2.0
    assert limiter.consume("u1").allowed
    clock.now = 100.0
    assert limiter.consume("u1").remaining == pytest.approx(1.0)  # capped at capacity
    limiter.reset()
    assert limiter.consume("u1").remaining == pytest.approx(1.0)


def test_token_bucket_evicts_least_recently_used_keys() -> None:
    limiter = TokenBucketRateLimiter(capacity=1, refill_per_second=0.001, max_keys=2)
    limiter.consume("a")
    limiter.consume("b")
    limiter.consume("c")  # evicts "a"
    assert limiter.consume("a").allowed  # fresh bucket after eviction
    assert not limiter.consume("c").allowed
    with pytest.raises(ValueError, match="capacity"):
        TokenBucketRateLimiter(capacity=0, refill_per_second=1)
    with pytest.raises(ValueError, match="refill"):
        TokenBucketRateLimiter(capacity=1, refill_per_second=0)


# ------------------------------------------------------------------------------ prompts
def test_seed_templates_are_valid() -> None:
    seeds = {(s.name, s.version) for s in load_seed_templates()}
    assert seeds == {("requirements_analysis", 1), ("test_case_generation", 1)}


def test_template_validation_rules() -> None:
    validate_template("requirements_analysis", VALID_TEMPLATE)
    with pytest.raises(PromptTemplateError, match="exactly one"):
        split_template("no markers")
    with pytest.raises(PromptTemplateError, match="before"):
        split_template("<<<USER>>>\nu\n<<<SYSTEM>>>\ns")
    with pytest.raises(PromptTemplateError, match="empty"):
        split_template("<<<SYSTEM>>>\n<<<USER>>>\nu")
    with pytest.raises(PromptTemplateError, match="missing required"):
        validate_template("requirements_analysis", "<<<SYSTEM>>>\n$schema\n<<<USER>>>\n$sections")
    with pytest.raises(PromptTemplateError, match="unknown placeholders"):
        validate_template("requirements_analysis", VALID_TEMPLATE + "$secret_key")
    with pytest.raises(PromptTemplateError, match="invalid placeholder"):
        validate_template("requirements_analysis", VALID_TEMPLATE + "costs $ 5")
    with pytest.raises(PromptTemplateError, match="Unknown prompt"):
        validate_template("nope", VALID_TEMPLATE)
    with pytest.raises(PromptTemplateError, match="front-matter"):
        parse_seed_file(VALID_TEMPLATE)


def test_render_substitutes_and_defaults_optional_variables() -> None:
    rendered = render(
        "requirements_analysis",
        7,
        VALID_TEMPLATE + "$project_name",
        {"schema": "{}", "sections": "entities", "evidence": "[E1] ..."},
    )
    assert rendered.version == 7
    assert rendered.system == "Be precise.\n{}"
    assert rendered.user == "entities\n[E1] ...\n"  # optional project_name defaults to ""
    with pytest.raises(PromptTemplateError):
        render("requirements_analysis", 1, VALID_TEMPLATE, {"schema": "{}"})


@pytest.fixture
async def session(tmp_path: Path) -> AsyncIterator[AsyncSession]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'p.db'}")
    await database.create_all()
    async with database.session_factory() as db_session:
        yield db_session
    await database.dispose()


async def test_prompt_service_seed_create_and_activate(session: AsyncSession) -> None:
    service = PromptService(session)
    assert await service.seed() == 2
    assert await service.seed() == 0  # idempotent
    active = await service.active("requirements_analysis")
    assert (active.version, active.created_by) == (1, "system:seed")

    draft = await service.create_version(
        "requirements_analysis", VALID_TEMPLATE, "v2", False, "a@x"
    )
    assert (draft.version, draft.is_active) == (2, False)
    assert (await service.active("requirements_analysis")).version == 1

    live = await service.create_version("requirements_analysis", VALID_TEMPLATE, None, True, "a@x")
    assert live.version == 3
    assert (await service.active("requirements_analysis")).version == 3

    await service.activate("requirements_analysis", 1)
    assert (await service.active("requirements_analysis")).version == 1
    assert len(draft.checksum) == 64

    with pytest.raises(UnprocessableError):
        await service.create_version(
            "requirements_analysis", "<<<SYSTEM>>>x<<<USER>>>y", None, False, "a"
        )
    with pytest.raises(NotFoundError):
        await service.create_version("unknown_prompt", VALID_TEMPLATE, None, False, "a")
    with pytest.raises(NotFoundError):
        await service.activate("requirements_analysis", 99)


async def test_prompt_service_requires_active_version(session: AsyncSession) -> None:
    with pytest.raises(NotFoundError):
        await PromptService(session).active("requirements_analysis")
    assert ConflictError("x").status_code == 409
