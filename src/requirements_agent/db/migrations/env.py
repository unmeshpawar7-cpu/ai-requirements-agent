"""Alembic environment (async engine; URL from DATABASE_URL / settings)."""

from __future__ import annotations

import asyncio
import os

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from requirements_agent.core.config import get_settings
from requirements_agent.db.models import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    # Explicit env var first (settings are cached per process), then .env / defaults.
    return (
        config.get_main_option("sqlalchemy.url")
        or os.getenv("DATABASE_URL")
        or get_settings().database_url
    )


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_url())
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
