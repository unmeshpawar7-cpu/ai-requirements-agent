"""Startup/shutdown routines and the AI usage report."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from requirements_agent.ai.vectorstore.base import VectorRecord
from requirements_agent.ai.vectorstore.memory import InMemoryVectorStore
from requirements_agent.domain.enums import AnalysisStatus, DocumentStatus, Role
from requirements_agent.repositories.repositories import (
    AnalysisRepository,
    ChunkRepository,
    DocumentRepository,
    UsageRepository,
)
from requirements_agent.schemas.dto import UsageGroupOut, UsageSummaryOut
from requirements_agent.services.accounts import AccountService
from requirements_agent.services.container import Container
from requirements_agent.services.documents import enqueue_processing
from requirements_agent.services.prompts import PromptService

logger = logging.getLogger(__name__)


async def on_startup(container: Container) -> None:
    settings = container.settings
    db = container.database
    if settings.database_auto_create:
        await db.create_all()
    if settings.uses_dev_jwt_secret:
        logger.warning("JWT_SECRET is the development default; set it outside local dev")

    async with db.session_factory() as session:
        seeded = await PromptService(session).seed()
        if seeded:
            logger.info("seeded prompt templates", extra={"count": seeded})
        if settings.admin_email and settings.admin_password:
            await AccountService(
                session,
                container.password_hasher,
                container.token_service,
                Role(settings.default_signup_role),
            ).seed_admin(settings.admin_email, settings.admin_password.get_secret_value())

        # In-process jobs do not survive a restart: fail interrupted analyses (their inputs
        # may have changed) and re-queue documents that never finished processing.
        stale = await AnalysisRepository(session).mark_stale(
            [AnalysisStatus.QUEUED, AnalysisStatus.RUNNING],
            "Interrupted by a service restart; please re-run the analysis",
            datetime.now(UTC),
        )
        await session.commit()
        if stale:
            logger.warning("marked interrupted analyses as failed", extra={"count": stale})
        unfinished = await DocumentRepository(session).ids_with_status(
            [DocumentStatus.PENDING, DocumentStatus.PROCESSING]
        )
    for document_id in unfinished:
        enqueue_processing(container, document_id)

    if (
        isinstance(container.vector_store, InMemoryVectorStore)
        and settings.vector_rebuild_on_startup
    ):
        await rebuild_memory_index(container)


async def rebuild_memory_index(container: Container) -> int:
    """Re-embed persisted chunks into the process-local store (memory backend only)."""
    async with container.database.session_factory() as session:
        chunks = await ChunkRepository(session).ready_chunks()
    if not chunks:
        return 0
    texts = [f"{c.section_path}\n\n{c.text}" if c.section_path else c.text for c in chunks]
    vectors = await container.embedder.embed(texts)
    await container.vector_store.upsert(
        [
            VectorRecord(c.id, c.document_id, c.project_id, v)
            for c, v in zip(chunks, vectors, strict=True)
        ]
    )
    logger.info("rebuilt in-memory vector index", extra={"chunks": len(chunks)})
    return len(chunks)


async def usage_summary(
    session: AsyncSession,
    *,
    since: datetime | None,
    until: datetime | None,
    project_id: uuid.UUID | None,
) -> UsageSummaryOut:
    rows = await UsageRepository(session).grouped(since=since, until=until, project_id=project_id)
    groups = [
        UsageGroupOut(
            operation=str(getattr(r[0], "value", r[0])),
            provider=r[1],
            model=r[2],
            prompt_name=r[3],
            prompt_version=r[4],
            calls=int(r[5]),
            failures=int(r[6] or 0),
            input_tokens=int(r[7] or 0),
            output_tokens=int(r[8] or 0),
            estimated_cost_usd=round(float(r[9] or 0.0), 6),
            avg_latency_ms=round(float(r[10] or 0.0), 2),
        )
        for r in rows
    ]
    return UsageSummaryOut(
        total_calls=sum(g.calls for g in groups),
        total_failures=sum(g.failures for g in groups),
        total_input_tokens=sum(g.input_tokens for g in groups),
        total_output_tokens=sum(g.output_tokens for g in groups),
        total_estimated_cost_usd=round(sum(g.estimated_cost_usd for g in groups), 6),
        groups=groups,
    )
