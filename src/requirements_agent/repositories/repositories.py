"""Data access for each aggregate. Repositories never commit; services own transactions."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import Select, case, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import undefer

from requirements_agent.ai.rag.retriever import ChunkRow
from requirements_agent.db.models import (
    AIUsage,
    Analysis,
    Document,
    DocumentChunk,
    Project,
    PromptTemplate,
    RequirementItem,
    TestCase,
    User,
)
from requirements_agent.domain.enums import AnalysisStatus, DocumentStatus, ItemType, ProjectStatus


async def paginate[T](
    session: AsyncSession, stmt: Select[tuple[T]], *, offset: int, limit: int
) -> tuple[list[T], int]:
    total = await session.scalar(select(func.count()).select_from(stmt.order_by(None).subquery()))
    rows = (await session.execute(stmt.offset(offset).limit(limit))).scalars().all()
    return list(rows), int(total or 0)


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, user_id: uuid.UUID) -> User | None:
        return await self.session.get(User, user_id)

    async def by_email(self, email: str) -> User | None:
        return await self.session.scalar(select(User).where(User.email == email.lower()))

    async def count(self) -> int:
        return int(await self.session.scalar(select(func.count(User.id))) or 0)

    async def list(self, offset: int, limit: int) -> tuple[list[User], int]:
        return await paginate(
            self.session,
            select(User).order_by(User.created_at, User.id),
            offset=offset,
            limit=limit,
        )


class ProjectRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, project_id: uuid.UUID) -> Project | None:
        return await self.session.get(Project, project_id)

    async def by_name(self, name: str) -> Project | None:
        return await self.session.scalar(select(Project).where(Project.name == name))

    def _counts(self) -> tuple[Any, Any]:
        documents = (
            select(func.count(Document.id))
            .where(Document.project_id == Project.id)
            .correlate(Project)
            .scalar_subquery()
        )
        analyses = (
            select(func.count(Analysis.id))
            .where(Analysis.project_id == Project.id)
            .correlate(Project)
            .scalar_subquery()
        )
        return documents, analyses

    async def list_with_counts(
        self, *, offset: int, limit: int, search: str | None, include_archived: bool
    ) -> tuple[list[tuple[Project, int, int]], int]:
        documents, analyses = self._counts()
        stmt = select(Project, documents, analyses)
        count_stmt = select(func.count(Project.id))
        if not include_archived:
            stmt = stmt.where(Project.status == ProjectStatus.ACTIVE)
            count_stmt = count_stmt.where(Project.status == ProjectStatus.ACTIVE)
        if search:
            pattern = f"%{search.lower()}%"
            stmt = stmt.where(func.lower(Project.name).like(pattern))
            count_stmt = count_stmt.where(func.lower(Project.name).like(pattern))
        stmt = stmt.order_by(Project.created_at.desc(), Project.id).offset(offset).limit(limit)
        rows = (await self.session.execute(stmt)).all()
        total = int(await self.session.scalar(count_stmt) or 0)
        return [(row[0], int(row[1]), int(row[2])) for row in rows], total

    async def counts_for(self, project_id: uuid.UUID) -> tuple[int, int]:
        documents = await self.session.scalar(
            select(func.count(Document.id)).where(Document.project_id == project_id)
        )
        analyses = await self.session.scalar(
            select(func.count(Analysis.id)).where(Analysis.project_id == project_id)
        )
        return int(documents or 0), int(analyses or 0)


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, document_id: uuid.UUID, *, with_content: bool = False) -> Document | None:
        stmt = select(Document).where(Document.id == document_id)
        if with_content:
            stmt = stmt.options(undefer(Document.raw_content))
        return await self.session.scalar(stmt)

    async def by_checksum(self, project_id: uuid.UUID, checksum: str) -> Document | None:
        return await self.session.scalar(
            select(Document).where(Document.project_id == project_id, Document.checksum == checksum)
        )

    async def list(
        self, project_id: uuid.UUID, *, offset: int, limit: int, status: DocumentStatus | None
    ) -> tuple[list[Document], int]:
        stmt = select(Document).where(Document.project_id == project_id)
        if status:
            stmt = stmt.where(Document.status == status)
        return await paginate(
            self.session,
            stmt.order_by(Document.created_at.desc(), Document.id),
            offset=offset,
            limit=limit,
        )

    async def count_ready(self, project_id: uuid.UUID, ids: Sequence[uuid.UUID] | None) -> int:
        stmt = select(func.count(Document.id)).where(
            Document.project_id == project_id, Document.status == DocumentStatus.READY
        )
        if ids:
            stmt = stmt.where(Document.id.in_(list(ids)))
        return int(await self.session.scalar(stmt) or 0)

    async def ids_with_status(self, statuses: Sequence[DocumentStatus]) -> list[uuid.UUID]:
        rows = await self.session.scalars(select(Document.id).where(Document.status.in_(statuses)))
        return list(rows)


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def delete_for_document(self, document_id: uuid.UUID) -> None:
        await self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )

    async def list(
        self, document_id: uuid.UUID, *, offset: int, limit: int
    ) -> tuple[list[DocumentChunk], int]:
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.ordinal)
        )
        return await paginate(self.session, stmt, offset=offset, limit=limit)

    async def ready_chunks(self) -> list[DocumentChunk]:
        stmt = (
            select(DocumentChunk)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == DocumentStatus.READY)
            .order_by(DocumentChunk.document_id, DocumentChunk.ordinal)
        )
        return list((await self.session.scalars(stmt)).all())


def make_chunk_loader(session_factory: async_sessionmaker[AsyncSession]):
    """Build the retriever's `ChunkLoader`: hydrate hits, keeping only chunks of ready docs."""

    async def load(chunk_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, ChunkRow]:
        if not chunk_ids:
            return {}
        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                Document.title,
                DocumentChunk.ordinal,
                DocumentChunk.section_path,
                DocumentChunk.text,
                DocumentChunk.token_count,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.id.in_(list(chunk_ids)), Document.status == DocumentStatus.READY)
        )
        async with session_factory() as session:
            rows = (await session.execute(stmt)).all()
        return {
            row[0]: ChunkRow(
                chunk_id=row[0],
                document_id=row[1],
                document_title=row[2],
                ordinal=row[3],
                section_path=row[4],
                text=row[5],
                token_count=row[6],
            )
            for row in rows
        }

    return load


class AnalysisRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, analysis_id: uuid.UUID) -> Analysis | None:
        return await self.session.get(Analysis, analysis_id)

    async def list(
        self, project_id: uuid.UUID, *, offset: int, limit: int
    ) -> tuple[list[Analysis], int]:
        stmt = (
            select(Analysis)
            .where(Analysis.project_id == project_id)
            .order_by(Analysis.created_at.desc(), Analysis.id)
        )
        return await paginate(self.session, stmt, offset=offset, limit=limit)

    async def items(
        self, analysis_id: uuid.UUID, item_type: ItemType | None = None
    ) -> list[RequirementItem]:
        stmt = select(RequirementItem).where(RequirementItem.analysis_id == analysis_id)
        if item_type:
            stmt = stmt.where(RequirementItem.item_type == item_type)
        return list((await self.session.scalars(stmt.order_by(RequirementItem.ordinal))).all())

    async def item_counts(
        self, analysis_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, dict[str, int]]:
        if not analysis_ids:
            return {}
        stmt = (
            select(RequirementItem.analysis_id, RequirementItem.item_type, func.count())
            .where(RequirementItem.analysis_id.in_(list(analysis_ids)))
            .group_by(RequirementItem.analysis_id, RequirementItem.item_type)
        )
        counts: dict[uuid.UUID, dict[str, int]] = {}
        for analysis_id, item_type, count in (await self.session.execute(stmt)).all():
            counts.setdefault(analysis_id, {})[ItemType(item_type).value] = int(count)
        return counts

    async def mark_stale(self, statuses: Sequence[Any], message: str, now: datetime) -> int:
        result = await self.session.execute(
            update(Analysis)
            .where(Analysis.status.in_(statuses))
            .values(status=AnalysisStatus.FAILED, error_message=message, completed_at=now)
        )
        return int(result.rowcount or 0)  # type: ignore[attr-defined]

    async def test_cases(self, analysis_id: uuid.UUID) -> list[TestCase]:
        stmt = (
            select(TestCase).where(TestCase.analysis_id == analysis_id).order_by(TestCase.ordinal)
        )
        return list((await self.session.scalars(stmt)).all())

    async def delete_test_cases(self, analysis_id: uuid.UUID) -> None:
        await self.session.execute(delete(TestCase).where(TestCase.analysis_id == analysis_id))


class PromptRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, name: str, version: int) -> PromptTemplate | None:
        return await self.session.scalar(
            select(PromptTemplate).where(
                PromptTemplate.name == name, PromptTemplate.version == version
            )
        )

    async def active(self, name: str) -> PromptTemplate | None:
        return await self.session.scalar(
            select(PromptTemplate).where(PromptTemplate.name == name, PromptTemplate.is_active)
        )

    async def max_version(self, name: str) -> int:
        value = await self.session.scalar(
            select(func.max(PromptTemplate.version)).where(PromptTemplate.name == name)
        )
        return int(value or 0)

    async def list(
        self, *, name: str | None, offset: int, limit: int
    ) -> tuple[list[PromptTemplate], int]:
        stmt = select(PromptTemplate)
        if name:
            stmt = stmt.where(PromptTemplate.name == name)
        stmt = stmt.order_by(PromptTemplate.name, PromptTemplate.version.desc())
        return await paginate(self.session, stmt, offset=offset, limit=limit)

    async def deactivate_all(self, name: str) -> None:
        await self.session.execute(
            update(PromptTemplate).where(PromptTemplate.name == name).values(is_active=False)
        )


class UsageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def grouped(
        self,
        *,
        since: datetime | None,
        until: datetime | None,
        project_id: uuid.UUID | None,
    ) -> list[Any]:
        stmt = select(
            AIUsage.operation,
            AIUsage.provider,
            AIUsage.model,
            AIUsage.prompt_name,
            AIUsage.prompt_version,
            func.count(AIUsage.id),
            func.coalesce(func.sum(case((AIUsage.success.is_(False), 1), else_=0)), 0),
            func.coalesce(func.sum(AIUsage.input_tokens), 0),
            func.coalesce(func.sum(AIUsage.output_tokens), 0),
            func.coalesce(func.sum(AIUsage.estimated_cost_usd), 0.0),
            func.coalesce(func.avg(AIUsage.latency_ms), 0.0),
        ).group_by(
            AIUsage.operation,
            AIUsage.provider,
            AIUsage.model,
            AIUsage.prompt_name,
            AIUsage.prompt_version,
        )
        if since:
            stmt = stmt.where(AIUsage.created_at >= since)
        if until:
            stmt = stmt.where(AIUsage.created_at < until)
        if project_id:
            stmt = stmt.where(AIUsage.project_id == project_id)
        stmt = stmt.order_by(
            AIUsage.provider, AIUsage.model, AIUsage.operation, AIUsage.prompt_name
        )
        return list((await self.session.execute(stmt)).all())
