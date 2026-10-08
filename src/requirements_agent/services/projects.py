"""Project lifecycle (create, read, update, soft archive)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from requirements_agent.core.errors import ConflictError, NotFoundError
from requirements_agent.db.models import Project, User
from requirements_agent.domain.enums import ProjectStatus
from requirements_agent.repositories.repositories import ProjectRepository
from requirements_agent.schemas.dto import ProjectCreate, ProjectOut, ProjectUpdate


def to_out(project: Project, documents: int = 0, analyses: int = 0) -> ProjectOut:
    return ProjectOut.model_validate(
        {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "status": project.status,
            "owner_id": project.owner_id,
            "document_count": documents,
            "analysis_count": analyses,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "archived_at": project.archived_at,
        }
    )


class ProjectService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.projects = ProjectRepository(session)

    async def get(self, project_id: uuid.UUID) -> Project:
        project = await self.projects.get(project_id)
        if project is None:
            raise NotFoundError("Project not found")
        return project

    async def get_active(self, project_id: uuid.UUID) -> Project:
        project = await self.get(project_id)
        if project.status is ProjectStatus.ARCHIVED:
            raise ConflictError("Project is archived and read-only")
        return project

    async def create(self, data: ProjectCreate, owner: User) -> Project:
        if await self.projects.by_name(data.name):
            raise ConflictError(f"A project named '{data.name}' already exists")
        project = Project(name=data.name, description=data.description, owner_id=owner.id)
        self.session.add(project)
        await self._commit_unique(data.name)
        return project

    async def update(self, project_id: uuid.UUID, data: ProjectUpdate) -> Project:
        project = await self.get_active(project_id)
        changes = data.model_dump(exclude_unset=True, by_alias=False)
        if "name" in changes and changes["name"] is not None:
            name = changes["name"].strip()
            existing = await self.projects.by_name(name)
            if existing and existing.id != project.id:
                raise ConflictError(f"A project named '{name}' already exists")
            project.name = name
        if "description" in changes:
            project.description = changes["description"]
        project.updated_at = datetime.now(UTC)
        await self._commit_unique(project.name)
        return project

    async def archive(self, project_id: uuid.UUID) -> None:
        project = await self.get(project_id)
        if project.status is ProjectStatus.ARCHIVED:
            return
        project.status = ProjectStatus.ARCHIVED
        project.archived_at = datetime.now(UTC)
        await self.session.commit()

    async def _commit_unique(self, name: str) -> None:
        try:
            await self.session.commit()
        except IntegrityError as exc:
            await self.session.rollback()
            raise ConflictError(f"A project named '{name}' already exists") from exc
