"""Prompt template versioning.

* Seed files (`ai/prompts/<name>/v<N>.md`) are inserted at startup if missing. A seeded version
  is immutable: if a file changes after seeding, the checksum mismatch is logged and the DB
  copy wins - publish a new version instead of editing history.
* Admins create new versions through the API (validated against the prompt's placeholder
  contract) and activate exactly one version per prompt name.
* Every analysis / test-case run records the prompt name + version it used.
"""

from __future__ import annotations

import logging

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from requirements_agent.ai.prompts.templates import (
    PROMPT_SPECS,
    PromptTemplateError,
    checksum,
    load_seed_templates,
    validate_template,
)
from requirements_agent.core.errors import ConflictError, NotFoundError, UnprocessableError
from requirements_agent.db.models import PromptTemplate
from requirements_agent.repositories.repositories import PromptRepository
from requirements_agent.schemas.common import FieldError

logger = logging.getLogger(__name__)

SEED_AUTHOR = "system:seed"


class PromptService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.prompts = PromptRepository(session)

    async def seed(self) -> int:
        inserted = 0
        for seed in load_seed_templates():
            existing = await self.prompts.get(seed.name, seed.version)
            if existing:
                if existing.checksum != seed.checksum:
                    logger.warning(
                        "seed prompt differs from stored version; keeping stored copy",
                        extra={"prompt": seed.name, "version": seed.version},
                    )
                continue
            has_active = await self.prompts.active(seed.name) is not None
            self.session.add(
                PromptTemplate(
                    name=seed.name,
                    version=seed.version,
                    template=seed.template,
                    description=seed.description,
                    is_active=not has_active,
                    checksum=seed.checksum,
                    created_by=SEED_AUTHOR,
                )
            )
            await self.session.flush()
            inserted += 1
        await self.session.commit()
        return inserted

    async def active(self, name: str) -> PromptTemplate:
        prompt = await self.prompts.active(name)
        if prompt is None:
            raise NotFoundError(f"No active version of prompt '{name}'")
        return prompt

    async def get(self, name: str, version: int) -> PromptTemplate:
        prompt = await self.prompts.get(name, version)
        if prompt is None:
            raise NotFoundError(f"Prompt '{name}' version {version} not found")
        return prompt

    async def create_version(
        self, name: str, template: str, description: str | None, activate: bool, author: str
    ) -> PromptTemplate:
        if name not in PROMPT_SPECS:
            raise NotFoundError(f"Unknown prompt '{name}'")
        try:
            validate_template(name, template)
        except PromptTemplateError as exc:
            raise UnprocessableError(
                "Invalid prompt template", errors=[FieldError(field="template", message=str(exc))]
            ) from exc
        version = await self.prompts.max_version(name) + 1
        if activate:
            await self.prompts.deactivate_all(name)
        prompt = PromptTemplate(
            name=name,
            version=version,
            template=template,
            description=description,
            is_active=activate,
            checksum=checksum(template),
            created_by=author,
        )
        self.session.add(prompt)
        try:
            await self.session.commit()
        except IntegrityError as exc:
            await self.session.rollback()
            raise ConflictError("A concurrent change created this version; retry") from exc
        return prompt

    async def activate(self, name: str, version: int) -> PromptTemplate:
        prompt = await self.get(name, version)
        if not prompt.is_active:
            await self.prompts.deactivate_all(name)
            await self.session.flush()
            prompt.is_active = True
            await self.session.commit()
        return prompt
