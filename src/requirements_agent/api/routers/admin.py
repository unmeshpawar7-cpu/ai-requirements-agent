"""Prompt version management, AI usage reporting and health checks."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Path, Query, Response, status
from fastapi.responses import JSONResponse

from requirements_agent.api.deps import Admin, ContainerDep, Paging, SessionDep
from requirements_agent.repositories.repositories import PromptRepository
from requirements_agent.schemas.common import Envelope, Page, error_responses, ok
from requirements_agent.schemas.dto import PromptVersionCreate, PromptVersionOut, UsageSummaryOut
from requirements_agent.services.lifecycle import usage_summary
from requirements_agent.services.prompts import PromptService

router = APIRouter()

PromptName = Annotated[str, Path(pattern=r"^[a-z][a-z0-9_]{1,99}$")]


@router.get(
    "/api/prompts",
    response_model=Envelope[Page[PromptVersionOut]],
    responses=error_responses(401, 403),
    tags=["Prompts"],
    summary="List prompt template versions (admin)",
)
async def list_prompts(
    _: Admin,
    session: SessionDep,
    paging: Paging,
    name: Annotated[str | None, Query(max_length=100)] = None,
) -> Envelope[Page[PromptVersionOut]]:
    prompts, total = await PromptRepository(session).list(
        name=name, offset=paging.offset, limit=paging.page_size
    )
    return ok(
        Page[PromptVersionOut].build(
            [PromptVersionOut.model_validate(p) for p in prompts],
            page=paging.page,
            page_size=paging.page_size,
            total_count=total,
        )
    )


@router.post(
    "/api/prompts/{name}/versions",
    status_code=status.HTTP_201_CREATED,
    response_model=Envelope[PromptVersionOut],
    responses=error_responses(401, 403, 404, 409, 422),
    tags=["Prompts"],
    summary="Create a new prompt version (validated against the placeholder contract)",
)
async def create_prompt_version(
    name: PromptName,
    body: PromptVersionCreate,
    admin: Admin,
    session: SessionDep,
    response: Response,
) -> Envelope[PromptVersionOut]:
    prompt = await PromptService(session).create_version(
        name, body.template, body.description, body.activate, author=admin.email
    )
    response.headers["Location"] = f"/api/prompts?name={name}"
    return ok(PromptVersionOut.model_validate(prompt), f"Created {name} v{prompt.version}")


@router.post(
    "/api/prompts/{name}/versions/{version}/activate",
    response_model=Envelope[PromptVersionOut],
    responses=error_responses(401, 403, 404),
    tags=["Prompts"],
    summary="Make a prompt version the active one",
)
async def activate_prompt_version(
    name: PromptName,
    version: Annotated[int, Path(ge=1)],
    _: Admin,
    session: SessionDep,
) -> Envelope[PromptVersionOut]:
    prompt = await PromptService(session).activate(name, version)
    return ok(PromptVersionOut.model_validate(prompt), f"Activated {name} v{version}")


@router.get(
    "/api/ai/usage",
    response_model=Envelope[UsageSummaryOut],
    responses=error_responses(401, 403, 422),
    tags=["AI usage"],
    summary="AI call aggregates by operation/provider/model/prompt (admin)",
)
async def ai_usage(
    _: Admin,
    session: SessionDep,
    since: datetime | None = None,
    until: datetime | None = None,
    project_id: Annotated[uuid.UUID | None, Query(alias="projectId")] = None,
) -> Envelope[UsageSummaryOut]:
    return ok(await usage_summary(session, since=since, until=until, project_id=project_id))


@router.get("/health/live", tags=["Health"], summary="Liveness probe")
async def live() -> dict[str, object]:
    return {"success": True, "data": {"status": "ok"}, "message": None}


@router.get("/health/ready", tags=["Health"], summary="Readiness probe (database reachable)")
async def ready(container: ContainerDep) -> JSONResponse:
    checks: dict[str, str] = {}
    try:
        await container.database.ping()
        checks["database"] = "ok"
    except Exception as exc:  # report, do not raise: this is a probe
        checks["database"] = f"error: {type(exc).__name__}"
    checks["vectorStore"] = container.vector_store.backend
    checks["llmProvider"] = f"{container.llm.name}:{container.llm.model}"
    checks["embeddings"] = f"{container.embedder.name}:{container.embedder.model}"
    healthy = checks["database"] == "ok"
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "success": healthy,
            "data": {"status": "ready" if healthy else "unavailable", "checks": checks},
            "message": None if healthy else "Dependency check failed",
        },
    )
