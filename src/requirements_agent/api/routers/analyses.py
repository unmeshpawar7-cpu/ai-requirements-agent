"""Analyses, generated artefacts (OpenAPI, Markdown) and test cases."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse

from requirements_agent.api.deps import (
    ContainerDep,
    Paging,
    RateLimitedAnalyst,
    SessionDep,
    Viewer,
)
from requirements_agent.db.models import Analysis
from requirements_agent.domain.enums import ItemType
from requirements_agent.repositories.repositories import AnalysisRepository
from requirements_agent.schemas.common import Envelope, Page, error_responses, ok
from requirements_agent.schemas.dto import (
    AnalysisCreate,
    AnalysisOut,
    AnalysisSummaryOut,
    EvidenceOut,
    RequirementItemOut,
    TestCaseGenerationOut,
    TestCaseOut,
)
from requirements_agent.services.analyses import AnalysisService
from requirements_agent.services.projects import ProjectService

router = APIRouter(prefix="/api")


def _summary(analysis: Analysis, counts: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "id": analysis.id,
        "project_id": analysis.project_id,
        "status": analysis.status,
        "sections": analysis.sections,
        "focus": analysis.focus,
        "prompt_name": analysis.prompt_name,
        "prompt_version": analysis.prompt_version,
        "provider": analysis.provider,
        "model": analysis.model,
        "confidence": analysis.confidence,
        "item_counts": counts or {},
        "error_message": analysis.error_message,
        "created_at": analysis.created_at,
        "completed_at": analysis.completed_at,
    }


@router.post(
    "/projects/{project_id}/analyses",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=Envelope[AnalysisSummaryOut],
    responses=error_responses(401, 403, 404, 409, 422, 429),
    tags=["Analyses"],
    summary="Start a requirement analysis (RAG + LLM); runs in the background",
)
async def create_analysis(
    project_id: uuid.UUID,
    body: AnalysisCreate,
    user: RateLimitedAnalyst,
    session: SessionDep,
    container: ContainerDep,
    response: Response,
) -> Envelope[AnalysisSummaryOut]:
    project = await ProjectService(session).get_active(project_id)
    analysis = await AnalysisService(session, container).create(project, body, user)
    response.headers["Location"] = f"/api/analyses/{analysis.id}"
    return ok(AnalysisSummaryOut.model_validate(_summary(analysis)), "Analysis queued")


@router.get(
    "/projects/{project_id}/analyses",
    response_model=Envelope[Page[AnalysisSummaryOut]],
    responses=error_responses(401, 404),
    tags=["Analyses"],
)
async def list_analyses(
    project_id: uuid.UUID, _: Viewer, session: SessionDep, paging: Paging
) -> Envelope[Page[AnalysisSummaryOut]]:
    await ProjectService(session).get(project_id)
    repo = AnalysisRepository(session)
    analyses, total = await repo.list(project_id, offset=paging.offset, limit=paging.page_size)
    counts = await repo.item_counts([a.id for a in analyses])
    return ok(
        Page[AnalysisSummaryOut].build(
            [AnalysisSummaryOut.model_validate(_summary(a, counts.get(a.id))) for a in analyses],
            page=paging.page,
            page_size=paging.page_size,
            total_count=total,
        )
    )


@router.get(
    "/analyses/{analysis_id}",
    response_model=Envelope[AnalysisOut],
    responses=error_responses(401, 404),
    tags=["Analyses"],
    summary="Get an analysis with its grounded artefacts grouped by type",
)
async def get_analysis(
    analysis_id: uuid.UUID, _: Viewer, session: SessionDep, container: ContainerDep
) -> Envelope[AnalysisOut]:
    analysis = await AnalysisService(session, container).get(analysis_id)
    items = await AnalysisRepository(session).items(analysis.id)
    grouped: dict[str, list[RequirementItemOut]] = {t.value: [] for t in ItemType}
    for item in items:
        grouped[item.item_type.value].append(RequirementItemOut.model_validate(item))
    data = _summary(analysis, {k: len(v) for k, v in grouped.items() if v}) | {
        "summary": analysis.summary,
        "open_questions": analysis.open_questions or [],
        "embedding_model": analysis.embedding_model,
        "retrieval_params": analysis.retrieval_params,
        "evidence": [EvidenceOut.model_validate(e) for e in analysis.evidence or []],
        "grounding_report": analysis.grounding_report,
        "repair_attempts": analysis.repair_attempts,
        "input_tokens": analysis.input_tokens,
        "output_tokens": analysis.output_tokens,
        "started_at": analysis.started_at,
        "items": grouped,
    }
    return ok(AnalysisOut.model_validate(data))


@router.get(
    "/analyses/{analysis_id}/openapi.json",
    responses=error_responses(401, 404, 409)
    | {200: {"description": "OpenAPI 3.1 document", "content": {"application/json": {}}}},
    tags=["Artefacts"],
    summary="Generated OpenAPI 3.1 document (raw JSON, not enveloped)",
)
async def openapi_document(
    analysis_id: uuid.UUID, _: Viewer, session: SessionDep, container: ContainerDep
) -> JSONResponse:
    service = AnalysisService(session, container)
    document = await service.openapi_document(await service.get(analysis_id))
    return JSONResponse(document)


@router.get(
    "/analyses/{analysis_id}/api-docs.md",
    response_class=PlainTextResponse,
    responses=error_responses(401, 404, 409)
    | {200: {"description": "Markdown API documentation", "content": {"text/markdown": {}}}},
    tags=["Artefacts"],
    summary="Generated Markdown API documentation",
)
async def markdown_docs(
    analysis_id: uuid.UUID, _: Viewer, session: SessionDep, container: ContainerDep
) -> PlainTextResponse:
    service = AnalysisService(session, container)
    text = await service.markdown_docs(await service.get(analysis_id))
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")


@router.post(
    "/analyses/{analysis_id}/test-cases",
    status_code=status.HTTP_201_CREATED,
    response_model=Envelope[TestCaseGenerationOut],
    responses=error_responses(401, 403, 404, 409, 429),
    tags=["Test cases"],
    summary="Generate test cases from the analysis' acceptance criteria (replaces previous)",
)
async def generate_test_cases(
    analysis_id: uuid.UUID,
    user: RateLimitedAnalyst,
    session: SessionDep,
    container: ContainerDep,
    response: Response,
) -> Envelope[TestCaseGenerationOut]:
    service = AnalysisService(session, container)
    analysis = await service.get(analysis_id)
    await ProjectService(session).get_active(analysis.project_id)
    rows, dropped = await service.generate_test_cases(analysis, user)
    response.headers["Location"] = f"/api/analyses/{analysis_id}/test-cases"
    return ok(
        TestCaseGenerationOut(
            analysis_id=analysis_id,
            generated=len(rows),
            dropped=dropped,
            test_cases=[TestCaseOut.model_validate(r) for r in rows],
        )
    )


@router.get(
    "/analyses/{analysis_id}/test-cases",
    response_model=Envelope[Page[TestCaseOut]],
    responses=error_responses(401, 404),
    tags=["Test cases"],
)
async def list_test_cases(
    analysis_id: uuid.UUID,
    _: Viewer,
    session: SessionDep,
    container: ContainerDep,
    paging: Paging,
) -> Envelope[Page[TestCaseOut]]:
    analysis = await AnalysisService(session, container).get(analysis_id)
    cases = await AnalysisRepository(session).test_cases(analysis.id)
    window = cases[paging.offset : paging.offset + paging.page_size]
    return ok(
        Page[TestCaseOut].build(
            [TestCaseOut.model_validate(c) for c in window],
            page=paging.page,
            page_size=paging.page_size,
            total_count=len(cases),
        )
    )


@router.get(
    "/analyses/{analysis_id}/test-cases.feature",
    response_class=PlainTextResponse,
    responses=error_responses(401, 404)
    | {200: {"description": "Gherkin feature file", "content": {"text/plain": {}}}},
    tags=["Test cases"],
    summary="Export test cases as a Gherkin .feature file",
)
async def feature_file(
    analysis_id: uuid.UUID, _: Viewer, session: SessionDep, container: ContainerDep
) -> PlainTextResponse:
    service = AnalysisService(session, container)
    text = await service.feature_file(await service.get(analysis_id))
    return PlainTextResponse(
        text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'inline; filename="analysis-{analysis_id}.feature"'},
    )
