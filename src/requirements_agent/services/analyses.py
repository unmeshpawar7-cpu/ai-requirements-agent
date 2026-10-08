"""Requirement analysis use case (RAG -> LLM -> validation -> grounding -> scoring -> persist),
test-case generation and artefact export."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from requirements_agent.ai.prompts.templates import render
from requirements_agent.ai.providers.base import ProviderError
from requirements_agent.ai.rag.context import EvidenceContext
from requirements_agent.ai.schemas import (
    AnalysisOutput,
    ApiEndpoint,
    Entity,
    TestCaseGenerationOutput,
    UserStory,
)
from requirements_agent.ai.structured import StructuredOutputError
from requirements_agent.ai.usage import ai_call_context
from requirements_agent.core.errors import ConflictError, NotFoundError
from requirements_agent.db.models import Analysis, Project, RequirementItem, TestCase, User
from requirements_agent.domain.enums import AnalysisStatus, ItemType, Section
from requirements_agent.generators.gherkin import ScenarioInfo, StoryInfo, export_feature
from requirements_agent.generators.markdown_docs import render_markdown
from requirements_agent.generators.openapi_builder import OpenAPIBuilder
from requirements_agent.repositories.repositories import (
    AnalysisRepository,
    DocumentRepository,
    ProjectRepository,
)
from requirements_agent.schemas.dto import AnalysisCreate
from requirements_agent.services.container import Container
from requirements_agent.services.prompts import PromptService

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = "requirements_analysis"
TEST_CASE_PROMPT = "test_case_generation"

# One retrieval query per requested section: each pulls the evidence that section needs, and
# MMR over the union keeps the final evidence set diverse.
SECTION_QUERIES: dict[Section, str] = {
    Section.FUNCTIONAL_REQUIREMENTS: (
        "the system shall must allow users to perform functional requirement feature capability"
    ),
    Section.NON_FUNCTIONAL_REQUIREMENTS: (
        "performance response time seconds security encryption authentication availability "
        "uptime scalability concurrent users usability accessibility compliance audit retention"
    ),
    Section.USER_STORIES: "as a user I want to so that role actor be able to",
    Section.ENTITIES: "each record has includes fields attributes data number date status",
    Section.API_ENDPOINTS: "create view list search update delete submit approve assign cancel",
    Section.IMPLEMENTATION_TASKS: "integration export import configure notify workflow",
}


def normalise_sections(sections: list[Section]) -> list[Section]:
    wanted = list(dict.fromkeys(sections))
    if Section.API_ENDPOINTS in wanted and Section.ENTITIES not in wanted:
        wanted.append(Section.ENTITIES)  # endpoints reference entity models
    return [s for s in Section if s in wanted]


class AnalysisService:
    def __init__(self, session: AsyncSession, container: Container) -> None:
        self.session = session
        self.container = container
        self.analyses = AnalysisRepository(session)

    async def get(self, analysis_id: uuid.UUID) -> Analysis:
        analysis = await self.analyses.get(analysis_id)
        if analysis is None:
            raise NotFoundError("Analysis not found")
        return analysis

    async def create(self, project: Project, request: AnalysisCreate, user: User) -> Analysis:
        settings = self.container.settings
        ready = await DocumentRepository(self.session).count_ready(project.id, request.document_ids)
        if ready == 0:
            raise ConflictError(
                "The project has no processed (ready) documents to analyse"
                + (" among the selected document ids" if request.document_ids else "")
            )
        prompt = await PromptService(self.session).active(ANALYSIS_PROMPT)
        sections = normalise_sections(request.sections)
        analysis = Analysis(
            project_id=project.id,
            status=AnalysisStatus.QUEUED,
            sections=[s.value for s in sections],
            focus=request.focus,
            document_ids=[str(d) for d in request.document_ids] if request.document_ids else None,
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            provider=self.container.llm.name,
            model=self.container.llm.model,
            embedding_model=f"{self.container.embedder.name}:{self.container.embedder.model}",
            retrieval_params={
                "topKPerQuery": settings.analysis_top_k_per_query,
                "maxEvidence": settings.analysis_max_evidence,
                "minScore": settings.analysis_min_score,
                "mmrLambda": settings.mmr_lambda,
                "contextTokens": settings.analysis_context_tokens,
                "vectorBackend": self.container.vector_store.backend,
            },
            requested_by_id=user.id,
        )
        self.session.add(analysis)
        await self.session.commit()
        analysis_id = analysis.id
        self.container.jobs.submit(
            f"analysis:{analysis_id}", lambda: run_analysis(self.container, analysis_id)
        )
        return analysis

    # ------------------------------------------------------------------ test cases
    async def generate_test_cases(
        self, analysis: Analysis, user: User
    ) -> tuple[list[TestCase], list[dict[str, str]]]:
        if analysis.status is not AnalysisStatus.COMPLETED:
            raise ConflictError("Test cases can only be generated for a completed analysis")
        story_items = await self.analyses.items(analysis.id, ItemType.USER_STORY)
        if not story_items:
            raise ConflictError("The analysis has no user stories to derive test cases from")
        stories = [_story_prompt_view(item.payload) for item in story_items]
        prompt = await PromptService(self.session).active(TEST_CASE_PROMPT)
        project = await ProjectRepository(self.session).get(analysis.project_id)
        rendered = render(
            prompt.name,
            prompt.version,
            prompt.template,
            {
                "project_name": project.name if project else "",
                "stories": json.dumps(stories, indent=2),
                "story_ids": ", ".join(s["id"] for s in stories),
                "schema": json.dumps(
                    TestCaseGenerationOutput.model_json_schema(by_alias=True), indent=1
                ),
            },
        )
        settings = self.container.settings
        with ai_call_context(
            project_id=analysis.project_id, analysis_id=analysis.id, user_id=user.id
        ):
            result = await self.container.structured.generate(
                rendered,
                TestCaseGenerationOutput,
                task_input={"stories": [item.payload for item in story_items]},
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_output_tokens,
            )

        criteria_count = {s["id"]: len(s["acceptanceCriteria"]) for s in stories}
        dropped: list[dict[str, str]] = []
        accepted = []
        seen: set[str] = set()
        for case in result.value.test_cases:
            if case.id in seen:
                dropped.append({"ref": case.id, "reason": "duplicate id"})
            elif case.story_id not in criteria_count:
                dropped.append({"ref": case.id, "reason": f"unknown story '{case.story_id}'"})
            elif case.criterion_index >= criteria_count[case.story_id]:
                dropped.append({"ref": case.id, "reason": "criterion index out of range"})
            else:
                seen.add(case.id)
                accepted.append(case)

        await self.analyses.delete_test_cases(analysis.id)
        rows = []
        for ordinal, case in enumerate(accepted):
            payload = case.model_dump(by_alias=True, mode="json")
            rows.append(
                TestCase(
                    analysis_id=analysis.id,
                    ref_code=case.id,
                    story_ref=case.story_id,
                    title=case.title,
                    case_type=case.type,
                    priority=case.priority,
                    payload=payload,
                    confidence=round(case.confidence * max(0.5, 1 - 0.15 * result.repairs), 3),
                    prompt_name=prompt.name,
                    prompt_version=prompt.version,
                    ordinal=ordinal,
                )
            )
        self.session.add_all(rows)
        await self.session.commit()
        return rows, dropped

    async def feature_file(self, analysis: Analysis) -> str:
        stories = [
            UserStory.model_validate(item.payload)
            for item in await self.analyses.items(analysis.id, ItemType.USER_STORY)
        ]
        cases = await self.analyses.test_cases(analysis.id)
        header = f"Generated by ai-requirements-agent from analysis {analysis.id}\n" + (
            f"Prompt {cases[0].prompt_name} v{cases[0].prompt_version}" if cases else ""
        )
        return export_feature(
            [StoryInfo(s.id, s.title, s.role, s.goal, s.benefit) for s in stories],
            [
                ScenarioInfo(
                    id=c.ref_code,
                    story_id=c.story_ref,
                    title=c.title,
                    case_type=c.case_type,
                    priority=c.priority,
                    given=c.payload.get("given", []),
                    when=c.payload.get("when", []),
                    then=c.payload.get("then", []),
                    preconditions=c.payload.get("preconditions", []),
                )
                for c in cases
            ],
            header=header,
        )

    # ------------------------------------------------------------------ artefacts
    async def openapi_document(self, analysis: Analysis) -> dict[str, Any]:
        if analysis.status is not AnalysisStatus.COMPLETED:
            raise ConflictError("The analysis has not completed yet")
        entities = [
            Entity.model_validate(i.payload)
            for i in await self.analyses.items(analysis.id, ItemType.ENTITY)
        ]
        endpoints = [
            ApiEndpoint.model_validate(i.payload)
            for i in await self.analyses.items(analysis.id, ItemType.API_ENDPOINT)
        ]
        project = await ProjectRepository(self.session).get(analysis.project_id)
        result = OpenAPIBuilder().build(
            title=f"{project.name if project else 'Generated'} API",
            version="0.1.0",
            description=(
                f"Draft API generated from the requirement documents of project "
                f"'{project.name if project else ''}'. Review before implementing."
            ),
            entities=entities,
            endpoints=endpoints,
            extensions={
                "generator": "ai-requirements-agent",
                "analysisId": str(analysis.id),
                "prompt": f"{analysis.prompt_name}@v{analysis.prompt_version}",
                "provider": f"{analysis.provider}:{analysis.model}",
                "warnings": [],
            },
        )
        result.document["x-generated"]["warnings"] = result.warnings
        return result.document

    async def markdown_docs(self, analysis: Analysis) -> str:
        return render_markdown(await self.openapi_document(analysis))


def _story_prompt_view(payload: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "title", "role", "goal", "benefit", "acceptanceCriteria")
    return {k: payload.get(k) for k in keys}


# ---------------------------------------------------------------------- background job
async def run_analysis(container: Container, analysis_id: uuid.UUID) -> None:
    factory = container.database.session_factory
    settings = container.settings
    async with factory() as session:
        analysis = await session.get(Analysis, analysis_id)
        if analysis is None or analysis.status is not AnalysisStatus.QUEUED:
            return
        analysis.status = AnalysisStatus.RUNNING
        analysis.started_at = datetime.now(UTC)
        await session.commit()
        project = await session.get(Project, analysis.project_id)
        prompt = await PromptService(session).get(analysis.prompt_name, analysis.prompt_version)
        project_id, project_name = analysis.project_id, project.name if project else ""
        sections = [Section(s) for s in analysis.sections]
        focus, user_id = analysis.focus, analysis.requested_by_id
        document_ids = [uuid.UUID(d) for d in analysis.document_ids or []] or None
        template, prompt_name, prompt_version = prompt.template, prompt.name, prompt.version

    try:
        with ai_call_context(project_id=project_id, analysis_id=analysis_id, user_id=user_id):
            queries = [SECTION_QUERIES[s] for s in sections]
            if focus:
                queries.insert(0, focus)
            hits = await container.retriever.multi_search(
                project_id,
                queries,
                top_k_per_query=settings.analysis_top_k_per_query,
                max_results=settings.analysis_max_evidence,
                min_score=settings.analysis_min_score,
                mmr_lambda=settings.mmr_lambda,
                document_ids=document_ids,
            )
            if not hits:
                raise _AnalysisFailure("No relevant evidence was retrieved for this analysis")
            context = container.context_builder.build(hits)
            rendered = render(
                prompt_name,
                prompt_version,
                template,
                {
                    "project_name": project_name,
                    "sections": ", ".join(s.value for s in sections),
                    "focus": focus or "none",
                    "evidence": context.text,
                    "evidence_ids": ", ".join(b.label for b in context.blocks),
                    "schema": json.dumps(AnalysisOutput.model_json_schema(by_alias=True), indent=1),
                },
            )
            result = await container.structured.generate(
                rendered,
                AnalysisOutput,
                task_input={
                    "project_name": project_name,
                    "sections": [s.value for s in sections],
                    "focus": focus,
                    "evidence": [
                        {
                            "label": b.label,
                            "text": b.text,
                            "section_path": b.section_path,
                            "document_title": b.document_title,
                        }
                        for b in context.blocks
                    ],
                },
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_output_tokens,
            )
        grounded, report = container.grounding.check(result.value, context, set(sections))
        await _persist_success(container, analysis_id, result, grounded, report, context)
    except StructuredOutputError as exc:
        detail = "; ".join(exc.errors[:3])
        await _persist_failure(container, analysis_id, f"{exc} ({detail})")
    except ProviderError as exc:
        await _persist_failure(container, analysis_id, f"LLM provider error: {exc}")
    except _AnalysisFailure as exc:
        await _persist_failure(container, analysis_id, str(exc))
    except Exception as exc:
        logger.exception("analysis crashed", extra={"analysisId": str(analysis_id)})
        await _persist_failure(container, analysis_id, f"Analysis failed ({type(exc).__name__})")


class _AnalysisFailure(Exception):  # noqa: N818 - internal control-flow signal
    pass


async def _persist_success(
    container: Container,
    analysis_id: uuid.UUID,
    result: Any,
    grounded: list[Any],
    report: Any,
    context: EvidenceContext,
) -> None:
    scores = []
    async with container.database.session_factory() as session:
        analysis = await session.get(Analysis, analysis_id)
        if analysis is None:
            return
        for ordinal, item in enumerate(grounded):
            breakdown = container.confidence.score(item, context, result.repairs)
            scores.append(breakdown.final)
            session.add(
                RequirementItem(
                    analysis_id=analysis_id,
                    item_type=item.item_type,
                    ref_code=item.ref_code[:120],
                    title=item.title,
                    payload=item.payload,
                    confidence=breakdown.final,
                    evidence_chunk_ids=item.evidence_chunk_ids,
                    grounded=item.grounded,
                    grounding_issues=item.issues,
                    confidence_breakdown=breakdown.to_dict(),
                    ordinal=ordinal,
                )
            )
        analysis.evidence = [
            {
                "label": b.label,
                "chunkId": str(b.chunk_id),
                "documentId": str(b.document_id),
                "documentTitle": b.document_title,
                "sectionPath": b.section_path,
                "score": round(b.score, 6),
            }
            for b in context.blocks
        ]
        analysis.summary = result.value.summary
        analysis.open_questions = result.value.open_questions
        analysis.grounding_report = report.to_dict() | {
            "evidenceBlocks": len(context.blocks),
            "contextTokens": context.token_count,
            "skippedForBudget": context.skipped_for_budget,
        }
        analysis.repair_attempts = result.repairs
        analysis.input_tokens = result.input_tokens
        analysis.output_tokens = result.output_tokens
        analysis.model = result.model
        analysis.confidence = round(sum(scores) / len(scores), 3) if scores else None
        analysis.status = AnalysisStatus.COMPLETED
        analysis.completed_at = datetime.now(UTC)
        await session.commit()
    logger.info(
        "analysis completed",
        extra={"analysisId": str(analysis_id), "items": len(grounded), "repairs": result.repairs},
    )


async def _persist_failure(container: Container, analysis_id: uuid.UUID, message: str) -> None:
    logger.warning("analysis failed", extra={"analysisId": str(analysis_id), "reason": message})
    async with container.database.session_factory() as session:
        analysis = await session.get(Analysis, analysis_id)
        if analysis is None:
            return
        analysis.status = AnalysisStatus.FAILED
        analysis.error_message = message[:2000]
        analysis.completed_at = datetime.now(UTC)
        await session.commit()
