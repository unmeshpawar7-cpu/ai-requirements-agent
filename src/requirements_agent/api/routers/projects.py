"""Projects, documents and semantic search endpoints."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Query, Response, UploadFile, status

from requirements_agent.ai.usage import ai_call_context
from requirements_agent.api.deps import Analyst, ContainerDep, Paging, SessionDep, Viewer
from requirements_agent.core.errors import PayloadTooLargeError
from requirements_agent.domain.enums import DocumentStatus
from requirements_agent.repositories.repositories import (
    ChunkRepository,
    DocumentRepository,
    ProjectRepository,
)
from requirements_agent.schemas.common import Envelope, Page, error_responses, ok
from requirements_agent.schemas.dto import (
    ChunkOut,
    DocumentOut,
    ProjectCreate,
    ProjectOut,
    ProjectUpdate,
    SearchHitOut,
    SearchRequest,
    SearchResponse,
    TextDocumentCreate,
)
from requirements_agent.services.documents import DocumentService
from requirements_agent.services.projects import ProjectService, to_out

router = APIRouter(prefix="/api")

# --------------------------------------------------------------------------- projects


@router.post(
    "/projects",
    status_code=status.HTTP_201_CREATED,
    response_model=Envelope[ProjectOut],
    responses=error_responses(401, 403, 409, 422),
    tags=["Projects"],
    summary="Create a project",
)
async def create_project(
    body: ProjectCreate, user: Analyst, session: SessionDep, response: Response
) -> Envelope[ProjectOut]:
    project = await ProjectService(session).create(body, user)
    response.headers["Location"] = f"/api/projects/{project.id}"
    return ok(to_out(project), "Project created")


@router.get(
    "/projects",
    response_model=Envelope[Page[ProjectOut]],
    responses=error_responses(401),
    tags=["Projects"],
    summary="List projects (active only unless includeArchived=true)",
)
async def list_projects(
    _: Viewer,
    session: SessionDep,
    paging: Paging,
    search: Annotated[str | None, Query(max_length=200)] = None,
    include_archived: Annotated[bool, Query(alias="includeArchived")] = False,
) -> Envelope[Page[ProjectOut]]:
    rows, total = await ProjectRepository(session).list_with_counts(
        offset=paging.offset,
        limit=paging.page_size,
        search=search,
        include_archived=include_archived,
    )
    return ok(
        Page[ProjectOut].build(
            [to_out(p, d, a) for p, d, a in rows],
            page=paging.page,
            page_size=paging.page_size,
            total_count=total,
        )
    )


@router.get(
    "/projects/{project_id}",
    response_model=Envelope[ProjectOut],
    responses=error_responses(401, 404),
    tags=["Projects"],
)
async def get_project(
    project_id: uuid.UUID, _: Viewer, session: SessionDep
) -> Envelope[ProjectOut]:
    project = await ProjectService(session).get(project_id)
    documents, analyses = await ProjectRepository(session).counts_for(project.id)
    return ok(to_out(project, documents, analyses))


@router.patch(
    "/projects/{project_id}",
    response_model=Envelope[ProjectOut],
    responses=error_responses(401, 403, 404, 409, 422),
    tags=["Projects"],
)
async def update_project(
    project_id: uuid.UUID, body: ProjectUpdate, _: Analyst, session: SessionDep
) -> Envelope[ProjectOut]:
    project = await ProjectService(session).update(project_id, body)
    documents, analyses = await ProjectRepository(session).counts_for(project.id)
    return ok(to_out(project, documents, analyses))


@router.delete(
    "/projects/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=error_responses(401, 403, 404),
    tags=["Projects"],
    summary="Archive a project (soft delete; data is kept read-only)",
)
async def archive_project(project_id: uuid.UUID, _: Analyst, session: SessionDep) -> Response:
    await ProjectService(session).archive(project_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# -------------------------------------------------------------------------- documents


@router.post(
    "/projects/{project_id}/documents",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=Envelope[DocumentOut],
    responses=error_responses(401, 403, 404, 409, 422)
    | {413: {"description": "File too large"}, 415: {"description": "Unsupported file type"}},
    tags=["Documents"],
    summary="Upload a requirement document (.txt, .md, .pdf, .docx); processed in background",
)
async def upload_document(
    project_id: uuid.UUID,
    user: Analyst,
    session: SessionDep,
    container: ContainerDep,
    response: Response,
    file: Annotated[UploadFile, File(description="Requirement document")],
) -> Envelope[DocumentOut]:
    project = await ProjectService(session).get_active(project_id)
    limit = container.settings.max_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise PayloadTooLargeError(f"Document exceeds the {limit // (1024 * 1024)} MB limit")
    document = await DocumentService(session, container).upload(
        project, filename=file.filename, content_type=file.content_type, data=data, user=user
    )
    response.headers["Location"] = f"/api/documents/{document.id}"
    return ok(DocumentOut.model_validate(document), "Document accepted for processing")


@router.post(
    "/projects/{project_id}/documents/text",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=Envelope[DocumentOut],
    responses=error_responses(401, 403, 404, 409, 422),
    tags=["Documents"],
    summary="Submit requirement text directly; processed in background",
)
async def create_text_document(
    project_id: uuid.UUID,
    body: TextDocumentCreate,
    user: Analyst,
    session: SessionDep,
    container: ContainerDep,
    response: Response,
) -> Envelope[DocumentOut]:
    project = await ProjectService(session).get_active(project_id)
    document = await DocumentService(session, container).create_from_text(
        project, title=body.title, content=body.content, fmt=body.format, user=user
    )
    response.headers["Location"] = f"/api/documents/{document.id}"
    return ok(DocumentOut.model_validate(document), "Document accepted for processing")


@router.get(
    "/projects/{project_id}/documents",
    response_model=Envelope[Page[DocumentOut]],
    responses=error_responses(401, 404),
    tags=["Documents"],
)
async def list_documents(
    project_id: uuid.UUID,
    _: Viewer,
    session: SessionDep,
    paging: Paging,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
) -> Envelope[Page[DocumentOut]]:
    await ProjectService(session).get(project_id)
    documents, total = await DocumentRepository(session).list(
        project_id, offset=paging.offset, limit=paging.page_size, status=status_filter
    )
    return ok(
        Page[DocumentOut].build(
            [DocumentOut.model_validate(d) for d in documents],
            page=paging.page,
            page_size=paging.page_size,
            total_count=total,
        )
    )


@router.get(
    "/documents/{document_id}",
    response_model=Envelope[DocumentOut],
    responses=error_responses(401, 404),
    tags=["Documents"],
    summary="Get a document's processing status and chunk count",
)
async def get_document(
    document_id: uuid.UUID, _: Viewer, session: SessionDep, container: ContainerDep
) -> Envelope[DocumentOut]:
    document = await DocumentService(session, container).get(document_id)
    return ok(DocumentOut.model_validate(document))


@router.get(
    "/documents/{document_id}/chunks",
    response_model=Envelope[Page[ChunkOut]],
    responses=error_responses(401, 404),
    tags=["Documents"],
    summary="Inspect the chunks a document was split into",
)
async def list_chunks(
    document_id: uuid.UUID,
    _: Viewer,
    session: SessionDep,
    container: ContainerDep,
    paging: Paging,
) -> Envelope[Page[ChunkOut]]:
    await DocumentService(session, container).get(document_id)
    chunks, total = await ChunkRepository(session).list(
        document_id, offset=paging.offset, limit=paging.page_size
    )
    return ok(
        Page[ChunkOut].build(
            [ChunkOut.model_validate(c) for c in chunks],
            page=paging.page,
            page_size=paging.page_size,
            total_count=total,
        )
    )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=error_responses(401, 403, 404, 409),
    tags=["Documents"],
    summary="Delete a document, its chunks and vectors",
)
async def delete_document(
    document_id: uuid.UUID, _: Analyst, session: SessionDep, container: ContainerDep
) -> Response:
    service = DocumentService(session, container)
    document = await service.get(document_id)
    await ProjectService(session).get_active(document.project_id)
    await service.delete(document)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ----------------------------------------------------------------------------- search


@router.post(
    "/projects/{project_id}/search",
    response_model=Envelope[SearchResponse],
    responses=error_responses(401, 404, 422),
    tags=["Search"],
    summary="Semantic search over the project's chunks (top-k + optional MMR)",
)
async def search(
    project_id: uuid.UUID,
    body: SearchRequest,
    user: Viewer,
    session: SessionDep,
    container: ContainerDep,
) -> Envelope[SearchResponse]:
    await ProjectService(session).get(project_id)
    with ai_call_context(project_id=project_id, user_id=user.id):
        hits = await container.retriever.search(
            project_id,
            body.query,
            top_k=body.top_k,
            min_score=body.min_score,
            use_mmr=body.use_mmr,
            mmr_lambda=body.mmr_lambda,
            document_ids=body.document_ids,
        )
    return ok(
        SearchResponse(
            query=body.query,
            embedding_provider=f"{container.embedder.name}:{container.embedder.model}",
            vector_backend=container.vector_store.backend,
            results=[
                SearchHitOut(
                    rank=h.rank,
                    score=h.score,
                    chunk_id=h.chunk_id,
                    document_id=h.document_id,
                    document_title=h.document_title,
                    ordinal=h.ordinal,
                    section_path=h.section_path,
                    text=h.text,
                )
                for h in hits
            ],
        )
    )
