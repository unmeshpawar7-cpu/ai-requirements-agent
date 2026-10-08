"""Document intake (upload / raw text) and background processing.

Intake (request path, fast): size and type checks, SHA-256 checksum, duplicate detection per
project (409), persist raw bytes with status `pending`, enqueue processing, return 202.

Processing (background job): `pending -> processing -> ready | failed`
extract text (thread pool, CPU-bound) -> normalise -> chunk -> embed -> persist chunks ->
index vectors -> `ready`. Any failure stores a client-safe error message and removes partial
chunks, so a retry starts clean.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from requirements_agent.ai.usage import ai_call_context
from requirements_agent.ai.vectorstore.base import VectorRecord
from requirements_agent.core.errors import (
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    UnprocessableError,
    UnsupportedMediaTypeError,
)
from requirements_agent.db.models import Document, DocumentChunk, Project, User
from requirements_agent.domain.enums import DocumentSource, DocumentStatus
from requirements_agent.ingestion.extraction import (
    CONTENT_TYPES,
    DocumentFormat,
    ExtractionError,
    UnsupportedFormatError,
    detect_format,
    extract_text,
)
from requirements_agent.ingestion.normalise import normalise_text
from requirements_agent.ingestion.tokens import estimate_tokens
from requirements_agent.repositories.repositories import ChunkRepository, DocumentRepository
from requirements_agent.services.container import Container

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, session: AsyncSession, container: Container) -> None:
        self.session = session
        self.container = container
        self.documents = DocumentRepository(session)

    async def get(self, document_id: uuid.UUID) -> Document:
        document = await self.documents.get(document_id)
        if document is None:
            raise NotFoundError("Document not found")
        return document

    async def upload(
        self,
        project: Project,
        *,
        filename: str | None,
        content_type: str | None,
        data: bytes,
        user: User,
    ) -> Document:
        try:
            fmt = detect_format(filename, content_type)
        except UnsupportedFormatError as exc:
            raise UnsupportedMediaTypeError(str(exc)) from exc
        title = (filename or "document").rsplit("/", 1)[-1]
        return await self._accept(
            project,
            title=title,
            filename=filename,
            fmt=fmt,
            data=data,
            source=DocumentSource.UPLOAD,
            user=user,
        )

    async def create_from_text(
        self, project: Project, *, title: str, content: str, fmt: str, user: User
    ) -> Document:
        document_format = DocumentFormat.MARKDOWN if fmt == "markdown" else DocumentFormat.TEXT
        return await self._accept(
            project,
            title=title.strip(),
            filename=None,
            fmt=document_format,
            data=content.encode("utf-8"),
            source=DocumentSource.TEXT,
            user=user,
        )

    async def _accept(
        self,
        project: Project,
        *,
        title: str,
        filename: str | None,
        fmt: DocumentFormat,
        data: bytes,
        source: DocumentSource,
        user: User,
    ) -> Document:
        limit = self.container.settings.max_upload_bytes
        if len(data) > limit:
            raise PayloadTooLargeError(f"Document exceeds the {limit // (1024 * 1024)} MB limit")
        if not data.strip():
            raise UnprocessableError("Document is empty")
        checksum = hashlib.sha256(data).hexdigest()
        duplicate = await self.documents.by_checksum(project.id, checksum)
        if duplicate:
            raise ConflictError(
                f"This document was already uploaded to the project (document {duplicate.id})"
            )
        document = Document(
            project_id=project.id,
            title=title[:300],
            filename=filename[:300] if filename else None,
            content_type=CONTENT_TYPES[fmt],
            source=source,
            size_bytes=len(data),
            checksum=checksum,
            status=DocumentStatus.PENDING,
            raw_content=data,
            uploaded_by_id=user.id,
        )
        self.session.add(document)
        try:
            await self.session.commit()
        except IntegrityError as exc:  # concurrent upload of the same file
            await self.session.rollback()
            raise ConflictError("This document was already uploaded to the project") from exc
        enqueue_processing(self.container, document.id)
        return document

    async def delete(self, document: Document) -> None:
        document_id = document.id
        await self.session.delete(document)
        await self.session.commit()
        await self.container.vector_store.delete_document(document_id)


def enqueue_processing(container: Container, document_id: uuid.UUID) -> None:
    container.jobs.submit(
        f"process-document:{document_id}", lambda: process_document(container, document_id)
    )


async def process_document(container: Container, document_id: uuid.UUID) -> None:
    """Background job: extract, normalise, chunk, embed and index one document."""
    factory = container.database.session_factory
    async with factory() as session:
        repo = DocumentRepository(session)
        document = await repo.get(document_id, with_content=True)
        if document is None or document.status is DocumentStatus.READY:
            return
        document.status = DocumentStatus.PROCESSING
        document.error_message = None
        await session.commit()
        project_id = document.project_id
        raw = document.raw_content or b""
        fmt = _format_of(document.content_type)

    try:
        with ai_call_context(project_id=project_id):
            extracted = await asyncio.to_thread(extract_text, raw, fmt)
            text = normalise_text(extracted.text)
            chunks = await asyncio.to_thread(container.chunker.split, text)
            if not chunks:
                raise ExtractionError("Document contains no usable text after normalisation")
            # Embed heading context with the body: a chunk that continues a section after an
            # overlap boundary still "knows" which requirement area it belongs to.
            vectors = await container.embedder.embed(
                [f"{c.section_path}\n\n{c.text}" if c.section_path else c.text for c in chunks]
            )

        async with factory() as session:
            chunk_repo = ChunkRepository(session)
            await chunk_repo.delete_for_document(document_id)
            rows = [
                DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=document_id,
                    project_id=project_id,
                    ordinal=c.ordinal,
                    text=c.text,
                    section_path=c.section_path[:500],
                    token_count=c.token_count,
                    char_start=c.char_start,
                    char_end=c.char_end,
                )
                for c in chunks
            ]
            session.add_all(rows)
            await session.commit()
            records = [
                VectorRecord(row.id, document_id, project_id, vector)
                for row, vector in zip(rows, vectors, strict=True)
            ]

        await container.vector_store.upsert(records)

        async with factory() as session:
            document = await DocumentRepository(session).get(document_id)
            if document is None:  # deleted while processing
                await container.vector_store.delete_document(document_id)
                return
            document.text_content = text
            document.char_count = len(text)
            document.token_estimate = estimate_tokens(text)
            document.chunk_count = len(chunks)
            document.status = DocumentStatus.READY
            document.processed_at = datetime.now(UTC)
            await session.commit()
        logger.info(
            "document processed",
            extra={"documentId": str(document_id), "chunks": len(chunks)},
        )
    except Exception as exc:
        message = str(exc) if isinstance(exc, ExtractionError) else "Processing failed"
        logger.warning(
            "document processing failed",
            extra={"documentId": str(document_id), "error": type(exc).__name__},
            exc_info=not isinstance(exc, ExtractionError),
        )
        await container.vector_store.delete_document(document_id)
        async with factory() as session:
            await ChunkRepository(session).delete_for_document(document_id)
            document = await DocumentRepository(session).get(document_id)
            if document is not None:
                document.status = DocumentStatus.FAILED
                document.error_message = message[:1000]
                document.chunk_count = 0
                document.processed_at = datetime.now(UTC)
            await session.commit()


def _format_of(content_type: str) -> DocumentFormat:
    for fmt, mime in CONTENT_TYPES.items():
        if mime == content_type:
            return fmt
    return DocumentFormat.TEXT
