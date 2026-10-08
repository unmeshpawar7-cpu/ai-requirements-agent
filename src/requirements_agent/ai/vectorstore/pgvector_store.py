"""PostgreSQL + pgvector store.

Embeddings live in `document_chunks.embedding` (`vector(dim)`), indexed with HNSW using
`vector_cosine_ops`. Queries order by the cosine distance operator `<=>` so the planner can use
the index; `hnsw.ef_search` is raised per query (transaction-local) when a large candidate set
is requested, because HNSW returns at most `ef_search` rows and project filtering happens
after the index scan.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, bindparam, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from requirements_agent.ai.vectorstore.base import VectorHit, VectorRecord
from requirements_agent.db.models import DocumentChunk


class PgVectorStore:
    backend = "pgvector"

    def __init__(self, session_factory: async_sessionmaker[AsyncSession], dimension: int) -> None:
        self._session_factory = session_factory
        self.dimension = dimension

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        rows = [
            {"id": record.chunk_id, "embedding": [float(x) for x in record.embedding]}
            for record in records
        ]
        async with self._session_factory() as session:
            # ORM "bulk UPDATE by primary key" -> a single executemany round trip.
            await session.execute(update(DocumentChunk), rows)
            await session.commit()

    async def search(
        self,
        project_id: uuid.UUID,
        query: Sequence[float],
        *,
        limit: int,
        min_score: float = 0.0,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[VectorHit]:
        if limit <= 0:
            return []
        query_vec = bindparam("query_vec", [float(x) for x in query], type_=Vector(self.dimension))
        distance = DocumentChunk.embedding.op("<=>", return_type=Float)(query_vec)
        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                distance.label("distance"),
                DocumentChunk.embedding,
            )
            .where(DocumentChunk.project_id == project_id)
            .where(DocumentChunk.embedding.is_not(None))
            .order_by(distance)
            .limit(limit)
        )
        if document_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(list(document_ids)))

        async with self._session_factory() as session, session.begin():
            ef_search = max(40, min(1000, limit * 2))
            await session.execute(
                text("SELECT set_config('hnsw.ef_search', :value, true)"),
                {"value": str(ef_search)},
            )
            rows = (await session.execute(stmt)).all()

        hits: list[VectorHit] = []
        for chunk_id, document_id, dist, embedding in rows:
            score = 1.0 - float(dist)
            if score < min_score:
                continue
            hits.append(
                VectorHit(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    score=score,
                    embedding=tuple(float(x) for x in embedding),
                )
            )
        return hits

    async def delete_document(self, document_id: uuid.UUID) -> None:
        # Embeddings are columns on the chunk rows, which are deleted by the FK cascade.
        async with self._session_factory() as session:
            await session.execute(
                update(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .values(embedding=None)
            )
            await session.commit()
