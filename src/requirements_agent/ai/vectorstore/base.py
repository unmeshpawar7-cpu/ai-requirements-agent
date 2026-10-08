"""Vector store abstraction.

A vector store indexes chunk embeddings and answers cosine-similarity queries scoped to a
project. Chunk text and metadata live in `document_chunks`; the store only deals with ids,
vectors and scores, so the retriever can hydrate hits from the database uniformly for every
backend.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class VectorRecord:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    project_id: uuid.UUID
    embedding: Sequence[float]


@dataclass(frozen=True, slots=True)
class VectorHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    score: float  # cosine similarity in [-1, 1]
    embedding: tuple[float, ...]


@runtime_checkable
class VectorStore(Protocol):
    backend: str

    async def upsert(self, records: Sequence[VectorRecord]) -> None: ...

    async def search(
        self,
        project_id: uuid.UUID,
        query: Sequence[float],
        *,
        limit: int,
        min_score: float = 0.0,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[VectorHit]:
        """Return up to `limit` hits with score >= min_score, best first."""
        ...

    async def delete_document(self, document_id: uuid.UUID) -> None: ...
