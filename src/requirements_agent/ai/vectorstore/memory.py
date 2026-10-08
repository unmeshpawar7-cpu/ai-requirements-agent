"""Process-local vector store backed by a numpy matrix (exact cosine search).

Used for tests and SQLite runs. It is O(n) per query, which is fine for thousands of chunks;
PostgreSQL + pgvector (HNSW) is the production backend.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence

import numpy as np

from requirements_agent.ai.vectorstore.base import VectorHit, VectorRecord


class InMemoryVectorStore:
    backend = "memory"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._ids: list[uuid.UUID] = []
        self._documents: list[uuid.UUID] = []
        self._projects: list[uuid.UUID] = []
        self._matrix = np.zeros((0, dimension), dtype=np.float32)
        self._position: dict[uuid.UUID, int] = {}
        self._lock = asyncio.Lock()

    def __len__(self) -> int:
        return len(self._ids)

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        async with self._lock:
            new_rows: list[np.ndarray] = []
            for record in records:
                vector = self._normalise(record.embedding)
                existing = self._position.get(record.chunk_id)
                if existing is not None:
                    self._matrix[existing] = vector
                    self._documents[existing] = record.document_id
                    self._projects[existing] = record.project_id
                    continue
                self._position[record.chunk_id] = len(self._ids) + len(new_rows)
                new_rows.append(vector)
                self._ids.append(record.chunk_id)
                self._documents.append(record.document_id)
                self._projects.append(record.project_id)
            if new_rows:
                self._matrix = np.vstack([self._matrix, np.stack(new_rows)])

    async def search(
        self,
        project_id: uuid.UUID,
        query: Sequence[float],
        *,
        limit: int,
        min_score: float = 0.0,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[VectorHit]:
        if limit <= 0 or not self._ids:
            return []
        q = self._normalise(query)
        allowed_docs = set(document_ids) if document_ids else None
        mask = np.fromiter(
            (
                p == project_id and (allowed_docs is None or d in allowed_docs)
                for p, d in zip(self._projects, self._documents, strict=True)
            ),
            dtype=bool,
            count=len(self._ids),
        )
        if not mask.any():
            return []
        candidate_idx = np.nonzero(mask)[0]
        scores = self._matrix[candidate_idx] @ q
        order = np.argsort(-scores, kind="stable")
        hits: list[VectorHit] = []
        for position in order:
            score = float(scores[position])
            if score < min_score:
                break
            row = int(candidate_idx[position])
            hits.append(
                VectorHit(
                    chunk_id=self._ids[row],
                    document_id=self._documents[row],
                    score=score,
                    embedding=tuple(float(x) for x in self._matrix[row]),
                )
            )
            if len(hits) >= limit:
                break
        return hits

    async def delete_document(self, document_id: uuid.UUID) -> None:
        async with self._lock:
            keep = [i for i, d in enumerate(self._documents) if d != document_id]
            if len(keep) == len(self._ids):
                return
            self._ids = [self._ids[i] for i in keep]
            self._documents = [self._documents[i] for i in keep]
            self._projects = [self._projects[i] for i in keep]
            self._matrix = self._matrix[keep] if keep else np.zeros((0, self.dimension), np.float32)
            self._position = {chunk_id: i for i, chunk_id in enumerate(self._ids)}

    def _normalise(self, vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (self.dimension,):
            raise ValueError(f"expected a {self.dimension}-d vector, got shape {arr.shape}")
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm else arr
