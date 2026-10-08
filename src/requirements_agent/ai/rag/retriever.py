"""Retrieval: embed query -> vector search (over-fetch) -> hydrate -> MMR -> top-k.

* `search` serves the semantic-search endpoint (single query).
* `multi_search` serves analyses: one query per requested artefact section (plus the caller's
  focus), merged by best score per chunk, then MMR across the union so the evidence set covers
  different parts of the corpus instead of k near-duplicates.

Only chunks of documents in status `ready` are returned (enforced by the loader), so partially
processed documents never leak into results.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from requirements_agent.ai.embeddings.base import EmbeddingProvider
from requirements_agent.ai.rag.mmr import mmr_rerank
from requirements_agent.ai.vectorstore.base import VectorHit, VectorStore


@dataclass(frozen=True, slots=True)
class ChunkRow:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    ordinal: int
    section_path: str
    text: str
    token_count: int


class ChunkLoader(Protocol):
    async def __call__(self, chunk_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, ChunkRow]: ...


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    ordinal: int
    section_path: str
    text: str
    token_count: int
    score: float
    rank: int


class Retriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        load_chunks: ChunkLoader,
        *,
        mmr_lambda: float = 0.7,
        fetch_multiplier: int = 4,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._load_chunks = load_chunks
        self._mmr_lambda = mmr_lambda
        self._fetch_multiplier = max(1, fetch_multiplier)

    async def search(
        self,
        project_id: uuid.UUID,
        query: str,
        *,
        top_k: int,
        min_score: float = 0.0,
        use_mmr: bool = True,
        mmr_lambda: float | None = None,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[RetrievedChunk]:
        return await self.multi_search(
            project_id,
            [query],
            top_k_per_query=top_k,
            max_results=top_k,
            min_score=min_score,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
            document_ids=document_ids,
        )

    async def multi_search(
        self,
        project_id: uuid.UUID,
        queries: Sequence[str],
        *,
        top_k_per_query: int,
        max_results: int,
        min_score: float = 0.0,
        use_mmr: bool = True,
        mmr_lambda: float | None = None,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[RetrievedChunk]:
        queries = [q for q in queries if q.strip()]
        if not queries or max_results <= 0:
            return []
        vectors = await self._embedder.embed(queries)
        # Over-fetch even without MMR: hydration may drop hits (documents not ready/deleted).
        fetch_k = top_k_per_query * self._fetch_multiplier

        best: dict[uuid.UUID, VectorHit] = {}
        for vector in vectors:
            for hit in await self._store.search(
                project_id, vector, limit=fetch_k, min_score=min_score, document_ids=document_ids
            ):
                current = best.get(hit.chunk_id)
                if current is None or hit.score > current.score:
                    best[hit.chunk_id] = hit
        if not best:
            return []

        rows = await self._load_chunks(list(best))
        hits = sorted(
            (h for h in best.values() if h.chunk_id in rows),
            key=lambda h: (-h.score, str(h.chunk_id)),
        )
        if use_mmr and len(hits) > 1:
            order = mmr_rerank(
                [h.score for h in hits],
                [h.embedding for h in hits],
                k=max_results,
                lambda_mult=self._mmr_lambda if mmr_lambda is None else mmr_lambda,
            )
            hits = [hits[i] for i in order]
        else:
            hits = hits[:max_results]

        results: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits, start=1):
            row = rows[hit.chunk_id]
            results.append(
                RetrievedChunk(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    document_title=row.document_title,
                    ordinal=row.ordinal,
                    section_path=row.section_path,
                    text=row.text,
                    token_count=row.token_count,
                    score=round(hit.score, 6),
                    rank=rank,
                )
            )
        return results
