from __future__ import annotations

import uuid

import numpy as np
import pytest

from requirements_agent.ai.embeddings.hashing import HashingEmbeddingProvider, tokenize
from requirements_agent.ai.rag.context import ContextBuilder
from requirements_agent.ai.rag.mmr import mmr_rerank, mmr_select
from requirements_agent.ai.rag.retriever import ChunkRow, RetrievedChunk, Retriever
from requirements_agent.ai.vectorstore.base import VectorRecord
from requirements_agent.ai.vectorstore.memory import InMemoryVectorStore
from requirements_agent.ingestion.tokens import estimate_tokens


def cosine(a: list[float], b: list[float]) -> float:
    return float(np.dot(a, b))


async def test_hashing_embeddings_are_deterministic_and_normalised() -> None:
    embedder = HashingEmbeddingProvider(256)
    first, second = await embedder.embed(
        ["Approvers approve invoices", "Approvers approve invoices"]
    )
    assert first == second
    assert len(first) == 256
    assert np.linalg.norm(first) == pytest.approx(1.0)
    assert HashingEmbeddingProvider(256).embed_one("Approvers approve invoices") == first


async def test_hashing_similarity_reflects_lexical_overlap() -> None:
    embedder = HashingEmbeddingProvider(384)
    query = embedder.embed_one("approve invoice")
    related = embedder.embed_one("Approvers must approve every invoice over the threshold")
    inflected = embedder.embed_one("invoice approval workflow")
    unrelated = embedder.embed_one("Technicians capture GPS location on the mobile app")
    assert cosine(query, related) > cosine(query, unrelated)
    assert cosine(query, inflected) > cosine(query, unrelated)  # char trigrams: approve~approval


def test_hashing_edge_cases() -> None:
    embedder = HashingEmbeddingProvider(64)
    assert embedder.embed_one("") == [0.0] * 64
    assert tokenize("The system SHALL export") == ["system", "export"]
    with pytest.raises(ValueError, match="dimension"):
        HashingEmbeddingProvider(4)


async def test_memory_store_search_scoping_threshold_and_delete() -> None:
    store = InMemoryVectorStore(3)
    project, other = uuid.uuid4(), uuid.uuid4()
    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
    ids = [uuid.uuid4() for _ in range(4)]
    await store.upsert(
        [
            VectorRecord(ids[0], doc_a, project, [1, 0, 0]),
            VectorRecord(ids[1], doc_a, project, [0.9, 0.1, 0]),
            VectorRecord(ids[2], doc_b, project, [0, 1, 0]),
            VectorRecord(ids[3], doc_b, other, [1, 0, 0]),
        ]
    )
    hits = await store.search(project, [1, 0, 0], limit=10)
    assert [h.chunk_id for h in hits] == [ids[0], ids[1], ids[2]]
    assert hits[0].score == pytest.approx(1.0)
    assert [h.chunk_id for h in await store.search(project, [1, 0, 0], limit=1)] == [ids[0]]
    assert len(await store.search(project, [1, 0, 0], limit=10, min_score=0.5)) == 2
    only_b = await store.search(project, [1, 0, 0], limit=10, document_ids=[doc_b])
    assert [h.chunk_id for h in only_b] == [ids[2]]

    await store.upsert([VectorRecord(ids[2], doc_b, project, [1, 0, 0])])  # replace in place
    assert len(store) == 4
    assert (await store.search(project, [1, 0, 0], limit=1, document_ids=[doc_b]))[0].score > 0.99

    await store.delete_document(doc_a)
    assert len(store) == 2
    assert {h.chunk_id for h in await store.search(project, [1, 0, 0], limit=10)} == {ids[2]}
    await store.delete_document(uuid.uuid4())  # unknown document is a no-op
    assert await store.search(uuid.uuid4(), [1, 0, 0], limit=5) == []
    with pytest.raises(ValueError, match="3-d"):
        await store.upsert([VectorRecord(uuid.uuid4(), doc_a, project, [1, 0])])


def test_mmr_trades_relevance_for_diversity() -> None:
    query = [1.0, 0.0]
    candidates = [[1.0, 0.0], [0.99, 0.01], [0.7, 0.7]]
    assert mmr_select(query, candidates, k=3, lambda_mult=1.0) == [0, 1, 2]
    # With diversity weighted in, the near-duplicate of the first pick is demoted.
    assert mmr_select(query, candidates, k=2, lambda_mult=0.3) == [0, 2]
    assert mmr_select(query, [], k=3) == []
    assert mmr_rerank([0.1, 0.9], [[1, 0], [0, 1]], k=1) == [1]
    with pytest.raises(ValueError, match="lambda"):
        mmr_select(query, candidates, k=1, lambda_mult=1.5)
    with pytest.raises(ValueError, match="same length"):
        mmr_rerank([0.1], candidates, k=1)


async def test_retriever_merges_queries_hydrates_and_filters() -> None:
    embedder = HashingEmbeddingProvider(128)
    store = InMemoryVectorStore(128)
    project, document = uuid.uuid4(), uuid.uuid4()
    texts = {
        uuid.uuid4(): "Approvers must approve invoices within five days.",
        uuid.uuid4(): "Approvers must approve invoices within five days!",
        uuid.uuid4(): "The portal must encrypt data at rest.",
        uuid.uuid4(): "Unrelated gardening notes about tomatoes.",
    }
    for chunk_id, text in texts.items():
        await store.upsert([VectorRecord(chunk_id, document, project, embedder.embed_one(text))])
    hidden = next(iter(texts))  # simulate a chunk whose document is not ready

    async def loader(ids: list[uuid.UUID]) -> dict[uuid.UUID, ChunkRow]:
        return {
            cid: ChunkRow(cid, document, "Doc", i, "Sec", texts[cid], 10)
            for i, cid in enumerate(ids)
            if cid != hidden
        }

    retriever = Retriever(embedder, store, loader, mmr_lambda=0.5)
    results = await retriever.multi_search(
        project, ["approve invoices", "encrypt data"], top_k_per_query=3, max_results=3
    )
    assert hidden not in {r.chunk_id for r in results}
    assert [r.rank for r in results] == [1, 2, 3]
    assert "encrypt" in " ".join(r.text for r in results)

    single = await retriever.search(project, "approve invoices", top_k=1, use_mmr=False)
    assert len(single) == 1
    assert await retriever.multi_search(project, ["  "], top_k_per_query=3, max_results=3) == []


DOCUMENT_ID = uuid.uuid4()


def _chunk(rank: int, text: str, title: str = "Doc", ordinal: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=DOCUMENT_ID,
        document_title=title,
        ordinal=ordinal,
        section_path="S",
        text=text,
        token_count=estimate_tokens(text),
        score=1.0 / rank,
        rank=rank,
    )


def test_context_builder_respects_budget_and_labels_blocks() -> None:
    chunks = [
        _chunk(1, "a" * 400, ordinal=2),
        _chunk(2, "b" * 2000, ordinal=1),  # too big for the budget -> skipped
        _chunk(3, "c" * 200, ordinal=0),
    ]
    context = ContextBuilder(max_tokens=250).build(chunks)
    assert context.token_count <= 250
    assert context.skipped_for_budget == 1
    assert [b.label for b in context.blocks] == ["E1", "E2"]
    # Emitted in reading order (ordinal), not relevance order.
    assert context.blocks[0].text.startswith("c")
    assert f"chunk={context.blocks[0].chunk_id}" in context.text
    assert context.max_score == pytest.approx(1.0)
    assert set(context.labels) == {"E1", "E2"}
    with pytest.raises(ValueError, match="positive"):
        ContextBuilder(0)
