"""PostgreSQL + pgvector tests. Skipped unless TEST_DATABASE_URL points at a disposable database,
e.g. postgresql+asyncpg://postgres:postgres@localhost:5432/reqagent_test (the CI job provides one).
Every test resets the schema, so never point this at a database you care about."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from asgi_lifespan import LifespanManager
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from requirements_agent.ai.embeddings.hashing import HashingEmbeddingProvider
from requirements_agent.ai.vectorstore.base import VectorRecord
from requirements_agent.ai.vectorstore.pgvector_store import PgVectorStore
from requirements_agent.db.models import Document, DocumentChunk, Project, User
from requirements_agent.db.session import Database
from requirements_agent.domain.enums import DocumentSource, DocumentStatus, Role
from requirements_agent.main import create_app
from requirements_agent.services.container import build_container
from tests.conftest import create_project, login, make_settings, register, upload_sample

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set"),
]


async def _reset_schema() -> None:
    engine = create_async_engine(DATABASE_URL)  # type: ignore[arg-type]
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await engine.dispose()


@pytest.fixture
async def database() -> AsyncIterator[Database]:
    await _reset_schema()
    db = Database(DATABASE_URL)  # type: ignore[arg-type]
    await db.create_all()
    yield db
    await db.dispose()


async def _seed_chunks(
    db: Database, texts: list[str]
) -> tuple[uuid.UUID, uuid.UUID, list[uuid.UUID]]:
    async with db.session_factory() as session:
        user = User(
            email=f"{uuid.uuid4().hex}@x.io", full_name="T", password_hash="x", role=Role.ADMIN
        )
        session.add(user)
        await session.flush()
        project = Project(name=f"p-{uuid.uuid4().hex}", owner_id=user.id)
        session.add(project)
        await session.flush()
        document = Document(
            project_id=project.id,
            title="doc",
            content_type="text/plain",
            source=DocumentSource.TEXT,
            size_bytes=1,
            checksum=uuid.uuid4().hex,
            status=DocumentStatus.READY,
        )
        session.add(document)
        await session.flush()
        chunk_ids = []
        for i, chunk_text in enumerate(texts):
            chunk = DocumentChunk(
                document_id=document.id,
                project_id=project.id,
                ordinal=i,
                text=chunk_text,
                token_count=10,
                char_start=0,
                char_end=len(chunk_text),
            )
            session.add(chunk)
            await session.flush()
            chunk_ids.append(chunk.id)
        await session.commit()
        return project.id, document.id, chunk_ids


async def test_pgvector_store_search_ranks_by_cosine_similarity(database: Database) -> None:
    texts = [
        "Approvers must approve invoices within five working days.",
        "Technicians capture customer signatures on the mobile app.",
        "The portal must be available 99.9% of the time.",
    ]
    embedder = HashingEmbeddingProvider(384)
    project_id, document_id, chunk_ids = await _seed_chunks(database, texts)
    other_project, _, _ = await _seed_chunks(database, ["Approvers must approve invoices."])
    store = PgVectorStore(database.session_factory, 384)
    vectors = await embedder.embed(texts)
    await store.upsert(
        [
            VectorRecord(c, document_id, project_id, v)
            for c, v in zip(chunk_ids, vectors, strict=True)
        ]
    )

    query = embedder.embed_one("approve invoices")
    hits = await store.search(project_id, query, limit=3)
    assert hits[0].chunk_id == chunk_ids[0]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert len(hits[0].embedding) == 384
    # Scoped to the project: the other project's identical text is never returned.
    assert {h.document_id for h in hits} == {document_id}
    assert await store.search(other_project, query, limit=5) == []  # no embeddings stored there

    filtered = await store.search(project_id, query, limit=3, min_score=0.99)
    assert filtered == []

    await store.delete_document(document_id)
    assert await store.search(project_id, query, limit=3) == []


async def test_hnsw_index_exists(database: Database) -> None:
    async with database.engine.connect() as conn:
        definition = await conn.scalar(
            text(
                "SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_document_chunks_embedding_hnsw'"
            )
        )
    assert definition is not None
    assert "hnsw" in definition and "vector_cosine_ops" in definition


def test_alembic_migration_upgrade_and_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_reset_schema())
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)  # type: ignore[arg-type]
    config = Config(str(ROOT / "alembic.ini"))
    command.upgrade(config, "head")

    async def tables() -> set[str]:
        engine = create_async_engine(DATABASE_URL)  # type: ignore[arg-type]
        async with engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            names = {r[0] for r in rows}
        await engine.dispose()
        return names

    assert {
        "users",
        "projects",
        "documents",
        "document_chunks",
        "analyses",
        "requirement_items",
        "test_cases",
        "prompt_templates",
        "ai_usage",
        "alembic_version",
    } <= asyncio.run(tables())
    command.downgrade(config, "base")
    assert asyncio.run(tables()) == {"alembic_version"}


async def test_full_flow_on_postgres(tmp_path: Path) -> None:
    await _reset_schema()
    settings = make_settings(tmp_path, database_url=DATABASE_URL, vector_backend="pgvector")
    container = build_container(settings)
    app = create_app(settings, container)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await register(client, "admin@example.com")
            headers = await login(client, "admin@example.com")
            project = await create_project(client, headers)
            document = await upload_sample(
                client, app, headers, project["id"], "invoice-approval-portal.txt"
            )
            assert document["status"] == "ready"
            search = await client.post(
                f"/api/projects/{project['id']}/search",
                json={"query": "multi-factor authentication approvers", "topK": 3},
                headers=headers,
            )
            assert search.json()["data"]["vectorBackend"] == "pgvector"
            assert search.json()["data"]["results"]

            analysis = await client.post(
                f"/api/projects/{project['id']}/analyses", json={}, headers=headers
            )
            await container.jobs.wait_idle()  # type: ignore[attr-defined]
            result = await client.get(
                f"/api/analyses/{analysis.json()['data']['id']}", headers=headers
            )
            assert result.json()["data"]["status"] == "completed"
            assert result.json()["data"]["items"]["functional_requirement"]
