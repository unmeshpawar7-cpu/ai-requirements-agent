"""API tests for prompt versioning, usage reporting, logging, startup recovery and seeding."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import httpx
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from requirements_agent.core.logging import JsonFormatter, request_id_var
from requirements_agent.db.models import Analysis, Document
from requirements_agent.domain.enums import AnalysisStatus, DocumentSource, DocumentStatus
from requirements_agent.main import create_app
from requirements_agent.services.container import build_container
from tests.conftest import (
    SAMPLES,
    create_project,
    login,
    make_settings,
    register,
    upload_sample,
    wait_for_jobs,
)

TEMPLATE = (
    "<<<SYSTEM>>>\nReturn JSON for $schema only.\n<<<USER>>>\nSections: $sections\n"
    "Project: $project_name\n$evidence\n"
)


async def test_prompt_versioning_api_is_used_by_new_analyses(
    client: httpx.AsyncClient, app: FastAPI, admin_headers: dict
) -> None:
    listed = await client.get("/api/prompts?name=requirements_analysis", headers=admin_headers)
    assert listed.status_code == 200
    assert [p["version"] for p in listed.json()["data"]["items"]] == [1]

    invalid = await client.post(
        "/api/prompts/requirements_analysis/versions",
        json={"template": "<<<SYSTEM>>>\nno placeholders here\n<<<USER>>>\nnothing"},
        headers=admin_headers,
    )
    assert invalid.status_code == 422
    assert invalid.json()["errors"][0]["field"] == "template"

    created = await client.post(
        "/api/prompts/requirements_analysis/versions",
        json={"template": TEMPLATE, "description": "Shorter system prompt", "activate": True},
        headers=admin_headers,
    )
    assert created.status_code == 201
    assert created.json()["data"]["version"] == 2
    assert created.json()["data"]["createdBy"] == "admin@example.com"

    project = await create_project(client, admin_headers)
    await upload_sample(client, app, admin_headers, project["id"], "invoice-approval-portal.txt")
    analysis = await client.post(
        f"/api/projects/{project['id']}/analyses", json={}, headers=admin_headers
    )
    assert analysis.json()["data"]["promptVersion"] == 2
    await wait_for_jobs(app)

    activated = await client.post(
        "/api/prompts/requirements_analysis/versions/1/activate", headers=admin_headers
    )
    assert activated.status_code == 200
    assert activated.json()["data"]["isActive"] is True
    missing = await client.post(
        "/api/prompts/requirements_analysis/versions/42/activate", headers=admin_headers
    )
    assert missing.status_code == 404
    unknown = await client.post(
        "/api/prompts/unknown_prompt/versions", json={"template": TEMPLATE}, headers=admin_headers
    )
    assert unknown.status_code == 404

    usage = await client.get(
        "/api/ai/usage", params={"projectId": project["id"]}, headers=admin_headers
    )
    groups = usage.json()["data"]["groups"]
    assert any(g["promptVersion"] == 2 for g in groups)
    assert usage.json()["data"]["totalCalls"] == sum(g["calls"] for g in groups)


async def test_json_log_formatter_includes_request_id_and_extras() -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    record.durationMs = 12.5
    token = request_id_var.set("req-123")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["message"] == "hello world"
    assert payload["requestId"] == "req-123"
    assert payload["durationMs"] == 12.5


async def test_startup_recovers_jobs_seeds_admin_and_rebuilds_index(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path, admin_email="root@example.com", admin_password="seeded-admin-password"
    )
    container = build_container(settings)
    app = create_app(settings, container)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            headers = await login(client, "root@example.com", "seeded-admin-password")
            me = await client.get("/api/auth/me", headers=headers)
            assert me.json()["data"]["role"] == "admin"
            # The env-seeded admin exists, so a self-registration is not promoted.
            assert (await register(client, "late@example.com"))["role"] == "viewer"
            project = await create_project(client, headers)
            await upload_sample(client, app, headers, project["id"], "field-service-work-orders.md")

        # Simulate a crash: a document stuck in `pending` and an analysis stuck in `running`.
        async with container.database.session_factory() as session:
            stuck_doc = Document(
                project_id=uuid.UUID(project["id"]),
                title="stuck",
                content_type="text/plain",
                source=DocumentSource.TEXT,
                size_bytes=10,
                checksum="f" * 64,
                status=DocumentStatus.PENDING,
                raw_content=(SAMPLES / "invoice-approval-portal.txt").read_bytes(),
            )
            stuck_analysis = Analysis(
                project_id=uuid.UUID(project["id"]),
                status=AnalysisStatus.RUNNING,
                sections=["entities"],
                prompt_name="requirements_analysis",
                prompt_version=1,
                provider="offline",
                model="heuristic-v1",
                embedding_model="hashing",
                retrieval_params={},
            )
            session.add_all([stuck_doc, stuck_analysis])
            await session.commit()
            stuck_doc_id, stuck_analysis_id = stuck_doc.id, stuck_analysis.id

    # "Restart": a new container (fresh in-memory vector index) on the same database.
    restarted = build_container(settings)
    app2 = create_app(settings, restarted)
    async with LifespanManager(app2):
        await wait_for_jobs(app2)
        assert len(restarted.vector_store) > 0  # type: ignore[arg-type]
        transport = httpx.ASGITransport(app=app2)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            headers = await login(client, "root@example.com", "seeded-admin-password")
            document = await client.get(f"/api/documents/{stuck_doc_id}", headers=headers)
            assert document.json()["data"]["status"] == "ready"
            analysis = await client.get(f"/api/analyses/{stuck_analysis_id}", headers=headers)
            assert analysis.json()["data"]["status"] == "failed"
            assert "restart" in analysis.json()["data"]["errorMessage"]
            search = await client.post(
                f"/api/projects/{project['id']}/search",
                json={"query": "dispatch board load time"},
                headers=headers,
            )
            assert search.json()["data"]["results"]
