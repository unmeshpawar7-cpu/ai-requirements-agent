"""API tests: upload -> background processing -> search -> analysis -> artefacts -> test cases."""

from __future__ import annotations

import io
from pathlib import Path

import docx
import httpx
import pytest
from fastapi import FastAPI
from openapi_spec_validator import validate

from tests.conftest import (
    SAMPLES,
    create_project,
    make_settings,
    upload_sample,
    wait_for_jobs,
)


async def test_upload_processes_document_in_background(
    client: httpx.AsyncClient, app: FastAPI, analyst_headers: dict
) -> None:
    project = await create_project(client, analyst_headers)
    accepted = await client.post(
        f"/api/projects/{project['id']}/documents",
        files={"file": ("srs.md", (SAMPLES / "field-service-work-orders.md").read_bytes())},
        headers=analyst_headers,
    )
    assert accepted.status_code == 202
    body = accepted.json()["data"]
    assert body["status"] in {"pending", "processing", "ready"}
    assert accepted.headers["location"] == f"/api/documents/{body['id']}"

    await wait_for_jobs(app)
    document = (await client.get(f"/api/documents/{body['id']}", headers=analyst_headers)).json()
    assert document["data"]["status"] == "ready"
    assert document["data"]["chunkCount"] > 3
    assert document["data"]["tokenEstimate"] > 0

    chunks = await client.get(
        f"/api/documents/{body['id']}/chunks?pageSize=100", headers=analyst_headers
    )
    items = chunks.json()["data"]["items"]
    assert len(items) == document["data"]["chunkCount"]
    assert all(c["tokenCount"] <= 350 for c in items)

    listed = await client.get(
        f"/api/projects/{project['id']}/documents?status=ready", headers=analyst_headers
    )
    assert listed.json()["data"]["totalCount"] == 1


async def test_duplicate_unsupported_and_failed_documents(
    client: httpx.AsyncClient, app: FastAPI, analyst_headers: dict
) -> None:
    project = await create_project(client, analyst_headers)
    url = f"/api/projects/{project['id']}/documents"
    content = (SAMPLES / "invoice-approval-portal.txt").read_bytes()
    first = await client.post(url, files={"file": ("a.txt", content)}, headers=analyst_headers)
    assert first.status_code == 202
    duplicate = await client.post(url, files={"file": ("b.txt", content)}, headers=analyst_headers)
    assert duplicate.status_code == 409

    unsupported = await client.post(
        url,
        files={"file": ("tool.exe", b"MZ\x90\x00", "application/octet-stream")},
        headers=analyst_headers,
    )
    assert unsupported.status_code == 415

    corrupt = await client.post(
        url, files={"file": ("broken.pdf", b"%PDF-1.4 not really a pdf")}, headers=analyst_headers
    )
    assert corrupt.status_code == 202
    await wait_for_jobs(app)
    failed = await client.get(
        f"/api/documents/{corrupt.json()['data']['id']}", headers=analyst_headers
    )
    assert failed.json()["data"]["status"] == "failed"
    assert failed.json()["data"]["errorMessage"]
    assert failed.json()["data"]["chunkCount"] == 0


async def test_text_and_docx_documents(
    client: httpx.AsyncClient, app: FastAPI, analyst_headers: dict
) -> None:
    project = await create_project(client, analyst_headers)
    text = await client.post(
        f"/api/projects/{project['id']}/documents/text",
        json={
            "title": "Notes",
            "content": "## Approvals\n\nApprovers must approve invoices within 5 working days.",
            "format": "markdown",
        },
        headers=analyst_headers,
    )
    assert text.status_code == 202

    document = docx.Document()
    document.add_heading("Rental booking", level=1)
    document.add_paragraph("The system shall allow customers to book equipment online.")
    buffer = io.BytesIO()
    document.save(buffer)
    uploaded = await client.post(
        f"/api/projects/{project['id']}/documents",
        files={"file": ("spec.docx", buffer.getvalue())},
        headers=analyst_headers,
    )
    assert uploaded.status_code == 202
    await wait_for_jobs(app)
    docs = await client.get(f"/api/projects/{project['id']}/documents", headers=analyst_headers)
    assert {d["status"] for d in docs.json()["data"]["items"]} == {"ready"}

    deleted = await client.delete(
        f"/api/documents/{uploaded.json()['data']['id']}", headers=analyst_headers
    )
    assert deleted.status_code == 204
    gone = await client.get(
        f"/api/documents/{uploaded.json()['data']['id']}", headers=analyst_headers
    )
    assert gone.status_code == 404


async def test_semantic_search(
    client: httpx.AsyncClient, app: FastAPI, analyst_headers: dict, viewer_headers: dict
) -> None:
    project = await create_project(client, analyst_headers)
    await upload_sample(client, app, analyst_headers, project["id"], "invoice-approval-portal.txt")
    response = await client.post(
        f"/api/projects/{project['id']}/search",
        json={"query": "escalate invoice not approved within working days", "topK": 3},
        headers=viewer_headers,
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["vectorBackend"] == "memory"
    assert 1 <= len(data["results"]) <= 3
    assert [r["rank"] for r in data["results"]] == list(range(1, len(data["results"]) + 1))
    assert "escalate" in " ".join(r["text"].lower() for r in data["results"])

    strict = await client.post(
        f"/api/projects/{project['id']}/search",
        json={"query": "escalate invoice", "minScore": 0.99},
        headers=viewer_headers,
    )
    assert strict.json()["data"]["results"] == []

    invalid = await client.post(
        f"/api/projects/{project['id']}/search",
        json={"query": "", "topK": 500},
        headers=viewer_headers,
    )
    assert invalid.status_code == 422
    assert {e["field"] for e in invalid.json()["errors"]} == {"query", "topK"}


async def test_analysis_end_to_end_with_offline_provider(
    client: httpx.AsyncClient, app: FastAPI, analyst_headers: dict, admin_headers: dict
) -> None:
    project = await create_project(client, analyst_headers)
    url = f"/api/projects/{project['id']}/analyses"
    no_docs = await client.post(url, json={}, headers=analyst_headers)
    assert no_docs.status_code == 409

    await upload_sample(client, app, analyst_headers, project["id"], "field-service-work-orders.md")
    accepted = await client.post(
        url, json={"focus": "work order dispatch"}, headers=analyst_headers
    )
    assert accepted.status_code == 202
    analysis_id = accepted.json()["data"]["id"]
    assert accepted.json()["data"]["status"] == "queued"
    assert accepted.json()["data"]["promptVersion"] == 1
    await wait_for_jobs(app)

    analysis = (await client.get(f"/api/analyses/{analysis_id}", headers=analyst_headers)).json()
    data = analysis["data"]
    assert data["status"] == "completed", data["errorMessage"]
    assert data["provider"] == "offline"
    assert 0 < data["confidence"] <= 1
    items = data["items"]
    assert items["functional_requirement"]
    assert items["non_functional_requirement"]
    assert items["user_story"]
    assert items["entity"]
    assert items["api_endpoint"]
    evidence_ids = {e["chunkId"] for e in data["evidence"]}
    for group in items.values():
        for item in group:
            assert set(item["evidenceChunkIds"]) <= evidence_ids
            assert 0 <= item["confidence"] <= 1
            assert item["confidenceBreakdown"]["final"] == item["confidence"]
    story = items["user_story"][0]["payload"]
    assert story["acceptanceCriteria"][0]["given"]
    assert data["groundingReport"]["kept"] == sum(len(v) for v in items.values())

    listed = await client.get(url, headers=analyst_headers)
    assert listed.json()["data"]["items"][0]["itemCounts"]["user_story"] == len(items["user_story"])

    spec = await client.get(f"/api/analyses/{analysis_id}/openapi.json", headers=analyst_headers)
    assert spec.status_code == 200
    document = spec.json()
    validate(document)
    assert document["openapi"] == "3.1.0"
    assert "/work-orders" in document["paths"]
    assert "WorkOrder" in document["components"]["schemas"]

    markdown = await client.get(f"/api/analyses/{analysis_id}/api-docs.md", headers=analyst_headers)
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert "## Endpoints" in markdown.text and "`WorkOrder`" in markdown.text

    generated = await client.post(
        f"/api/analyses/{analysis_id}/test-cases", headers=analyst_headers
    )
    assert generated.status_code == 201, generated.text
    assert generated.json()["data"]["generated"] > 0
    cases = await client.get(
        f"/api/analyses/{analysis_id}/test-cases?pageSize=5", headers=analyst_headers
    )
    assert len(cases.json()["data"]["items"]) <= 5
    feature = await client.get(
        f"/api/analyses/{analysis_id}/test-cases.feature", headers=analyst_headers
    )
    assert "Feature:" in feature.text and "Scenario:" in feature.text and "Given " in feature.text

    usage = (await client.get("/api/ai/usage", headers=admin_headers)).json()["data"]
    operations = {g["operation"] for g in usage["groups"]}
    assert {"chat", "embedding"} <= operations
    chat = next(g for g in usage["groups"] if g["promptName"] == "requirements_analysis")
    assert chat["calls"] >= 1 and chat["inputTokens"] > 0 and chat["estimatedCostUsd"] == 0.0


async def test_artefacts_require_completed_analysis(
    client: httpx.AsyncClient, app: FastAPI, analyst_headers: dict
) -> None:
    project = await create_project(client, analyst_headers)
    await upload_sample(client, app, analyst_headers, project["id"], "invoice-approval-portal.txt")
    accepted = await client.post(
        f"/api/projects/{project['id']}/analyses",
        json={"sections": ["functional_requirements"]},
        headers=analyst_headers,
    )
    analysis_id = accepted.json()["data"]["id"]
    await wait_for_jobs(app)
    analysis = (await client.get(f"/api/analyses/{analysis_id}", headers=analyst_headers)).json()
    assert set(analysis["data"]["sections"]) == {"functional_requirements"}
    assert analysis["data"]["itemCounts"].keys() == {"functional_requirement"}
    # No stories were requested, so test-case generation has nothing to work from.
    no_stories = await client.post(
        f"/api/analyses/{analysis_id}/test-cases", headers=analyst_headers
    )
    assert no_stories.status_code == 409


async def test_viewer_cannot_start_analysis(
    client: httpx.AsyncClient, analyst_headers: dict, viewer_headers: dict
) -> None:
    project = await create_project(client, analyst_headers)
    response = await client.post(
        f"/api/projects/{project['id']}/analyses", json={}, headers=viewer_headers
    )
    assert response.status_code == 403


@pytest.fixture
def settings(tmp_path: Path):  # type: ignore[no-untyped-def]
    # A tiny bucket so the rate-limit test can exhaust it; other tests in this module stay
    # well below 3 AI calls per user.
    return make_settings(tmp_path, rate_limit_ai_capacity=3, rate_limit_ai_refill_per_minute=0.5)


async def test_rate_limit_returns_429(
    client: httpx.AsyncClient, app: FastAPI, analyst_headers: dict
) -> None:
    project = await create_project(client, analyst_headers)
    url = f"/api/projects/{project['id']}/analyses"
    statuses = [(await client.post(url, json={}, headers=analyst_headers)) for _ in range(4)]
    assert [r.status_code for r in statuses[:3]] == [409, 409, 409]  # allowed, but no documents
    limited = statuses[3]
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1
    assert limited.json()["success"] is False
    assert "Rate limit" in limited.json()["message"]
