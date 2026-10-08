"""API tests: auth flow, RBAC, project lifecycle, pagination and the error envelope."""

from __future__ import annotations

import uuid

import httpx

from tests.conftest import PASSWORD, create_project, login, register


async def test_health_endpoints(client: httpx.AsyncClient) -> None:
    live = await client.get("/health/live")
    ready = await client.get("/health/ready")
    assert live.status_code == 200
    assert live.json()["data"]["status"] == "ok"
    assert ready.status_code == 200
    checks = ready.json()["data"]["checks"]
    assert checks["database"] == "ok"
    assert checks["vectorStore"] == "memory"
    assert checks["llmProvider"] == "offline:heuristic-v1"


async def test_first_user_is_admin_and_later_users_are_viewers(client: httpx.AsyncClient) -> None:
    first = await register(client, "first@example.com")
    second = await register(client, "second@example.com")
    assert first["role"] == "admin"
    assert second["role"] == "viewer"

    headers = await login(client, "second@example.com")
    me = await client.get("/api/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["data"]["email"] == "second@example.com"


async def test_register_validation_and_duplicates(client: httpx.AsyncClient) -> None:
    await register(client, "dup@example.com")
    duplicate = await client.post(
        "/api/auth/register",
        json={"email": "DUP@example.com", "password": PASSWORD, "fullName": "Dup"},
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["success"] is False

    weak = await client.post(
        "/api/auth/register", json={"email": "not-an-email", "password": "short", "fullName": ""}
    )
    assert weak.status_code == 422
    body = weak.json()
    assert body["message"] == "Validation failed"
    fields = {e["field"] for e in body["errors"]}
    assert {"email", "password", "fullName"} <= fields
    assert body["traceId"]


async def test_login_failures_and_token_validation(client: httpx.AsyncClient) -> None:
    await register(client, "user@example.com")
    wrong = await client.post(
        "/api/auth/token", json={"email": "user@example.com", "password": "wrong-password"}
    )
    unknown = await client.post(
        "/api/auth/token", json={"email": "ghost@example.com", "password": PASSWORD}
    )
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["message"] == unknown.json()["message"]  # no account enumeration

    missing = await client.get("/api/auth/me")
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    tampered = await client.get("/api/auth/me", headers={"Authorization": "Bearer abc.def.ghi"})
    assert tampered.status_code == 401


async def test_role_based_access(
    client: httpx.AsyncClient, admin_headers: dict, viewer_headers: dict, analyst_headers: dict
) -> None:
    denied = await client.post("/api/projects", json={"name": "Nope"}, headers=viewer_headers)
    assert denied.status_code == 403
    assert "analyst" in denied.json()["message"]

    assert (await client.get("/api/prompts", headers=analyst_headers)).status_code == 403
    assert (await client.get("/api/ai/usage", headers=analyst_headers)).status_code == 403
    assert (await client.get("/api/users", headers=admin_headers)).status_code == 200


async def test_admin_cannot_demote_self(client: httpx.AsyncClient, admin_headers: dict) -> None:
    me = (await client.get("/api/auth/me", headers=admin_headers)).json()["data"]
    response = await client.patch(
        f"/api/users/{me['id']}", json={"role": "viewer"}, headers=admin_headers
    )
    assert response.status_code == 409


async def test_deactivated_user_is_rejected(client: httpx.AsyncClient, admin_headers: dict) -> None:
    user = await register(client, "temp@example.com")
    headers = await login(client, "temp@example.com")
    await client.patch(f"/api/users/{user['id']}", json={"isActive": False}, headers=admin_headers)
    assert (await client.get("/api/auth/me", headers=headers)).status_code == 401


async def test_project_lifecycle(
    client: httpx.AsyncClient, analyst_headers: dict, viewer_headers: dict
) -> None:
    created = await client.post(
        "/api/projects", json={"name": "FieldOps", "description": "d"}, headers=analyst_headers
    )
    assert created.status_code == 201
    project = created.json()["data"]
    assert created.headers["location"] == f"/api/projects/{project['id']}"
    assert project["status"] == "active"

    duplicate = await client.post(
        "/api/projects", json={"name": "FieldOps"}, headers=analyst_headers
    )
    assert duplicate.status_code == 409

    patched = await client.patch(
        f"/api/projects/{project['id']}", json={"description": "updated"}, headers=analyst_headers
    )
    assert patched.status_code == 200
    assert patched.json()["data"]["description"] == "updated"

    fetched = await client.get(f"/api/projects/{project['id']}", headers=viewer_headers)
    assert fetched.json()["data"]["documentCount"] == 0

    archived = await client.delete(f"/api/projects/{project['id']}", headers=analyst_headers)
    assert archived.status_code == 204
    listed = await client.get("/api/projects", headers=viewer_headers)
    assert all(p["id"] != project["id"] for p in listed.json()["data"]["items"])
    with_archived = await client.get(
        "/api/projects", params={"includeArchived": "true"}, headers=viewer_headers
    )
    assert any(
        p["id"] == project["id"] and p["status"] == "archived"
        for p in with_archived.json()["data"]["items"]
    )
    blocked = await client.patch(
        f"/api/projects/{project['id']}", json={"name": "Renamed"}, headers=analyst_headers
    )
    assert blocked.status_code == 409


async def test_pagination_and_page_size_cap(
    client: httpx.AsyncClient, analyst_headers: dict
) -> None:
    for i in range(5):
        await create_project(client, analyst_headers, name=f"Paged {i}")
    first = (await client.get("/api/projects?page=1&pageSize=2", headers=analyst_headers)).json()
    third = (await client.get("/api/projects?page=3&pageSize=2", headers=analyst_headers)).json()
    assert first["data"]["totalCount"] == 5
    assert first["data"]["totalPages"] == 3
    assert len(first["data"]["items"]) == 2
    assert len(third["data"]["items"]) == 1
    capped = (await client.get("/api/projects?pageSize=500", headers=analyst_headers)).json()
    assert capped["data"]["pageSize"] == 100
    searched = await client.get("/api/projects?search=paged 4", headers=analyst_headers)
    assert searched.json()["data"]["totalCount"] == 1
    invalid = await client.get("/api/projects?page=0", headers=analyst_headers)
    assert invalid.status_code == 422


async def test_not_found_and_request_id(client: httpx.AsyncClient, viewer_headers: dict) -> None:
    missing = await client.get(
        f"/api/projects/{uuid.uuid4()}",
        headers=viewer_headers | {"X-Request-ID": "trace-12345678"},
    )
    assert missing.status_code == 404
    assert missing.headers["x-request-id"] == "trace-12345678"
    assert missing.json() == {
        "success": False,
        "message": "Project not found",
        "errors": [],
        "traceId": "trace-12345678",
    }
    unknown_route = await client.get("/api/does-not-exist")
    assert unknown_route.status_code == 404
    assert unknown_route.json()["success"] is False
    bad_uuid = await client.get("/api/projects/not-a-uuid", headers=viewer_headers)
    assert bad_uuid.status_code == 422
