from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from requirements_agent.core.config import Settings
from requirements_agent.main import create_app
from requirements_agent.services.container import Container, build_container
from requirements_agent.services.jobs import AsyncioJobRunner

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
PASSWORD = "correct-horse-battery"


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "database_url": f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        "database_auto_create": True,
        "bcrypt_rounds": 4,
        "log_json": False,
        "log_level": "WARNING",
        "rate_limit_ai_capacity": 100,
        "jwt_secret": "test-secret-0123456789-0123456789-abcdef",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def container(settings: Settings) -> Container:
    return build_container(settings)


@pytest.fixture
async def app(settings: Settings, container: Container) -> AsyncIterator[FastAPI]:
    application = create_app(settings, container)
    async with LifespanManager(application):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


async def register(client: httpx.AsyncClient, email: str, password: str = PASSWORD) -> dict:
    response = await client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "fullName": email.split("@")[0]},
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]


async def login(client: httpx.AsyncClient, email: str, password: str = PASSWORD) -> dict:
    response = await client.post("/api/auth/token", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['data']['accessToken']}"}


@pytest.fixture
async def admin_headers(client: httpx.AsyncClient) -> dict:
    await register(client, "admin@example.com")
    return await login(client, "admin@example.com")


@pytest.fixture
async def analyst_headers(client: httpx.AsyncClient, admin_headers: dict) -> dict:
    user = await register(client, "analyst@example.com")
    response = await client.patch(
        f"/api/users/{user['id']}", json={"role": "analyst"}, headers=admin_headers
    )
    assert response.status_code == 200, response.text
    return await login(client, "analyst@example.com")


@pytest.fixture
async def viewer_headers(client: httpx.AsyncClient, admin_headers: dict) -> dict:
    await register(client, "viewer@example.com")
    return await login(client, "viewer@example.com")


async def create_project(client: httpx.AsyncClient, headers: dict, name: str | None = None) -> dict:
    response = await client.post(
        "/api/projects",
        json={"name": name or f"Project {uuid.uuid4().hex[:8]}", "description": "test"},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]


async def wait_for_jobs(app: FastAPI) -> None:
    runner = app.state.container.jobs
    assert isinstance(runner, AsyncioJobRunner)
    await runner.wait_idle(timeout=60)
    await asyncio.sleep(0)


async def upload_sample(
    client: httpx.AsyncClient, app: FastAPI, headers: dict, project_id: str, name: str
) -> dict:
    path = SAMPLES / name
    content_type = "text/markdown" if name.endswith(".md") else "text/plain"
    response = await client.post(
        f"/api/projects/{project_id}/documents",
        files={"file": (name, path.read_bytes(), content_type)},
        headers=headers,
    )
    assert response.status_code == 202, response.text
    await wait_for_jobs(app)
    document = await client.get(f"/api/documents/{response.json()['data']['id']}", headers=headers)
    return document.json()["data"]
