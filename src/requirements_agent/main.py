"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from requirements_agent import __version__
from requirements_agent.api.routers import admin, analyses, auth, projects
from requirements_agent.core.config import Settings, get_settings
from requirements_agent.core.errors import register_exception_handlers
from requirements_agent.core.logging import configure_logging
from requirements_agent.core.middleware import RequestContextMiddleware
from requirements_agent.services.container import Container, build_container
from requirements_agent.services.lifecycle import on_startup

DESCRIPTION = """
Upload requirement documents, index them for retrieval (RAG) and generate grounded engineering
artefacts: functional and non-functional requirements, user stories with Given/When/Then
acceptance criteria, entities, API endpoint suggestions, implementation tasks, test cases, an
OpenAPI 3.1 spec and Markdown API docs.

* Every response uses the `{ success, data, message }` envelope; errors add `errors` and `traceId`.
* Authenticate with `POST /api/auth/token` and send `Authorization: Bearer <token>`.
* Roles: `viewer` (read), `analyst` (upload/analyse), `admin` (prompts, usage, users).
"""

TAGS = [
    {"name": "Auth & users", "description": "Registration, tokens and user administration"},
    {"name": "Projects", "description": "Containers for requirement documents and analyses"},
    {"name": "Documents", "description": "Upload and background processing of documents"},
    {"name": "Search", "description": "Semantic search over indexed chunks"},
    {"name": "Analyses", "description": "RAG-grounded requirement analysis"},
    {"name": "Artefacts", "description": "Deterministic OpenAPI / Markdown generation"},
    {"name": "Test cases", "description": "Test cases from acceptance criteria + Gherkin export"},
    {"name": "Prompts", "description": "Versioned prompt templates (admin)"},
    {"name": "AI usage", "description": "Token, cost and latency accounting (admin)"},
    {"name": "Health", "description": "Liveness and readiness probes"},
]


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    container = container or build_container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await on_startup(container)
        try:
            yield
        finally:
            await container.jobs.shutdown(timeout=10)
            await container.database.dispose()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    app.state.container = container
    app.add_middleware(RequestContextMiddleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "Location", "Retry-After"],
        )
    register_exception_handlers(app)
    for module in (auth, projects, analyses, admin):
        app.include_router(module.router)
    return app


def run() -> None:  # pragma: no cover - console entry point
    import uvicorn

    uvicorn.run(
        "requirements_agent.main:create_app",
        factory=True,
        host="0.0.0.0",  # noqa: S104 - container entry point
        port=8000,
        proxy_headers=True,
    )
