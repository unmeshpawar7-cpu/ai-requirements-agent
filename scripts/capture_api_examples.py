"""Run the real application in-process (SQLite + offline providers) against the sample
documents and record actual requests/responses into docs/api-examples.md. Full generated
artefacts are written to samples/generated/.

Run: python scripts/capture_api_examples.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx
from asgi_lifespan import LifespanManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from requirements_agent.core.config import Settings
from requirements_agent.main import create_app
from requirements_agent.services.container import build_container

SAMPLES = ROOT / "samples"
GENERATED = SAMPLES / "generated"
REQUESTS = SAMPLES / "requests"
BASE = "http://localhost:8000"


def trim(value: Any, *, max_items: int = 2, max_chars: int = 280) -> Any:
    """Shorten long arrays/strings so the examples stay readable (marked explicitly)."""
    if isinstance(value, dict):
        return {k: trim(v, max_items=max_items, max_chars=max_chars) for k, v in value.items()}
    if isinstance(value, list):
        head = [trim(v, max_items=max_items, max_chars=max_chars) for v in value[:max_items]]
        if len(value) > max_items:
            head.append(f"... {len(value) - max_items} more item(s) omitted")
        return head
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + f"... [{len(value) - max_chars} chars omitted]"
    return value


class Recorder:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self.sections: list[str] = []
        self.token: str | None = None

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def call(
        self,
        title: str,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        file: Path | None = None,
        note: str = "",
        auth: bool = True,
        max_items: int = 2,
        text_lines: int | None = None,
        transform: Any = None,
    ) -> httpx.Response:
        headers = self.headers() if auth else {}
        if file is not None:
            response = await self.client.request(
                method, path, files={"file": (file.name, file.read_bytes())}, headers=headers
            )
        else:
            response = await self.client.request(method, path, json=body, headers=headers)

        curl = [f"curl -s -X {method} '{BASE}{path}'"]
        if auth:
            curl.append('-H "Authorization: Bearer $TOKEN"')
        if file is not None:
            curl.append(f"-F 'file=@samples/{file.name}'")
        elif body is not None:
            curl.append("-H 'Content-Type: application/json'")
            curl.append(f"-d '{json.dumps(body)}'")
        lines = [f"### {title}", ""]
        if note:
            lines += [note, ""]
        lines += ["```bash", " \\\n  ".join(curl), "```", ""]
        status_line = f"`{response.status_code} {response.reason_phrase}`"
        location = response.headers.get("location")
        lines.append(
            f"Response: {status_line}" + (f" - `Location: {location}`" if location else "")
        )
        lines.append("")
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            payload = transform(response.json()) if transform else response.json()
            pretty = json.dumps(trim(payload, max_items=max_items), indent=2)
            lines += ["```json", pretty, "```", ""]
        else:
            text = response.text.splitlines()
            shown = text[:text_lines] if text_lines else text
            language = "gherkin" if "feature" in path else "markdown"
            lines += [f"```{language}", *shown]
            if text_lines and len(text) > text_lines:
                lines.append(f"... ({len(text) - text_lines} more lines)")
            lines += ["```", ""]
        self.sections.append("\n".join(lines))
        return response


async def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    REQUESTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            database_url=f"sqlite+aiosqlite:///{tmp}/examples.db",
            database_auto_create=True,
            log_level="WARNING",
            bcrypt_rounds=4,
            rate_limit_ai_capacity=1,
            rate_limit_ai_refill_per_minute=1,
        )
        container = build_container(settings)
        app = create_app(settings, container)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=BASE) as client:
                rec = Recorder(client)
                await record_flow(rec, container)
    write_markdown(rec.sections)


async def record_flow(rec: Recorder, container: Any) -> None:
    register = {"email": "ana@example.com", "password": "correct-horse-battery", "fullName": "Ana"}
    await rec.call(
        "Register (first account becomes admin)",
        "POST",
        "/api/auth/register",
        body=register,
        auth=False,
    )
    token = await rec.call(
        "Get an access token",
        "POST",
        "/api/auth/token",
        body={"email": register["email"], "password": register["password"]},
        auth=False,
    )
    rec.token = token.json()["data"]["accessToken"]

    project_body = {"name": "FieldOps", "description": "Field-service work order management"}
    project = await rec.call("Create a project", "POST", "/api/projects", body=project_body)
    project_id = project.json()["data"]["id"]

    upload = await rec.call(
        "Upload a requirement document (processed in the background)",
        "POST",
        f"/api/projects/{project_id}/documents",
        file=SAMPLES / "field-service-work-orders.md",
    )
    document_id = upload.json()["data"]["id"]
    await container.jobs.wait_idle()
    await rec.call("Poll the document until it is ready", "GET", f"/api/documents/{document_id}")

    await rec.call(
        "Duplicate upload is rejected",
        "POST",
        f"/api/projects/{project_id}/documents",
        file=SAMPLES / "field-service-work-orders.md",
        note="The SHA-256 checksum is unique per project.",
    )

    search_body = {"query": "dispatch board load time", "topK": 2, "useMmr": True}
    await rec.call(
        "Semantic search (top-k + MMR)",
        "POST",
        f"/api/projects/{project_id}/search",
        body=search_body,
    )

    analysis_body = {
        "sections": [
            "functional_requirements",
            "non_functional_requirements",
            "user_stories",
            "entities",
            "api_endpoints",
            "implementation_tasks",
        ],
        "focus": "work order scheduling and dispatch",
    }
    accepted = await rec.call(
        "Start an analysis (202, runs in the background)",
        "POST",
        f"/api/projects/{project_id}/analyses",
        body=analysis_body,
    )
    analysis_id = accepted.json()["data"]["id"]
    await rec.call(
        "AI endpoints are rate limited per user",
        "POST",
        f"/api/projects/{project_id}/analyses",
        body=analysis_body,
        note="This capture runs with RATE_LIMIT_AI_CAPACITY=1 to show the 429 envelope.",
    )
    await container.jobs.wait_idle()
    analysis = await rec.call(
        "Get the analysis (grounded artefacts grouped by type)",
        "GET",
        f"/api/analyses/{analysis_id}",
        note="Arrays are trimmed to one entry here; the full response is in "
        "`samples/generated/field-service/analysis.json`.",
        max_items=1,
    )
    openapi = await rec.call(
        "Generated OpenAPI 3.1 document",
        "GET",
        f"/api/analyses/{analysis_id}/openapi.json",
        max_items=3,
        transform=openapi_excerpt,
        note="Excerpt (two paths, two schemas); the full document is in "
        "`samples/generated/field-service/openapi.json` and passes openapi-spec-validator.",
    )
    docs = await rec.call(
        "Generated Markdown API docs",
        "GET",
        f"/api/analyses/{analysis_id}/api-docs.md",
        text_lines=30,
    )

    container.ai_rate_limiter.reset()
    await rec.call(
        "Generate test cases from acceptance criteria",
        "POST",
        f"/api/analyses/{analysis_id}/test-cases",
    )
    feature = await rec.call(
        "Export test cases as Gherkin",
        "GET",
        f"/api/analyses/{analysis_id}/test-cases.feature",
        text_lines=25,
    )
    await rec.call("AI usage report (admin)", "GET", "/api/ai/usage", max_items=3)

    await rec.call(
        "Validation error envelope",
        "POST",
        f"/api/projects/{project_id}/search",
        body={"query": "", "topK": 500},
    )
    await rec.call("Missing token", "GET", "/api/projects", auth=False)

    out = GENERATED / "field-service"
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(analysis.json(), indent=2) + "\n")
    (out / "openapi.json").write_text(json.dumps(openapi.json(), indent=2) + "\n")
    (out / "api-docs.md").write_text(docs.text)
    (out / "test-cases.feature").write_text(feature.text)
    for name, body in {
        "create-project.json": project_body,
        "search.json": search_body,
        "create-analysis.json": analysis_body,
    }.items():
        (REQUESTS / name).write_text(json.dumps(body, indent=2) + "\n")


def openapi_excerpt(document: dict[str, Any]) -> dict[str, Any]:
    paths = dict(list(document["paths"].items())[:2])
    schemas = document["components"]["schemas"]
    names = [n for n in schemas if not n.endswith(("Create", "Update", "Page"))][:2]
    return {
        "openapi": document["openapi"],
        "info": document["info"],
        "paths": paths,
        "components": {"schemas": {n: schemas[n] for n in names}},
        "x-generated": document.get("x-generated"),
    }


def write_markdown(sections: list[str]) -> None:
    header = """# API examples

Real requests and responses captured by `scripts/capture_api_examples.py`, which runs the
application in-process (SQLite, `LLM_PROVIDER=offline`, `EMBEDDING_PROVIDER=hashing`) against
`samples/field-service-work-orders.md`. Ids and timestamps are from that run. Long arrays and
strings are trimmed and marked as such; complete outputs are in `samples/generated/`.

Because the offline provider is a rule-based heuristic (not an LLM), the artefacts below show
what the pipeline *structure* produces without any API key. With `LLM_PROVIDER=openai` or
`anthropic` the same endpoints return model-generated content that goes through the same
validation, grounding and confidence scoring.

```bash
export TOKEN=...   # from POST /api/auth/token
```
"""
    (ROOT / "docs" / "api-examples.md").write_text(header + "\n" + "\n".join(sections))


if __name__ == "__main__":
    asyncio.run(main())
