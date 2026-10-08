# AI Requirements Agent

A RAG-grounded FastAPI backend that turns software requirement documents into validated
engineering artefacts: functional and non-functional requirements, user stories with
Given/When/Then criteria, data entities, REST endpoint suggestions, implementation tasks,
test cases, an OpenAPI 3.1 spec and Markdown API docs, each traceable to the source text.

[![CI](https://github.com/unmeshpawar7-cpu/ai-requirements-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/unmeshpawar7-cpu/ai-requirements-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791)

## Problem

Requirement documents arrive as Word files, PDFs and wiki pages written in prose. Turning them
into backlog items, data models, API contracts and test cases is slow manual work, and the
link between a ticket and the sentence that justified it is usually lost. Asking a chat
assistant to "write the user stories" is fast but untrustworthy: the output is unstructured,
it may invent requirements, and nobody can tell which parts are grounded in the document.

## Solution

The service ingests documents, indexes them for retrieval, and runs a controlled generation
pipeline:

* **Retrieval-augmented generation**: the model only sees numbered evidence blocks retrieved
  from the project's own documents (multi-query retrieval + MMR, within a token budget).
* **Structured output with validation**: answers must match Pydantic schemas; invalid output
  is repaired with the exact validation errors fed back (bounded), or the run fails clearly.
* **Grounding**: every requirement and story must cite evidence it was actually given;
  uncited items are dropped, unknown citations and dangling cross-references are removed and
  reported.
* **Confidence scoring**: a documented formula combining retrieval strength, lexical support,
  model confidence and repair count, with the breakdown stored per item.
* **Deterministic generators**: OpenAPI 3.1, Markdown docs and Gherkin `.feature` files are
  produced by code, not by the model, so they are reproducible and diffable.
* **Operability**: versioned prompts, per-call token/cost/latency accounting, rate limiting,
  JWT + RBAC, background processing with explicit status, health probes, Docker and CI.

It runs fully offline by default. The default `offline` provider is a deterministic
**rule-based heuristic analyser, not an LLM** (see [AI Architecture](#ai-architecture)); set
`LLM_PROVIDER=openai` or `anthropic` for model-generated analysis through the same pipeline.

## Architecture

```mermaid
flowchart LR
    client[Client / Swagger UI] --> api[FastAPI routers<br/>JWT, RBAC, rate limit, envelope]
    api --> svc[Services]
    svc --> repo[Repositories] --> pg[(PostgreSQL 16 + pgvector)]
    svc --> jobs[Async job runner]
    jobs --> ing[Extract -> normalise -> chunk]
    jobs --> emb[Embeddings<br/>hashing / OpenAI]
    emb --> vs[VectorStore<br/>pgvector HNSW / in-memory]
    jobs --> rag[Retriever + MMR -> context builder]
    rag --> llm[LLMProvider<br/>offline / OpenAI / Anthropic]
    llm --> val[Pydantic validation + repair]
    val --> gr[Grounding + confidence]
    gr --> pg
    svc --> gen[OpenAPI / Markdown / Gherkin generators]
    llm -.-> usage[(ai_usage)]
    emb -.-> usage
```

### RAG pipeline

```mermaid
flowchart TD
    A[.txt .md .pdf .docx / raw text] --> B[Text extraction] --> C[Normalisation]
    C --> D[Heading-aware chunking<br/>350-token budget, 50-token overlap]
    D --> E[Embed section path + chunk] --> F[(pgvector HNSW<br/>cosine)]
    Q[Analysis: sections + focus] --> R[One query per section]
    R --> F --> M[Merge best scores, MMR re-rank]
    M --> K[Evidence blocks E1..En<br/>6000-token budget]
    K --> P[Prompt requirements_analysis vN] --> L[LLM]
    L --> V{Schema valid?}
    V -- no, repairs left --> L
    V -- yes --> G[Grounding check] --> S[Confidence] --> DB[(requirement_items)]
```

### Analysis sequence

```mermaid
sequenceDiagram
    actor Analyst
    participant API
    participant Job as Analysis job
    participant RAG as Retriever + Context
    participant LLM as LLM provider
    participant DB as PostgreSQL
    Analyst->>API: POST /api/projects/{id}/analyses
    API->>DB: analysis(queued, prompt v1)
    API-->>Analyst: 202 Accepted + Location
    Job->>RAG: multi-query retrieval, MMR, evidence budget
    Job->>LLM: prompt + schema + evidence
    LLM-->>Job: JSON (validated, repaired if needed)
    Job->>Job: grounding, confidence
    Job->>DB: requirement_items, analysis(completed)
    Analyst->>API: GET /api/analyses/{id} | /openapi.json | /api-docs.md
```

More diagrams (deployment, ingestion sequence, data flow) are in
[docs/architecture.md](docs/architecture.md).

## Key Features

- Upload `.txt`, `.md`, `.pdf`, `.docx` (multipart) or raw text; SHA-256 duplicate detection per
  project (409); 10 MB limit (413); unsupported types rejected (415).
- Background processing with status `pending -> processing -> ready | failed` and an error
  message on failure; restart recovery for interrupted work.
- Heading-aware, token-budgeted chunking with overlap and exact character offsets.
- Semantic search endpoint (top-k, similarity threshold, optional MMR).
- Asynchronous analyses with selectable sections and an optional focus.
- Grounded artefacts with per-item evidence chunk ids and confidence breakdowns.
- Generated OpenAPI 3.1 (validated by `openapi-spec-validator` in tests), Markdown API docs and
  Gherkin export of generated test cases.
- Prompt template versioning (seeded from files, create/activate via API, version stored on
  every result).
- AI usage accounting: tokens, estimated cost, latency, outcome per call; aggregates endpoint.
- JWT auth, bcrypt, roles `admin` / `analyst` / `viewer`, per-user token-bucket rate limit (429).
- Consistent success/error envelope with trace ids, pagination on every list endpoint,
  structured JSON logs.

## Technology Stack

| Area | Choice |
| --- | --- |
| API | Python 3.12, FastAPI, Pydantic v2, pydantic-settings, Uvicorn |
| Data | SQLAlchemy 2.x async, asyncpg, PostgreSQL 16, pgvector (HNSW), Alembic; aiosqlite for tests |
| AI | Provider protocol with OpenAI / Anthropic SDKs (optional extras) and an offline heuristic provider; hashing or OpenAI embeddings; numpy |
| Documents | pypdf, python-docx |
| Security | PyJWT (HS256), bcrypt |
| Resilience | tenacity (exponential backoff + jitter for transient provider errors) |
| Quality | pytest, pytest-asyncio, httpx ASGITransport, pytest-cov, openapi-spec-validator, ruff |
| Delivery | Multi-stage Docker image (non-root), docker compose with `pgvector/pgvector:pg16`, GitHub Actions |

## Project Structure

```text
ai-requirements-agent/
├── src/requirements_agent/
│   ├── api/            routers (auth, projects/documents/search, analyses, admin) + deps
│   ├── schemas/        API DTOs, envelope, pagination
│   ├── services/       use cases, composition root, job runner, startup recovery
│   ├── domain/         enums (roles, statuses, sections, item types)
│   ├── repositories/   SQLAlchemy queries per aggregate
│   ├── ingestion/      extraction, normalisation, token estimate, chunker
│   ├── ai/
│   │   ├── embeddings/   hashing + OpenAI providers
│   │   ├── vectorstore/  pgvector + in-memory stores
│   │   ├── rag/          retriever, MMR, context builder
│   │   ├── providers/    LLM protocol, OpenAI, Anthropic, offline
│   │   ├── offline/      rule-based analyser + test-case rules
│   │   ├── prompts/      versioned templates (<name>/v1.md) + renderer
│   │   └── schemas.py, structured.py, grounding.py, confidence.py, usage.py
│   ├── generators/     OpenAPI 3.1, Markdown docs, Gherkin
│   ├── core/           config, logging, errors, middleware, security, rate limiter
│   └── db/             models, column types, session, Alembic migrations
├── tests/              unit/, integration/ (full API), postgres/ (pgvector, migrations)
├── docs/               architecture.md, ai-design.md, api-examples.md, adr/
├── samples/            requirement documents, request JSON, generated outputs
├── scripts/            capture_api_examples.py, build_sample_docx.py
├── docker/             Dockerfile, entrypoint.sh
├── docker-compose.yml, alembic.ini, Makefile, pyproject.toml
└── .github/workflows/ci.yml
```

## API Endpoints

All `/api` responses use `{ "success", "data", "message" }`; errors use
`{ "success": false, "message", "errors": [{ "field", "message" }], "traceId" }`.
Lists accept `?page=1&pageSize=20` (max 100) and return `items, page, pageSize, totalCount,
totalPages`. Interactive docs: `/docs`.

| Method | Route | Auth | Description |
| --- | --- | --- | --- |
| POST | `/api/auth/register` | public | Register; first account becomes admin |
| POST | `/api/auth/token` | public | Email + password -> JWT |
| GET | `/api/auth/me` | any role | Current user |
| GET | `/api/users` | admin | List users |
| PATCH | `/api/users/{id}` | admin | Change role / active flag |
| POST | `/api/projects` | analyst | Create project (201 + Location) |
| GET | `/api/projects` | viewer | List projects (`search`, `includeArchived`) |
| GET | `/api/projects/{id}` | viewer | Project with document/analysis counts |
| PATCH | `/api/projects/{id}` | analyst | Update name/description |
| DELETE | `/api/projects/{id}` | analyst | Soft archive (204); archived projects are read-only |
| POST | `/api/projects/{id}/documents` | analyst | Multipart upload (202) |
| POST | `/api/projects/{id}/documents/text` | analyst | Raw text/Markdown (202) |
| GET | `/api/projects/{id}/documents` | viewer | List documents (`status` filter) |
| GET | `/api/documents/{id}` | viewer | Status, chunk count, error message |
| GET | `/api/documents/{id}/chunks` | viewer | Inspect chunks |
| DELETE | `/api/documents/{id}` | analyst | Delete document, chunks and vectors |
| POST | `/api/projects/{id}/search` | viewer | Semantic search (`query, topK, minScore, useMmr`) |
| POST | `/api/projects/{id}/analyses` | analyst, rate limited | Start analysis (202) |
| GET | `/api/projects/{id}/analyses` | viewer | List analyses with item counts |
| GET | `/api/analyses/{id}` | viewer | Analysis, evidence, grounding report, items by type |
| GET | `/api/analyses/{id}/openapi.json` | viewer | Generated OpenAPI 3.1 (raw JSON) |
| GET | `/api/analyses/{id}/api-docs.md` | viewer | Generated Markdown docs |
| POST | `/api/analyses/{id}/test-cases` | analyst, rate limited | Generate test cases (201) |
| GET | `/api/analyses/{id}/test-cases` | viewer | List test cases |
| GET | `/api/analyses/{id}/test-cases.feature` | viewer | Gherkin export |
| GET | `/api/prompts` | admin | List prompt versions (`name` filter) |
| POST | `/api/prompts/{name}/versions` | admin | Create a validated version (`activate` optional) |
| POST | `/api/prompts/{name}/versions/{v}/activate` | admin | Activate a version |
| GET | `/api/ai/usage` | admin | Usage aggregates (`since`, `until`, `projectId`) |
| GET | `/health/live`, `/health/ready` | public | Liveness / readiness (DB check, 503 when down) |

## Database Design

```mermaid
erDiagram
    users ||--o{ projects : owns
    users ||--o{ documents : uploads
    users ||--o{ analyses : requests
    projects ||--o{ documents : contains
    projects ||--o{ document_chunks : "scopes (denormalised)"
    documents ||--o{ document_chunks : "split into"
    projects ||--o{ analyses : has
    analyses ||--o{ requirement_items : produces
    analyses ||--o{ test_cases : "has"
    projects ||--o{ ai_usage : "accounted to"
    analyses ||--o{ ai_usage : "accounted to"

    users {
        uuid id PK
        string email UK
        string password_hash
        string role "admin|analyst|viewer"
        bool is_active
    }
    projects {
        uuid id PK
        string name UK
        string status "active|archived"
        uuid owner_id FK
        timestamptz archived_at
    }
    documents {
        uuid id PK
        uuid project_id FK
        string checksum "UK with project_id"
        string status "pending|processing|ready|failed"
        bytea raw_content
        text text_content
        int chunk_count
        text error_message
    }
    document_chunks {
        uuid id PK
        uuid document_id FK
        uuid project_id FK
        int ordinal "UK with document_id"
        text text
        string section_path
        int char_start
        int char_end
        vector embedding "vector(384), HNSW cosine"
    }
    analyses {
        uuid id PK
        uuid project_id FK
        string status "queued|running|completed|failed"
        jsonb sections
        string prompt_name
        int prompt_version
        string provider
        jsonb evidence
        jsonb grounding_report
        float confidence
        int repair_attempts
    }
    requirement_items {
        uuid id PK
        uuid analysis_id FK
        string item_type
        string ref_code "UK with analysis_id, item_type"
        jsonb payload
        jsonb evidence_chunk_ids
        float confidence
        jsonb confidence_breakdown
        bool grounded
    }
    test_cases {
        uuid id PK
        uuid analysis_id FK
        string ref_code "UK with analysis_id"
        string story_ref
        jsonb payload
        int prompt_version
    }
    prompt_templates {
        uuid id PK
        string name "UK with version"
        int version
        text template
        bool is_active "one active per name"
        string checksum
        string created_by
    }
    ai_usage {
        uuid id PK
        string operation "chat|embedding"
        string provider
        string model
        string prompt_name
        int input_tokens
        int output_tokens
        float estimated_cost_usd
        float latency_ms
        bool success
        string error_type
    }
```

The PostgreSQL schema comes from a hand-written Alembic migration
(`src/requirements_agent/db/migrations/versions/20261008_0001_initial_schema.py`): it creates
the `vector` extension, check constraints for every status column, the unique constraints
above, a partial unique index for the active prompt version, and the HNSW index
`USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)`. Upgrade and
downgrade are exercised by the Postgres test suite. On SQLite the same ORM models are created
with `metadata.create_all`, and the `embedding` column falls back to JSON (unused there; the
in-memory vector store is rebuilt from persisted chunks at startup).

## AI Architecture

| Concern | Implementation |
| --- | --- |
| Providers | `LLMProvider` protocol: `OfflineHeuristicProvider` (default), `OpenAIProvider` (`OPENAI_MODEL`, default `gpt-4o-mini`), `AnthropicProvider` (`ANTHROPIC_MODEL`, default `claude-sonnet-4-5`); SDKs are optional extras imported lazily. Embeddings: `HashingEmbeddingProvider` (default) or `OpenAIEmbeddingProvider` (`text-embedding-3-small`, `dimensions=384`). |
| Retrieval | Multi-query (one per requested section + focus), over-fetch, best-score merge, MMR (`lambda=0.7`), similarity floor, only `ready` documents |
| Context | Numbered evidence blocks `[E3] chunk=<uuid> doc=... section=... score=...`, greedy admission in relevance order within `ANALYSIS_CONTEXT_TOKENS`, emitted in reading order |
| Prompt versioning | Files `ai/prompts/<name>/v<N>.md` seeded into `prompt_templates` (checksum, created_by); admins create/activate versions; name + version stored on analyses, test cases and usage rows |
| Structured output | JSON Schema from Pydantic models in the prompt; strict parsing (`extra="forbid"`, enums, id patterns); repair loop with exact errors, max `STRUCTURED_MAX_REPAIRS` (2); transport retries separate |
| Hallucination mitigation | Evidence-only prompting; citations required; uncited FR/NFR/stories dropped; derived artefacts flagged; unknown citations and dangling references removed and reported; open questions instead of guesses |
| Confidence | `clip01((0.35*R + 0.35*S + 0.30*M) * max(0.5, 1-0.15*repairs) * G)` with retrieval strength R, lexical support S, model confidence M, grounding factor G; breakdown stored per item |
| Cost / tokens | ~4 chars/token estimate when providers do not report usage; per-call `ai_usage` rows with estimated cost from `AI_PRICE_TABLE`; budgets, output cap, temperature 0.1, per-user rate limit |

**Honest note on the offline provider.** `LLM_PROVIDER=offline` does not understand language;
it applies documented rules (modal verbs -> FRs, quality keywords + metrics -> NFRs, "As a..."
and actor patterns -> stories, "X has a, b and c" -> entities, verbs on entities -> endpoints,
vague words -> open questions) and emits the same schema with citations. Derived acceptance
criteria are template rewrites of the source sentence, and its confidence values are
rule-strength constants. It is there so the pipeline is runnable and testable end to end
without keys; the hashing embeddings are likewise lexical, not semantic. Details and limits:
[docs/ai-design.md](docs/ai-design.md).

## Security

- Passwords hashed with bcrypt (configurable cost; inputs over bcrypt's 72-byte limit are
  rejected, not truncated); login runs a dummy hash for unknown emails and returns the same
  401 message, so accounts cannot be enumerated by response or timing.
- JWT (HS256) with `iss`, `aud`, `exp`, `nbf`, `jti`; the secret must be at least 32
  characters and must be set explicitly when `ENVIRONMENT=production` (startup fails otherwise).
- Role checks read the user from the database on every request, so deactivation and demotion
  take effect immediately. Admins cannot demote or deactivate themselves.
- Bootstrap: the first registered account becomes `admin`; alternatively set `ADMIN_EMAIL` +
  `ADMIN_PASSWORD` to seed one at startup. Later sign-ups get `DEFAULT_SIGNUP_ROLE` (`viewer`).
- Input validation on every request; upload size/type limits; client-supplied
  `X-Request-ID` is only echoed if well-formed.
- No prompt text, document content or secrets in logs or `ai_usage`; API keys only from env.
- Generic 500 responses (details only in logs); container runs as a non-root user.

## Testing

```bash
make install          # venv + editable install with dev extras
make test             # unit + API tests (SQLite, offline providers, no network)
make coverage
# pgvector tests: point at a disposable database (schema is dropped per test)
TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/reqagent_test make test
```

What is covered:

- **Unit**: extraction (txt incl. encodings, generated PDF, DOCX headings/lists/tables),
  normalisation, chunk boundaries/overlap/token budget/offsets, hashing embeddings
  (determinism, normalisation, similarity sanity), in-memory store, MMR, multi-query
  retriever, context budget, repair-with-feedback loop (scripted provider returning invalid
  then valid JSON), grounding rules, confidence formula, OpenAPI builder (spec validation,
  determinism), Markdown and Gherkin export, prompt versioning, JWT/bcrypt, RBAC, token bucket,
  OpenAI/Anthropic adapters with fake SDK clients, offline analyser on the samples, usage
  accounting, job runner.
- **API (httpx AsyncClient over the full app)**: auth flow, project lifecycle and pagination,
  upload -> background processing -> ready/failed, search, end-to-end analysis with the
  offline provider, OpenAPI/Markdown/Gherkin endpoints, prompt versioning, usage, startup
  recovery after a simulated crash, and 401/403/404/409/415/422/429 paths.
- **PostgreSQL** (`@pytest.mark.postgres`, skipped without `TEST_DATABASE_URL`): pgvector
  ranking/scoping/thresholds, HNSW index, Alembic upgrade/downgrade, full flow on pgvector.

Measured locally (Python 3.13): **95 passed, 4 skipped, 94% line+branch coverage** without
Postgres; **99 passed, 95% coverage** with `TEST_DATABASE_URL` pointing at PostgreSQL 16 +
pgvector 0.6. CI runs the full suite against a `pgvector/pgvector:pg16` service with
`--cov-fail-under=80`. `ruff check` and `ruff format --check` are clean.

## Running Locally

```bash
git clone https://github.com/unmeshpawar7-cpu/ai-requirements-agent.git
cd ai-requirements-agent
make install
cp .env.example .env            # SQLite + offline providers by default
make run                        # http://localhost:8000/docs
```

Then register, get a token, create a project and upload `samples/field-service-work-orders.md`
(see [docs/api-examples.md](docs/api-examples.md) for the exact calls). To use real models:

```bash
pip install -e ".[openai]"      # or ".[anthropic]"
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... EMBEDDING_PROVIDER=openai make run
```

To run against PostgreSQL instead of SQLite, set `DATABASE_URL=postgresql+asyncpg://...`,
`DATABASE_AUTO_CREATE=false` and run `make migrate` first.

## Docker

```bash
cp .env.example .env            # set JWT_SECRET and POSTGRES_PASSWORD
docker compose up --build       # api on :8000, pgvector/pgvector:pg16 on :5432
```

The image is multi-stage (build venv -> slim runtime), runs as a non-root user, has a
`HEALTHCHECK` on `/health/live`, and its entrypoint applies Alembic migrations before starting
Uvicorn (`RUN_MIGRATIONS=false` to skip). Add provider SDKs with
`docker compose build --build-arg EXTRAS=openai,anthropic`.

## Example API Request

```bash
curl -s -X POST "http://localhost:8000/api/projects/$PROJECT_ID/analyses" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"sections": ["functional_requirements", "non_functional_requirements", "user_stories",
                    "entities", "api_endpoints", "implementation_tasks"],
       "focus": "work order scheduling and dispatch"}'
# 202 Accepted, Location: /api/analyses/{id}; then poll GET /api/analyses/{id}
```

## Example API Response

Excerpt of a real `GET /api/analyses/{id}` response produced with the **offline** provider
from `samples/field-service-work-orders.md` (full output in
`samples/generated/field-service/analysis.json`):

```json
{
  "success": true,
  "data": {
    "status": "completed",
    "promptName": "requirements_analysis",
    "promptVersion": 1,
    "provider": "offline",
    "model": "heuristic-v1",
    "confidence": 0.695,
    "itemCounts": { "functional_requirement": 14, "non_functional_requirement": 9,
                    "user_story": 10, "entity": 3, "api_endpoint": 8, "implementation_task": 11 },
    "groundingReport": { "kept": 55, "dropped": [], "flagged": [], "unknownCitations": [],
                         "evidenceBlocks": 14, "contextTokens": 1884, "skippedForBudget": 0 },
    "items": {
      "user_story": [
        {
          "refCode": "US-002",
          "title": "Create work orders from customer service requests",
          "confidence": 0.637,
          "grounded": true,
          "evidenceChunkIds": ["ee22d4e5-a034-456b-816b-7f23d16796a4"],
          "confidenceBreakdown": { "retrieval": 0.5955, "support": 0.6667, "model": 0.65,
                                   "repairFactor": 1.0, "groundingFactor": 1.0, "final": 0.637 },
          "payload": {
            "id": "US-002",
            "role": "dispatcher",
            "goal": "create work orders from customer service requests",
            "benefit": null,
            "acceptanceCriteria": [{
              "given": "an authenticated dispatcher with access to the feature",
              "when": "the dispatcher creates work orders from customer service requests",
              "then": "the system confirms that the work orders were created"
            }],
            "relatedRequirements": ["FR-001"]
          }
        }
      ]
    }
  },
  "message": null
}
```

The generated OpenAPI document for this run (`samples/generated/field-service/openapi.json`)
contains `WorkOrder`, `Technician` and `PartUsed` schemas and paths such as
`POST /work-orders/{id}/assign`, each operation annotated with `x-evidence` chunk ids.

## Future Enhancements

- External job queue (Arq/Celery) and object storage for raw uploads; SSE/webhooks instead of polling.
- Hybrid retrieval (BM25 + vectors) and a cross-encoder re-ranker; pgvector iterative scans
  for heavily filtered HNSW queries.
- Calibrate the confidence score against reviewer accept/reject labels; add an evaluation
  set (precision/recall of extracted requirements per prompt version).
- Human-in-the-loop review states for items (accept, edit, reject) feeding prompt evaluation.
- Diffing analyses between document versions; export to Jira/Azure DevOps.
- Refresh tokens, per-project membership/ACLs, Redis-backed distributed rate limiting.
- OCR for scanned PDFs; table-aware chunking.

## Engineering Highlights

- **Trust is enforced in code, not in the prompt.** The prompt asks for citations, but
  grounding, reference integrity and schema validation are deterministic checks that apply to
  every provider. The model can propose; it cannot write unvalidated data into the database.
- **Short evidence handles over UUIDs.** Models cite `E3`; grounding maps handles to chunk ids
  and rejects anything not provided. This trades one indirection for far fewer copy errors.
- **Repairs and transport retries are separate budgets.** A flaky network does not burn the
  repair budget, and a model that cannot follow the schema fails in bounded time with the
  validation errors attached.
- **Deterministic where possible.** OpenAPI, Markdown and Gherkin are generated by code from
  validated artefacts, so outputs are reproducible and reviewable; AI is used only for the
  language-understanding step.
- **Multi-query retrieval per artefact type.** A single "analyse this" query retrieves one
  topic; one query per section plus MMR over the union covers NFRs, data definitions and
  actors that a single embedding would miss.
- **pgvector over a separate vector DB** (ADR 0001): one transactional store for chunks,
  metadata and vectors; the `VectorStore` protocol keeps the door open if scale demands more.
- **Offline-first** (ADR 0002): the default providers are deterministic and network-free, so
  the repository is demoable and the test suite runs the real pipeline without mocks of the
  pipeline itself; only the LLM boundary is faked where behaviour must be scripted.
- **In-process jobs with database-backed state** (ADR 0004): no extra infrastructure today,
  recoverable after restarts, and a one-method seam for an external queue tomorrow.
- **Measured, not claimed.** Coverage figures above are from actual runs; no performance
  numbers are claimed because none were benchmarked.

See also: [architecture](docs/architecture.md), [AI design](docs/ai-design.md),
[API examples](docs/api-examples.md), [ADRs](docs/adr/).

## License

MIT - see [LICENSE](LICENSE).
