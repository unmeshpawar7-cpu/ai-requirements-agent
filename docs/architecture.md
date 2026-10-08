# Architecture

AI Requirements Agent is a single FastAPI service backed by PostgreSQL + pgvector. It is
structured in layers with dependencies pointing inwards; everything that talks to an external
system (database, LLM, embedding API) sits behind a small protocol so it can be swapped or
faked in tests.

| Layer | Package | Responsibility |
| --- | --- | --- |
| API | `api/` | Routers, dependency injection, auth/RBAC, paging, rate limiting |
| Schemas | `schemas/` | Pydantic DTOs (camelCase on the wire), envelope, pagination |
| Services | `services/` | Use cases and transactions; background jobs; composition root |
| Domain | `domain/` | Enumerations shared by every layer (roles, statuses, sections) |
| Repositories | `repositories/` | SQLAlchemy queries per aggregate (no commits) |
| Ingestion | `ingestion/` | Extraction, normalisation, token estimation, chunking |
| AI | `ai/` | Embeddings, vector stores, retrieval, prompts, providers, structured output, grounding, confidence, usage |
| Generators | `generators/` | Deterministic OpenAPI 3.1, Markdown and Gherkin builders |
| Core | `core/` | Settings, logging, errors, middleware, security, rate limiter |
| DB | `db/` | ORM models, portable column types, session, Alembic migrations |

## Component diagram

```mermaid
flowchart LR
    client[API client / Swagger UI] -->|HTTPS + JWT| mw[Request-context middleware<br/>request id, access log, 500 envelope]
    mw --> routers[FastAPI routers]
    routers --> deps[Dependencies<br/>auth, RBAC, paging, token bucket]
    routers --> services

    subgraph services[Services]
        accounts[AccountService]
        projects[ProjectService]
        documents[DocumentService]
        analyses[AnalysisService]
        prompts[PromptService]
        lifecycle[Startup recovery and usage report]
    end

    services --> repos[Repositories]
    repos --> pg[(PostgreSQL 16<br/>+ pgvector)]

    documents -->|submit job| jobs[AsyncioJobRunner]
    analyses -->|submit job| jobs

    subgraph ingestion[Ingestion]
        extract[Extraction<br/>txt / md / pdf / docx] --> normalise[Normalisation] --> chunker[Heading-aware chunker]
    end

    subgraph ai[AI layer]
        embed[EmbeddingProvider<br/>hashing / OpenAI]
        store[VectorStore<br/>PgVectorStore / InMemory]
        retriever[Retriever<br/>multi-query + MMR]
        context[ContextBuilder<br/>evidence blocks + budget]
        structured[StructuredGenerator<br/>validate + repair]
        llm[LLMProvider<br/>offline / OpenAI / Anthropic]
        grounding[GroundingChecker]
        confidence[ConfidenceScorer]
        usage[Usage recorder]
    end

    jobs --> ingestion
    jobs --> embed
    jobs --> retriever
    retriever --> embed
    retriever --> store
    store --> pg
    retriever --> context --> structured --> llm
    structured --> usage
    embed --> usage
    usage --> pg
    structured --> grounding --> confidence
    analyses --> generators[OpenAPI / Markdown / Gherkin generators]
```

## Document ingestion sequence

```mermaid
sequenceDiagram
    autonumber
    actor Analyst
    participant API as FastAPI
    participant DS as DocumentService
    participant DB as PostgreSQL
    participant Jobs as JobRunner
    participant Ing as Extract/Normalise/Chunk
    participant Emb as EmbeddingProvider
    participant VS as VectorStore

    Analyst->>API: POST /api/projects/{id}/documents (multipart)
    API->>DS: size + type checks, SHA-256
    DS->>DB: duplicate checksum in project?
    alt duplicate
        API-->>Analyst: 409 Conflict (envelope)
    else new
        DS->>DB: INSERT document (status=pending, raw bytes)
        DS->>Jobs: submit process-document job
        API-->>Analyst: 202 Accepted + Location
    end
    Jobs->>DB: status=processing
    Jobs->>Ing: extract text (thread pool), normalise, chunk
    Jobs->>Emb: embed "section path + chunk text"
    Jobs->>DB: INSERT document_chunks
    Jobs->>VS: upsert vectors (pgvector: UPDATE embedding column)
    Jobs->>DB: status=ready, counts, processed_at
    Note over Jobs,DB: on any error: delete partial chunks,<br/>status=failed + client-safe error message
    Analyst->>API: GET /api/documents/{id}
    API-->>Analyst: status, chunkCount, tokenEstimate
```

## Analysis sequence

```mermaid
sequenceDiagram
    autonumber
    actor Analyst
    participant API as FastAPI
    participant AS as AnalysisService
    participant Jobs as JobRunner
    participant R as Retriever
    participant C as ContextBuilder
    participant S as StructuredGenerator
    participant L as LLMProvider
    participant G as GroundingChecker
    participant Q as ConfidenceScorer
    participant DB as PostgreSQL

    Analyst->>API: POST /api/projects/{id}/analyses {sections, focus}
    API->>API: token bucket (429 if empty)
    API->>AS: create()
    AS->>DB: ready documents? active prompt version?
    AS->>DB: INSERT analysis (queued, prompt name+version)
    AS->>Jobs: submit analysis job
    API-->>Analyst: 202 Accepted + Location
    Jobs->>DB: status=running
    Jobs->>R: one query per section (+ focus)
    R->>R: embed queries, vector search (over-fetch), merge by best score, hydrate, MMR
    R-->>Jobs: ranked chunks
    Jobs->>C: build evidence blocks [E1..En] within token budget
    Jobs->>S: rendered prompt + JSON schema + evidence
    loop at most 1 + STRUCTURED_MAX_REPAIRS attempts
        S->>L: complete(request)
        L-->>S: raw text
        S->>S: parse JSON, Pydantic validate
        S->>DB: ai_usage row (tokens, cost, latency, outcome)
        alt invalid
            S->>S: append answer + validation errors, ask for a fix
        end
    end
    S-->>Jobs: AnalysisOutput, or StructuredOutputError (analysis failed)
    Jobs->>G: resolve citations, drop/flag, check cross references
    Jobs->>Q: score every kept item
    Jobs->>DB: INSERT requirement_items, analysis=completed
    Analyst->>API: GET /api/analyses/{id}, /openapi.json, /api-docs.md
```

## Deployment

```mermaid
flowchart TB
    subgraph host[Docker host / Kubernetes namespace]
        subgraph apiC[api container - non-root, healthcheck /health/live]
            entry[entrypoint.sh<br/>alembic upgrade head] --> uvicorn[uvicorn<br/>requirements_agent.main:create_app]
            uvicorn --> jobsC[in-process job runner]
        end
        subgraph dbC[db container - pgvector/pgvector:pg16]
            pgdb[(reqagent database<br/>vector extension + HNSW index)]
        end
        vol[(pgdata volume)]
    end
    users[Clients] -->|:8000| uvicorn
    uvicorn -->|asyncpg| pgdb
    pgdb --- vol
    uvicorn -.->|optional, HTTPS| openai[OpenAI API]
    uvicorn -.->|optional, HTTPS| anthropic[Anthropic API]
```

`docker compose up --build` starts both containers; the API waits for the database
healthcheck, applies migrations, then serves on port 8000. With the default `offline` and
`hashing` providers there are no outbound calls at all.

## Data flow and storage

```mermaid
flowchart LR
    raw[Uploaded bytes] -->|documents.raw_content| doc[(documents)]
    doc -->|extract + normalise| text[documents.text_content]
    text -->|chunk| chunks[(document_chunks<br/>text, section path, offsets)]
    chunks -->|embed| vec[document_chunks.embedding<br/>vector 384]
    vec -->|cosine search + MMR| evidence[analyses.evidence<br/>E-handle to chunk id + score]
    evidence -->|LLM + validation + grounding| items[(requirement_items<br/>payload, evidence_chunk_ids,<br/>confidence + breakdown)]
    items -->|deterministic| openapi[OpenAPI 3.1 / Markdown]
    items -->|stories| tcs[(test_cases)] -->|deterministic| gherkin[.feature]
    calls[Every AI call] --> usage[(ai_usage<br/>no prompt content)]
```

Key properties:

* **Traceability end to end.** Every requirement item stores the chunk ids it was derived
  from; chunks store character offsets into the normalised document text; OpenAPI operations
  carry `x-evidence`. A reviewer can go from an endpoint suggestion back to the sentence that
  motivated it.
* **State lives in the database.** Jobs receive ids only; document and analysis status
  columns are the source of truth, which makes restarts recoverable (see below) and makes it
  straightforward to move jobs to an external queue later.
* **No raw content in logs.** Logs carry ids, counts, token estimates and error types.

## Failure handling

| Situation | Behaviour |
| --- | --- |
| Corrupt / encrypted / scanned PDF, empty DOCX | Document `failed` with a specific message; partial chunks removed |
| Unexpected error while processing | Document `failed` with a generic message; stack trace only in logs |
| Model returns invalid JSON / schema violations | Bounded repair loop; then analysis `failed` with the first validation errors |
| Provider timeout / 429 / 5xx | Retried with exponential backoff and jitter inside the provider (tenacity) |
| Provider 4xx (bad key, bad request) | Not retried; analysis `failed` with the error type |
| Process restart | Startup marks `queued`/`running` analyses `failed` (inputs may have changed) and re-queues `pending`/`processing` documents; the in-memory vector index is rebuilt from persisted chunks |
| Database down | `/health/ready` returns 503; requests fail with the 500 envelope |
