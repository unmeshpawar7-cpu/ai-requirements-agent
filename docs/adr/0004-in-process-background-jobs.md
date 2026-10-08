# ADR 0004: In-process asyncio job runner behind a JobRunner interface

- Status: accepted
- Date: 2026-10-08

## Context

Document processing (PDF parsing, chunking, embedding) and analyses (retrieval + LLM calls,
possibly tens of seconds) must not block HTTP requests. Options: FastAPI `BackgroundTasks`,
an in-process asyncio runner, or an external queue (Celery/RQ/Arq + Redis, cloud task queues).

## Decision

Use `AsyncioJobRunner` (`services/jobs.py`) behind a two-method `JobRunner` protocol
(`submit`, `shutdown`): jobs run as asyncio tasks with bounded concurrency
(`JOB_CONCURRENCY`), inherit the request's context (request id in logs), and CPU-bound
extraction/chunking runs in a thread pool. Endpoints return `202 Accepted` with a `Location`
header; clients poll the resource status. Jobs receive only ids and keep all state in the
database (`pending -> processing -> ready | failed`, `queued -> running -> completed | failed`).

`BackgroundTasks` was rejected because it is tied to the request lifecycle: no concurrency
limit, no drain on shutdown, and with in-process test transports (httpx `ASGITransport`) the
task finishes before the client sees the response, which hides the asynchronous behaviour
the API promises.

## Consequences

- No extra infrastructure; easy to test (`wait_idle()`), graceful drain on shutdown.
- Work is lost if the process dies. Startup recovery re-queues unfinished documents and marks
  interrupted analyses as failed with a clear message; clients can re-run them.
- Throughput scales with API replicas only; each replica runs its own jobs. Moving to an
  external queue means implementing `JobRunner.submit` with a queue client and running the
  same job functions in a worker process - no service code changes.
