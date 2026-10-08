# ADR 0001: PostgreSQL + pgvector instead of a dedicated vector database

- Status: accepted
- Date: 2026-10-08

## Context

The service needs similarity search over document chunks, scoped per project, alongside
relational data (users, projects, documents, analyses, prompt versions, usage). Expected scale
per deployment is thousands to low millions of chunks. Options considered: a dedicated vector
database (Qdrant, Weaviate, Pinecone), Elasticsearch/OpenSearch k-NN, or pgvector inside the
PostgreSQL database we need anyway.

## Decision

Store embeddings in `document_chunks.embedding vector(384)` with an HNSW index
(`vector_cosine_ops`, `m=16`, `ef_construction=64`), behind a `VectorStore` protocol with two
implementations: `PgVectorStore` (production) and `InMemoryVectorStore` (numpy, exact search;
tests and SQLite runs). Project scoping uses a denormalised `project_id` column; queries order
by `<=>` so the planner uses the index, and raise `hnsw.ef_search` per transaction for large
candidate sets.

## Consequences

- One database to operate, back up and secure; chunk rows, metadata and vectors are updated in
  the same transactions, and deleting a document cascades to its vectors.
- Joins and filters (project, document status) are plain SQL.
- HNSW filtering happens after the index scan; with very many projects a query may return
  fewer than `k` rows. Mitigations if needed: raise `ef_search`, pgvector >= 0.8 iterative
  scans, or partitioning by project.
- The embedding dimension is fixed by the migration; changing models with a different width
  needs a migration and re-embedding (OpenAI `text-embedding-3-*` are requested with
  `dimensions=384` to avoid this).
- If scale or hybrid-search requirements outgrow pgvector, the `VectorStore` protocol is the
  seam for a dedicated engine.
