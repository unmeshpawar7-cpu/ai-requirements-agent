"""Initial schema: users, projects, documents, chunks (pgvector), analyses, items, test cases,
prompt templates and AI usage.

Hand-written for PostgreSQL 16 + pgvector >= 0.5 (HNSW). The embedding dimension is read from
EMBEDDING_DIMENSION at migration time; changing it later needs a new migration plus a full
re-embedding of all chunks.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-10-08
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from requirements_agent.core.config import get_settings

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)


def _status(name: str, values: Sequence[str]) -> sa.CheckConstraint:
    quoted = ", ".join(f"'{v}'" for v in values)
    return sa.CheckConstraint(f"{name} IN ({quoted})", name=f"ck_{name}_valid")


def upgrade() -> None:
    dimension = get_settings().embedding_dimension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("full_name", sa.String(200), nullable=False),
        sa.Column("password_hash", sa.String(100), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", TS, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("email", name="uq_users_email"),
        _status("role", ["admin", "analyst", "viewer"]),
    )

    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column(
            "owner_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="RESTRICT", name="fk_projects_owner_id_users"),
            nullable=False,
        ),
        sa.Column("archived_at", TS),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", TS, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("name", name="uq_projects_name"),
        _status("status", ["active", "archived"]),
    )
    op.create_index("ix_projects_owner_id", "projects", ["owner_id"])
    op.create_index("ix_projects_status_created_at", "projects", ["status", "created_at"])

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "project_id",
            sa.Uuid(),
            sa.ForeignKey(
                "projects.id", ondelete="CASCADE", name="fk_documents_project_id_projects"
            ),
            nullable=False,
        ),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("filename", sa.String(300)),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text()),
        sa.Column("raw_content", sa.LargeBinary()),
        sa.Column("text_content", sa.Text()),
        sa.Column("char_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("token_estimate", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "uploaded_by_id",
            sa.Uuid(),
            sa.ForeignKey(
                "users.id", ondelete="SET NULL", name="fk_documents_uploaded_by_id_users"
            ),
        ),
        sa.Column("processed_at", TS),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", TS, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("project_id", "checksum", name="uq_documents_project_checksum"),
        _status("status", ["pending", "processing", "ready", "failed"]),
        sa.CheckConstraint("source IN ('upload', 'text')", name="ck_document_source_valid"),
    )
    op.create_index("ix_documents_project_status", "documents", ["project_id", "status"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey(
                "documents.id", ondelete="CASCADE", name="fk_document_chunks_document_id_documents"
            ),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            sa.Uuid(),
            sa.ForeignKey(
                "projects.id", ondelete="CASCADE", name="fk_document_chunks_project_id_projects"
            ),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("section_path", sa.String(500), nullable=False, server_default=""),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(dimension)),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_document_chunks_document_ordinal"),
    )
    op.create_index("ix_document_chunks_project_id", "document_chunks", ["project_id"])
    # Approximate nearest-neighbour index for cosine distance (`<=>`).
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.create_table(
        "analyses",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "project_id",
            sa.Uuid(),
            sa.ForeignKey(
                "projects.id", ondelete="CASCADE", name="fk_analyses_project_id_projects"
            ),
            nullable=False,
        ),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("sections", postgresql.JSONB(), nullable=False),
        sa.Column("focus", sa.Text()),
        sa.Column("document_ids", postgresql.JSONB()),
        sa.Column("prompt_name", sa.String(100), nullable=False),
        sa.Column("prompt_version", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("embedding_model", sa.String(100), nullable=False),
        sa.Column("retrieval_params", postgresql.JSONB(), nullable=False),
        sa.Column("evidence", postgresql.JSONB()),
        sa.Column("summary", sa.Text()),
        sa.Column("open_questions", postgresql.JSONB()),
        sa.Column("grounding_report", postgresql.JSONB()),
        sa.Column("repair_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("confidence", sa.Float()),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text()),
        sa.Column(
            "requested_by_id",
            sa.Uuid(),
            sa.ForeignKey(
                "users.id", ondelete="SET NULL", name="fk_analyses_requested_by_id_users"
            ),
        ),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", TS),
        sa.Column("completed_at", TS),
        _status("status", ["queued", "running", "completed", "failed"]),
    )
    op.create_index("ix_analyses_project_created_at", "analyses", ["project_id", "created_at"])

    op.create_table(
        "requirement_items",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "analysis_id",
            sa.Uuid(),
            sa.ForeignKey(
                "analyses.id", ondelete="CASCADE", name="fk_requirement_items_analysis_id_analyses"
            ),
            nullable=False,
        ),
        sa.Column("item_type", sa.String(32), nullable=False),
        sa.Column("ref_code", sa.String(120), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("evidence_chunk_ids", postgresql.JSONB(), nullable=False),
        sa.Column("grounded", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("grounding_issues", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("confidence_breakdown", postgresql.JSONB()),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.UniqueConstraint(
            "analysis_id", "item_type", "ref_code", name="uq_requirement_items_analysis_ref"
        ),
        sa.CheckConstraint(
            "item_type IN ('functional_requirement', 'non_functional_requirement', 'user_story', "
            "'entity', 'api_endpoint', 'implementation_task')",
            name="ck_item_type_valid",
        ),
    )
    op.create_index(
        "ix_requirement_items_analysis_type", "requirement_items", ["analysis_id", "item_type"]
    )

    op.create_table(
        "test_cases",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "analysis_id",
            sa.Uuid(),
            sa.ForeignKey(
                "analyses.id", ondelete="CASCADE", name="fk_test_cases_analysis_id_analyses"
            ),
            nullable=False,
        ),
        sa.Column("ref_code", sa.String(32), nullable=False),
        sa.Column("story_ref", sa.String(32), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("case_type", sa.String(20), nullable=False),
        sa.Column("priority", sa.String(10), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("prompt_name", sa.String(100), nullable=False),
        sa.Column("prompt_version", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("analysis_id", "ref_code", name="uq_test_cases_analysis_ref"),
    )
    op.create_index("ix_test_cases_analysis_id", "test_cases", ["analysis_id"])

    op.create_table(
        "prompt_templates",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("template", sa.Text(), nullable=False),
        sa.Column("description", sa.String(500)),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("created_by", sa.String(320), nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("name", "version", name="uq_prompt_templates_name_version"),
    )
    op.create_index(
        "uq_prompt_templates_active_name",
        "prompt_templates",
        ["name"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )

    op.create_table(
        "ai_usage",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.Column("operation", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("prompt_name", sa.String(100)),
        sa.Column("prompt_version", sa.Integer()),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_type", sa.String(100)),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "project_id",
            sa.Uuid(),
            sa.ForeignKey(
                "projects.id", ondelete="SET NULL", name="fk_ai_usage_project_id_projects"
            ),
        ),
        sa.Column(
            "analysis_id",
            sa.Uuid(),
            sa.ForeignKey(
                "analyses.id", ondelete="SET NULL", name="fk_ai_usage_analysis_id_analyses"
            ),
        ),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="SET NULL", name="fk_ai_usage_user_id_users"),
        ),
        sa.CheckConstraint("operation IN ('chat', 'embedding')", name="ck_ai_operation_valid"),
    )
    op.create_index("ix_ai_usage_created_at", "ai_usage", ["created_at"])
    op.create_index("ix_ai_usage_provider_model", "ai_usage", ["provider", "model"])
    op.create_index("ix_ai_usage_prompt", "ai_usage", ["prompt_name", "prompt_version"])


def downgrade() -> None:
    for table in (
        "ai_usage",
        "prompt_templates",
        "test_cases",
        "requirement_items",
        "analyses",
        "document_chunks",
        "documents",
        "projects",
        "users",
    ):
        op.drop_table(table)
    # The `vector` extension is left installed: other schemas in the database may use it.
