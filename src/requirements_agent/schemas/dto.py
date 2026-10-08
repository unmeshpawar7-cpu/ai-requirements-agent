"""Request/response DTOs for the public API (camelCase on the wire)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import EmailStr, Field, field_validator

from requirements_agent.core.security import BCRYPT_MAX_BYTES
from requirements_agent.domain.enums import (
    AnalysisStatus,
    DocumentSource,
    DocumentStatus,
    ItemType,
    ProjectStatus,
    Role,
    Section,
)
from requirements_agent.schemas.common import ApiModel

# ------------------------------------------------------------------------------- auth


class RegisterRequest(ApiModel):
    email: EmailStr = Field(examples=["ana.analyst@example.com"])
    password: str = Field(min_length=10, examples=["correct-horse-battery"])
    full_name: str = Field(min_length=1, max_length=200, examples=["Ana Analyst"])

    @field_validator("password")
    @classmethod
    def _bcrypt_limit(cls, value: str) -> str:
        if len(value.encode("utf-8")) > BCRYPT_MAX_BYTES:
            raise ValueError(f"must be at most {BCRYPT_MAX_BYTES} bytes")
        return value


class TokenRequest(ApiModel):
    email: EmailStr = Field(examples=["ana.analyst@example.com"])
    password: str = Field(min_length=1, max_length=200, examples=["correct-horse-battery"])


class TokenResponse(ApiModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 - OAuth2 token type, not a secret
    expires_in: int


class UserOut(ApiModel):
    id: uuid.UUID
    email: str
    full_name: str
    role: Role
    is_active: bool
    created_at: datetime


class UserUpdate(ApiModel):
    role: Role | None = None
    is_active: bool | None = None


# --------------------------------------------------------------------------- projects


class ProjectCreate(ApiModel):
    name: str = Field(min_length=2, max_length=200, examples=["FieldOps work orders"])
    description: str | None = Field(default=None, max_length=5000)

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("must contain at least 2 non-blank characters")
        return value


class ProjectUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=2, max_length=200)
    description: str | None = Field(default=None, max_length=5000)


class ProjectOut(ApiModel):
    id: uuid.UUID
    name: str
    description: str | None
    status: ProjectStatus
    owner_id: uuid.UUID
    document_count: int = 0
    analysis_count: int = 0
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


# -------------------------------------------------------------------------- documents


class TextDocumentCreate(ApiModel):
    title: str = Field(min_length=1, max_length=300, examples=["Approval routing notes"])
    content: str = Field(
        min_length=20,
        max_length=2_000_000,
        examples=["The system shall route invoices over 10,000 EUR to the finance controller."],
    )
    format: str = Field(default="markdown", pattern="^(markdown|text)$")


class DocumentOut(ApiModel):
    id: uuid.UUID
    project_id: uuid.UUID
    title: str
    filename: str | None
    content_type: str
    source: DocumentSource
    size_bytes: int
    checksum: str
    status: DocumentStatus
    error_message: str | None
    char_count: int
    token_estimate: int
    chunk_count: int
    created_at: datetime
    processed_at: datetime | None


class ChunkOut(ApiModel):
    id: uuid.UUID
    ordinal: int
    section_path: str
    token_count: int
    char_start: int
    char_end: int
    text: str


# ----------------------------------------------------------------------------- search


class SearchRequest(ApiModel):
    query: str = Field(min_length=2, max_length=2000, examples=["approval thresholds"])
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    use_mmr: bool = True
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    document_ids: list[uuid.UUID] | None = None


class SearchHitOut(ApiModel):
    rank: int
    score: float
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    ordinal: int
    section_path: str
    text: str


class SearchResponse(ApiModel):
    query: str
    embedding_provider: str
    vector_backend: str
    results: list[SearchHitOut]


# --------------------------------------------------------------------------- analyses


ALL_SECTIONS = list(Section)


class AnalysisCreate(ApiModel):
    sections: list[Section] = Field(
        default_factory=lambda: list(ALL_SECTIONS),
        min_length=1,
        description="Artefact sections to generate. `api_endpoints` implies `entities`.",
    )
    focus: str | None = Field(
        default=None,
        max_length=500,
        description="Optional topic that steers retrieval (e.g. 'approval routing').",
    )
    document_ids: list[uuid.UUID] | None = Field(
        default=None, description="Restrict the analysis to these documents."
    )


class AnalysisSummaryOut(ApiModel):
    id: uuid.UUID
    project_id: uuid.UUID
    status: AnalysisStatus
    sections: list[str]
    focus: str | None
    prompt_name: str
    prompt_version: int
    provider: str
    model: str
    confidence: float | None
    item_counts: dict[str, int] = Field(default_factory=dict)
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None


class RequirementItemOut(ApiModel):
    id: uuid.UUID
    item_type: ItemType
    ref_code: str
    title: str
    confidence: float
    grounded: bool
    grounding_issues: list[str]
    evidence_chunk_ids: list[str]
    confidence_breakdown: dict[str, float] | None
    payload: dict[str, Any]


class EvidenceOut(ApiModel):
    label: str
    chunk_id: str
    document_id: str
    document_title: str
    section_path: str
    score: float


class AnalysisOut(AnalysisSummaryOut):
    summary: str | None
    open_questions: list[str] = Field(default_factory=list)
    embedding_model: str
    retrieval_params: dict[str, Any]
    evidence: list[EvidenceOut] = Field(default_factory=list)
    grounding_report: dict[str, Any] | None
    repair_attempts: int
    input_tokens: int
    output_tokens: int
    started_at: datetime | None
    items: dict[str, list[RequirementItemOut]] = Field(default_factory=dict)


class TestCaseOut(ApiModel):
    __test__ = False

    id: uuid.UUID
    ref_code: str
    story_ref: str
    title: str
    case_type: str
    priority: str
    confidence: float
    prompt_name: str
    prompt_version: int
    payload: dict[str, Any]
    created_at: datetime


class TestCaseGenerationOut(ApiModel):
    __test__ = False

    analysis_id: uuid.UUID
    generated: int
    dropped: list[dict[str, str]]
    test_cases: list[TestCaseOut]


# ---------------------------------------------------------------------------- prompts


class PromptVersionOut(ApiModel):
    id: uuid.UUID
    name: str
    version: int
    description: str | None
    is_active: bool
    checksum: str
    created_by: str
    created_at: datetime
    template: str


class PromptVersionCreate(ApiModel):
    template: str = Field(min_length=20, max_length=100_000)
    description: str | None = Field(default=None, max_length=500)
    activate: bool = False


# ------------------------------------------------------------------------------ usage


class UsageGroupOut(ApiModel):
    operation: str
    provider: str
    model: str
    prompt_name: str | None
    prompt_version: int | None
    calls: int
    failures: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    avg_latency_ms: float


class UsageSummaryOut(ApiModel):
    total_calls: int
    total_failures: int
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: float
    groups: list[UsageGroupOut]
