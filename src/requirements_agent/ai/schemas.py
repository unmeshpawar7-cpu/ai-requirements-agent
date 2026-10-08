"""Structured-output schemas shared by every LLM provider (remote and offline).

The JSON Schema generated from these models is embedded in the prompt, and every model
response is validated against them before anything is persisted. Unknown fields are rejected
(`extra="forbid"`) so a model cannot smuggle unvalidated data through; the repair loop feeds
the exact validation errors back to the model.

`evidence` lists are intentionally *not* required by the schema: an uncited item is a
grounding problem (handled by `ai.grounding`, which drops or flags it), not a syntax problem.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

Priority = Literal["must", "should", "could", "wont"]
NFRCategory = Literal[
    "performance",
    "security",
    "availability",
    "scalability",
    "usability",
    "reliability",
    "maintainability",
    "compliance",
    "other",
]
FieldType = Literal[
    "string",
    "text",
    "integer",
    "number",
    "decimal",
    "boolean",
    "date",
    "datetime",
    "uuid",
    "enum",
    "email",
]
RelationKind = Literal["one-to-one", "one-to-many", "many-to-one", "many-to-many"]
HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


def clip_unit(value: object, default: float = 0.5) -> float:
    """Coerce a model-reported score into [0, 1]; unparseable or NaN values become `default`."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return min(1.0, max(0.0, number))


class AIModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class Cited(AIModel):
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence ids from the provided context (e.g. 'E3') that support this item. "
        "Every item must cite at least one.",
    )
    confidence: float = Field(
        default=0.5,
        description="Your confidence (0-1) that this item is stated or directly implied by the "
        "cited evidence.",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _clip_confidence(cls, value: object) -> float:
        return clip_unit(value)


class FunctionalRequirement(Cited):
    id: str = Field(pattern=r"^FR-\d{1,4}$", examples=["FR-001"])
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=3, max_length=2000)
    priority: Priority = "should"


class NonFunctionalRequirement(Cited):
    id: str = Field(pattern=r"^NFR-\d{1,4}$", examples=["NFR-001"])
    category: NFRCategory
    description: str = Field(min_length=3, max_length=2000)
    metric: str | None = Field(
        default=None, max_length=300, description="Measurable target, if the source states one."
    )


class AcceptanceCriterion(AIModel):
    given: str = Field(min_length=2, max_length=500)
    when: str = Field(min_length=2, max_length=500)
    then: str = Field(min_length=2, max_length=500)


class UserStory(Cited):
    id: str = Field(pattern=r"^US-\d{1,4}$", examples=["US-001"])
    title: str = Field(min_length=3, max_length=200)
    role: str = Field(min_length=2, max_length=100)
    goal: str = Field(min_length=3, max_length=500)
    benefit: str | None = Field(
        default=None, max_length=500, description="Omit when the source gives no rationale."
    )
    acceptance_criteria: list[AcceptanceCriterion] = Field(min_length=1, max_length=10)
    related_requirements: list[str] = Field(
        default_factory=list, description="Ids of functional requirements this story realises."
    )


class EntityField(AIModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,62}$", description="snake_case field name")
    type: FieldType
    required: bool = True
    description: str | None = Field(default=None, max_length=300)
    enum_values: list[str] = Field(default_factory=list, max_length=30)


class Relationship(AIModel):
    target: str = Field(pattern=r"^[A-Z][A-Za-z0-9]{0,63}$")
    kind: RelationKind
    description: str | None = Field(default=None, max_length=300)


class Entity(Cited):
    name: str = Field(pattern=r"^[A-Z][A-Za-z0-9]{0,63}$", description="PascalCase entity name")
    description: str = Field(min_length=3, max_length=1000)
    fields: list[EntityField] = Field(min_length=1, max_length=60)
    relationships: list[Relationship] = Field(default_factory=list, max_length=20)


class QueryParam(AIModel):
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,62}$")
    type: FieldType = "string"
    required: bool = False
    description: str | None = Field(default=None, max_length=300)


class ApiEndpoint(Cited):
    method: HttpMethod
    path: str = Field(
        pattern=r"^/[a-z0-9\-/_]*(?:\{[A-Za-z_][A-Za-z0-9_]*\}[a-z0-9\-/_]*)*$",
        max_length=200,
        examples=["/work-orders/{id}"],
    )
    summary: str = Field(min_length=3, max_length=300)
    operation_id: str | None = Field(default=None, pattern=r"^[a-z][A-Za-z0-9]{0,99}$")
    request_model: str | None = Field(
        default=None, description="Name of an entity used as the request body"
    )
    response_model: str | None = Field(
        default=None, description="Name of an entity returned by the endpoint"
    )
    response_is_collection: bool = False
    success_status: Literal[200, 201, 202, 204] = 200
    query_params: list[QueryParam] = Field(default_factory=list, max_length=20)
    related_stories: list[str] = Field(default_factory=list)


class ImplementationTask(Cited):
    id: str = Field(pattern=r"^T-\d{1,4}$", examples=["T-001"])
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=3, max_length=2000)
    estimate: Literal["S", "M", "L"] = "M"
    depends_on: list[str] = Field(default_factory=list)
    related_items: list[str] = Field(default_factory=list)


class AnalysisOutput(AIModel):
    """Top-level object the `requirements_analysis` prompt must return."""

    summary: str = Field(min_length=3, max_length=4000)
    functional_requirements: list[FunctionalRequirement] = Field(default_factory=list)
    non_functional_requirements: list[NonFunctionalRequirement] = Field(default_factory=list)
    user_stories: list[UserStory] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    api_endpoints: list[ApiEndpoint] = Field(default_factory=list)
    implementation_tasks: list[ImplementationTask] = Field(default_factory=list)
    open_questions: list[str] = Field(
        default_factory=list,
        description="Ambiguities or gaps in the source that need clarification.",
    )


class GeneratedTestCase(AIModel):
    id: str = Field(pattern=r"^TC-\d{1,4}$", examples=["TC-001"])
    title: str = Field(min_length=3, max_length=300)
    story_id: str = Field(pattern=r"^US-\d{1,4}$")
    criterion_index: int = Field(ge=0, description="0-based index of the acceptance criterion")
    type: Literal["positive", "negative", "edge"]
    priority: Literal["high", "medium", "low"] = "medium"
    preconditions: list[str] = Field(default_factory=list)
    given: list[str] = Field(min_length=1)
    when: list[str] = Field(min_length=1)
    then: list[str] = Field(min_length=1)
    confidence: float = 0.5

    @field_validator("confidence", mode="before")
    @classmethod
    def _clip_confidence(cls, value: object) -> float:
        return clip_unit(value)


class TestCaseGenerationOutput(AIModel):
    __test__ = False

    test_cases: list[GeneratedTestCase] = Field(default_factory=list)
