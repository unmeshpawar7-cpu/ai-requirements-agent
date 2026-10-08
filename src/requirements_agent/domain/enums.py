"""Domain enumerations shared by the persistence, service and API layers."""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class ProjectStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentSource(StrEnum):
    UPLOAD = "upload"
    TEXT = "text"


class AnalysisStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Section(StrEnum):
    """Artefact sections a caller can ask an analysis to generate."""

    FUNCTIONAL_REQUIREMENTS = "functional_requirements"
    NON_FUNCTIONAL_REQUIREMENTS = "non_functional_requirements"
    USER_STORIES = "user_stories"
    ENTITIES = "entities"
    API_ENDPOINTS = "api_endpoints"
    IMPLEMENTATION_TASKS = "implementation_tasks"


class ItemType(StrEnum):
    FUNCTIONAL_REQUIREMENT = "functional_requirement"
    NON_FUNCTIONAL_REQUIREMENT = "non_functional_requirement"
    USER_STORY = "user_story"
    ENTITY = "entity"
    API_ENDPOINT = "api_endpoint"
    IMPLEMENTATION_TASK = "implementation_task"


SECTION_ITEM_TYPE: dict[Section, ItemType] = {
    Section.FUNCTIONAL_REQUIREMENTS: ItemType.FUNCTIONAL_REQUIREMENT,
    Section.NON_FUNCTIONAL_REQUIREMENTS: ItemType.NON_FUNCTIONAL_REQUIREMENT,
    Section.USER_STORIES: ItemType.USER_STORY,
    Section.ENTITIES: ItemType.ENTITY,
    Section.API_ENDPOINTS: ItemType.API_ENDPOINT,
    Section.IMPLEMENTATION_TASKS: ItemType.IMPLEMENTATION_TASK,
}


class AIOperation(StrEnum):
    CHAT = "chat"
    EMBEDDING = "embedding"


ROLE_RANK: dict[Role, int] = {Role.VIEWER: 0, Role.ANALYST: 1, Role.ADMIN: 2}
