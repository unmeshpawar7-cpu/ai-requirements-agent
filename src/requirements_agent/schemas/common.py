"""Shared API DTOs: response envelope, error envelope and pagination."""

from __future__ import annotations

import math
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

T = TypeVar("T")

MAX_PAGE_SIZE = 100


class ApiModel(BaseModel):
    """Base for every API DTO: camelCase on the wire, snake_case in Python."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        serialize_by_alias=True,
    )


class Envelope(ApiModel, Generic[T]):
    """Success envelope: `{ "success": true, "data": ..., "message": null }`."""

    success: bool = True
    data: T | None = None
    message: str | None = None


class FieldError(ApiModel):
    field: str | None = Field(default=None, examples=["name"])
    message: str = Field(examples=["Field required"])


class ErrorEnvelope(ApiModel):
    """Error envelope returned for every 4xx/5xx response."""

    success: bool = False
    message: str = Field(examples=["Validation failed"])
    errors: list[FieldError] = Field(default_factory=list)
    trace_id: str | None = Field(default=None, examples=["5f0c2a9e8f6d4f0e9a1b2c3d4e5f6a7b"])


class Page(ApiModel, Generic[T]):
    items: list[T]
    page: int
    page_size: int
    total_count: int
    total_pages: int

    @classmethod
    def build(cls, items: list[T], *, page: int, page_size: int, total_count: int) -> Page[T]:
        total_pages = math.ceil(total_count / page_size) if total_count else 0
        return cls(
            items=items,
            page=page,
            page_size=page_size,
            total_count=total_count,
            total_pages=total_pages,
        )


class PageParams(ApiModel):
    page: int = 1
    page_size: int = 20

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


def ok(data: T | None = None, message: str | None = None) -> Envelope[T]:
    return Envelope[T](success=True, data=data, message=message)


ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ErrorEnvelope, "description": "Bad request"},
    401: {"model": ErrorEnvelope, "description": "Missing or invalid bearer token"},
    403: {"model": ErrorEnvelope, "description": "Authenticated but not allowed"},
    404: {"model": ErrorEnvelope, "description": "Resource not found"},
    409: {"model": ErrorEnvelope, "description": "Conflict with current state"},
    422: {"model": ErrorEnvelope, "description": "Request validation failed"},
    429: {"model": ErrorEnvelope, "description": "Rate limit exceeded"},
}


def error_responses(*codes: int) -> dict[int | str, dict[str, object]]:
    """Pick documented error responses for a route's OpenAPI entry."""
    return {code: ERROR_RESPONSES[code] for code in codes}
