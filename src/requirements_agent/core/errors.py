"""Application exceptions and the global handlers that map them onto the error envelope."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from requirements_agent.core.logging import get_request_id
from requirements_agent.schemas.common import ErrorEnvelope, FieldError

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for errors that carry an HTTP status and a client-safe message."""

    status_code: int = 400
    default_message: str = "Bad request"

    def __init__(
        self,
        message: str | None = None,
        *,
        errors: Sequence[FieldError] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.errors = list(errors or [])
        self.headers = headers or {}
        super().__init__(self.message)


class BadRequestError(AppError):
    status_code = 400
    default_message = "Bad request"


class UnauthorizedError(AppError):
    status_code = 401
    default_message = "Not authenticated"

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("headers", {"WWW-Authenticate": "Bearer"})
        super().__init__(message, **kwargs)


class ForbiddenError(AppError):
    status_code = 403
    default_message = "You do not have permission to perform this action"


class NotFoundError(AppError):
    status_code = 404
    default_message = "Resource not found"


class ConflictError(AppError):
    status_code = 409
    default_message = "Conflict"


class PayloadTooLargeError(AppError):
    status_code = 413
    default_message = "Payload too large"


class UnsupportedMediaTypeError(AppError):
    status_code = 415
    default_message = "Unsupported media type"


class UnprocessableError(AppError):
    status_code = 422
    default_message = "Validation failed"


class RateLimitedError(AppError):
    status_code = 429
    default_message = "Rate limit exceeded"

    def __init__(self, retry_after_seconds: float, message: str | None = None) -> None:
        retry_after = max(1, int(retry_after_seconds + 0.999))
        super().__init__(
            message or f"Rate limit exceeded; retry in {retry_after}s",
            headers={"Retry-After": str(retry_after)},
        )


def error_response(
    status_code: int,
    message: str,
    errors: Sequence[FieldError] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorEnvelope(message=message, errors=list(errors or []), trace_id=get_request_id())
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json", by_alias=True),
        headers=headers,
    )


def _loc_to_field(loc: Sequence[Any]) -> str | None:
    parts = [str(p) for p in loc]
    if parts and parts[0] in ("body", "query", "path", "header", "cookie", "form"):
        parts = parts[1:]
    return ".".join(parts) or None


async def _handle_app_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    if exc.status_code >= 500:
        logger.error("application error", extra={"error": exc.message})
    return error_response(exc.status_code, exc.message, exc.errors, exc.headers)


async def _handle_validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [
        FieldError(field=_loc_to_field(err.get("loc", ())), message=err.get("msg", "Invalid"))
        for err in exc.errors()
    ]
    return error_response(422, "Validation failed", errors)


async def _handle_http_exception(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    try:
        default = HTTPStatus(exc.status_code).phrase
    except ValueError:
        default = "HTTP error"
    message = exc.detail if isinstance(exc.detail, str) else default
    return error_response(exc.status_code, message, headers=getattr(exc, "headers", None))


async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled exception", exc_info=exc)
    return error_response(500, "Internal server error")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(Exception, _handle_unexpected)
