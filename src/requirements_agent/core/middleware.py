"""Pure-ASGI middleware: correlation id, access logging and last-resort error handling.

Implemented without `BaseHTTPMiddleware` so that context variables (the request id) and
streaming responses behave correctly.
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from requirements_agent.core.errors import error_response
from requirements_agent.core.logging import request_id_var

logger = logging.getLogger("requirements_agent.access")

REQUEST_ID_HEADER = "x-request-id"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,128}$")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or []).get(REQUEST_ID_HEADER.encode())
        candidate = incoming.decode("latin-1") if incoming else ""
        # Only trust well-formed ids from clients; never echo arbitrary header content.
        request_id = candidate if _VALID_REQUEST_ID.match(candidate) else uuid.uuid4().hex
        request_id_var.set(request_id)

        started = time.perf_counter()
        status_holder = {"status": 500, "started": False}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                status_holder["started"] = True
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception("unhandled exception while processing request")
            if not status_holder["started"]:
                response = error_response(500, "Internal server error")
                await response(scope, receive, send_wrapper)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "request completed",
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status_holder["status"],
                    "durationMs": duration_ms,
                },
            )
