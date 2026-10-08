"""LLM provider abstraction.

Providers receive an `LLMRequest` (rendered prompt messages + the JSON schema the answer must
match) and return raw text. They never validate or trust the content: parsing, schema
validation, repair and grounding are done by the caller (`ai.structured`, `ai.grounding`).

`LLMRequest.task_input` carries the same evidence in structured form. Remote LLMs ignore it (they
read the rendered prompt); the offline heuristic provider uses it instead of parsing prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


class ProviderError(Exception):
    """Base error for provider failures (after transport-level retries)."""


class ProviderUnavailableError(ProviderError):
    """The provider is misconfigured (missing API key or SDK)."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(slots=True)
class LLMRequest:
    task: str
    system: str
    messages: list[ChatMessage]
    response_schema: dict[str, Any]
    schema_name: str
    temperature: float = 0.1
    max_output_tokens: int = 4096
    task_input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None
    tokens_estimated: bool = False


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


_TRANSIENT_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
    "OverloadedError",
    "ConnectError",
    "ReadTimeout",
    "TimeoutException",
}


def is_transient_error(exc: BaseException) -> bool:
    """Classify SDK exceptions without importing optional SDKs.

    Retries are worth it for timeouts, connection errors, 429s and 5xx; never for 4xx such as
    authentication or bad-request errors.
    """
    if type(exc).__name__ in _TRANSIENT_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)
