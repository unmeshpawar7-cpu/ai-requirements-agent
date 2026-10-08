"""Structured output: call the model, parse JSON, validate with Pydantic, repair, give up.

Never trust raw model output. Each attempt is parsed and validated against the target schema;
on failure the model's previous answer and the exact validation errors are appended to the
conversation and the model is asked to return a corrected object. After `max_repairs` failed
repairs the run fails with `StructuredOutputError` (the caller marks the analysis as failed).

Transport failures (timeouts, 5xx) are retried inside the providers with exponential backoff
and are not counted as repairs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from requirements_agent.ai.prompts.templates import RenderedPrompt
from requirements_agent.ai.providers.base import ChatMessage, LLMProvider, LLMRequest
from requirements_agent.ai.usage import UsageEvent, UsageRecorder
from requirements_agent.domain.enums import AIOperation

MAX_REPORTED_ERRORS = 20
MAX_ECHOED_RESPONSE_CHARS = 24_000


class StructuredOutputError(Exception):
    def __init__(self, message: str, errors: list[str], attempts: int) -> None:
        super().__init__(message)
        self.errors = errors
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class StructuredResult[T: BaseModel]:
    value: T
    attempts: int
    input_tokens: int
    output_tokens: int
    provider: str
    model: str

    @property
    def repairs(self) -> int:
        return self.attempts - 1


def extract_json(text: str) -> Any:
    """Parse a JSON object from model text, tolerating Markdown fences and leading prose."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
        candidate = candidate.rsplit("```", 1)[0]
    candidate = candidate.strip()
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in response")
        candidate = candidate[start : end + 1]
    return json.loads(candidate)


def format_validation_errors(exc: ValidationError) -> list[str]:
    messages = []
    for error in exc.errors()[:MAX_REPORTED_ERRORS]:
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        messages.append(f"{location}: {error['msg']}")
    remaining = exc.error_count() - MAX_REPORTED_ERRORS
    if remaining > 0:
        messages.append(f"... and {remaining} more error(s)")
    return messages


def repair_instruction(errors: list[str]) -> str:
    bullet_list = "\n".join(f"- {e}" for e in errors)
    return (
        "Your previous response did not pass validation against the JSON Schema:\n"
        f"{bullet_list}\n\n"
        "Return the complete corrected JSON object only. Keep every valid item, fix the listed "
        "problems, and do not add fields that are not in the schema."
    )


class StructuredGenerator:
    def __init__(self, provider: LLMProvider, recorder: UsageRecorder, *, max_repairs: int = 2):
        self._provider = provider
        self._recorder = recorder
        self._max_repairs = max_repairs

    async def generate[T: BaseModel](
        self,
        prompt: RenderedPrompt,
        schema: type[T],
        *,
        task_input: dict[str, Any] | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 4096,
    ) -> StructuredResult[T]:
        json_schema = schema.model_json_schema(by_alias=True)
        messages = [ChatMessage(role="user", content=prompt.user)]
        total_in = total_out = 0
        last_errors: list[str] = []
        max_attempts = self._max_repairs + 1

        for attempt in range(1, max_attempts + 1):
            request = LLMRequest(
                task=prompt.name,
                system=prompt.system,
                messages=list(messages),
                response_schema=json_schema,
                schema_name=schema.__name__,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                task_input=task_input or {},
            )
            started = time.perf_counter()
            try:
                response = await self._provider.complete(request)
            except Exception as exc:
                await self._record(prompt, attempt, started, 0, 0, False, type(exc).__name__)
                raise
            total_in += response.input_tokens
            total_out += response.output_tokens

            error_type: str | None = None
            try:
                value = schema.model_validate(extract_json(response.text))
            except ValidationError as exc:
                last_errors, error_type = format_validation_errors(exc), "SchemaValidationError"
            except ValueError as exc:  # includes json.JSONDecodeError
                last_errors, error_type = (
                    [f"<root>: response is not valid JSON ({exc})"],
                    "InvalidJSON",
                )

            await self._record(
                prompt,
                attempt,
                started,
                response.input_tokens,
                response.output_tokens,
                error_type is None,
                error_type,
                model=response.model,
            )
            if error_type is None:
                return StructuredResult(
                    value=value,
                    attempts=attempt,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    provider=self._provider.name,
                    model=response.model,
                )
            messages.append(
                ChatMessage(role="assistant", content=response.text[:MAX_ECHOED_RESPONSE_CHARS])
            )
            messages.append(ChatMessage(role="user", content=repair_instruction(last_errors)))

        raise StructuredOutputError(
            f"Model output failed schema validation after {max_attempts} attempt(s)",
            errors=last_errors,
            attempts=max_attempts,
        )

    async def _record(
        self,
        prompt: RenderedPrompt,
        attempt: int,
        started: float,
        input_tokens: int,
        output_tokens: int,
        success: bool,
        error_type: str | None,
        *,
        model: str | None = None,
    ) -> None:
        await self._recorder.record(
            UsageEvent(
                operation=AIOperation.CHAT,
                provider=self._provider.name,
                model=model or self._provider.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=(time.perf_counter() - started) * 1000,
                success=success,
                prompt_name=prompt.name,
                prompt_version=prompt.version,
                error_type=error_type,
                attempt=attempt,
            )
        )
