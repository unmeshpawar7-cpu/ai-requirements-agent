"""AI call accounting: token estimates, estimated cost, latency and outcome per call.

What is logged: provider, model, prompt name/version, token counts, estimated cost, latency,
success and the *type* of error. What is never logged: prompt text, document content, model
output or secrets.

Correlation (project / analysis / user) is carried in a context variable so that deeply nested
components (e.g. the embedding provider used by the retriever) do not need extra parameters.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from requirements_agent.ai.embeddings.base import EmbeddingProvider
from requirements_agent.db.models import AIUsage
from requirements_agent.domain.enums import AIOperation
from requirements_agent.ingestion.tokens import estimate_tokens

logger = logging.getLogger("requirements_agent.ai.usage")


@dataclass(frozen=True, slots=True)
class AICallContext:
    project_id: uuid.UUID | None = None
    analysis_id: uuid.UUID | None = None
    user_id: uuid.UUID | None = None


_call_context: ContextVar[AICallContext] = ContextVar("ai_call_context", default=AICallContext())  # noqa: B039 - immutable default


@contextmanager
def ai_call_context(**fields: uuid.UUID | None) -> Iterator[AICallContext]:
    token = _call_context.set(replace(_call_context.get(), **fields))
    try:
        yield _call_context.get()
    finally:
        _call_context.reset(token)


def current_ai_context() -> AICallContext:
    return _call_context.get()


@dataclass(frozen=True, slots=True)
class UsageEvent:
    operation: AIOperation
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    prompt_name: str | None = None
    prompt_version: int | None = None
    error_type: str | None = None
    attempt: int = 1


class PriceTable:
    """USD per 1K tokens keyed by `"<provider>:<model>"`; unknown models cost 0 (and are logged)."""

    def __init__(self, table: dict[str, dict[str, float]]) -> None:
        self._table = table

    def estimate(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        prices = self._table.get(f"{provider}:{model}")
        if prices is None:
            return 0.0
        cost = (input_tokens / 1000) * prices.get("input_per_1k", 0.0) + (
            output_tokens / 1000
        ) * prices.get("output_per_1k", 0.0)
        return round(cost, 8)


class UsageRecorder(Protocol):
    async def record(self, event: UsageEvent) -> None: ...


class DatabaseUsageRecorder:
    """Writes one `ai_usage` row per call in its own short transaction, so failed analyses are
    still accounted for. Recording failures are logged and swallowed: accounting must never
    break the user-facing operation."""

    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], prices: PriceTable
    ) -> None:
        self._session_factory = session_factory
        self._prices = prices

    async def record(self, event: UsageEvent) -> None:
        context = current_ai_context()
        cost = self._prices.estimate(
            event.provider, event.model, event.input_tokens, event.output_tokens
        )
        logger.info(
            "ai call",
            extra={
                "operation": event.operation.value,
                "provider": event.provider,
                "model": event.model,
                "promptName": event.prompt_name,
                "promptVersion": event.prompt_version,
                "inputTokens": event.input_tokens,
                "outputTokens": event.output_tokens,
                "estimatedCostUsd": cost,
                "latencyMs": round(event.latency_ms, 2),
                "success": event.success,
                "errorType": event.error_type,
                "attempt": event.attempt,
            },
        )
        try:
            async with self._session_factory() as session:
                session.add(
                    AIUsage(
                        operation=event.operation,
                        provider=event.provider,
                        model=event.model,
                        prompt_name=event.prompt_name,
                        prompt_version=event.prompt_version,
                        input_tokens=event.input_tokens,
                        output_tokens=event.output_tokens,
                        estimated_cost_usd=cost,
                        latency_ms=event.latency_ms,
                        success=event.success,
                        error_type=event.error_type,
                        attempt=event.attempt,
                        project_id=context.project_id,
                        analysis_id=context.analysis_id,
                        user_id=context.user_id,
                    )
                )
                await session.commit()
        except Exception:
            logger.warning("failed to persist ai usage record", exc_info=True)


class InMemoryUsageRecorder:
    """Test double that keeps events in a list."""

    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    async def record(self, event: UsageEvent) -> None:
        self.events.append(event)


class TrackedEmbeddingProvider:
    """Decorator that records an `ai_usage` row for every embedding call."""

    def __init__(self, inner: EmbeddingProvider, recorder: UsageRecorder) -> None:
        self._inner = inner
        self._recorder = recorder
        self.name = inner.name
        self.model = inner.model
        self.dimension = inner.dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        started = time.perf_counter()
        input_tokens = sum(estimate_tokens(t) for t in texts)
        try:
            vectors = await self._inner.embed(texts)
        except Exception as exc:
            await self._record(started, input_tokens, success=False, error=type(exc).__name__)
            raise
        await self._record(started, input_tokens, success=True, error=None)
        return vectors

    async def _record(
        self, started: float, input_tokens: int, *, success: bool, error: str | None
    ) -> None:
        await self._recorder.record(
            UsageEvent(
                operation=AIOperation.EMBEDDING,
                provider=self.name,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=0,
                latency_ms=(time.perf_counter() - started) * 1000,
                success=success,
                error_type=error,
            )
        )
