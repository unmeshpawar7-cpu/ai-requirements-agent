"""Anthropic Messages API provider (optional extra: `pip install .[anthropic]`)."""

from __future__ import annotations

from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from requirements_agent.ai.providers.base import (
    LLMRequest,
    LLMResponse,
    ProviderError,
    ProviderUnavailableError,
    is_transient_error,
)
from requirements_agent.ingestion.tokens import estimate_tokens


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = "claude-sonnet-4-5",
        timeout_seconds: float = 60.0,
        max_attempts: int = 3,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._max_attempts = max_attempts
        if client is not None:
            self._client = client
            return
        if not api_key:
            raise ProviderUnavailableError(
                "ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic"
            )
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise ProviderUnavailableError(
                "The 'anthropic' package is not installed; install with `pip install .[anthropic]`"
            ) from exc
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_seconds, max_retries=0)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=1, max=20) + wait_random(0, 1),
            retry=retry_if_exception(is_transient_error),
            reraise=True,
        ):
            with attempt:
                message = await self._client.messages.create(
                    model=self.model,
                    system=request.system,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_output_tokens,
                )
        blocks = getattr(message, "content", None) or []
        text = "".join(
            getattr(block, "text", "") for block in blocks if getattr(block, "type", "") == "text"
        )
        if not text:
            raise ProviderError("Anthropic returned no text content")
        usage = getattr(message, "usage", None)
        prompt_chars = request.system + "".join(m.content for m in request.messages)
        return LLMResponse(
            text=text,
            model=getattr(message, "model", None) or self.model,
            input_tokens=getattr(usage, "input_tokens", None) or estimate_tokens(prompt_chars),
            output_tokens=getattr(usage, "output_tokens", None) or estimate_tokens(text),
            finish_reason=getattr(message, "stop_reason", None),
            tokens_estimated=usage is None,
        )
