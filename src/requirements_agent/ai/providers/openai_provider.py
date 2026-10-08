"""OpenAI Chat Completions provider (optional extra: `pip install .[openai]`).

Uses JSON mode (`response_format={"type": "json_object"}`) with the schema embedded in the
prompt rather than strict structured outputs: strict mode requires every property to be
required and `additionalProperties: false` at every level, which the shared Pydantic schemas
(optional fields with defaults) do not satisfy. Validation is enforced on our side anyway.
"""

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


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
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
            raise ProviderUnavailableError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise ProviderUnavailableError(
                "The 'openai' package is not installed; install with `pip install .[openai]`"
            ) from exc
        # SDK-level retries are disabled; tenacity below owns the retry policy.
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        messages = [{"role": "system", "content": request.system}] + [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=1, max=20) + wait_random(0, 1),
            retry=retry_if_exception(is_transient_error),
            reraise=True,
        ):
            with attempt:
                completion = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_output_tokens,
                    response_format={"type": "json_object"},
                )
        if not completion.choices:
            raise ProviderError("OpenAI returned no choices")
        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = getattr(completion, "usage", None)
        prompt_chars = request.system + "".join(m.content for m in request.messages)
        return LLMResponse(
            text=text,
            model=getattr(completion, "model", None) or self.model,
            input_tokens=getattr(usage, "prompt_tokens", None) or estimate_tokens(prompt_chars),
            output_tokens=getattr(usage, "completion_tokens", None) or estimate_tokens(text),
            finish_reason=getattr(choice, "finish_reason", None),
            tokens_estimated=usage is None,
        )
