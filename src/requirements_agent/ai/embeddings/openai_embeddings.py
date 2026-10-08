"""OpenAI embeddings (optional extra: `pip install .[openai]`)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from requirements_agent.ai.providers.base import ProviderUnavailableError, is_transient_error


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = "text-embedding-3-small",
        dimension: int = 384,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        batch_size: int = 64,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.dimension = dimension
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        if client is not None:
            self._client = client
            return
        if not api_key:
            raise ProviderUnavailableError(
                "OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai"
            )
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise ProviderUnavailableError(
                "The 'openai' package is not installed; install with `pip install .[openai]`"
            ) from exc
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=0.5, max=8),
            retry=retry_if_exception(is_transient_error),
            reraise=True,
        ):
            with attempt:
                # text-embedding-3-* support shortening via `dimensions`, which keeps the
                # pgvector column size independent of the model's native width.
                response = await self._client.embeddings.create(
                    model=self.model, input=batch, dimensions=self.dimension
                )
        ordered = sorted(response.data, key=lambda item: item.index)
        result = []
        for item in ordered:
            vec = np.asarray(item.embedding, dtype=np.float64)
            norm = float(np.linalg.norm(vec))
            result.append((vec / norm).tolist() if norm else vec.tolist())
        return result
