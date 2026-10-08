"""Embedding provider abstraction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns texts into fixed-dimension, L2-normalised vectors."""

    name: str
    model: str
    dimension: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts; output order matches input order."""
        ...
