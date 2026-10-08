"""Build providers and the vector store from settings (`LLM_PROVIDER`, `EMBEDDING_PROVIDER`,
`VECTOR_BACKEND`)."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from requirements_agent.ai.embeddings.base import EmbeddingProvider
from requirements_agent.ai.embeddings.hashing import HashingEmbeddingProvider
from requirements_agent.ai.providers.base import LLMProvider
from requirements_agent.ai.providers.offline_provider import OfflineHeuristicProvider
from requirements_agent.ai.vectorstore.base import VectorStore
from requirements_agent.ai.vectorstore.memory import InMemoryVectorStore
from requirements_agent.core.config import Settings


def _secret(value: object) -> str | None:
    getter = getattr(value, "get_secret_value", None)
    return getter() if getter else None


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        from requirements_agent.ai.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=_secret(settings.openai_api_key),
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            max_attempts=settings.llm_max_transport_retries,
        )
    if settings.llm_provider == "anthropic":
        from requirements_agent.ai.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=_secret(settings.anthropic_api_key),
            model=settings.anthropic_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_attempts=settings.llm_max_transport_retries,
        )
    return OfflineHeuristicProvider()


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        from requirements_agent.ai.embeddings.openai_embeddings import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=_secret(settings.openai_api_key),
            model=settings.openai_embedding_model,
            dimension=settings.embedding_dimension,
            base_url=settings.openai_base_url,
            max_attempts=settings.llm_max_transport_retries,
        )
    return HashingEmbeddingProvider(dimension=settings.embedding_dimension)


def build_vector_store(
    settings: Settings, session_factory: async_sessionmaker[AsyncSession]
) -> VectorStore:
    if settings.resolved_vector_backend == "pgvector":
        from requirements_agent.ai.vectorstore.pgvector_store import PgVectorStore

        return PgVectorStore(session_factory, settings.embedding_dimension)
    return InMemoryVectorStore(settings.embedding_dimension)
