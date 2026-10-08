"""Composition root: builds every long-lived component once per application instance."""

from __future__ import annotations

from dataclasses import dataclass

from requirements_agent.ai.confidence import ConfidenceScorer
from requirements_agent.ai.embeddings.base import EmbeddingProvider
from requirements_agent.ai.factory import (
    build_embedding_provider,
    build_llm_provider,
    build_vector_store,
)
from requirements_agent.ai.grounding import GroundingChecker
from requirements_agent.ai.providers.base import LLMProvider
from requirements_agent.ai.rag.context import ContextBuilder
from requirements_agent.ai.rag.retriever import Retriever
from requirements_agent.ai.structured import StructuredGenerator
from requirements_agent.ai.usage import (
    DatabaseUsageRecorder,
    PriceTable,
    TrackedEmbeddingProvider,
    UsageRecorder,
)
from requirements_agent.ai.vectorstore.base import VectorStore
from requirements_agent.core.config import Settings
from requirements_agent.core.rate_limit import TokenBucketRateLimiter
from requirements_agent.core.security import PasswordHasher, TokenService
from requirements_agent.db.session import Database
from requirements_agent.ingestion.chunker import Chunker
from requirements_agent.repositories.repositories import make_chunk_loader
from requirements_agent.services.jobs import AsyncioJobRunner, JobRunner


@dataclass
class Container:
    settings: Settings
    database: Database
    password_hasher: PasswordHasher
    token_service: TokenService
    usage_recorder: UsageRecorder
    embedder: EmbeddingProvider
    vector_store: VectorStore
    llm: LLMProvider
    chunker: Chunker
    retriever: Retriever
    context_builder: ContextBuilder
    structured: StructuredGenerator
    grounding: GroundingChecker
    confidence: ConfidenceScorer
    jobs: JobRunner
    ai_rate_limiter: TokenBucketRateLimiter


def build_container(
    settings: Settings,
    *,
    database: Database | None = None,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    jobs: JobRunner | None = None,
) -> Container:
    """Wire the application. Keyword overrides exist for tests (fake LLMs, custom stores)."""
    database = database or Database(settings.database_url, echo=settings.database_echo)
    recorder = DatabaseUsageRecorder(database.session_factory, PriceTable(settings.ai_price_table))
    raw_embedder = embedder or build_embedding_provider(settings)
    if raw_embedder.dimension != settings.embedding_dimension:
        raise ValueError(
            f"embedding provider dimension {raw_embedder.dimension} != EMBEDDING_DIMENSION "
            f"{settings.embedding_dimension}"
        )
    tracked_embedder = TrackedEmbeddingProvider(raw_embedder, recorder)
    store = vector_store or build_vector_store(settings, database.session_factory)
    provider = llm or build_llm_provider(settings)
    retriever = Retriever(
        tracked_embedder,
        store,
        make_chunk_loader(database.session_factory),
        mmr_lambda=settings.mmr_lambda,
        fetch_multiplier=settings.retrieval_fetch_multiplier,
    )
    return Container(
        settings=settings,
        database=database,
        password_hasher=PasswordHasher(rounds=settings.bcrypt_rounds),
        token_service=TokenService(settings),
        usage_recorder=recorder,
        embedder=tracked_embedder,
        vector_store=store,
        llm=provider,
        chunker=Chunker(settings.chunk_max_tokens, settings.chunk_overlap_tokens),
        retriever=retriever,
        context_builder=ContextBuilder(settings.analysis_context_tokens),
        structured=StructuredGenerator(
            provider, recorder, max_repairs=settings.structured_max_repairs
        ),
        grounding=GroundingChecker(),
        confidence=ConfidenceScorer(),
        jobs=jobs or AsyncioJobRunner(settings.job_concurrency),
        ai_rate_limiter=TokenBucketRateLimiter(
            settings.rate_limit_ai_capacity, settings.rate_limit_ai_refill_per_minute / 60.0
        ),
    )
