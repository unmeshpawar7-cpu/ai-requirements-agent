"""FastAPI dependencies: container, DB session, authentication, RBAC, paging, rate limits."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Annotated, Any

from fastapi import Depends, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from requirements_agent.core.errors import ForbiddenError, RateLimitedError, UnauthorizedError
from requirements_agent.core.security import InvalidTokenError
from requirements_agent.db.models import User
from requirements_agent.domain.enums import ROLE_RANK, Role
from requirements_agent.schemas.common import MAX_PAGE_SIZE, PageParams
from requirements_agent.services.container import Container

_bearer = HTTPBearer(auto_error=False, description="JWT from POST /api/auth/token")


def get_container(request: Request) -> Container:
    return request.app.state.container  # type: ignore[no-any-return]


ContainerDep = Annotated[Container, Depends(get_container)]


async def get_session(container: ContainerDep) -> AsyncIterator[AsyncSession]:
    async with container.database.session() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_current_user(
    container: ContainerDep,
    session: SessionDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise UnauthorizedError("Missing bearer token")
    try:
        claims = container.token_service.decode(credentials.credentials)
    except InvalidTokenError as exc:
        raise UnauthorizedError("Invalid or expired token") from exc
    user = await session.get(User, claims.subject)
    if user is None or not user.is_active:
        raise UnauthorizedError("Account not found or disabled")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_role(minimum: Role) -> Callable[..., Coroutine[Any, Any, User]]:
    """Hierarchical RBAC: admin > analyst > viewer. The role is read from the database on
    every request, so demotions take effect immediately (not at token expiry)."""

    async def dependency(user: CurrentUser) -> User:
        if ROLE_RANK[user.role] < ROLE_RANK[minimum]:
            raise ForbiddenError(f"This action requires the '{minimum.value}' role")
        return user

    return dependency


Viewer = Annotated[User, Depends(require_role(Role.VIEWER))]
Analyst = Annotated[User, Depends(require_role(Role.ANALYST))]
Admin = Annotated[User, Depends(require_role(Role.ADMIN))]


def page_params(
    page: Annotated[int, Query(ge=1, description="1-based page number")] = 1,
    page_size: Annotated[
        int, Query(alias="pageSize", ge=1, description="Items per page (capped at 100)")
    ] = 20,
) -> PageParams:
    return PageParams(page=page, page_size=min(page_size, MAX_PAGE_SIZE))


Paging = Annotated[PageParams, Depends(page_params)]


async def ai_rate_limit(user: Analyst, container: ContainerDep) -> User:
    decision = container.ai_rate_limiter.consume(str(user.id))
    if not decision.allowed:
        raise RateLimitedError(decision.retry_after_seconds)
    return user


RateLimitedAnalyst = Annotated[User, Depends(ai_rate_limit)]
