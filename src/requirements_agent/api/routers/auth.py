"""Authentication and user administration endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Response, status

from requirements_agent.api.deps import Admin, ContainerDep, CurrentUser, Paging, SessionDep
from requirements_agent.domain.enums import Role
from requirements_agent.repositories.repositories import UserRepository
from requirements_agent.schemas.common import Envelope, Page, error_responses, ok
from requirements_agent.schemas.dto import (
    RegisterRequest,
    TokenRequest,
    TokenResponse,
    UserOut,
    UserUpdate,
)
from requirements_agent.services.accounts import AccountService

router = APIRouter(prefix="/api", tags=["Auth & users"])


def _service(session: SessionDep, container: ContainerDep) -> AccountService:
    return AccountService(
        session,
        container.password_hasher,
        container.token_service,
        Role(container.settings.default_signup_role),
    )


@router.post(
    "/auth/register",
    status_code=status.HTTP_201_CREATED,
    response_model=Envelope[UserOut],
    responses=error_responses(409, 422),
    summary="Register an account (the first account becomes admin)",
)
async def register(
    body: RegisterRequest, session: SessionDep, container: ContainerDep, response: Response
) -> Envelope[UserOut]:
    user = await _service(session, container).register(body.email, body.password, body.full_name)
    response.headers["Location"] = f"/api/users/{user.id}"
    return ok(UserOut.model_validate(user), "Account created")


@router.post(
    "/auth/token",
    response_model=Envelope[TokenResponse],
    responses=error_responses(401, 422),
    summary="Exchange email + password for a JWT access token",
)
async def token(
    body: TokenRequest, session: SessionDep, container: ContainerDep
) -> Envelope[TokenResponse]:
    access_token, _ = await _service(session, container).login(body.email, body.password)
    return ok(
        TokenResponse(access_token=access_token, expires_in=container.token_service.ttl_seconds)
    )


@router.get("/auth/me", response_model=Envelope[UserOut], responses=error_responses(401))
async def me(user: CurrentUser) -> Envelope[UserOut]:
    return ok(UserOut.model_validate(user))


@router.get(
    "/users",
    response_model=Envelope[Page[UserOut]],
    responses=error_responses(401, 403),
    summary="List users (admin)",
)
async def list_users(_: Admin, session: SessionDep, paging: Paging) -> Envelope[Page[UserOut]]:
    users, total = await UserRepository(session).list(paging.offset, paging.page_size)
    return ok(
        Page[UserOut].build(
            [UserOut.model_validate(u) for u in users],
            page=paging.page,
            page_size=paging.page_size,
            total_count=total,
        )
    )


@router.patch(
    "/users/{user_id}",
    response_model=Envelope[UserOut],
    responses=error_responses(401, 403, 404, 409, 422),
    summary="Change a user's role or active flag (admin)",
)
async def update_user(
    user_id: uuid.UUID,
    body: UserUpdate,
    actor: Admin,
    session: SessionDep,
    container: ContainerDep,
) -> Envelope[UserOut]:
    user = await _service(session, container).update_user(
        user_id, role=body.role, is_active=body.is_active, actor=actor
    )
    return ok(UserOut.model_validate(user))
