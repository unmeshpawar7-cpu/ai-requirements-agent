"""Registration, login, admin seeding and user administration."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from requirements_agent.core.errors import ConflictError, NotFoundError, UnauthorizedError
from requirements_agent.core.security import PasswordHasher, TokenService
from requirements_agent.db.models import User
from requirements_agent.domain.enums import Role
from requirements_agent.repositories.repositories import UserRepository

logger = logging.getLogger(__name__)


class AccountService:
    def __init__(
        self,
        session: AsyncSession,
        hasher: PasswordHasher,
        tokens: TokenService,
        default_role: Role = Role.VIEWER,
    ) -> None:
        self.session = session
        self.users = UserRepository(session)
        self.hasher = hasher
        self.tokens = tokens
        self.default_role = default_role

    async def register(self, email: str, password: str, full_name: str) -> User:
        """Create an account. The very first account becomes `admin` (bootstrap); later
        self-registrations get `DEFAULT_SIGNUP_ROLE` (viewer unless configured)."""
        email = email.lower()
        if await self.users.by_email(email):
            raise ConflictError("An account with this email already exists")
        role = Role.ADMIN if await self.users.count() == 0 else self.default_role
        user = User(
            email=email,
            full_name=full_name.strip(),
            password_hash=self.hasher.hash(password),
            role=role,
            is_active=True,
        )
        self.session.add(user)
        try:
            await self.session.commit()
        except IntegrityError as exc:  # concurrent registration with the same email
            await self.session.rollback()
            raise ConflictError("An account with this email already exists") from exc
        logger.info("user registered", extra={"userId": str(user.id), "role": role.value})
        return user

    async def login(self, email: str, password: str) -> tuple[str, User]:
        user = await self.users.by_email(email.lower())
        # verify() runs bcrypt against a dummy hash for unknown users (timing equalisation).
        valid = self.hasher.verify(password, user.password_hash if user else None)
        if not user or not valid or not user.is_active:
            raise UnauthorizedError("Invalid email or password")
        return self.tokens.create_access_token(user.id, user.role), user

    async def seed_admin(self, email: str, password: str) -> None:
        if await self.users.by_email(email.lower()):
            return
        self.session.add(
            User(
                email=email.lower(),
                full_name="Administrator",
                password_hash=self.hasher.hash(password),
                role=Role.ADMIN,
                is_active=True,
            )
        )
        await self.session.commit()
        logger.info("seeded admin account from ADMIN_EMAIL")

    async def update_user(
        self, user_id: uuid.UUID, *, role: Role | None, is_active: bool | None, actor: User
    ) -> User:
        user = await self.users.get(user_id)
        if user is None:
            raise NotFoundError("User not found")
        if user.id == actor.id and (
            (role is not None and role != Role.ADMIN) or is_active is False
        ):
            raise ConflictError("Admins cannot demote or deactivate themselves")
        if role is not None:
            user.role = role
        if is_active is not None:
            user.is_active = is_active
        await self.session.commit()
        return user
