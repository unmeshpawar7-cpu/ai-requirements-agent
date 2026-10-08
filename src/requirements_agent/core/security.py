"""Password hashing (bcrypt) and JWT access tokens (PyJWT)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from requirements_agent.core.config import Settings
from requirements_agent.domain.enums import Role

BCRYPT_MAX_BYTES = 72


class PasswordHasher:
    """bcrypt wrapper. bcrypt only uses the first 72 bytes, so longer inputs are rejected
    at the API boundary instead of being silently truncated."""

    def __init__(self, rounds: int = 12) -> None:
        self._rounds = rounds
        # Used to equalise timing when a login targets an unknown account.
        self._dummy_hash = self.hash("timing-equaliser-password")

    def hash(self, password: str) -> str:
        encoded = password.encode("utf-8")
        if len(encoded) > BCRYPT_MAX_BYTES:
            raise ValueError("password exceeds bcrypt's 72-byte limit")
        return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=self._rounds)).decode("ascii")

    def verify(self, password: str, password_hash: str | None) -> bool:
        encoded = password.encode("utf-8")
        if len(encoded) > BCRYPT_MAX_BYTES:
            return False
        target = (password_hash or self._dummy_hash).encode("ascii")
        try:
            matches = bcrypt.checkpw(encoded, target)
        except ValueError:
            return False
        return matches and password_hash is not None


@dataclass(frozen=True, slots=True)
class TokenClaims:
    subject: uuid.UUID
    role: Role
    expires_at: datetime
    token_id: str


class InvalidTokenError(Exception):
    """Raised for expired, malformed or tampered tokens."""


class TokenService:
    def __init__(self, settings: Settings) -> None:
        self._secret = settings.jwt_secret.get_secret_value()
        self._algorithm = settings.jwt_algorithm
        self._issuer = settings.jwt_issuer
        self._audience = settings.jwt_audience
        self._ttl = timedelta(minutes=settings.jwt_expires_minutes)

    @property
    def ttl_seconds(self) -> int:
        return int(self._ttl.total_seconds())

    def create_access_token(
        self, user_id: uuid.UUID, role: Role, *, now: datetime | None = None
    ) -> str:
        issued_at = now or datetime.now(UTC)
        claims = {
            "sub": str(user_id),
            "role": role.value,
            "iss": self._issuer,
            "aud": self._audience,
            "iat": issued_at,
            "nbf": issued_at,
            "exp": issued_at + self._ttl,
            "jti": uuid.uuid4().hex,
        }
        return jwt.encode(claims, self._secret, algorithm=self._algorithm)

    def decode(self, token: str) -> TokenClaims:
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm],
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["exp", "iat", "sub", "aud", "iss"]},
            )
            return TokenClaims(
                subject=uuid.UUID(payload["sub"]),
                role=Role(payload["role"]),
                expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC),
                token_id=str(payload.get("jti", "")),
            )
        except (jwt.PyJWTError, KeyError, ValueError) as exc:
            raise InvalidTokenError(str(exc)) from exc
