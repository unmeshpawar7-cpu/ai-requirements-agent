"""In-process token-bucket rate limiter.

Each key (user id) owns a bucket of `capacity` tokens that refills continuously at
`refill_per_second`. A request consumes one token; an empty bucket yields the number of
seconds until a token is available, which the API returns as `Retry-After`.

Scope and limits (documented trade-off): state is per process, so N replicas allow N times
the configured rate. Swap for a Redis-backed limiter when running more than one replica.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    remaining: float
    retry_after_seconds: float


class TokenBucketRateLimiter:
    def __init__(
        self,
        capacity: int,
        refill_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if refill_per_second <= 0:
            raise ValueError("refill_per_second must be > 0")
        self.capacity = float(capacity)
        self.refill_per_second = refill_per_second
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._lock = threading.Lock()

    def consume(self, key: str, cost: float = 1.0) -> RateLimitDecision:
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self.capacity, updated_at=now)
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
                bucket.updated_at = now
            self._buckets.move_to_end(key)
            self._evict_if_needed()

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return RateLimitDecision(True, bucket.tokens, 0.0)
            deficit = cost - bucket.tokens
            return RateLimitDecision(False, bucket.tokens, deficit / self.refill_per_second)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()

    def _evict_if_needed(self) -> None:
        # Least-recently-used keys go first; an evicted key simply starts with a full bucket,
        # which only matters for keys that have been idle longest.
        while len(self._buckets) > self._max_keys:
            self._buckets.popitem(last=False)
