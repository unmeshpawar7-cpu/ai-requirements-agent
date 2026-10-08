"""Background job runner.

`JobRunner` is the seam: today `AsyncioJobRunner` runs jobs as asyncio tasks inside the API
process (bounded concurrency, graceful drain on shutdown). Jobs only receive ids and open their
own database sessions, and their state lives in the database (document/analysis status), so
moving them to an external queue (Arq, Celery, Cloud Tasks) means implementing `submit` only.

Durability trade-off: an in-process job dies with the process. Startup recovery re-queues
documents left in `pending`/`processing` and fails analyses left `queued`/`running` (see
`services.lifecycle`).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextvars import copy_context
from typing import Protocol

logger = logging.getLogger(__name__)

JobFn = Callable[[], Awaitable[None]]


class JobRunner(Protocol):
    def submit(self, name: str, fn: JobFn) -> None: ...

    async def shutdown(self, timeout: float = 10.0) -> None: ...


class AsyncioJobRunner:
    def __init__(self, concurrency: int = 2) -> None:
        self._semaphore = asyncio.Semaphore(concurrency)
        self._tasks: set[asyncio.Task[None]] = set()
        self._accepting = True

    @property
    def pending(self) -> int:
        return len(self._tasks)

    def submit(self, name: str, fn: JobFn) -> None:
        if not self._accepting:
            raise RuntimeError("job runner is shutting down")
        # Copy the caller's context so the request id appears in the job's log lines.
        task = asyncio.get_running_loop().create_task(
            self._run(name, fn), name=f"job:{name}", context=copy_context()
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, name: str, fn: JobFn) -> None:
        async with self._semaphore:
            logger.info("job started", extra={"job": name})
            try:
                await fn()
            except asyncio.CancelledError:
                logger.warning("job cancelled", extra={"job": name})
                raise
            except Exception:
                # Jobs record their own failure state; this is the last-resort log.
                logger.exception("job crashed", extra={"job": name})
            else:
                logger.info("job finished", extra={"job": name})

    async def wait_idle(self, timeout: float = 30.0) -> None:
        """Wait until every submitted job has finished (used by tests and the demo script)."""
        async with asyncio.timeout(timeout):
            while self._tasks:
                await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def shutdown(self, timeout: float = 10.0) -> None:
        self._accepting = False
        if not self._tasks:
            return
        done, pending = await asyncio.wait(list(self._tasks), timeout=timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            logger.warning("cancelled unfinished jobs on shutdown", extra={"count": len(pending)})
        del done
