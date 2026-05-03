"""Request cancellation — stop inference when the client disconnects.

A ``Cancellation`` is a one-shot, thread-safe flag. The streaming endpoint
spawns a watchdog that polls ``Request.is_disconnected()`` and trips the flag
when the client drops; both adapters check the flag (llama.cpp via
``stopping_criteria``, MLX via the producer loop) and bail out early so we
don't burn GPU on a response nobody's reading.

Why a plain flag instead of asyncio.Event:
- The flag is read from a worker thread (llama.cpp's stopping_criteria
  callback), and asyncio.Event is not thread-safe across the event-loop /
  thread boundary.
- ``__bool__`` lets adapters write ``if cancel:`` without importing the type.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol


class _SupportsDisconnect(Protocol):
    async def is_disconnected(self) -> bool: ...


class Cancellation:
    """A thread-safe boolean flag with a friendly truthiness check."""

    __slots__ = ("_flag", "reason")

    def __init__(self) -> None:
        self._flag = threading.Event()
        self.reason: str | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        if not self._flag.is_set():
            self.reason = reason
            self._flag.set()

    def __bool__(self) -> bool:
        return self._flag.is_set()

    @property
    def cancelled(self) -> bool:
        return self._flag.is_set()


@asynccontextmanager
async def watch_disconnect(
    request: _SupportsDisconnect,
    poll_interval: float = 0.1,
) -> AsyncIterator[Cancellation]:
    """Spin up a background poll on ``request.is_disconnected()``.

    Yields a :class:`Cancellation` that trips when the client drops. The
    watchdog task is always cancelled on exit so we never leak it.
    """
    cancel = Cancellation()

    async def _watch() -> None:
        try:
            while not cancel:
                if await request.is_disconnected():
                    cancel.cancel(reason="client_disconnect")
                    return
                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            return

    task = asyncio.create_task(_watch())
    try:
        yield cancel
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            # Watchdog cancellation is expected; any other exception is
            # already surfaced via the request itself, so swallow it here.
            pass
