"""The total-elapsed cap on one generation, shared by every route that generates.

Lives outside ``api/chat.py`` so the eval runner can apply the same cap to a
judge call without importing a route module.
"""

from __future__ import annotations

import asyncio
import sys
import time

from .adapters.base import GenerationTimeoutError
from .config import settings
from .observability import get_logger

log = get_logger("generation")

# A clock that keeps counting while the host is asleep. ``time.monotonic`` does
# not, on macOS or Linux, so the difference between the two over one call is
# how long the machine was suspended during it. None where neither exists.
_SUSPEND_INCLUSIVE_CLOCK: int | None = (
    time.CLOCK_MONOTONIC if sys.platform == "darwin" else getattr(time, "CLOCK_BOOTTIME", None)
)


def _suspend_inclusive_now() -> float | None:
    if _SUSPEND_INCLUSIVE_CLOCK is None:
        return None
    return time.clock_gettime(_SUSPEND_INCLUSIVE_CLOCK)


def generation_deadline_seconds() -> float | None:
    """Total elapsed budget for one generation, or None when disabled."""
    seconds = settings.chat_completion_timeout_seconds
    return seconds if seconds > 0 else None


async def generate_within_deadline(adapter, messages, params, model_name: str):
    """Run one generation under a TOTAL-ELAPSED cap, where that is honest.

    WHY THE ADAPTER'S OWN TIMEOUT WAS NOT ENOUGH. Every HTTP adapter passes
    ``chat_completion_timeout_seconds`` to ``httpx.Timeout(...)``, whose read
    component is PER READ OPERATION rather than total elapsed. A response that
    streams steadily — which is exactly what a slow model looks like — resets
    that clock on every chunk and never trips it, so the setting described as a
    "server-side timeout for HTTP-backed /v1/chat/completions calls" did not
    bound the call at all.

    That was not merely a long request. The scheduler lease is held for the
    duration, and a client that gives up first does not stop the generation, so
    every abandoned request leaked a slot. Measured on this engine
    (CHAT_COMPLETION_TIMEOUT_SECONDS=300): one benchmark case ran 1010s across
    two legs — past a 420s client deadline — and the three cases after it were
    refused with ``tenant_queue_timeout`` on an otherwise idle process, CPU
    under 1% and memory 73% free. Only a restart cleared it, three times in one
    afternoon, each time taking the dependent platform down.

    WHY IT IS CONDITIONAL. ``asyncio.wait_for`` cancels the await. For an
    adapter waiting on a socket that closes the upstream request and the work
    really stops. For one running blocking native code in a worker thread it
    abandons the RESULT while the thread computes on — the resource stays busy,
    and a timeout raised on its behalf would claim something that did not
    happen. ``GenerationTimeoutError``'s own docstring draws that line, so the
    deadline applies only to adapters that declare
    ``generation_is_cancellable``. Non-cancellable backends keep exactly their
    previous behaviour, including the unbounded duration: fixing those needs
    interruptible native calls, not a lie at this layer.

    WHAT THE DEADLINE COUNTS is the event loop's clock. uvloop, which uvicorn
    picks when it is installed, reads libuv's clock, and on macOS that keeps
    running while the host sleeps. So on the native laptop deployment a call in
    flight when the machine sleeps fails the moment it wakes, however little
    of its budget it used. On 2026-09-29 four judge calls 504'd 15–17 minutes
    after they started, each after 3–43 s of generation, each at the second of
    a DarkWake. ``awake_seconds`` and ``host_suspended_seconds`` on the log
    line tell that case apart from a generation that really ran too long.
    """
    deadline = generation_deadline_seconds()
    if deadline is None or not getattr(adapter, "generation_is_cancellable", False):
        return await adapter.generate(messages, params)
    started_awake = time.monotonic()
    started_total = _suspend_inclusive_now()
    try:
        return await asyncio.wait_for(adapter.generate(messages, params), deadline)
    except asyncio.TimeoutError as exc:
        awake = time.monotonic() - started_awake
        ended_total = _suspend_inclusive_now()
        suspended = (
            None
            if started_total is None or ended_total is None
            else round(max(0.0, ended_total - started_total - awake), 3)
        )
        log.warning(
            "generation.deadline_exceeded",
            backend=adapter.backend_name,
            model=model_name,
            deadline_seconds=deadline,
            awake_seconds=round(awake, 3),
            host_suspended_seconds=suspended,
        )
        # The typed error the routes already map to a 504, raised only where
        # cancellation genuinely ended the work.
        raise GenerationTimeoutError(
            timeout_seconds=deadline,
            backend=adapter.backend_name,
            model=model_name,
        ) from exc
