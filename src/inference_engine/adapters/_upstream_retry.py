"""Bounded retries for idempotent HTTP upstream calls.

Why this exists
---------------

Resilience in this engine used to be fallback-only: any exception out of an
adapter sent the request to the *next model in the signed route*. A single
transient 503 from a vLLM upstream therefore changed which model answered the
caller — a silent model substitution caused by a dropped packet. Retrying the
same deployment first is the whole point of this module; falling back to a
different model stays the second resort, not the first.

What is retried
---------------

Only failures where the upstream produced no usable answer AND a second
attempt has a real chance of a different outcome:

* transport failures that never delivered a response (connect error, connect
  timeout, connection reset, protocol error);
* status codes an upstream uses to say "later" or "I broke": 408, 425, 429,
  500, 502, 503, 504, 529.

Everything else is left alone on purpose. A 400/404/422 is the request being
wrong, and resending it burns the caller's deadline to receive the same
rejection. A *read* timeout is not retried either: the upstream was working
when the clock ran out, so a second attempt re-pays the full wait rather than
recovering from a blip — the deadline is the caller's budget, not ours to
spend twice. Nor is this module's own deadline expiring: there is by
definition nothing left to spend on another attempt.

The same set answers a second question — whether a failure counts against the
deployment's health — via :func:`_upstream.is_health_signal`, whose docstring
carries the reasoning for the timeout split.

Deadline
--------

Every attempt and every backoff sleep is drawn from one
``CHAT_COMPLETION_TIMEOUT_SECONDS`` budget (:class:`UpstreamDeadline`), on the
blocking path and the streaming path alike: :func:`deadline_scope` bounds each
await on the upstream by what is LEFT of the budget, and raises up front once
it is gone.

The exact guarantee, because the loose version of it would be false: no
attempt, no backoff sleep and no upstream read is ever STARTED after the budget
is spent, and any read in flight is bounded by what remains. The request itself
still ends a hair past the budget — unwinding the timeout scope and mapping the
error take real time. Measured against a mock upstream that would have streamed
for 10 s, a 300 ms budget ended the stream at 301–303 ms over five runs. What
retrying cannot do is carry a request materially past the operator's timeout;
what it does not promise is ending on the microsecond.

When the remaining budget cannot fit the backoff plus a usable attempt, the
retry is refused and the original error surfaces — a real 503 is more useful
to a caller than a timeout produced by our own optimism.

Streaming
---------

Beyond the budget this module has no opinion about streams: the "never retry
after the first token" rule lives in each adapter's ``stream()``, because only
the adapter knows whether a chunk has already reached the consumer. A partial
stream cannot be restarted and concatenated.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from datetime import UTC, datetime
from typing import Any

import httpx

from ..config import settings
from ..observability import get_logger

log = get_logger("adapter.upstream_retry")


# Status codes worth a second attempt. 501 is deliberately absent: an upstream
# that does not implement the route will not implement it a moment later.
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})

# Transport failures that leave no response behind. ``ConnectTimeout`` is a
# ``TimeoutException`` subclass and is listed first in ``is_retryable`` so it
# does not fall into the read-timeout rule below it.
RETRYABLE_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)

# A retry needs room for the backoff *and* an attempt that can plausibly do
# something. Retrying with milliseconds left would convert a typed upstream
# error into a timeout, which tells the caller less than the error did.
_MIN_ATTEMPT_HEADROOM_SECONDS = 0.25


class _RetryCounters:
    """Process-wide retry tally, rendered by ``/v1/metrics``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_backend: dict[str, int] = {}

    def record(self, backend: str) -> None:
        with self._lock:
            self._by_backend[backend] = self._by_backend.get(backend, 0) + 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._by_backend)

    def reset(self) -> None:
        with self._lock:
            self._by_backend.clear()


retry_counters = _RetryCounters()


class UpstreamDeadline:
    """One total-elapsed budget shared by an upstream call and its retries.

    ``None`` total means the operator disabled the timeout
    (``CHAT_COMPLETION_TIMEOUT_SECONDS=0``); ``remaining`` is then ``None`` and
    every attempt runs unbounded, exactly as it did before retries existed.
    """

    __slots__ = ("_deadline_at",)

    def __init__(self, total_seconds: float | None) -> None:
        if total_seconds is None:
            self._deadline_at: float | None = None
        else:
            self._deadline_at = time.monotonic() + total_seconds

    @property
    def remaining(self) -> float | None:
        if self._deadline_at is None:
            return None
        return self._deadline_at - time.monotonic()


def deadline_scope(deadline: UpstreamDeadline) -> AbstractAsyncContextManager[Any]:
    """Bound ONE await on the upstream by what is LEFT of the request budget.

    Raises ``TimeoutError`` up front when the budget is already spent, so no
    caller can open a fresh upstream operation on a deadline that has passed.
    A ``None`` total (the operator disabled the timeout) yields an unbounded
    scope, exactly as it behaved before retries existed.

    ONLY wrap an await that cannot suspend at a ``yield``. ``asyncio.timeout``
    works by cancelling the running task; a scope left open across a generator
    yield would fire while the *consumer's* frame is on the stack, where the
    ``__aexit__`` that turns the cancellation into ``TimeoutError`` is not.
    """
    remaining = deadline.remaining
    if remaining is None:
        return nullcontext()
    if remaining <= 0:
        raise TimeoutError("upstream deadline exhausted")
    return asyncio.timeout(remaining)


@dataclass(frozen=True)
class RetryPlan:
    """The decision for one failed attempt. ``delay_seconds`` is a sleep."""

    retry: bool
    delay_seconds: float = 0.0
    reason: str = ""


def is_retryable(exc: BaseException) -> bool:
    """Is a second attempt at this failure worth the caller's deadline?

    Also the breaker's health classifier — see
    :func:`_upstream.is_health_signal`, which delegates here and documents why
    the two questions get the same answer.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    if isinstance(exc, httpx.ConnectTimeout):
        # The one timeout that is evidence rather than a clock: no connection
        # was established, so nothing about the caller's budget is implicated.
        return True
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        # Read / write / pool timeout, or this module's own ``UpstreamDeadline``
        # expiring: the upstream had the request and the clock ran out on it.
        # See the module docstring.
        return False
    return isinstance(exc, RETRYABLE_TRANSPORT_ERRORS)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Parse ``Retry-After`` off a status error, in seconds. ``None`` if absent.

    Both wire forms are accepted: delta-seconds and an HTTP-date. A date in
    the past yields 0.0 (retry immediately), never a negative delay.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    raw = response.headers.get("retry-after") if hasattr(response, "headers") else None
    if not raw:
        return None
    raw = raw.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with equal jitter for the ``attempt``-th failure.

    Equal jitter (half fixed, half random) rather than full jitter: it keeps
    the decorrelation that stops a fleet from retrying in lockstep without
    letting a retry fire back essentially immediately.
    """
    base = max(0.0, settings.upstream_retry_base_delay_seconds)
    ceiling = max(0.0, settings.upstream_retry_max_delay_seconds)
    if base <= 0.0:
        return 0.0
    raw = min(ceiling, base * (2 ** max(0, attempt - 1)))
    half = raw / 2
    return half + random.uniform(0.0, half)  # noqa: S311 — jitter, not a secret


def plan_retry(
    *,
    attempt: int,
    exc: BaseException,
    remaining_seconds: float | None,
) -> RetryPlan:
    """Decide whether the ``attempt``-th (1-based) failure gets another go."""
    if not settings.upstream_retry_enabled:
        return RetryPlan(retry=False, reason="retries_disabled")
    max_attempts = max(1, settings.upstream_retry_max_attempts)
    if attempt >= max_attempts:
        return RetryPlan(retry=False, reason="attempts_exhausted")
    if not is_retryable(exc):
        return RetryPlan(retry=False, reason="not_retryable")

    delay = _backoff_delay(attempt)
    after = retry_after_seconds(exc)
    if after is not None:
        # Honour what the upstream asked for even when it exceeds our own
        # ceiling — the deadline check below is what refuses an impossible one.
        delay = max(delay, after)

    if remaining_seconds is not None:
        if remaining_seconds <= 0:
            return RetryPlan(retry=False, reason="deadline_exceeded")
        if delay + _MIN_ATTEMPT_HEADROOM_SECONDS > remaining_seconds:
            return RetryPlan(retry=False, reason="deadline_too_short")

    return RetryPlan(retry=True, delay_seconds=delay, reason="retryable")


def log_retry(
    *,
    backend: str,
    model: str,
    attempt: int,
    plan: RetryPlan,
    exc: BaseException,
    operation: str,
) -> None:
    retry_counters.record(backend)
    status = getattr(getattr(exc, "response", None), "status_code", None)
    log.warning(
        "upstream.retry",
        backend=backend,
        model=model,
        operation=operation,
        attempt=attempt,
        delay_seconds=round(plan.delay_seconds, 3),
        error_type=type(exc).__name__,
        upstream_status_code=status,
    )


def log_retry_refused(
    *,
    backend: str,
    model: str,
    attempt: int,
    plan: RetryPlan,
    exc: BaseException,
    operation: str,
) -> None:
    """Why a failed attempt was NOT reissued.

    The counterpart to :func:`log_retry`, and the only place ``plan.reason``
    reaches an operator: from the outside a 502 that was retried once and a 502
    that was refused a retry look identical, and which one it was decides
    whether the answer is "the upstream is broken" or "our budget ran out".
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    log.info(
        "upstream.retry_refused",
        backend=backend,
        model=model,
        operation=operation,
        attempt=attempt,
        reason=plan.reason,
        error_type=type(exc).__name__,
        upstream_status_code=status,
    )


__all__ = [
    "RETRYABLE_STATUS_CODES",
    "RETRYABLE_TRANSPORT_ERRORS",
    "RetryPlan",
    "UpstreamDeadline",
    "deadline_scope",
    "is_retryable",
    "log_retry",
    "log_retry_refused",
    "plan_retry",
    "retry_after_seconds",
    "retry_counters",
]
