"""Shared API helpers for tenant-aware scheduling."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from fastapi import HTTPException

from ..adapters import InferenceAdapter
from ..auth import Identity
from ..config import settings
from ..scheduler import SchedulerLease, TenantQueueFullError, TenantQueueTimeoutError
from . import _usage
from .state import app_state

# The response-header twin of ``scheduler_span_attrs`` below. Same numbers, on
# the wire instead of on a span, because the two answer different questions: a
# span says "why was that request slow?" once it has finished, and a client
# rendering a live progress row needs the answer while the request is still in
# flight. Nothing else the engine emits carries a *per-request* queue wait —
# ``/metrics`` is aggregate, and a trace lands in the caller's backend after
# the fact.
#
# All of them describe the admission that STARTED the response — the first
# lease the request acquired. A request that falls back to another model, or
# retries a schema repair, is admitted again; that later wait is not counted
# here. On the streaming path it cannot be: these headers are flushed when the
# stream opens, which is strictly before any re-admission can happen. One rule
# that holds on both paths beats a number whose meaning depends on which path
# served it.
QUEUE_WAIT_MS_HEADER = "x-engine-queue-wait-ms"
QUEUE_DEPTH_HEADER = "x-engine-queue-depth"
TENANT_QUEUE_DEPTH_HEADER = "x-engine-tenant-queue-depth"
RESOURCE_HEADER = "x-engine-resource"


@dataclass
class AdmissionTelemetry:
    """Per-request holder for the admission that started the response.

    Opened on the way in and mutated on the way through, rather than set by the
    route: ``BaseHTTPMiddleware`` sits between the two and runs everything below
    it in a child task, so a ContextVar the route *rebinds* never travels back
    up, while a container it *mutates* does. ``usage_ledger`` opens its record
    the same way and for the same reason.
    """

    lease: SchedulerLease | None = None


_current_admission: ContextVar[AdmissionTelemetry | None] = ContextVar(
    "scheduler_admission", default=None
)


def resource_key(adapter: InferenceAdapter, model_name: str) -> str:
    return f"{adapter.backend_name}:{model_name}"


def resource_limit(adapter: InferenceAdapter) -> int:
    if adapter.backend_name in {"vllm", "openrouter"}:
        return settings.scheduler_vllm_resource_max_in_flight
    return settings.scheduler_resource_max_in_flight


def scheduler_span_attrs(lease: SchedulerLease | None) -> dict:
    if lease is None:
        return {}
    return {
        "scheduler.enabled": lease.enabled,
        "scheduler.tenant": lease.tenant,
        "scheduler.resource": lease.resource_key,
        "scheduler.workload": lease.workload,
        "scheduler.priority": lease.priority,
        "scheduler.estimated_tokens": lease.estimated_tokens,
        "scheduler.wait_ms": round(lease.wait_ms, 2),
        "scheduler.queue_depth_at_submit": lease.queue_depth_at_submit,
        "scheduler.tenant_queue_depth_at_submit": lease.tenant_queue_depth_at_submit,
    }


def begin_admission() -> AdmissionTelemetry:
    """Open this request's holder. Called by the response-header middleware."""
    telemetry = AdmissionTelemetry()
    _current_admission.set(telemetry)
    return telemetry


def bind_admission(lease: SchedulerLease) -> None:
    """Record the admission that started this response. First lease wins.

    A no-op outside a request scope, which is what lets the route coroutines be
    called directly, as much of the test suite does.
    """
    telemetry = _current_admission.get()
    if telemetry is None or telemetry.lease is not None:
        return
    telemetry.lease = lease


def _header_safe(value: str) -> bool:
    """Whether a value can be written into a response header at all.

    A resource key is a backend name joined to a model id, and a model id is
    whatever the upstream registry reported. One that will not encode raises
    while the response is being written — turning a completion the caller
    already paid for into a 500 over a telemetry field. Drop the header
    instead: this channel must degrade, never fail the request.
    """
    return bool(value) and value.isascii() and value.isprintable()


def admission_headers(telemetry: AdmissionTelemetry | None) -> dict[str, str]:
    """The wire form of one admission. Empty when the request never queued."""
    lease = telemetry.lease if telemetry is not None else None
    if lease is None:
        return {}
    headers = {
        QUEUE_WAIT_MS_HEADER: str(round(lease.wait_ms)),
        # Both depths count this request, so an admitted request under a live
        # scheduler always reports at least 1: it means "admitted with nobody
        # else waiting", not "one ahead of you". A 0 means the scheduler is
        # disabled and nothing was queued at all. A client that renders the
        # raw number as "requests ahead of you" is off by one, which is the
        # exact class of misleading progress this channel exists to remove.
        QUEUE_DEPTH_HEADER: str(lease.queue_depth_at_submit),
        TENANT_QUEUE_DEPTH_HEADER: str(lease.tenant_queue_depth_at_submit),
    }
    if _header_safe(lease.resource_key):
        # The resource this request was admitted against. On a response that
        # fell back, the body's ``model`` is what actually served — these
        # headers describe the admission, not the outcome.
        headers[RESOURCE_HEADER] = lease.resource_key
    return headers


def scheduling_http_error(exc: TenantQueueFullError | TenantQueueTimeoutError) -> HTTPException:
    headers = {"Retry-After": str(exc.retry_after_seconds)}
    if isinstance(exc, TenantQueueFullError):
        _usage.bind_error_type("tenant_queue_full")
        return HTTPException(
            status_code=429,
            detail={
                "message": "tenant queue is full",
                "type": "tenant_queue_full",
                "tenant": exc.tenant,
                "queue_depth": exc.queue_depth,
            },
            headers=headers,
        )
    _usage.bind_error_type("tenant_queue_timeout")
    return HTTPException(
        status_code=503,
        detail={
            "message": "timed out waiting for tenant scheduler capacity",
            "type": "tenant_queue_timeout",
            "tenant": exc.tenant,
            "timeout_seconds": exc.timeout_seconds,
        },
        headers=headers,
    )


async def acquire_slot(
    *,
    identity: Identity,
    adapter: InferenceAdapter,
    model_name: str,
    workload: str,
    priority: float,
    estimated_tokens: int,
):
    try:
        lease = await app_state.scheduler.acquire(
            tenant=identity.tenant,
            key_id=identity.key_id,
            resource_key=resource_key(adapter, model_name),
            resource_limit=resource_limit(adapter),
            workload=workload,
            priority=priority,
            estimated_tokens=estimated_tokens,
        )
    except (TenantQueueFullError, TenantQueueTimeoutError) as exc:
        raise scheduling_http_error(exc) from exc
    # Every scheduled route funnels through here, so binding at this one seam
    # is what keeps a route added later from silently reporting no admission.
    bind_admission(lease)
    return lease
