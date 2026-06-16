"""Shared API helpers for tenant-aware scheduling."""

from __future__ import annotations

from fastapi import HTTPException

from ..adapters import InferenceAdapter
from ..auth import Identity
from ..config import settings
from ..scheduler import SchedulerLease, TenantQueueFullError, TenantQueueTimeoutError


def resource_key(adapter: InferenceAdapter, model_name: str) -> str:
    return f"{adapter.backend_name}:{model_name}"


def resource_limit(adapter: InferenceAdapter) -> int:
    if adapter.backend_name == "vllm":
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


def scheduling_http_error(exc: TenantQueueFullError | TenantQueueTimeoutError) -> HTTPException:
    headers = {"Retry-After": str(exc.retry_after_seconds)}
    if isinstance(exc, TenantQueueFullError):
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
    from .state import app_state  # noqa: PLC0415 — avoid import-time singleton cycles

    try:
        return await app_state.scheduler.acquire(
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
