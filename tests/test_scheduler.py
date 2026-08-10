from __future__ import annotations

import asyncio

import pytest

from inference_engine.api._scheduling import resource_limit
from inference_engine.config import Settings, settings
from inference_engine.scheduler import (
    TenantQueueFullError,
    TenantQueueTimeoutError,
    TenantScheduler,
)


class _FakeAdapter:
    """Just the attribute ``resource_limit`` dispatches on."""

    def __init__(self, backend_name: str) -> None:
        self.backend_name = backend_name


@pytest.fixture(autouse=True)
def _scheduler_settings(monkeypatch):
    monkeypatch.setattr(settings, "scheduler_enabled", True)
    monkeypatch.setattr(settings, "scheduler_global_max_in_flight", 1)
    monkeypatch.setattr(settings, "scheduler_tenant_reserved_in_flight", 1)
    monkeypatch.setattr(settings, "scheduler_resource_max_in_flight", 1)
    monkeypatch.setattr(settings, "scheduler_vllm_resource_max_in_flight", 4)
    monkeypatch.setattr(settings, "scheduler_max_queue_per_tenant", 8)
    monkeypatch.setattr(settings, "scheduler_queue_timeout_seconds", 2.0)
    monkeypatch.setattr(settings, "scheduler_wait_aging_priority_per_second", 0.0)
    monkeypatch.setattr(settings, "scheduler_tenant_fairness_weight", 2.0)


async def _acquire(
    scheduler: TenantScheduler,
    tenant: str,
    *,
    resource_limit: int = 1,
):
    return await scheduler.acquire(
        tenant=tenant,
        key_id=f"{tenant}-key",
        resource_key="llama_cpp:model",
        resource_limit=resource_limit,
        workload="chat.generate",
        priority=0.0,
        estimated_tokens=10,
    )


@pytest.mark.asyncio
async def test_later_tenant_gets_turn_before_bulk_tail() -> None:
    scheduler = TenantScheduler()
    first_a = await _acquire(scheduler, "tenant-a")

    second_a_task = asyncio.create_task(_acquire(scheduler, "tenant-a"))
    await asyncio.sleep(0)
    first_b_task = asyncio.create_task(_acquire(scheduler, "tenant-b"))
    await asyncio.sleep(0)

    assert not second_a_task.done()
    assert not first_b_task.done()

    await scheduler.release(first_a)
    first_b = await asyncio.wait_for(first_b_task, timeout=0.5)

    assert first_b.tenant == "tenant-b"
    assert not second_a_task.done()

    await scheduler.release(first_b)
    second_a = await asyncio.wait_for(second_a_task, timeout=0.5)
    assert second_a.tenant == "tenant-a"
    await scheduler.release(second_a)


@pytest.mark.asyncio
async def test_single_tenant_can_borrow_idle_capacity(monkeypatch) -> None:
    monkeypatch.setattr(settings, "scheduler_global_max_in_flight", 3)
    scheduler = TenantScheduler()

    leases = await asyncio.gather(
        _acquire(scheduler, "tenant-a", resource_limit=3),
        _acquire(scheduler, "tenant-a", resource_limit=3),
        _acquire(scheduler, "tenant-a", resource_limit=3),
    )

    snapshot = scheduler.snapshot()
    assert snapshot.total_in_flight == 3
    assert snapshot.in_flight_by_tenant == {"tenant-a": 3}

    for lease in leases:
        await scheduler.release(lease)


@pytest.mark.asyncio
async def test_queue_full_rejects_per_tenant(monkeypatch) -> None:
    monkeypatch.setattr(settings, "scheduler_max_queue_per_tenant", 1)
    scheduler = TenantScheduler()

    running = await _acquire(scheduler, "tenant-a")
    queued_task = asyncio.create_task(_acquire(scheduler, "tenant-a"))
    await asyncio.sleep(0)

    with pytest.raises(TenantQueueFullError):
        await _acquire(scheduler, "tenant-a")

    await scheduler.release(running)
    queued = await asyncio.wait_for(queued_task, timeout=0.5)
    await scheduler.release(queued)


@pytest.mark.asyncio
async def test_one_held_ollama_slot_queues_then_refuses_the_rest(monkeypatch) -> None:
    """What holding an ``ollama_http`` dispatch slot actually costs everyone else.

    The README's timeout section leans on this, so it is measured rather than
    asserted. ``ollama_http`` takes the default resource cap of 1, so a second
    request for the same model cannot dispatch while the first holds the slot —
    and it does not wait forever either: it is refused with
    ``TenantQueueTimeoutError`` (a 503 ``tenant_queue_timeout`` on the wire)
    once it has waited ``SCHEDULER_QUEUE_TIMEOUT_SECONDS``.
    """
    monkeypatch.setattr(settings, "scheduler_queue_timeout_seconds", 0.2)
    # The cap the README names: ollama_http takes SCHEDULER_RESOURCE_MAX_IN_FLIGHT
    # (shipped default 1) while vLLM takes its own, larger, setting.
    assert Settings.model_fields["scheduler_resource_max_in_flight"].default == 1
    assert (
        resource_limit(_FakeAdapter("ollama_http"))
        == settings.scheduler_resource_max_in_flight
        == 1
    )
    assert (
        resource_limit(_FakeAdapter("vllm")) == settings.scheduler_vllm_resource_max_in_flight
    )

    scheduler = TenantScheduler()
    held = await _acquire(scheduler, "tenant-a")

    waiter = asyncio.create_task(_acquire(scheduler, "tenant-a"))
    await asyncio.sleep(0)
    assert not waiter.done()

    with pytest.raises(TenantQueueTimeoutError):
        await asyncio.wait_for(waiter, timeout=2.0)

    # The holder was never disturbed: nothing here bounds the call itself.
    assert scheduler.snapshot().in_flight_by_resource == {"llama_cpp:model": 1}
    await scheduler.release(held)
