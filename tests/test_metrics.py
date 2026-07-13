from __future__ import annotations

import pytest

from inference_engine.api.metrics import metrics
from inference_engine.api.state import app_state
from inference_engine.config import settings
from inference_engine.scheduler import TenantScheduler


class FakeObserver:
    metrics_snapshot = {
        "running": 1,
        "attempts_total": 4,
        "successes_total": 3,
        "failures_total": 1,
        "consecutive_failures": 0,
        "pending": 0,
        "last_success_unixtime": 1_789_000_000.0,
    }


@pytest.mark.asyncio
async def test_metrics_include_scheduler_pressure(monkeypatch) -> None:
    monkeypatch.setattr(settings, "scheduler_enabled", True)
    monkeypatch.setattr(settings, "scheduler_global_max_in_flight", 1)
    monkeypatch.setattr(settings, "scheduler_tenant_reserved_in_flight", 1)
    monkeypatch.setattr(settings, "scheduler_max_queue_per_tenant", 8)
    monkeypatch.setattr(app_state, "scheduler", TenantScheduler())

    lease = await app_state.scheduler.acquire(
        tenant="tenant-a",
        key_id="key-a",
        resource_key="llama_cpp:model",
        resource_limit=1,
        workload="chat.generate",
        priority=0.0,
        estimated_tokens=12,
    )
    try:
        body = await metrics()
    finally:
        await app_state.scheduler.release(lease)

    assert "inference_engine_scheduler_enabled 1" in body
    assert "inference_engine_scheduler_in_flight 1" in body
    assert 'inference_engine_scheduler_in_flight_by_tenant{tenant="tenant-a"} 1' in body
    assert (
        'inference_engine_scheduler_in_flight_by_resource{resource="llama_cpp:model"} 1'
        in body
    )


@pytest.mark.asyncio
async def test_metrics_include_model_plane_observer_delivery_state(monkeypatch) -> None:
    monkeypatch.setattr(app_state, "model_plane_observer", FakeObserver())

    body = await metrics()

    assert "inference_engine_model_plane_observer_enabled 1" in body
    assert "inference_engine_model_plane_observer_running 1" in body
    assert "inference_engine_model_plane_observer_attempts_total 4" in body
    assert "inference_engine_model_plane_observer_successes_total 3" in body
    assert "inference_engine_model_plane_observer_failures_total 1" in body
