from __future__ import annotations

import pytest

from inference_engine import usage_ledger
from inference_engine.api.metrics import metrics
from inference_engine.api.state import app_state
from inference_engine.config import settings
from inference_engine.model_routing_runtime import (
    ModelRoutingEnforcementError,
    ModelRoutingRateLimiter,
)
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


@pytest.mark.asyncio
async def test_metrics_expose_usage_ledger_pressure(monkeypatch) -> None:
    monkeypatch.setattr(settings, "usage_ledger_enabled", True)
    monkeypatch.setattr(settings, "usage_ledger_max_buffer", 1)
    usage_ledger._reset_for_tests()
    try:
        usage_ledger.usage_ledger.submit({"usage_record_id": "usage-1"})
        usage_ledger.usage_ledger.submit({"usage_record_id": "usage-2"})
        body = await metrics()
    finally:
        usage_ledger._reset_for_tests()

    assert "inference_engine_usage_ledger_enabled 1" in body
    assert "inference_engine_usage_ledger_emitted_total 1" in body
    assert "inference_engine_usage_ledger_dropped_total 1" in body
    assert "inference_engine_usage_ledger_sink_failures_total 0" in body
    assert "inference_engine_usage_ledger_buffered 1" in body


@pytest.mark.asyncio
async def test_metrics_expose_budget_window_entry_headroom(monkeypatch) -> None:
    limiter = ModelRoutingRateLimiter(max_window_entries=4, clock=lambda: 10.0)
    monkeypatch.setattr(app_state, "model_routing_rate_limiter", limiter)

    for _ in range(4):
        limiter.consume(
            digest="sha256:policy",
            route_id="route",
            org_id="org",
            tenant="tenant",
            limit=None,
            policy_id="policy-1",
            tokens=1,
            max_tokens_per_minute=1_000_000,
        )
    with pytest.raises(ModelRoutingEnforcementError):
        limiter.consume(
            digest="sha256:policy",
            route_id="route",
            org_id="org",
            tenant="tenant",
            limit=None,
            policy_id="policy-1",
            tokens=1,
            max_tokens_per_minute=1_000_000,
        )

    body = await metrics()

    assert "inference_engine_model_routing_rate_limit_max_window_entries 4" in body
    assert "inference_engine_model_routing_rate_limit_window_entries_peak 4" in body
    assert "inference_engine_model_routing_rate_limit_state_capacity_denials_total 1" in body


@pytest.mark.asyncio
async def test_metrics_report_the_usage_ledger_off_by_default() -> None:
    body = await metrics()

    assert "inference_engine_usage_ledger_enabled 0" in body
