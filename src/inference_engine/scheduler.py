"""Tenant-aware admission and fair dispatch for shared inference backends."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from .config import settings


class TenantQueueFullError(RuntimeError):
    def __init__(self, tenant: str, queue_depth: int) -> None:
        self.tenant = tenant
        self.queue_depth = queue_depth
        self.retry_after_seconds = settings.scheduler_retry_after_seconds
        super().__init__(f"tenant queue is full for {tenant!r}")


class TenantQueueTimeoutError(TimeoutError):
    def __init__(self, tenant: str, timeout_seconds: float) -> None:
        self.tenant = tenant
        self.timeout_seconds = timeout_seconds
        self.retry_after_seconds = settings.scheduler_retry_after_seconds
        super().__init__(f"timed out waiting for tenant queue slot for {tenant!r}")


@dataclass(frozen=True)
class SchedulerLease:
    lease_id: int
    tenant: str
    resource_key: str
    workload: str
    priority: float
    estimated_tokens: int
    wait_ms: float
    queue_depth_at_submit: int
    tenant_queue_depth_at_submit: int
    enabled: bool = True


@dataclass(frozen=True)
class SchedulerSnapshot:
    enabled: bool
    total_queued: int
    total_in_flight: int
    queued_by_tenant: dict[str, int]
    in_flight_by_tenant: dict[str, int]
    in_flight_by_resource: dict[str, int]
    accepted_total: int
    rejected_total: int
    timed_out_total: int
    completed_total: int
    wait_seconds_total: float
    wait_observations_total: int
    max_queue_wait_seconds: float


@dataclass
class _Pending:
    tenant: str
    key_id: str
    resource_key: str
    resource_limit: int
    workload: str
    priority: float
    estimated_tokens: int
    submitted_at: float
    sequence: int
    future: asyncio.Future[SchedulerLease]
    queue_depth_at_submit: int = 0
    tenant_queue_depth_at_submit: int = 0
    started: bool = False
    lease: SchedulerLease | None = None


@dataclass
class TenantScheduler:
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _queues: dict[str, deque[_Pending]] = field(default_factory=lambda: defaultdict(deque))
    _in_flight_total: int = 0
    _in_flight_by_tenant: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _in_flight_by_resource: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _last_dispatch_at: dict[str, float] = field(default_factory=dict)
    _active_leases: set[int] = field(default_factory=set)
    _sequence: int = 0
    _accepted_total: int = 0
    _rejected_total: int = 0
    _timed_out_total: int = 0
    _completed_total: int = 0
    _wait_seconds_total: float = 0.0
    _wait_observations_total: int = 0
    _max_queue_wait_seconds: float = 0.0

    @asynccontextmanager
    async def slot(
        self,
        *,
        tenant: str,
        key_id: str,
        resource_key: str,
        workload: str,
        priority: float = 0.0,
        estimated_tokens: int = 0,
        resource_limit: int | None = None,
    ) -> AsyncIterator[SchedulerLease]:
        lease = await self.acquire(
            tenant=tenant,
            key_id=key_id,
            resource_key=resource_key,
            workload=workload,
            priority=priority,
            estimated_tokens=estimated_tokens,
            resource_limit=resource_limit,
        )
        try:
            yield lease
        finally:
            await self.release(lease)

    async def acquire(
        self,
        *,
        tenant: str,
        key_id: str,
        resource_key: str,
        workload: str,
        priority: float = 0.0,
        estimated_tokens: int = 0,
        resource_limit: int | None = None,
    ) -> SchedulerLease:
        if not settings.scheduler_enabled:
            return SchedulerLease(
                lease_id=-1,
                tenant=tenant,
                resource_key=resource_key,
                workload=workload,
                priority=priority,
                estimated_tokens=estimated_tokens,
                wait_ms=0.0,
                queue_depth_at_submit=0,
                tenant_queue_depth_at_submit=0,
                enabled=False,
            )

        loop = asyncio.get_running_loop()
        pending = _Pending(
            tenant=tenant,
            key_id=key_id,
            resource_key=resource_key,
            resource_limit=max(1, resource_limit or settings.scheduler_resource_max_in_flight),
            workload=workload,
            priority=priority,
            estimated_tokens=max(0, estimated_tokens),
            submitted_at=time.perf_counter(),
            sequence=0,
            future=loop.create_future(),
        )

        async with self._lock:
            tenant_queue = self._queues[tenant]
            if len(tenant_queue) >= settings.scheduler_max_queue_per_tenant:
                self._rejected_total += 1
                raise TenantQueueFullError(tenant, len(tenant_queue))

            self._sequence += 1
            pending.sequence = self._sequence
            pending.queue_depth_at_submit = self._total_queued_locked() + 1
            pending.tenant_queue_depth_at_submit = len(tenant_queue) + 1
            tenant_queue.append(pending)
            self._accepted_total += 1
            self._dispatch_locked()

        timeout = (
            settings.scheduler_queue_timeout_seconds
            if settings.scheduler_queue_timeout_seconds > 0
            else None
        )
        try:
            return await asyncio.wait_for(asyncio.shield(pending.future), timeout=timeout)
        except TimeoutError as exc:
            await self._cancel_pending(pending, timed_out=True)
            raise TenantQueueTimeoutError(tenant, settings.scheduler_queue_timeout_seconds) from exc
        except asyncio.CancelledError:
            await self._cancel_pending(pending, timed_out=False)
            raise

    async def release(self, lease: SchedulerLease) -> None:
        if not lease.enabled:
            return
        async with self._lock:
            if lease.lease_id not in self._active_leases:
                return
            self._active_leases.remove(lease.lease_id)
            self._in_flight_total = max(0, self._in_flight_total - 1)
            self._decrement(self._in_flight_by_tenant, lease.tenant)
            self._decrement(self._in_flight_by_resource, lease.resource_key)
            self._completed_total += 1
            self._dispatch_locked()

    def snapshot(self) -> SchedulerSnapshot:
        queued_by_tenant = {
            tenant: len(queue)
            for tenant, queue in self._queues.items()
            if queue
        }
        return SchedulerSnapshot(
            enabled=settings.scheduler_enabled,
            total_queued=sum(queued_by_tenant.values()),
            total_in_flight=self._in_flight_total,
            queued_by_tenant=queued_by_tenant,
            in_flight_by_tenant={
                tenant: count
                for tenant, count in self._in_flight_by_tenant.items()
                if count
            },
            in_flight_by_resource={
                resource: count
                for resource, count in self._in_flight_by_resource.items()
                if count
            },
            accepted_total=self._accepted_total,
            rejected_total=self._rejected_total,
            timed_out_total=self._timed_out_total,
            completed_total=self._completed_total,
            wait_seconds_total=self._wait_seconds_total,
            wait_observations_total=self._wait_observations_total,
            max_queue_wait_seconds=self._max_queue_wait_seconds,
        )

    async def _cancel_pending(self, pending: _Pending, *, timed_out: bool) -> None:
        async with self._lock:
            if pending.started:
                if pending.lease is not None:
                    self._release_started_locked(pending.lease)
                return
            queue = self._queues.get(pending.tenant)
            if queue is not None:
                try:
                    queue.remove(pending)
                except ValueError:
                    pass
                if not queue:
                    self._queues.pop(pending.tenant, None)
            if timed_out:
                self._timed_out_total += 1
            self._dispatch_locked()

    def _release_started_locked(self, lease: SchedulerLease) -> None:
        if lease.lease_id not in self._active_leases:
            return
        self._active_leases.remove(lease.lease_id)
        self._in_flight_total = max(0, self._in_flight_total - 1)
        self._decrement(self._in_flight_by_tenant, lease.tenant)
        self._decrement(self._in_flight_by_resource, lease.resource_key)
        self._dispatch_locked()

    def _dispatch_locked(self) -> None:
        while self._in_flight_total < settings.scheduler_global_max_in_flight:
            pending = self._choose_next_locked()
            if pending is None:
                return
            queue = self._queues[pending.tenant]
            popped = queue.popleft()
            if popped is not pending:
                raise RuntimeError("tenant scheduler queue corruption")
            if not queue:
                self._queues.pop(pending.tenant, None)

            now = time.perf_counter()
            wait_seconds = max(0.0, now - pending.submitted_at)
            lease = SchedulerLease(
                lease_id=pending.sequence,
                tenant=pending.tenant,
                resource_key=pending.resource_key,
                workload=pending.workload,
                priority=pending.priority,
                estimated_tokens=pending.estimated_tokens,
                wait_ms=wait_seconds * 1000.0,
                queue_depth_at_submit=pending.queue_depth_at_submit,
                tenant_queue_depth_at_submit=pending.tenant_queue_depth_at_submit,
            )
            pending.started = True
            pending.lease = lease
            self._active_leases.add(lease.lease_id)
            self._in_flight_total += 1
            self._in_flight_by_tenant[pending.tenant] += 1
            self._in_flight_by_resource[pending.resource_key] += 1
            self._last_dispatch_at[pending.tenant] = now
            self._wait_seconds_total += wait_seconds
            self._wait_observations_total += 1
            self._max_queue_wait_seconds = max(self._max_queue_wait_seconds, wait_seconds)
            if not pending.future.done():
                pending.future.set_result(lease)

    def _choose_next_locked(self) -> _Pending | None:
        now = time.perf_counter()
        candidates = [
            queue[0]
            for tenant, queue in self._queues.items()
            if queue and self._can_start_locked(tenant, queue[0])
        ]
        if not candidates:
            return None

        def score(pending: _Pending) -> tuple[float, int]:
            waited = max(0.0, now - pending.submitted_at)
            if pending.tenant in self._last_dispatch_at:
                tenant_idle = max(0.0, now - self._last_dispatch_at[pending.tenant])
            else:
                tenant_idle = 60.0
            numeric_score = (
                pending.priority
                + waited * settings.scheduler_wait_aging_priority_per_second
                + min(tenant_idle, 60.0) * settings.scheduler_tenant_fairness_weight
            )
            return numeric_score, -pending.sequence

        return max(candidates, key=score)

    def _can_start_locked(self, tenant: str, pending: _Pending) -> bool:
        if self._in_flight_total >= settings.scheduler_global_max_in_flight:
            return False
        if self._in_flight_by_resource[pending.resource_key] >= pending.resource_limit:
            return False
        tenant_running = self._in_flight_by_tenant[tenant]
        if tenant_running < settings.scheduler_tenant_reserved_in_flight:
            return True
        return not self._has_other_waiting_tenants_locked(tenant)

    def _has_other_waiting_tenants_locked(self, tenant: str) -> bool:
        return any(other != tenant and bool(queue) for other, queue in self._queues.items())

    def _total_queued_locked(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    @staticmethod
    def _decrement(counts: dict[str, int], key: str) -> None:
        current = counts.get(key, 0) - 1
        if current > 0:
            counts[key] = current
        else:
            counts.pop(key, None)
