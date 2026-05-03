"""Concurrency contracts: lock granularity, dedup, parallel cold loads.

These tests use deterministic timing-based fakes so we can assert ordering
properties (overlap vs. serialise) without depending on real model latency.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.cancellation import Cancellation, watch_disconnect
from inference_engine.manager import ModelManager
from inference_engine.registry import ModelDescriptor


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _desc(name: str, *, size: int = 10) -> ModelDescriptor:
    return ModelDescriptor(
        name=name,
        tag="1",
        namespace="ns",
        registry="reg",
        model_path=Path(f"/tmp/{name}"),
        format="gguf",
        size_bytes=size,
    )


@dataclass
class _Registry:
    items: dict[str, ModelDescriptor]

    def get(self, k: str) -> ModelDescriptor | None:
        return self.items.get(k)

    def list_models(self) -> list[ModelDescriptor]:
        return list(self.items.values())


class _TimedAdapter(InferenceAdapter):
    """Adapter that records load/generate enter+exit timestamps."""

    backend_name = "timed"
    # Class-level counters for cross-instance accounting.
    instances_created = 0
    load_calls: list[str] = []

    def __init__(self, load_seconds: float = 0.0, gen_seconds: float = 0.0) -> None:
        _TimedAdapter.instances_created += 1
        self.load_seconds = load_seconds
        self.gen_seconds = gen_seconds
        self._descriptor: ModelDescriptor | None = None
        self._lock = asyncio.Lock()
        self.gen_windows: list[tuple[float, float]] = []  # (entered, exited)

    @property
    def is_loaded(self) -> bool:
        return self._descriptor is not None

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return self._descriptor

    async def load(self, descriptor: ModelDescriptor) -> None:
        _TimedAdapter.load_calls.append(descriptor.qualified_name)
        await asyncio.sleep(self.load_seconds)
        self._descriptor = descriptor

    async def unload(self) -> None:
        self._descriptor = None

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> GenerationResult:
        async with self._lock:
            entered = time.perf_counter()
            await asyncio.sleep(self.gen_seconds)
            exited = time.perf_counter()
            self.gen_windows.append((entered, exited))
        return GenerationResult(text="ok", finish_reason="stop", prompt_tokens=1, completion_tokens=1)

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="x", finish_reason="stop")


@pytest.fixture(autouse=True)
def _reset_class_counters() -> None:
    _TimedAdapter.instances_created = 0
    _TimedAdapter.load_calls = []


def _windows_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


# ---------------------------------------------------------------------------
# ModelManager — load dedup + parallel-load + cache-hit-during-load
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_get_same_model_dedupes_load() -> None:
    """N coros asking for the same unloaded model must trigger exactly one load."""
    desc = _desc("alpha")
    factory_calls: list[ModelDescriptor] = []

    def factory(d: ModelDescriptor) -> InferenceAdapter:
        factory_calls.append(d)
        return _TimedAdapter(load_seconds=0.05)

    mgr = ModelManager(_Registry({"alpha:1": desc}), factory, memory_budget_bytes=100)

    # Fire 10 concurrent gets for the same key.
    results = await asyncio.gather(*(mgr.get("alpha:1") for _ in range(10)))

    # Same adapter object returned to every caller.
    assert all(r[0] is results[0][0] for r in results)
    # The factory was called exactly once.
    assert len(factory_calls) == 1
    assert _TimedAdapter.load_calls == ["alpha:1"]


@pytest.mark.asyncio
async def test_concurrent_loads_for_different_models_run_in_parallel() -> None:
    """Two cold loads for different models must overlap, not serialise."""
    a = _desc("a")
    b = _desc("b")

    load_starts: dict[str, float] = {}
    load_ends: dict[str, float] = {}

    class _RecordingAdapter(_TimedAdapter):
        async def load(self, descriptor: ModelDescriptor) -> None:
            load_starts[descriptor.qualified_name] = time.perf_counter()
            await asyncio.sleep(0.1)
            load_ends[descriptor.qualified_name] = time.perf_counter()
            self._descriptor = descriptor

    mgr = ModelManager(
        _Registry({"a:1": a, "b:1": b}),
        adapter_factory=lambda d: _RecordingAdapter(),
        memory_budget_bytes=1000,
    )

    started = time.perf_counter()
    await asyncio.gather(mgr.get("a:1"), mgr.get("b:1"))
    total = time.perf_counter() - started

    # Serial would be ~0.20s; parallel is ~0.10s + a few ms of overhead.
    assert total < 0.18, f"loads should run in parallel; took {total:.3f}s"
    assert _windows_overlap(
        (load_starts["a:1"], load_ends["a:1"]),
        (load_starts["b:1"], load_ends["b:1"]),
    )


@pytest.mark.asyncio
async def test_cache_hit_does_not_block_on_concurrent_cold_load() -> None:
    """While model A is mid-load, get(B) on an already-loaded B must return immediately."""
    a = _desc("a")
    b = _desc("b")
    mgr = ModelManager(
        _Registry({"a:1": a, "b:1": b}),
        adapter_factory=lambda d: _TimedAdapter(load_seconds=0.2),  # slow load
        memory_budget_bytes=1000,
    )

    # Pre-warm B so it's a cache hit.
    await mgr.get("b:1")
    assert "b:1" in mgr.loaded_models()

    # Kick off a slow A load, then race a B cache hit against it.
    a_task = asyncio.create_task(mgr.get("a:1"))
    await asyncio.sleep(0.01)  # let a's load start

    t0 = time.perf_counter()
    await mgr.get("b:1")
    cache_hit_latency = time.perf_counter() - t0

    # If the manager held a global lock across A's load, this would be ~0.2s.
    assert cache_hit_latency < 0.05, f"cache hit blocked on cold load: {cache_hit_latency:.3f}s"

    await a_task  # cleanup


# ---------------------------------------------------------------------------
# Adapter — same-instance generate calls must serialise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_generate_on_same_adapter_serializes() -> None:
    """Two generate() calls on the same adapter must not overlap (llama_cpp.Llama isn't thread-safe)."""
    adapter = _TimedAdapter(gen_seconds=0.1)
    adapter._descriptor = _desc("x")

    await asyncio.gather(
        adapter.generate([], GenerationParams()),
        adapter.generate([], GenerationParams()),
    )

    assert len(adapter.gen_windows) == 2
    a, b = sorted(adapter.gen_windows, key=lambda w: w[0])
    assert not _windows_overlap(a, b), f"generate calls overlapped: {a} vs {b}"
    # And the second started no earlier than the first ended.
    assert b[0] >= a[1] - 1e-3


# ---------------------------------------------------------------------------
# watch_disconnect — many concurrent contexts cleaned up cleanly
# ---------------------------------------------------------------------------


@dataclass
class _AlwaysConnected:
    polls: int = 0

    async def is_disconnected(self) -> bool:
        self.polls += 1
        return False


@pytest.mark.asyncio
async def test_many_concurrent_watchdogs_are_reaped() -> None:
    """50 watch_disconnect contexts come and go — none of them leak as pending tasks."""
    loop = asyncio.get_running_loop()
    initial_tasks = len([t for t in asyncio.all_tasks(loop) if not t.done()])

    async def one() -> None:
        async with watch_disconnect(_AlwaysConnected(), poll_interval=0.005):
            await asyncio.sleep(0.02)

    await asyncio.gather(*(one() for _ in range(50)))
    # Give any laggards a chance to settle.
    await asyncio.sleep(0.05)

    final_tasks = len([t for t in asyncio.all_tasks(loop) if not t.done()])
    # The currently-running test task is the only expected live one.
    assert final_tasks <= initial_tasks + 1, (
        f"watchdog leaked tasks: started={initial_tasks} ended={final_tasks}"
    )
