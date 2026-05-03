"""Cancellation flag mechanics + watch_disconnect watchdog."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

import pytest

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.cancellation import Cancellation, watch_disconnect
from inference_engine.registry import ModelDescriptor


# ---------------------------------------------------------------------------
# Cancellation primitive
# ---------------------------------------------------------------------------


def test_initial_state_is_not_cancelled() -> None:
    c = Cancellation()
    assert not c
    assert not c.cancelled
    assert c.reason is None


def test_cancel_sets_flag_and_reason() -> None:
    c = Cancellation()
    c.cancel(reason="client_disconnect")
    assert c
    assert c.cancelled
    assert c.reason == "client_disconnect"


def test_double_cancel_keeps_first_reason() -> None:
    c = Cancellation()
    c.cancel(reason="first")
    c.cancel(reason="second")
    assert c.reason == "first"


# ---------------------------------------------------------------------------
# watch_disconnect — watchdog around a fake Request.is_disconnected()
# ---------------------------------------------------------------------------


@dataclass
class _FakeRequest:
    drop_after_polls: int = 1
    _polls: int = 0

    async def is_disconnected(self) -> bool:
        self._polls += 1
        return self._polls > self.drop_after_polls


@pytest.mark.asyncio
async def test_watchdog_trips_on_disconnect() -> None:
    req = _FakeRequest(drop_after_polls=2)
    async with watch_disconnect(req, poll_interval=0.01) as cancel:
        # Wait long enough for the watchdog to poll and fire.
        for _ in range(50):
            if cancel:
                break
            await asyncio.sleep(0.01)
        assert cancel
        assert cancel.reason == "client_disconnect"


@pytest.mark.asyncio
async def test_watchdog_does_not_fire_when_client_stays() -> None:
    req = _FakeRequest(drop_after_polls=10_000)  # effectively never
    async with watch_disconnect(req, poll_interval=0.01) as cancel:
        await asyncio.sleep(0.05)
        assert not cancel


@pytest.mark.asyncio
async def test_watchdog_is_cancelled_on_exit() -> None:
    """Leaving the context manager must reap the polling task."""

    class _SlowReq:
        async def is_disconnected(self) -> bool:
            await asyncio.sleep(10)  # would block forever if not cancelled
            return False

    async with watch_disconnect(_SlowReq(), poll_interval=0.01) as cancel:
        assert not cancel
    # If the watchdog wasn't cancelled, the test would hang here on shutdown.


# ---------------------------------------------------------------------------
# Adapter signature contract — both adapters must accept a `cancel=` kwarg.
# We don't load real models; we just confirm the override resolves correctly.
# ---------------------------------------------------------------------------


class _SignatureCheckAdapter(InferenceAdapter):
    """Concrete adapter that records whether cancel was passed through."""

    backend_name = "sig-check"

    def __init__(self) -> None:
        self.last_cancel: Cancellation | None = None

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return None

    async def load(self, descriptor: ModelDescriptor) -> None:
        return None

    async def unload(self) -> None:
        return None

    async def generate(
        self,
        messages: Iterable,
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> GenerationResult:
        self.last_cancel = cancel
        return GenerationResult(text="", finish_reason="stop", prompt_tokens=0, completion_tokens=0)

    async def stream(
        self,
        messages: Iterable,
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.last_cancel = cancel
        yield StreamChunk(text="x", finish_reason="stop")


@pytest.mark.asyncio
async def test_adapter_receives_cancel_object() -> None:
    adapter = _SignatureCheckAdapter()
    cancel = Cancellation()
    await adapter.generate([], GenerationParams(), cancel=cancel)
    assert adapter.last_cancel is cancel

    async for _ in adapter.stream([], GenerationParams(), cancel=cancel):
        pass
    assert adapter.last_cancel is cancel
