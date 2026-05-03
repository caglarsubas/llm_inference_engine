"""Integration test for chat.py streaming cancellation wire.

Drives ``_stream_response`` with a fake adapter so we exercise the full
disconnect → watchdog → cancel.cancel() → adapter.break-out → span attrs path
without depending on real model speed (which is too fast on the M5 Max for a
real timing race).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

import pytest

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.api.chat import _stream_response
from inference_engine.auth import Identity
from inference_engine.cancellation import Cancellation
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import ChatMessage


class _SlowAdapter(InferenceAdapter):
    """Yields a chunk every 50 ms, breaks early when ``cancel`` is set."""

    backend_name = "fake-slow"

    def __init__(self) -> None:
        self.received_cancel: Cancellation | None = None

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return None

    async def load(self, descriptor: ModelDescriptor) -> None: ...
    async def unload(self) -> None: ...

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> GenerationResult:
        return GenerationResult(text="", finish_reason="stop", prompt_tokens=0, completion_tokens=0)

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        self.received_cancel = cancel
        for i in range(100):
            if cancel is not None and bool(cancel):
                # Consumer/watchdog tripped the flag — stop emitting.
                break
            await asyncio.sleep(0.05)
            yield StreamChunk(text=f"tok{i} ")
        yield StreamChunk(text="", finish_reason="stop")


@dataclass
class _FakeRequest:
    """Quacks like FastAPI Request for watch_disconnect."""

    drop_after_seconds: float
    _start: float = 0.0

    async def is_disconnected(self) -> bool:
        if self._start == 0.0:
            self._start = asyncio.get_event_loop().time()
        return asyncio.get_event_loop().time() - self._start >= self.drop_after_seconds


@pytest.mark.asyncio
async def test_stream_cancellation_propagates_from_disconnect_to_adapter() -> None:
    """Client 'drops' after 0.2s; verify the slow adapter sees the cancel and the run records it."""
    adapter = _SlowAdapter()
    request = _FakeRequest(drop_after_seconds=0.2)
    identity = Identity(tenant="dev", key_id="sk-x")
    messages = [ChatMessage(role="user", content="hi")]

    chunks_received = 0
    async for _ in _stream_response(
        adapter=adapter,
        model_name="fake:1",
        messages=messages,
        params=GenerationParams(),
        identity=identity,
        request=request,
    ):
        chunks_received += 1

    assert adapter.received_cancel is not None
    assert adapter.received_cancel.cancelled, "adapter should have observed the cancel"
    assert adapter.received_cancel.reason == "client_disconnect"
    # We yielded the role=assistant chunk + at least a few content chunks before the drop.
    assert chunks_received >= 2
    # The "[DONE]" trailer should have been *suppressed* because we cancelled.
    # _stream_response emits role + N content + final + [DONE] on a clean close;
    # on cancellation it returns early after the content chunks.
    # We can't easily assert exact count — just bound it to a reasonable window.
    assert chunks_received < 50, "should not have streamed full 100 chunks"
