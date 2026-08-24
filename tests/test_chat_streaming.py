"""Integration tests for the chat.py streaming wire.

Both drive ``_stream_response`` with a fake adapter so the real generator is in
the path without depending on model speed (too fast on the M5 Max for a real
timing race):

* cancellation — the full disconnect → watchdog → cancel.cancel() →
  adapter.break-out → span attrs path.
* reasoning deltas — that a thinking model's tokens reach the client labelled
  and *while* they are produced, which is what lets a caller draw a live
  progress row without ever rendering the chain-of-thought.
"""

from __future__ import annotations

import asyncio
import json
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


class _ReasoningPreludeAdapter(InferenceAdapter):
    """Emits what a Nemotron-family chat template actually puts on the wire.

    The template opens ``<think>`` itself, before the model starts sampling, so
    the opening tag never reaches the engine — the reasoning text simply *is*
    the first thing that arrives, and only the closing tag marks where the
    answer begins. This is the shape Ollama's OpenAI shim forwards today:
    everything lands in ``delta.content`` upstream and the engine is the layer
    that splits it.
    """

    backend_name = "fake-reasoning"

    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces

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
        for piece in self._pieces:
            yield StreamChunk(text=piece)
        yield StreamChunk(text="", finish_reason="stop")


_THINKING_PIECES = [
    "Checking ",
    "the pipeline ",
    "metrics. ",
    "</think>",
    "Recall ",
    "is 0.82.",
]


async def _collect_deltas(model_name: str) -> list[dict]:
    """Drive ``_stream_response`` and return each chunk's ``delta`` object."""
    adapter = _ReasoningPreludeAdapter(list(_THINKING_PIECES))
    deltas = []
    async for event in _stream_response(
        adapter=adapter,
        model_name=model_name,
        messages=[ChatMessage(role="user", content="how did it score?")],
        params=GenerationParams(),
        identity=Identity(tenant="dev", key_id="sk-x"),
        request=_FakeRequest(drop_after_seconds=60.0),
    ):
        payload = event.get("data")
        if not payload or payload == "[DONE]":
            continue
        for choice in json.loads(payload).get("choices", []):
            deltas.append(choice.get("delta") or {})
    return deltas


@pytest.mark.asyncio
async def test_reasoning_tokens_stream_as_labelled_deltas_throughout() -> None:
    """Reasoning is a live channel on the streaming path, not an end-of-turn field.

    A client can render a "thinking" row while it fills, and can render it
    *without* ever showing the chain-of-thought, because the reasoning text
    arrives on its own key. Both properties are contract for DeclarAI's
    progress rows; this pins them.
    """
    deltas = await _collect_deltas("nemotron-3-nano:30b")

    reasoning = [d["reasoning_content"] for d in deltas if d.get("reasoning_content")]
    content = [d["content"] for d in deltas if d.get("content")]

    # Arrives incrementally. A single delta would mean the whole block was held
    # to the end, which is the behaviour this test exists to forbid.
    assert len(reasoning) > 1, reasoning
    assert "".join(reasoning) == "Checking the pipeline metrics. "
    assert "".join(content) == "Recall is 0.82."

    # The channels never cross: no answer text labelled as reasoning, and no
    # vendor markup on either.
    assert not any("</think>" in text or "<think>" in text for text in reasoning + content)


@pytest.mark.asyncio
async def test_reasoning_labelling_rests_on_the_model_name_heuristic() -> None:
    """The same upstream bytes are labelled by ``infer_model_capabilities`` alone.

    Nothing in ``StreamChunk`` carries a reasoning channel, so the split is the
    engine's own parse of a pre-opened ``<think>``, keyed off the model id. Pin
    the dependency: a backend that starts stripping thinking into a field of
    its own, or a reasoning model whose name misses ``_REASONING_MARKERS``,
    silently delivers chain-of-thought as answer text — and this is the test
    that says so.
    """
    deltas = await _collect_deltas("llama3.2:1b")

    content = "".join(d["content"] for d in deltas if d.get("content"))
    assert not [d for d in deltas if d.get("reasoning_content")]
    assert content.startswith("Checking the pipeline metrics.")
