"""Streaming tool-call audit — reassembler unit tests + chat-stream integration."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

import pytest

from inference_engine.adapters import (
    GenerationParams,
    InferenceAdapter,
    StreamChunk,
)
from inference_engine.adapters.base import GenerationResult
from inference_engine.api._tool_audit import ToolCallReassembler
from inference_engine.api.chat import _stream_response
from inference_engine.auth import Identity
from inference_engine.cancellation import Cancellation
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import ChatMessage


# ---------------------------------------------------------------------------
# ToolCallReassembler — merge OpenAI-style streamed deltas
# ---------------------------------------------------------------------------


def test_reassembler_merges_arguments_across_chunks() -> None:
    r = ToolCallReassembler()
    # Chunk 1 — id + name + start of arguments
    r.feed([{"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city"'}}])
    # Chunk 2 — argument continuation
    r.feed([{"index": 0, "function": {"arguments": ': "SF"}'}}])

    [out] = r.assembled()
    assert out["id"] == "call_1"
    assert out["type"] == "function"
    assert out["function"]["name"] == "get_weather"
    assert out["function"]["arguments"] == '{"city": "SF"}'


def test_reassembler_handles_multiple_parallel_calls() -> None:
    r = ToolCallReassembler()
    r.feed([
        {"index": 0, "id": "call_a", "type": "function",
         "function": {"name": "f1", "arguments": ""}},
        {"index": 1, "id": "call_b", "type": "function",
         "function": {"name": "f2", "arguments": ""}},
    ])
    r.feed([{"index": 0, "function": {"arguments": '{"x": 1}'}}])
    r.feed([{"index": 1, "function": {"arguments": '{"y": 2}'}}])

    out = r.assembled()
    assert len(out) == 2
    assert out[0]["id"] == "call_a"
    assert out[0]["function"]["arguments"] == '{"x": 1}'
    assert out[1]["id"] == "call_b"
    assert out[1]["function"]["arguments"] == '{"y": 2}'


def test_reassembler_no_calls_when_only_text() -> None:
    r = ToolCallReassembler()
    r.feed(None)
    r.feed([])
    assert not r.has_calls()
    assert r.assembled() == []


def test_reassembler_orders_by_index_not_arrival() -> None:
    """If chunks arrive in unusual order, output is still sorted by index."""
    r = ToolCallReassembler()
    r.feed([{"index": 1, "id": "call_b", "function": {"name": "f2", "arguments": ""}}])
    r.feed([{"index": 0, "id": "call_a", "function": {"name": "f1", "arguments": ""}}])
    out = r.assembled()
    assert [c["id"] for c in out] == ["call_a", "call_b"]


def test_reassembler_tolerates_missing_function_block() -> None:
    """Some chunks legitimately have no function field at all."""
    r = ToolCallReassembler()
    r.feed([{"index": 0, "id": "call_x", "type": "function",
             "function": {"name": "f", "arguments": ""}}])
    r.feed([{"index": 0}])  # heartbeat-style chunk
    r.feed([{"index": 0, "function": {"arguments": "{}"}}])
    [out] = r.assembled()
    assert out["function"]["arguments"] == "{}"


# ---------------------------------------------------------------------------
# Stream integration — fake adapter feeds canned deltas, observe spans + SSE
# ---------------------------------------------------------------------------


@dataclass
class _FakeRequest:
    async def is_disconnected(self) -> bool:
        return False


class _ToolStreamingAdapter(InferenceAdapter):
    """Yields a canned sequence: a text intro, then 3 tool-call delta chunks."""

    backend_name = "fake-tools"

    def __init__(self) -> None:
        self._descriptor: ModelDescriptor | None = None

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return self._descriptor

    async def load(self, descriptor: ModelDescriptor) -> None:
        return None

    async def unload(self) -> None:
        return None

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> GenerationResult:
        return GenerationResult(text="", finish_reason="stop", prompt_tokens=0, completion_tokens=0)

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        # Brief text preamble.
        yield StreamChunk(text="Calling get_weather... ")
        # First delta — id + name + start of arguments.
        yield StreamChunk(
            text="",
            tool_call_deltas=[{
                "index": 0, "id": "call_xyz", "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city"'},
            }],
        )
        # Second delta — argument continuation.
        yield StreamChunk(
            text="",
            tool_call_deltas=[{"index": 0, "function": {"arguments": ': "SF"}'}}],
        )
        # Final chunk — finish_reason flips to tool_calls.
        yield StreamChunk(text="", finish_reason="tool_calls")


@pytest.mark.asyncio
async def test_stream_emits_tool_call_event_at_end(_session_exporter) -> None:
    """A streamed response with tool_calls emits a single gen_ai.tool_call
    span event at end-of-stream with the reassembled arguments."""
    _session_exporter.clear()
    adapter = _ToolStreamingAdapter()

    chunks: list[dict] = []
    async for chunk in _stream_response(
        adapter=adapter,
        model_name="fake-tools:1",
        messages=[ChatMessage(role="user", content="weather?")],
        params=GenerationParams(),
        identity=Identity(tenant="dev", key_id="sk-x"),
        request=_FakeRequest(),
    ):
        chunks.append(chunk)

    spans = [s for s in _session_exporter.get_finished_spans() if s.name == "chat.stream"]
    assert len(spans) == 1
    chat_span = spans[0]

    # Aggregate count bound on span.
    assert chat_span.attributes["tool_audit.tool_calls_out"] == 1

    # One reassembled tool_call event with full arguments.
    tool_events = [e for e in chat_span.events if e.name == "gen_ai.tool_call"]
    assert len(tool_events) == 1
    e = tool_events[0]
    assert e.attributes["gen_ai.tool.call.id"] == "call_xyz"
    assert e.attributes["gen_ai.tool.name"] == "get_weather"
    assert e.attributes["gen_ai.tool.call.arguments"] == '{"city": "SF"}'


@pytest.mark.asyncio
async def test_stream_passes_tool_call_deltas_through_to_sse() -> None:
    """SSE clients see every tool-call delta in OpenAI wire format."""
    adapter = _ToolStreamingAdapter()
    chunks: list[dict] = []
    async for chunk in _stream_response(
        adapter=adapter,
        model_name="fake-tools:1",
        messages=[ChatMessage(role="user", content="weather?")],
        params=GenerationParams(),
        identity=Identity(tenant="dev", key_id="sk-x"),
        request=_FakeRequest(),
    ):
        chunks.append(chunk)

    # Pull the data: payloads off the SSE-shaped chunks (skip the trailing [DONE]).
    payloads: list[dict] = []
    import json
    for c in chunks:
        if c.get("data") == "[DONE]":
            continue
        payloads.append(json.loads(c["data"]))

    # Find the chunks that carried tool_calls deltas.
    tool_chunks = [
        p for p in payloads
        if p["choices"][0].get("delta", {}).get("tool_calls")
    ]
    assert len(tool_chunks) == 2

    # First delta has the id + name; second only has argument continuation.
    deltas_1 = tool_chunks[0]["choices"][0]["delta"]["tool_calls"]
    deltas_2 = tool_chunks[1]["choices"][0]["delta"]["tool_calls"]
    assert deltas_1[0]["id"] == "call_xyz"
    assert deltas_1[0]["function"]["name"] == "get_weather"
    assert deltas_1[0]["function"]["arguments"] == '{"city"'
    # Second chunk's id/name fields are None (only arguments fragment).
    assert deltas_2[0].get("id") is None
    assert deltas_2[0]["function"]["arguments"] == ': "SF"}'


@pytest.mark.asyncio
async def test_stream_no_tool_calls_emits_nothing(_session_exporter) -> None:
    """A pure-text stream emits no gen_ai.tool_* events."""
    _session_exporter.clear()

    class _PlainAdapter(_ToolStreamingAdapter):
        async def stream(self, messages, params, cancel=None):
            yield StreamChunk(text="hi there")
            yield StreamChunk(text="", finish_reason="stop")

    async for _ in _stream_response(
        adapter=_PlainAdapter(),
        model_name="x:1",
        messages=[ChatMessage(role="user", content="hi")],
        params=GenerationParams(),
        identity=Identity(tenant="dev", key_id="sk-x"),
        request=_FakeRequest(),
    ):
        pass

    span_obj = next(
        s for s in _session_exporter.get_finished_spans() if s.name == "chat.stream"
    )
    assert span_obj.attributes["tool_audit.tool_calls_out"] == 0
    assert not any(e.name.startswith("gen_ai.tool_") for e in span_obj.events)
