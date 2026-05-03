"""VLLMRegistry + VLLMAdapter — config parse + HTTP client behaviour.

Adapter tests use ``httpx.MockTransport`` so we exercise the real HTTP code
path (request/response shaping, SSE parsing, error mapping, cancellation
loop) without standing up a vLLM server. The transport is a thin function
that returns canned responses keyed by URL — same way real vLLM would.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from inference_engine.adapters import EmbeddingsNotSupportedError
from inference_engine.adapters.base import GenerationParams
from inference_engine.adapters.vllm_adapter import VLLMAdapter
from inference_engine.cancellation import Cancellation
from inference_engine.registry import ModelDescriptor, VLLMRegistry
from inference_engine.schemas import (
    ChatMessage,
    ToolCall,
    ToolCallFunction,
)


# ---------------------------------------------------------------------------
# Registry — config parsing
# ---------------------------------------------------------------------------


def test_registry_missing_file_returns_empty(tmp_path: Path) -> None:
    reg = VLLMRegistry(tmp_path / "ghost.json")
    assert reg.list_models() == []
    assert reg.get("anything") is None


def test_registry_parses_minimal_entry(tmp_path: Path) -> None:
    path = tmp_path / "vllm.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "llama-3.2-1b-instruct",
                    "endpoint": "http://vllm:8000",
                    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                }
            ]
        )
    )

    reg = VLLMRegistry(path)
    descs = reg.list_models()
    assert len(descs) == 1
    d = descs[0]
    assert d.format == "vllm"
    assert d.qualified_name == "llama-3.2-1b-instruct:vllm"
    assert d.endpoint == "http://vllm:8000"
    assert d.params["model_id"] == "meta-llama/Llama-3.2-1B-Instruct"


def test_registry_strips_trailing_slash_on_endpoint(tmp_path: Path) -> None:
    """Trailing slash on endpoint causes httpx to build paths like ``//v1/...``;
    normalise it at parse time so adapters don't need to defensively trim."""
    path = tmp_path / "vllm.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "x",
                    "endpoint": "http://vllm:8000/",
                    "model_id": "any",
                }
            ]
        )
    )
    reg = VLLMRegistry(path)
    assert reg.list_models()[0].endpoint == "http://vllm:8000"


def test_registry_rejects_non_array_top_level(tmp_path: Path) -> None:
    path = tmp_path / "vllm.json"
    path.write_text(json.dumps({"models": []}))
    reg = VLLMRegistry(path)
    with pytest.raises(ValueError, match="JSON array"):
        reg.list_models()


def test_registry_rejects_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "vllm.json"
    path.write_text(json.dumps([{"name": "x"}]))  # endpoint + model_id missing
    reg = VLLMRegistry(path)
    with pytest.raises(ValueError, match="missing required field"):
        reg.list_models()


def test_registry_get_resolves_short_name(tmp_path: Path) -> None:
    path = tmp_path / "vllm.json"
    path.write_text(
        json.dumps(
            [{"name": "alpha", "endpoint": "http://x:8000", "model_id": "ax"}]
        )
    )
    reg = VLLMRegistry(path)
    assert reg.get("alpha:vllm") is not None
    assert reg.get("alpha") is not None
    assert reg.get("nope") is None


# ---------------------------------------------------------------------------
# Adapter — HTTP behaviour against MockTransport
# ---------------------------------------------------------------------------


def _make_descriptor(endpoint: str = "http://vllm:8000", model_id: str = "test-model") -> ModelDescriptor:
    return ModelDescriptor(
        name="test",
        tag="vllm",
        namespace="vllm",
        registry="local",
        model_path=Path(f"vllm://{endpoint}/{model_id}"),
        format="vllm",
        params={"model_id": model_id},
        size_bytes=0,
        endpoint=endpoint,
    )


def _install_transport(adapter: VLLMAdapter, handler) -> None:
    """Replace the adapter's httpx client with one that routes to ``handler``.

    Used after ``adapter.load(...)`` so we keep the descriptor + endpoint state
    the adapter set up but swap the transport.
    """
    assert adapter._client is not None  # noqa: SLF001 — test scaffolding
    adapter._client = httpx.AsyncClient(  # noqa: SLF001
        base_url=adapter._client.base_url,
        transport=httpx.MockTransport(handler),
        timeout=30.0,
    )


def _ok_chat_response(*, content: str = "hi", finish: str = "stop", tool_calls=None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["content"] = None
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {"index": 0, "message": msg, "finish_reason": finish}
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


@pytest.mark.asyncio
async def test_load_rejects_wrong_format() -> None:
    adapter = VLLMAdapter()
    bad = _make_descriptor()
    bad = ModelDescriptor(
        name=bad.name, tag=bad.tag, namespace=bad.namespace, registry=bad.registry,
        model_path=bad.model_path, format="gguf", params=bad.params,
        size_bytes=bad.size_bytes, endpoint=bad.endpoint,
    )
    with pytest.raises(ValueError, match="vllm"):
        await adapter.load(bad)


@pytest.mark.asyncio
async def test_load_rejects_missing_endpoint() -> None:
    desc = _make_descriptor()
    desc = ModelDescriptor(
        name=desc.name, tag=desc.tag, namespace=desc.namespace, registry=desc.registry,
        model_path=desc.model_path, format="vllm", params=desc.params,
        size_bytes=desc.size_bytes, endpoint=None,
    )
    with pytest.raises(ValueError, match="missing endpoint"):
        await VLLMAdapter().load(desc)


@pytest.mark.asyncio
async def test_load_rejects_missing_model_id() -> None:
    desc = _make_descriptor()
    desc = ModelDescriptor(
        name=desc.name, tag=desc.tag, namespace=desc.namespace, registry=desc.registry,
        model_path=desc.model_path, format="vllm", params={},
        size_bytes=desc.size_bytes, endpoint=desc.endpoint,
    )
    with pytest.raises(ValueError, match="model_id"):
        await VLLMAdapter().load(desc)


@pytest.mark.asyncio
async def test_generate_round_trip() -> None:
    """The adapter posts to /v1/chat/completions with the right model id and
    decodes the response into a GenerationResult."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        assert req.url.path == "/v1/chat/completions"
        return httpx.Response(200, json=_ok_chat_response(content="hello world"))

    adapter = VLLMAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    result = await adapter.generate(
        [ChatMessage(role="user", content="hi")], GenerationParams(max_tokens=8)
    )
    assert result.text == "hello world"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3
    assert captured[0]["model"] == "test-model"
    assert captured[0]["stream"] is False
    assert captured[0]["max_tokens"] == 8


@pytest.mark.asyncio
async def test_generate_passes_tool_calls_through() -> None:
    """A request with tool_calls in the assistant message should round-trip
    cleanly (vLLM reads them just like OpenAI does)."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json=_ok_chat_response(
                tool_calls=[
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                    }
                ]
            ),
        )

    adapter = VLLMAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    msgs = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_prev",
                    function=ToolCallFunction(name="search", arguments="{}"),
                )
            ],
        ),
        ChatMessage(role="tool", tool_call_id="call_prev", content="ok"),
    ]
    result = await adapter.generate(msgs, GenerationParams())
    assert result.tool_calls is not None
    assert result.tool_calls[0]["id"] == "call_x"
    # The outbound message included the prior assistant tool_calls block.
    sent = captured[0]
    sent_assistant = next(m for m in sent["messages"] if m["role"] == "assistant")
    assert sent_assistant["tool_calls"][0]["id"] == "call_prev"


def _sse(events: list[dict]) -> Iterator[bytes]:
    """Build an SSE stream body from a list of OpenAI-shaped event dicts."""
    for e in events:
        yield f"data: {json.dumps(e)}\n\n".encode()
    yield b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_yields_text_and_finish() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = b"".join(
            _sse(
                [
                    {"choices": [{"index": 0, "delta": {"content": "hello "}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {"content": "world"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
            )
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    adapter = VLLMAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    chunks = []
    async for piece in adapter.stream([ChatMessage(role="user", content="x")], GenerationParams()):
        chunks.append(piece)

    text = "".join(c.text for c in chunks)
    assert text == "hello world"
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_stream_yields_tool_call_deltas() -> None:
    """vLLM streams tool_calls as per-index fragments — the adapter relays
    them so the chat-stream reassembler (round 20) can reconstruct."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = b"".join(
            _sse(
                [
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_xyz",
                                            "type": "function",
                                            "function": {"name": "get_weather", "arguments": ""},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": '{"city":"SF"}'}}
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    },
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                ]
            )
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    adapter = VLLMAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    deltas: list[dict] = []
    finish = None
    async for piece in adapter.stream([ChatMessage(role="user", content="x")], GenerationParams()):
        if piece.tool_call_deltas:
            deltas.extend(piece.tool_call_deltas)
        if piece.finish_reason:
            finish = piece.finish_reason

    assert finish == "tool_calls"
    assert len(deltas) == 2
    assert deltas[0]["id"] == "call_xyz"
    assert deltas[1]["function"]["arguments"] == '{"city":"SF"}'


@pytest.mark.asyncio
async def test_stream_stops_on_cancel() -> None:
    """Setting the cancel flag mid-stream stops yielding new chunks."""

    def handler(req: httpx.Request) -> httpx.Response:
        # 100 chunks — way more than we'll consume.
        body = b"".join(
            _sse(
                [
                    {"choices": [{"index": 0, "delta": {"content": f"t{i}"}, "finish_reason": None}]}
                    for i in range(100)
                ]
                + [{"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
            )
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    adapter = VLLMAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    cancel = Cancellation()
    n = 0
    async for _ in adapter.stream(
        [ChatMessage(role="user", content="x")], GenerationParams(), cancel=cancel
    ):
        n += 1
        if n >= 5:
            cancel.cancel(reason="test")

    # We requested ~100 events but cancelled after 5; we shouldn't see all of them.
    assert n < 50, f"cancel didn't stop the stream loop (got {n} chunks)"


@pytest.mark.asyncio
async def test_complete_round_trip() -> None:
    """/v1/completions hits a different vLLM endpoint and skips chat templating."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append({"path": req.url.path, "body": json.loads(req.content)})
        return httpx.Response(
            200,
            json={
                "id": "cmpl-x",
                "object": "text_completion",
                "model": "test-model",
                "choices": [{"text": " Paris.", "index": 0, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        )

    adapter = VLLMAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    result = await adapter.complete(
        "The capital of France is", GenerationParams(max_tokens=4)
    )
    assert result.text == " Paris."
    assert result.finish_reason == "length"
    assert captured[0]["path"] == "/v1/completions"
    assert captured[0]["body"]["prompt"] == "The capital of France is"


@pytest.mark.asyncio
async def test_embed_raises_not_supported() -> None:
    """vLLM-served models report 501 for /v1/embeddings — same shape as MLX."""
    adapter = VLLMAdapter()
    await adapter.load(_make_descriptor())
    with pytest.raises(EmbeddingsNotSupportedError):
        await adapter.embed(["x"])


@pytest.mark.asyncio
async def test_unload_closes_client_and_clears_state() -> None:
    adapter = VLLMAdapter()
    await adapter.load(_make_descriptor())
    assert adapter.is_loaded

    await adapter.unload()
    assert not adapter.is_loaded
    assert adapter.loaded_model is None
    assert adapter._client is None  # noqa: SLF001 — test introspection
    # Prefix-cache surface still answers (returns disabled), no exception.
    assert adapter.prefix_cache_enabled is False
    assert adapter.prefix_cache_last_action == "disabled"


@pytest.mark.asyncio
async def test_http_error_propagates_as_httpstatuserror() -> None:
    """A 500 from vLLM bubbles up so the chat route can map it to the client."""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "out of capacity"})

    adapter = VLLMAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    with pytest.raises(httpx.HTTPStatusError):
        await adapter.generate([ChatMessage(role="user", content="x")], GenerationParams())
