"""VLLMRegistry + VLLMAdapter — config parse + HTTP client behaviour.

Adapter tests use ``httpx.MockTransport`` so we exercise the real HTTP code
path (request/response shaping, SSE parsing, error mapping, cancellation
loop) without standing up a vLLM server. The transport is a thin function
that returns canned responses keyed by URL — same way real vLLM would.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from inference_engine.adapters import (
    EmbeddingsNotSupportedError,
    GenerationTimeoutError,
    UpstreamGenerationError,
)
from inference_engine.adapters.base import GenerationParams
from inference_engine.adapters.vllm_adapter import VLLMAdapter
from inference_engine.cancellation import Cancellation
from inference_engine.config import settings
from inference_engine.registry import ModelDescriptor, VLLMRegistry
from inference_engine.schemas import (
    ChatMessage,
    ToolCall,
    ToolCallFunction,
)


ROOT = Path(__file__).resolve().parents[1]


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


def test_registry_parses_chat_template_kwargs(tmp_path: Path) -> None:
    path = tmp_path / "vllm.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "minicpm-v-4.5-gguf-q4-k-m",
                    "tag": "dmr",
                    "endpoint": "http://127.0.0.1:12434/engines",
                    "model_id": "docker.io/local/minicpm-v-4.5-gguf:q4_k_m",
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            ]
        )
    )

    desc = VLLMRegistry(path).list_models()[0]

    assert desc.qualified_name == "minicpm-v-4.5-gguf-q4-k-m:dmr"
    assert desc.params["chat_template_kwargs"] == {"enable_thinking": False}


def test_registry_parses_vlm_benchmark_metadata(tmp_path: Path) -> None:
    path = tmp_path / "vllm.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "glm-4.1v-9b-thinking",
                    "endpoint": "http://vllm-glm-4-1v-9b-thinking:8000",
                    "model_id": "zai-org/GLM-4.1V-9B-Thinking",
                    "family": "GLM-V",
                    "profile": "vision-reasoning",
                    "parameter_count_b": 9,
                    "open_weight": True,
                    "proprietary": False,
                    "commercial_use": "MIT; verify provider terms before production use",
                    "modality": "text+image->text",
                    "supports_json_mode": True,
                    "supports_strict_image_json": False,
                    "strict_image_json_status": "pending_smoke",
                    "strict_image_json_checked_at": "2026-06-19",
                    "strict_image_json_detail": "not yet smoke validated",
                    "benchmark_only": True,
                }
            ]
        )
    )

    desc = VLLMRegistry(path).list_models()[0]

    assert desc.qualified_name == "glm-4.1v-9b-thinking:vllm"
    assert desc.params["provider"] == "vllm"
    assert desc.params["modality"] == "text+image->text"
    assert desc.params["supports_json_mode"] is True
    assert desc.params["supports_strict_image_json"] is False
    assert desc.params["strict_image_json_status"] == "pending_smoke"
    assert desc.params["strict_image_json_checked_at"] == "2026-06-19"
    assert desc.params["strict_image_json_detail"] == "not yet smoke validated"
    assert desc.params["commercial_use"] == "MIT; verify provider terms before production use"
    assert desc.params["benchmark_only"] is True
    assert desc.params["parameter_count_b"] == 9
    assert desc.params["open_weight"] is True
    assert desc.params["proprietary"] is False


def test_demanded_vlm_manifest_covers_fraudguard_issue_40_candidates() -> None:
    descs = VLLMRegistry(ROOT / ".vllm_models.demanded.example.json").list_models()
    by_id = {d.qualified_name: d for d in descs}
    requested = {
        "qwen3-vl-8b-instruct:vllm": "Qwen/Qwen3-VL-8B-Instruct",
        "qwen3-vl-32b-instruct:vllm": "Qwen/Qwen3-VL-32B-Instruct",
        "internvl3.5-8b:vllm": "OpenGVLab/InternVL3_5-8B",
        "internvl3.5-14b:vllm": "OpenGVLab/InternVL3_5-14B",
        "glm-4.1v-9b-thinking:vllm": "zai-org/GLM-4.1V-9B-Thinking",
        "kimi-vl-a3b-thinking:vllm": "moonshotai/Kimi-VL-A3B-Thinking",
        "ovis2.5-9b:vllm": "AIDC-AI/Ovis2.5-9B",
        "molmo-7b-d:vllm": "allenai/Molmo-7B-D-0924",
        "fakeshield-22b:vllm": "zhipeixu/fakeshield-v1-22b",
        "sida-7b:vllm": "saberzl/SIDA-7B",
        "sida-13b:vllm": "saberzl/SIDA-13B",
    }

    assert requested.keys() <= by_id.keys()
    for engine_id, upstream_id in requested.items():
        desc = by_id[engine_id]
        assert desc.params["model_id"] == upstream_id
        assert "image" in desc.params["modality"]
        assert desc.params["supports_json_mode"] is True
        assert desc.params["supports_strict_image_json"] is False
        assert desc.params["strict_image_json_status"] == "pending_smoke"
        assert desc.params["strict_image_json_checked_at"] == "2026-06-19"
        assert desc.params["commercial_use"]


def test_fakeshield_issue_43_live_fixture_has_required_metadata() -> None:
    descs = VLLMRegistry(ROOT / ".vllm_models.fakeshield.example.json").list_models()
    assert len(descs) == 1
    desc = descs[0]

    assert desc.qualified_name == "fakeshield-22b:vllm"
    assert desc.endpoint == "http://vllm-fakeshield-22b:8000"
    assert desc.params["model_id"] == "zhipeixu/fakeshield-v1-22b"
    assert desc.params["family"] == "FakeShield"
    assert desc.params["profile"] == "forensics"
    assert desc.params["modality"] == "text+image->text"
    assert desc.params["supports_json_mode"] is True
    assert desc.params["supports_strict_image_json"] is False
    assert desc.params["strict_image_json_status"] == "pending_smoke"
    assert desc.params["strict_image_json_checked_at"] == "2026-06-20"
    assert "Issue #43" in desc.params["strict_image_json_detail"]
    assert desc.params["commercial_use"].startswith("Apache-2.0")
    assert desc.params["benchmark_only"] is True


def test_registry_reports_demanded_models_missing_from_live_config(tmp_path: Path) -> None:
    live = tmp_path / "vllm.json"
    live.write_text(
        json.dumps(
            [
                {
                    "name": "qwen3-vl-8b-instruct",
                    "endpoint": "http://vllm-qwen3-vl-8b:8000",
                    "model_id": "Qwen/Qwen3-VL-8B-Instruct",
                }
            ]
        )
    )

    skipped = VLLMRegistry(
        live,
        demanded_config_path=ROOT / ".vllm_models.demanded.example.json",
    ).list_skipped()
    skipped_by_id = {skip.qualified_name: skip for skip in skipped}

    assert "qwen3-vl-8b-instruct:vllm" not in skipped_by_id
    assert "qwen3-vl-32b-instruct:vllm" in skipped_by_id
    assert skipped_by_id["qwen3-vl-32b-instruct:vllm"].reason == (
        "demanded_not_configured"
    )
    assert "Qwen/Qwen3-VL-32B-Instruct" in skipped_by_id[
        "qwen3-vl-32b-instruct:vllm"
    ].detail


def test_registry_rejects_bad_chat_template_kwargs(tmp_path: Path) -> None:
    path = tmp_path / "vllm.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "x",
                    "endpoint": "http://vllm:8000",
                    "model_id": "any",
                    "chat_template_kwargs": "enable_thinking=false",
                }
            ]
        )
    )

    with pytest.raises(ValueError, match="chat_template_kwargs"):
        VLLMRegistry(path).list_models()


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
async def test_generate_passes_multimodal_content_parts_through() -> None:
    """VLM callers use OpenAI content parts; vLLM should receive them unchanged."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=_ok_chat_response(content='{"vehicle_visible":true}'))

    adapter = VLLMAdapter()
    await adapter.load(_make_descriptor(model_id="Qwen/Qwen3-VL-8B-Instruct"))
    _install_transport(adapter, handler)

    await adapter.generate(
        [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Inspect this vehicle photo and return JSON."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ==",
                            "detail": "low",
                        },
                    },
                ],
            )
        ],
        GenerationParams(max_tokens=64, temperature=0.0, json_mode=True),
    )

    assert captured[0]["model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[0]["messages"][0]["content"] == [
        {"type": "text", "text": "Inspect this vehicle photo and return JSON."},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ==",
                "detail": "low",
            },
        },
    ]


@pytest.mark.asyncio
async def test_generate_merges_descriptor_and_request_chat_template_kwargs() -> None:
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=_ok_chat_response(content="hello"))

    adapter = VLLMAdapter()
    desc = _make_descriptor()
    desc.params["chat_template_kwargs"] = {"enable_thinking": False, "foo": "model"}
    await adapter.load(desc)
    _install_transport(adapter, handler)

    await adapter.generate(
        [ChatMessage(role="user", content="hi")],
        GenerationParams(
            max_tokens=8,
            chat_template_kwargs={"foo": "request", "bar": True},
        ),
    )

    assert captured[0]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "foo": "request",
        "bar": True,
    }


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


@pytest.mark.asyncio
async def test_generate_timeout_raises_typed_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("upstream was too slow", request=req)

    adapter = VLLMAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    with pytest.raises(GenerationTimeoutError) as ei:
        await adapter.generate([ChatMessage(role="user", content="x")], GenerationParams())

    assert ei.value.backend == "vllm"
    assert ei.value.model == "test-model"
    assert ei.value.error_detail()["type"] == "generation_timeout"


@pytest.mark.asyncio
async def test_generate_wall_clock_deadline_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "chat_completion_timeout_seconds", 0.01)

    async def handler(req: httpx.Request) -> httpx.Response:  # noqa: ARG001
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=_ok_chat_response(content="too late"))

    adapter = VLLMAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    with pytest.raises(GenerationTimeoutError) as ei:
        await adapter.generate([ChatMessage(role="user", content="x")], GenerationParams())

    assert ei.value.timeout_seconds == 0.01
    assert ei.value.backend == "vllm"
    assert ei.value.model == "test-model"


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
async def test_http_error_maps_to_typed_upstream_error() -> None:
    """A 500 from vLLM keeps enough detail for the route/client payload."""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "out of capacity"})

    adapter = VLLMAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    with pytest.raises(UpstreamGenerationError) as ei:
        await adapter.generate([ChatMessage(role="user", content="x")], GenerationParams())

    assert ei.value.error_type == "upstream_http_error"
    assert ei.value.upstream_status_code == 500
    assert ei.value.backend == "vllm"
    assert ei.value.model == "test-model"
    assert "out of capacity" in ei.value.detail
