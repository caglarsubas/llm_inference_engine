"""Per-adapter token accounting, logprobs, and tokenizer surface.

The chat-route tests use a fake adapter, so these cover the other half: that
each backend actually produces the terminal usage chunk, cached-token count,
and logprob payload the route expects. HTTP backends go through
``httpx.MockTransport``; MLX runs against stubbed ``mlx_lm`` modules.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from inference_engine.adapters.base import GenerationParams
from inference_engine.adapters.ollama_http import OllamaHttpAdapter
from inference_engine.adapters.vllm_adapter import VLLMAdapter
from inference_engine.config import settings
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import ChatMessage


def _sse(events: list[dict]) -> Iterator[bytes]:
    for event in events:
        yield f"data: {json.dumps(event)}\n\n".encode()
    yield b"data: [DONE]\n\n"


def _vllm_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        name="test",
        tag="vllm",
        namespace="vllm",
        registry="local",
        model_path=Path("vllm://vllm:8000/test-model"),
        format="vllm",
        params={"model_id": "test-model"},
        endpoint="http://vllm:8000",
    )


def _install_transport(adapter, handler) -> None:
    adapter._client = httpx.AsyncClient(  # noqa: SLF001 — test scaffolding
        base_url=adapter._client.base_url,  # noqa: SLF001
        transport=httpx.MockTransport(handler),
        timeout=30.0,
    )


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vllm_stream_requests_usage_from_upstream() -> None:
    """Our include_usage support is backed by the upstream's real counts."""
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(
            200,
            content=b"".join(_sse([{"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}])),
            headers={"content-type": "text/event-stream"},
        )

    adapter = VLLMAdapter()
    await adapter.load(_vllm_descriptor())
    _install_transport(adapter, handler)

    async for _ in adapter.stream([ChatMessage(role="user", content="x")], GenerationParams()):
        pass

    assert seen["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_vllm_stream_converts_usage_trailer_to_a_chunk() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = b"".join(
            _sse(
                [
                    {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    # vLLM's include_usage frame: empty choices, usage only.
                    {"choices": [], "usage": {"prompt_tokens": 31, "completion_tokens": 7}},
                ]
            )
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    adapter = VLLMAdapter()
    await adapter.load(_vllm_descriptor())
    _install_transport(adapter, handler)

    chunks = [
        piece
        async for piece in adapter.stream(
            [ChatMessage(role="user", content="x")], GenerationParams()
        )
    ]

    assert chunks[-1].prompt_tokens == 31
    assert chunks[-1].completion_tokens == 7
    assert chunks[-1].text == ""


@pytest.mark.asyncio
async def test_vllm_stream_survives_a_choiceless_frame_without_usage() -> None:
    """Some upstreams emit keep-alive frames with neither choices nor usage."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = b"".join(
            _sse(
                [
                    {"choices": []},
                    {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]},
                ]
            )
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    adapter = VLLMAdapter()
    await adapter.load(_vllm_descriptor())
    _install_transport(adapter, handler)

    chunks = [
        piece
        async for piece in adapter.stream(
            [ChatMessage(role="user", content="x")], GenerationParams()
        )
    ]
    assert "".join(c.text for c in chunks) == "ok"


@pytest.mark.asyncio
async def test_vllm_generate_reads_upstream_cached_tokens() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-x",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 64},
                },
            },
        )

    adapter = VLLMAdapter()
    await adapter.load(_vllm_descriptor())
    _install_transport(adapter, handler)

    result = await adapter.generate([ChatMessage(role="user", content="x")], GenerationParams())
    assert result.cached_tokens == 64


@pytest.mark.asyncio
async def test_vllm_generate_reports_zero_cached_tokens_when_absent() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-x",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    adapter = VLLMAdapter()
    await adapter.load(_vllm_descriptor())
    _install_transport(adapter, handler)

    result = await adapter.generate([ChatMessage(role="user", content="x")], GenerationParams())
    assert result.cached_tokens == 0


@pytest.mark.asyncio
async def test_vllm_generate_passes_logprobs_through() -> None:
    entries = [{"token": "hi", "logprob": -0.1, "bytes": None, "top_logprobs": []}]

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                        "logprobs": {"content": entries},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    adapter = VLLMAdapter()
    await adapter.load(_vllm_descriptor())
    _install_transport(adapter, handler)

    result = await adapter.generate(
        [ChatMessage(role="user", content="x")], GenerationParams(logprobs=True)
    )
    assert result.logprobs == entries


@pytest.mark.asyncio
async def test_vllm_sampling_knobs_reach_the_request_body() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    adapter = VLLMAdapter()
    await adapter.load(_vllm_descriptor())
    _install_transport(adapter, handler)

    await adapter.generate(
        [ChatMessage(role="user", content="x")],
        GenerationParams(
            frequency_penalty=0.3,
            presence_penalty=0.2,
            repetition_penalty=1.05,
            logit_bias={"5": -50.0},
            logprobs=True,
            top_logprobs=2,
            parallel_tool_calls=False,
        ),
    )

    assert seen["frequency_penalty"] == 0.3
    assert seen["presence_penalty"] == 0.2
    assert seen["repetition_penalty"] == 1.05
    assert seen["logit_bias"] == {"5": -50.0}
    assert seen["logprobs"] is True
    assert seen["top_logprobs"] == 2
    assert seen["parallel_tool_calls"] is False


@pytest.mark.asyncio
async def test_vllm_omits_sampling_knobs_the_caller_did_not_set() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    adapter = VLLMAdapter()
    await adapter.load(_vllm_descriptor())
    _install_transport(adapter, handler)

    await adapter.generate([ChatMessage(role="user", content="x")], GenerationParams())

    for absent in (
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "logit_bias",
        "logprobs",
        "parallel_tool_calls",
    ):
        assert absent not in seen, f"{absent} should not be sent when unset"


# ---------------------------------------------------------------------------
# Ollama HTTP
# ---------------------------------------------------------------------------


def _ollama_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        name="test",
        tag="latest",
        namespace="library",
        registry="ollama",
        model_path=Path("ollama://ollama:11434/test"),
        format="ollama_http",
        params={"model_id": "test:latest"},
        endpoint="http://ollama:11434",
    )


@pytest.mark.asyncio
async def test_ollama_stream_requests_and_parses_usage() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        body = b"".join(
            _sse(
                [
                    {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
                ]
            )
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    adapter = OllamaHttpAdapter()
    await adapter.load(_ollama_descriptor())
    _install_transport(adapter, handler)

    chunks = [
        piece
        async for piece in adapter.stream(
            [ChatMessage(role="user", content="x")], GenerationParams()
        )
    ]

    assert seen["stream_options"] == {"include_usage": True}
    assert chunks[-1].prompt_tokens == 12
    assert chunks[-1].completion_tokens == 3


# ---------------------------------------------------------------------------
# MLX — stubbed mlx_lm modules
# ---------------------------------------------------------------------------


class _StubTokenizer:
    model_max_length = 8192

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(f"{m['role'][0]}:{m['content']}" for m in messages)


def _install_mlx(monkeypatch, *, stream_tokens: list[int] | None = None) -> dict:
    record: dict = {"generate_kwargs": [], "stream_kwargs": []}
    tokens = stream_tokens if stream_tokens is not None else [1000, 1001, 1002]

    fake_cache_mod = types.SimpleNamespace(
        make_prompt_cache=lambda model: {},
        trim_prompt_cache=lambda cache, n: None,
        can_trim_prompt_cache=lambda cache: True,
    )
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache_mod)

    def _generate(**kwargs):
        record["generate_kwargs"].append(kwargs)
        return "ok"

    def _stream_generate(**kwargs):
        record["stream_kwargs"].append(kwargs)
        for token in tokens:
            yield types.SimpleNamespace(text="x", token=token, finish_reason=None)

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(
            load=lambda path: (object(), _StubTokenizer()),
            generate=_generate,
            stream_generate=_stream_generate,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.sample_utils",
        types.SimpleNamespace(
            make_sampler=lambda **kw: "sampler",
            make_logits_processors=lambda **kw: ["processor"],
        ),
    )
    return record


def _mlx_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        name="m",
        tag="mlx",
        namespace="ns",
        registry="reg",
        model_path=Path("/tmp/m"),
        format="mlx",
        size_bytes=1024,
    )


@pytest.fixture
def mlx_kit(monkeypatch):
    record = _install_mlx(monkeypatch)
    monkeypatch.setattr(settings, "mlx_prefix_cache_enabled", True)
    from inference_engine.adapters.mlx_lm import MLXAdapter  # noqa: PLC0415

    return MLXAdapter(), record


@pytest.mark.asyncio
async def test_mlx_stream_emits_exact_terminal_token_counts(mlx_kit) -> None:
    adapter, _ = mlx_kit
    await adapter.load(_mlx_descriptor())

    chunks = [
        piece
        async for piece in adapter.stream(
            [ChatMessage(role="user", content="abc")], GenerationParams()
        )
    ]

    usage = chunks[-1]
    assert usage.text == ""
    # The stub yields three token ids, and the prompt is the templated string.
    assert usage.completion_tokens == 3
    assert usage.prompt_tokens == len(adapter._tokenize(adapter._format_prompt(  # noqa: SLF001
        [ChatMessage(role="user", content="abc")]
    )))


@pytest.mark.asyncio
async def test_mlx_reports_cached_tokens_when_a_slot_is_trimmed(mlx_kit, monkeypatch) -> None:
    """At capacity, a repeat prompt trims the best slot and reuses its prefix."""
    adapter, _ = mlx_kit
    monkeypatch.setattr(settings, "mlx_prefix_cache_max_slots", 1)
    await adapter.load(_mlx_descriptor())
    messages = [ChatMessage(role="user", content="abcdef")]

    first = await adapter.generate(messages, GenerationParams())
    assert first.cached_tokens == 0

    second = await adapter.generate(messages, GenerationParams())
    assert second.cached_tokens > 0
    assert second.cached_tokens <= second.prompt_tokens


@pytest.mark.asyncio
async def test_mlx_reports_zero_cached_tokens_when_it_allocates_a_fresh_slot(
    mlx_kit, monkeypatch
) -> None:
    """With spare capacity the adapter deliberately starts a new slot on partial
    overlap rather than trimming, so that call really does pay full prefill —
    reporting 0 is the honest answer, not a missed measurement.

    (The prior call's slot holds prompt *plus* generated tokens, which is why a
    repeated prompt is a partial rather than full match.)
    """
    adapter, _ = mlx_kit
    monkeypatch.setattr(settings, "mlx_prefix_cache_max_slots", 4)
    await adapter.load(_mlx_descriptor())
    messages = [ChatMessage(role="user", content="abcdef")]

    await adapter.generate(messages, GenerationParams())
    second = await adapter.generate(messages, GenerationParams())

    assert second.cached_tokens == 0
    assert adapter.prefix_cache_last_action == "miss"


@pytest.mark.asyncio
async def test_mlx_wires_logits_processors_when_a_penalty_is_set(mlx_kit) -> None:
    adapter, record = mlx_kit
    await adapter.load(_mlx_descriptor())

    await adapter.generate(
        [ChatMessage(role="user", content="hi")],
        GenerationParams(repetition_penalty=1.2),
    )
    assert record["generate_kwargs"][-1]["logits_processors"] == ["processor"]


@pytest.mark.asyncio
async def test_mlx_omits_logits_processors_when_no_penalty_is_set(mlx_kit) -> None:
    adapter, record = mlx_kit
    await adapter.load(_mlx_descriptor())

    await adapter.generate([ChatMessage(role="user", content="hi")], GenerationParams())
    assert "logits_processors" not in record["generate_kwargs"][-1]


@pytest.mark.asyncio
async def test_mlx_tokenizer_surface_round_trips(mlx_kit) -> None:
    adapter, _ = mlx_kit
    await adapter.load(_mlx_descriptor())

    tokens = adapter.tokenize("hey")
    assert tokens == [ord(c) for c in "hey"]
    assert adapter.detokenize(tokens) == "hey"
    assert adapter.max_model_len == 8192


@pytest.mark.asyncio
async def test_mlx_format_chat_prompt_applies_the_template(mlx_kit) -> None:
    adapter, _ = mlx_kit
    await adapter.load(_mlx_descriptor())

    rendered = adapter.format_chat_prompt([ChatMessage(role="user", content="hi")])
    assert rendered == "u:hi"


def test_mlx_tokenizer_requires_a_loaded_model(mlx_kit) -> None:
    adapter, _ = mlx_kit
    with pytest.raises(RuntimeError, match="model not loaded"):
        adapter.tokenize("hey")
    assert adapter.max_model_len is None


@pytest.mark.asyncio
async def test_mlx_survives_an_older_sample_utils_without_logits_processors(
    monkeypatch,
) -> None:
    """Older mlx-lm has no make_logits_processors — skip, don't fail the call."""
    _install_mlx(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.sample_utils",
        types.SimpleNamespace(make_sampler=lambda **kw: "sampler"),
    )
    from inference_engine.adapters.mlx_lm import MLXAdapter  # noqa: PLC0415

    adapter = MLXAdapter()
    await adapter.load(_mlx_descriptor())
    result = await adapter.generate(
        [ChatMessage(role="user", content="hi")], GenerationParams(repetition_penalty=1.2)
    )
    assert result.text == "ok"
