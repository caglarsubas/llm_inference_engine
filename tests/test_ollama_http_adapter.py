from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from inference_engine.adapters.base import (
    ContextLengthExceededError,
    EmbeddingsNotSupportedError,
    GenerationParams,
    UpstreamGenerationError,
)
from inference_engine.adapters.ollama_http import OllamaHttpAdapter
from inference_engine.api.models import _context_lengths
from inference_engine.config import settings
from inference_engine.registry import ModelDescriptor, get_upstream_breaker
from inference_engine.registry.breaker import CLOSED
from inference_engine.schemas import ChatMessage, ToolCall, ToolCallFunction


def _make_descriptor(
    endpoint: str = "http://ollama:11434",
    *,
    context_length: int | None = None,
) -> ModelDescriptor:
    params: dict = {"model_id": "gemma4:31b"}
    if context_length is not None:
        params["context_length"] = context_length
    return ModelDescriptor(
        name="gemma4",
        tag="31b",
        namespace="library",
        registry="registry.ollama.ai",
        model_path=Path(f"ollama_http://{endpoint}/gemma4:31b"),
        format="ollama_http",
        params=params,
        size_bytes=0,
        endpoint=endpoint,
    )


def _install_transport(adapter: OllamaHttpAdapter, handler) -> None:
    assert adapter._client is not None  # noqa: SLF001 - test scaffolding
    adapter._client = httpx.AsyncClient(  # noqa: SLF001
        base_url=adapter._client.base_url,
        transport=httpx.MockTransport(handler),
        timeout=30.0,
    )


def _chat_response(content: str, *, finish_reason: str = "stop") -> dict:
    """A native ``/api/chat`` non-streaming response."""
    return {
        "model": "gemma4:31b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": finish_reason,
        "prompt_eval_count": 7,
        "eval_count": 3,
    }


@pytest.mark.asyncio
async def test_blank_multimodal_json_response_retries_without_hard_json_mode() -> None:
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        if len(captured) == 1:
            return httpx.Response(200, json=_chat_response("", finish_reason="length"))
        return httpx.Response(
            200,
            json=_chat_response(
                '{"vehicle_visible":true,"damage_visible":true,'
                '"anomaly_score":0.9,"confidence":0.98}'
            ),
        )

    adapter = OllamaHttpAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    result = await adapter.generate(
        [
            ChatMessage(role="system", content="Return JSON only."),
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Assess this vehicle photo. Return JSON with keys "
                            "vehicle_visible, damage_visible, anomaly_score, confidence."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc", "detail": "low"},
                    },
                ],
            ),
        ],
        GenerationParams(max_tokens=128, temperature=0.0, json_mode=True),
    )

    assert result.text == (
        '{"vehicle_visible":true,"damage_visible":true,'
        '"anomaly_score":0.9,"confidence":0.98}'
    )
    assert len(captured) == 2
    assert captured[0]["format"] == "json"
    assert captured[0]["options"]["num_predict"] == 128
    assert "format" not in captured[1]
    assert captured[1]["options"]["num_predict"] == 256
    assert captured[1]["messages"][0]["role"] == "system"
    assert "compact valid JSON object" in captured[1]["messages"][0]["content"]
    # The image survived the reprompt, unwrapped out of its data URL.
    assert captured[1]["messages"][-1]["images"] == ["abc"]


@pytest.mark.asyncio
async def test_text_only_blank_json_response_does_not_retry() -> None:
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=_chat_response("", finish_reason="length"))

    adapter = OllamaHttpAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    result = await adapter.generate(
        [ChatMessage(role="user", content="Return JSON with one field.")],
        GenerationParams(max_tokens=128, temperature=0.0, json_mode=True),
    )

    assert result.text == ""
    assert len(captured) == 1


def test_ollama_declares_enforcement_as_a_prior_not_a_guarantee() -> None:
    """Recent Ollama does constrain decoding, so True is the right starting belief.

    An earlier fix asserted False here, which was a blunt instrument: it made
    every Ollama deployment pay a retry, including modern ones that honour the
    schema. Its own docstring said the honest change was to DETECT the
    capability per deployment, which `structured_output_capability` now does —
    so the class-level value goes back to being a claim about the SOFTWARE, and
    the runtime demotes the endpoints where the claim does not hold.
    """
    assert OllamaHttpAdapter.supports_structured_outputs is True


def test_ollama_deployment_id_distinguishes_endpoints() -> None:
    """Two hosts behind one adapter class can be different releases.

    A demotion earned by a stale endpoint must not silence a modern one, so the
    endpoint has to be part of the observation key.
    """
    stale = OllamaHttpAdapter()
    stale._endpoint = "http://old-host:11434"
    modern = OllamaHttpAdapter()
    modern._endpoint = "http://new-host:11434"

    assert stale.deployment_id() != modern.deployment_id()
    assert "old-host" in stale.deployment_id()
    # An adapter that never loaded still yields a stable, non-colliding key.
    assert OllamaHttpAdapter().deployment_id() == "ollama_http:unbound"


def test_ollama_still_sends_the_schema_it_was_given() -> None:
    """Not trusting the backend is not the same as not asking.

    The request must still carry the schema, so a deployment whose Ollama DOES
    honour it gets constrained decoding; the flag above only governs whether
    the gateway re-checks the answer. Native ``/api/chat`` takes the bare
    schema in ``format`` — the same sampler the shim's ``response_format``
    wraps, reached directly.
    """
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    adapter = OllamaHttpAdapter()
    params = GenerationParams(
        json_mode=True,
        json_schema=schema,
        json_schema_name="probe",
        json_schema_strict=True,
    )
    body = adapter._native_body([], params, stream=False)

    assert body["format"] == schema


def test_json_mode_without_a_schema_asks_for_plain_json() -> None:
    body = OllamaHttpAdapter()._native_body([], GenerationParams(json_mode=True), stream=False)
    assert body["format"] == "json"


# ---------------------------------------------------------------------------
# context window — settings.n_ctx reaches the upstream (issue #111)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_goes_to_the_native_endpoint_carrying_num_ctx(monkeypatch) -> None:
    """The whole point of not using the OpenAI shim.

    The shim accepts no runtime options, so a deployment that set N_CTX got
    whatever window Ollama sized from free VRAM — up to 262,144, and 34 GiB of
    KV cache on an 8.9B model. Native /api/chat is the only endpoint that takes
    ``options``.
    """
    monkeypatch.setattr(settings, "n_ctx", 32768)
    seen: list[tuple[str, dict]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, json.loads(req.content)))
        return httpx.Response(200, json=_chat_response("ok"))

    adapter = OllamaHttpAdapter()
    await adapter.load(_make_descriptor(context_length=262144))
    _install_transport(adapter, handler)

    await adapter.generate(
        [ChatMessage(role="user", content="hi")], GenerationParams(max_tokens=16)
    )

    path, body = seen[0]
    assert path == "/api/chat"
    assert body["options"]["num_ctx"] == 32768
    assert body["options"]["num_predict"] == 16


@pytest.mark.asyncio
async def test_num_ctx_is_clamped_to_what_the_model_was_trained_for(monkeypatch) -> None:
    """A ceiling, not a fixed size — same rule the llama.cpp path applies."""
    monkeypatch.setattr(settings, "n_ctx", 32768)
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=_chat_response("ok"))

    adapter = OllamaHttpAdapter()
    await adapter.load(_make_descriptor(context_length=8192))
    _install_transport(adapter, handler)

    await adapter.generate([ChatMessage(role="user", content="hi")], GenerationParams())
    assert seen[0]["options"]["num_ctx"] == 8192


@pytest.mark.asyncio
async def test_an_unprobed_window_leaves_the_ceiling_alone(monkeypatch) -> None:
    """No ``context_length`` means the registry couldn't ask, not "zero"."""
    monkeypatch.setattr(settings, "n_ctx", 16384)
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=_chat_response("ok"))

    adapter = OllamaHttpAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    await adapter.generate([ChatMessage(role="user", content="hi")], GenerationParams())
    assert seen[0]["options"]["num_ctx"] == 16384


@pytest.mark.asyncio
async def test_models_route_reports_the_window_it_will_actually_serve(monkeypatch) -> None:
    """The advertised and effective windows differ silently otherwise."""
    monkeypatch.setattr(settings, "n_ctx", 32768)
    assert _context_lengths(_make_descriptor(context_length=262144)) == (262144, 32768)


# ---------------------------------------------------------------------------
# context overflow — the typed 400 the llama.cpp path already offers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_upstream_overflow_becomes_the_typed_context_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": (
                    "input length (41000 tokens) exceeds the model's maximum "
                    "context length (32768 tokens)"
                )
            },
        )

    adapter = OllamaHttpAdapter()
    await adapter.load(_make_descriptor(context_length=262144))
    _install_transport(adapter, handler)

    with pytest.raises(ContextLengthExceededError) as ei:
        await adapter.generate([ChatMessage(role="user", content="x" * 10)], GenerationParams())

    assert ei.value.requested_tokens == 41000
    assert ei.value.context_window == 32768
    assert ei.value.error_detail()["type"] == "context_length_exceeded"


@pytest.mark.asyncio
async def test_an_overflow_without_counts_still_reports_the_window_we_asked_for(
    monkeypatch,
) -> None:
    """Ollama has several overflow phrasings; only one carries the integers."""
    monkeypatch.setattr(settings, "n_ctx", 32768)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": "input exceeds maximum context length and cannot be truncated further"},
        )

    adapter = OllamaHttpAdapter()
    await adapter.load(_make_descriptor(context_length=262144))
    _install_transport(adapter, handler)

    with pytest.raises(ContextLengthExceededError) as ei:
        await adapter.generate([ChatMessage(role="user", content="x")], GenerationParams())

    assert ei.value.requested_tokens is None
    assert ei.value.context_window == 32768


@pytest.mark.asyncio
async def test_an_oversized_prompt_does_not_open_the_breaker(monkeypatch) -> None:
    """The prompt is the caller's fault; the deployment is fine.

    Counting it would let one oversized caller take a healthy Ollama away
    from every other tenant on it.
    """
    monkeypatch.setattr(settings, "upstream_breaker_failure_threshold", 1)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "the input length exceeds the context length"})

    adapter = OllamaHttpAdapter()
    await adapter.load(_make_descriptor(endpoint="http://overflow-host:11434"))
    _install_transport(adapter, handler)

    for _ in range(3):
        with pytest.raises(ContextLengthExceededError):
            await adapter.generate([ChatMessage(role="user", content="x")], GenerationParams())

    assert get_upstream_breaker().state(adapter.upstream_deployment_key()) == CLOSED


def test_an_unrelated_upstream_error_is_not_mistaken_for_an_overflow() -> None:
    adapter = OllamaHttpAdapter()
    request = httpx.Request("POST", "http://ollama:11434/api/chat")
    exc = httpx.HTTPStatusError(
        "boom",
        request=request,
        response=httpx.Response(500, json={"error": "model runner has stopped"}, request=request),
    )
    assert adapter._as_context_error(exc) is None


# ---------------------------------------------------------------------------
# native <-> OpenAI translation
# ---------------------------------------------------------------------------


def test_tool_calls_are_translated_in_both_directions() -> None:
    """Native carries arguments as an object; the OpenAI contract as a string."""
    adapter = OllamaHttpAdapter()
    sent = adapter._to_messages(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCallFunction(name="lookup", arguments='{"city":"Paris"}'),
                    )
                ],
            ),
            ChatMessage(role="tool", content="sunny", tool_call_id="call_1", name="lookup"),
        ]
    )
    assert sent[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert sent[1]["tool_call_id"] == "call_1"
    assert sent[1]["tool_name"] == "lookup"

    result = OllamaHttpAdapter._result_from_payload(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_x", "function": {"name": "lookup", "arguments": {"city": "Paris"}}}
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 11,
            "eval_count": 4,
        }
    )
    assert result.tool_calls == [
        {
            "id": "call_x",
            "type": "function",
            "index": 0,
            "function": {"name": "lookup", "arguments": '{"city": "Paris"}'},
        }
    ]
    assert (result.prompt_tokens, result.completion_tokens) == (11, 4)
    assert result.finish_reason == "stop"


def test_a_tool_call_without_an_upstream_id_still_gets_one() -> None:
    """Older Ollama omits it, and the agent's reply has to match something."""
    result = OllamaHttpAdapter._result_from_payload(
        {
            "message": {"tool_calls": [{"function": {"name": "f", "arguments": {}}}]},
            "done": True,
        }
    )
    assert result.tool_calls[0]["id"] == "call_0"


@pytest.mark.asyncio
async def test_the_stream_reads_ndjson_and_the_trailing_counts() -> None:
    """Native streaming is newline-delimited JSON with no [DONE] sentinel."""
    frames = [
        {"message": {"role": "assistant", "content": "hi"}, "done": False},
        {"message": {"role": "assistant", "content": " there"}, "done": False},
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 21,
            "eval_count": 5,
        },
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        assert json.loads(req.content)["stream"] is True
        return httpx.Response(
            200,
            content=b"".join(json.dumps(f).encode() + b"\n" for f in frames),
            headers={"content-type": "application/x-ndjson"},
        )

    adapter = OllamaHttpAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    chunks = [
        piece
        async for piece in adapter.stream(
            [ChatMessage(role="user", content="x")], GenerationParams()
        )
    ]

    assert "".join(c.text for c in chunks) == "hi there"
    assert chunks[-1].finish_reason == "stop"
    assert (chunks[-1].prompt_tokens, chunks[-1].completion_tokens) == (21, 5)


@pytest.mark.asyncio
async def test_two_streamed_tool_calls_keep_separate_indices() -> None:
    """The route reassembles streamed calls by index.

    Native Ollama emits a whole call per frame rather than argument fragments,
    so two calls in two frames that both fell back to index 0 would be merged
    into one with both argument payloads concatenated.
    """
    frames = [
        {"message": {"tool_calls": [{"function": {"name": "a", "arguments": {"x": 1}}}]}},
        {"message": {"tool_calls": [{"function": {"name": "b", "arguments": {"y": 2}}}]}},
        {"message": {"content": ""}, "done": True, "done_reason": "stop"},
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"".join(json.dumps(f).encode() + b"\n" for f in frames),
            headers={"content-type": "application/x-ndjson"},
        )

    adapter = OllamaHttpAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    deltas = [
        d
        async for piece in adapter.stream(
            [ChatMessage(role="user", content="x")], GenerationParams()
        )
        for d in (piece.tool_call_deltas or [])
    ]

    assert [d["index"] for d in deltas] == [0, 1]
    assert [d["function"]["name"] for d in deltas] == ["a", "b"]
    assert len({d["id"] for d in deltas}) == 2


@pytest.mark.asyncio
async def test_a_mid_stream_error_object_is_surfaced_not_swallowed() -> None:
    """A 200 response line can still end in an upstream failure."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = (
            json.dumps({"message": {"content": "par"}, "done": False}).encode()
            + b"\n"
            + json.dumps({"error": "model runner has stopped"}).encode()
            + b"\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})

    adapter = OllamaHttpAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)

    with pytest.raises(UpstreamGenerationError):
        async for _ in adapter.stream(
            [ChatMessage(role="user", content="x")], GenerationParams()
        ):
            pass


# ---------------------------------------------------------------------------
# embeddings — served through the shim's /v1/embeddings
# ---------------------------------------------------------------------------


def _embedding_response(vectors: list[list[float]], *, prompt_tokens: int = 7) -> dict:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ],
        "model": "embeddinggemma:300m",
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


async def _loaded_adapter(handler) -> OllamaHttpAdapter:
    adapter = OllamaHttpAdapter()
    await adapter.load(_make_descriptor())
    _install_transport(adapter, handler)
    return adapter


@pytest.mark.asyncio
async def test_embed_posts_to_embeddings_path_and_returns_vectors() -> None:
    seen: list[tuple[str, dict]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, json.loads(req.content)))
        return httpx.Response(200, json=_embedding_response([[0.1, 0.2], [0.3, 0.4]]))

    adapter = await _loaded_adapter(handler)
    result = await adapter.embed(["one", "two"])

    assert [path for path, _ in seen] == ["/v1/embeddings"]
    assert seen[0][1]["input"] == ["one", "two"]
    assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert result.prompt_tokens == 7
    assert adapter.last_embed_action == "upstream"


@pytest.mark.asyncio
async def test_embed_orders_vectors_by_index_not_wire_order() -> None:
    """The contract is one vector per input IN REQUEST ORDER.

    The shim is not obliged to preserve order on the wire, so a response whose
    entries arrive reversed must still map back onto the caller's inputs.
    Taking them as-received would silently mis-pair every vector.
    """

    def handler(_req: httpx.Request) -> httpx.Response:
        payload = _embedding_response([[9.0], [1.0]])
        payload["data"] = list(reversed(payload["data"]))
        return httpx.Response(200, json=payload)

    adapter = await _loaded_adapter(handler)
    result = await adapter.embed(["first", "second"])

    assert result.embeddings == [[9.0], [1.0]]


@pytest.mark.asyncio
async def test_embed_rejects_a_response_with_the_wrong_vector_count() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_embedding_response([[0.1]]))

    adapter = await _loaded_adapter(handler)
    with pytest.raises(UpstreamGenerationError) as excinfo:
        await adapter.embed(["one", "two"])
    assert excinfo.value.error_type == "upstream_contract_error"


@pytest.mark.asyncio
async def test_embed_short_circuits_on_empty_input_without_calling_upstream() -> None:
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        return httpx.Response(200, json=_embedding_response([]))

    adapter = await _loaded_adapter(handler)
    result = await adapter.embed([])

    assert result.embeddings == []
    assert result.prompt_tokens == 0
    assert calls == []


@pytest.mark.asyncio
async def test_embed_maps_upstream_http_error_to_typed_upstream_error() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "not an embedding model"}})

    adapter = await _loaded_adapter(handler)
    with pytest.raises(UpstreamGenerationError) as excinfo:
        await adapter.embed(["one"])
    assert excinfo.value.error_type == "upstream_http_error"
    assert excinfo.value.upstream_status_code == 400


@pytest.mark.asyncio
async def test_upstream_501_becomes_embeddings_not_supported_not_a_server_error() -> None:
    """Ollama answers 501 for a model that is not an embedder at all.

    That is a fact about the model, so it has to reach the route as the typed
    ``EmbeddingsNotSupportedError`` it already renders as HTTP 501 with the
    backend name. Letting it escape as a generic upstream error turned a
    "load a real embedding model" answer into an opaque 500.
    """

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(501, json={"error": {"message": "does not support embeddings"}})

    adapter = await _loaded_adapter(handler)
    with pytest.raises(EmbeddingsNotSupportedError):
        await adapter.embed(["one"])


@pytest.mark.asyncio
async def test_model_level_501_does_not_trip_the_deployment_breaker() -> None:
    """A non-embedding model must not make the whole deployment look unhealthy.

    Chat on this same endpoint is fine; counting the refusal as a health
    failure would take the upstream out of candidate selection for everyone.
    """
    from inference_engine.registry.breaker import OPEN, get_upstream_breaker

    get_upstream_breaker().reset()

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(501, json={"error": {"message": "does not support embeddings"}})

    adapter = await _loaded_adapter(handler)
    key = adapter.upstream_deployment_key()

    for _ in range(5):
        with pytest.raises(EmbeddingsNotSupportedError):
            await adapter.embed(["one"])

    assert get_upstream_breaker().state(key) != OPEN
