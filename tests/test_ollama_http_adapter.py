from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from inference_engine.adapters.base import (
    EmbeddingsNotSupportedError,
    GenerationParams,
    UpstreamGenerationError,
)
from inference_engine.adapters.ollama_http import OllamaHttpAdapter
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import ChatMessage


def _make_descriptor(endpoint: str = "http://ollama:11434") -> ModelDescriptor:
    return ModelDescriptor(
        name="gemma4",
        tag="31b",
        namespace="library",
        registry="registry.ollama.ai",
        model_path=Path(f"ollama_http://{endpoint}/gemma4:31b"),
        format="ollama_http",
        params={"model_id": "gemma4:31b"},
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
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
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
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[0]["max_tokens"] == 128
    assert "response_format" not in captured[1]
    assert captured[1]["max_tokens"] == 256
    assert captured[1]["messages"][0]["role"] == "system"
    assert "compact valid JSON object" in captured[1]["messages"][0]["content"]


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

    The request must still carry `response_format`, so a deployment whose
    Ollama DOES honour it gets constrained decoding; the flag above only
    governs whether the gateway re-checks the answer.
    """
    adapter = OllamaHttpAdapter()
    params = GenerationParams(
        json_mode=True,
        json_schema={"type": "object", "properties": {"a": {"type": "integer"}}},
        json_schema_name="probe",
        json_schema_strict=True,
    )
    kwargs = adapter._completion_kwargs(params)

    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "probe"
    assert kwargs["response_format"]["json_schema"]["strict"] is True


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
