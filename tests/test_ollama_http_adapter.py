from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from inference_engine.adapters.base import GenerationParams
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
