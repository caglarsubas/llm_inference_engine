"""Generation-timeout mapping for chat routes and HTTP-backed adapters."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult, GenerationTimeoutError
from inference_engine.adapters.ollama_http import OllamaHttpAdapter
from inference_engine.api.state import app_state
from inference_engine.config import settings
from inference_engine.manager import ModelNotFoundError
from inference_engine.api.chat import _blocking_response, _stream_response
from inference_engine.auth import Identity
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import ChatMessage


class _TimeoutAdapter(InferenceAdapter):
    backend_name = "fake-timeout"

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return None

    async def load(self, descriptor: ModelDescriptor) -> None: ...
    async def unload(self) -> None: ...

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel=None
    ) -> GenerationResult:
        raise GenerationTimeoutError(
            timeout_seconds=12.5,
            backend=self.backend_name,
            model="gemma4:26b",
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel=None
    ) -> AsyncIterator[StreamChunk]:
        raise GenerationTimeoutError(
            timeout_seconds=12.5,
            backend=self.backend_name,
            model="gemma4:26b",
        )
        yield  # pragma: no cover - marks this as an async generator


class _LocalTimeoutAdapter(_TimeoutAdapter):
    backend_name = "llama_cpp"


class _LocalErrorAdapter(_TimeoutAdapter):
    backend_name = "vllm"

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel=None
    ) -> GenerationResult:
        raise RuntimeError("upstream died")


class _OpenRouterFallbackAdapter(_TimeoutAdapter):
    backend_name = "openrouter"
    request_key_source = "openrouter-api-key"

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel=None
    ) -> GenerationResult:
        return GenerationResult(
            text="answered by OpenRouter",
            finish_reason="stop",
            prompt_tokens=7,
            completion_tokens=5,
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel=None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="streamed by OpenRouter", finish_reason="stop")


def _openrouter_descriptor(name: str = "gemma4:openrouter") -> ModelDescriptor:
    return ModelDescriptor(
        name=name.split(":")[0],
        tag=name.split(":")[1],
        namespace="openrouter",
        registry="openrouter",
        model_path=Path(f"openrouter://{name}"),
        format="openrouter",
        params={"request_key_source": "openrouter-api-key"},
        size_bytes=0,
    )


def _install_openrouter_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openrouter_fallback_enabled", True)
    monkeypatch.setattr(settings, "openrouter_fallback_model", "")
    monkeypatch.setattr(settings, "openrouter_fallback_backends", "llama_cpp,ollama_http,vllm")

    fallback_adapter = _OpenRouterFallbackAdapter()

    async def _fake_get(model_id: str):
        if model_id == "gemma4:openrouter":
            return fallback_adapter, _openrouter_descriptor(model_id)
        raise ModelNotFoundError(model_id)

    monkeypatch.setattr(app_state.manager, "get", _fake_get)


@pytest.mark.asyncio
async def test_blocking_timeout_maps_to_typed_504() -> None:
    identity = Identity(tenant="dev", key_id="sk-x")
    with pytest.raises(HTTPException) as ei:
        await _blocking_response(
            _TimeoutAdapter(),
            "gemma4:26b",
            [ChatMessage(role="user", content="score this answer")],
            GenerationParams(),
            identity,
        )

    assert ei.value.status_code == 504
    assert ei.value.detail["type"] == "generation_timeout"
    assert ei.value.detail["timeout_seconds"] == 12.5
    assert ei.value.detail["backend"] == "fake-timeout"
    assert ei.value.detail["model"] == "gemma4:26b"


@pytest.mark.asyncio
async def test_blocking_timeout_uses_openrouter_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_openrouter_fallback(monkeypatch)
    identity = Identity(tenant="dev", key_id="sk-x")

    response = await _blocking_response(
        _LocalTimeoutAdapter(),
        "gemma4:26b",
        [ChatMessage(role="user", content="score this answer")],
        GenerationParams(),
        identity,
    )

    assert response.model == "gemma4:openrouter"
    assert response.request_key_source == "openrouter-api-key"
    assert response.fallback_from_model == "gemma4:26b"
    assert response.fallback_from_backend == "llama_cpp"
    assert response.fallback_reason == "generation_timeout"
    assert response.fallback_error_type == "GenerationTimeoutError"
    assert response.choices[0].message.content == "answered by OpenRouter"


@pytest.mark.asyncio
async def test_blocking_backend_error_uses_openrouter_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_openrouter_fallback(monkeypatch)
    identity = Identity(tenant="dev", key_id="sk-x")

    response = await _blocking_response(
        _LocalErrorAdapter(),
        "gemma4:26b",
        [ChatMessage(role="user", content="score this answer")],
        GenerationParams(),
        identity,
    )

    assert response.model == "gemma4:openrouter"
    assert response.request_key_source == "openrouter-api-key"
    assert response.fallback_from_backend == "vllm"
    assert response.fallback_reason == "backend_error"
    assert response.fallback_error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_streaming_timeout_emits_terminal_error_event() -> None:
    identity = Identity(tenant="dev", key_id="sk-x")

    class _Req:
        async def is_disconnected(self) -> bool:
            return False

    events = [
        chunk
        async for chunk in _stream_response(
            _TimeoutAdapter(),
            "gemma4:26b",
            [ChatMessage(role="user", content="score this answer")],
            GenerationParams(),
            identity,
            _Req(),
        )
    ]

    error_events = [e for e in events if e.get("event") == "error"]
    assert len(error_events) == 1
    assert "generation_timeout" in error_events[0]["data"]
    assert not any(e.get("data") == "[DONE]" for e in events)


@pytest.mark.asyncio
async def test_streaming_timeout_uses_openrouter_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_openrouter_fallback(monkeypatch)
    identity = Identity(tenant="dev", key_id="sk-x")

    class _Req:
        async def is_disconnected(self) -> bool:
            return False

    events = [
        chunk
        async for chunk in _stream_response(
            _LocalTimeoutAdapter(),
            "gemma4:26b",
            [ChatMessage(role="user", content="score this answer")],
            GenerationParams(),
            identity,
            _Req(),
        )
    ]

    assert not any(e.get("event") == "error" for e in events)
    assert events[-1]["data"] == "[DONE]"
    payloads = [
        json.loads(e["data"])
        for e in events
        if e.get("data") and e.get("data") != "[DONE]"
    ]
    assert payloads
    assert all(p["model"] == "gemma4:openrouter" for p in payloads)
    assert all(p["request_key_source"] == "openrouter-api-key" for p in payloads)
    assert payloads[0]["fallback_from_model"] == "gemma4:26b"
    assert payloads[0]["fallback_from_backend"] == "llama_cpp"
    assert any(
        choice["delta"].get("content") == "streamed by OpenRouter"
        for payload in payloads
        for choice in payload["choices"]
    )


def _ollama_descriptor(endpoint: str = "http://ollama:11434") -> ModelDescriptor:
    return ModelDescriptor(
        name="gemma4",
        tag="26b",
        namespace="ollama",
        registry="ollama_http",
        model_path=Path(f"ollama_http://{endpoint}/gemma4:26b"),
        format="ollama_http",
        params={"model_id": "gemma4:26b"},
        size_bytes=0,
        endpoint=endpoint,
    )


def _install_timeout_transport(adapter: OllamaHttpAdapter) -> None:
    assert adapter._client is not None  # noqa: SLF001 - test scaffolding

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("upstream was too slow", request=req)

    adapter._client = httpx.AsyncClient(  # noqa: SLF001
        base_url=adapter._client.base_url,
        transport=httpx.MockTransport(handler),
        timeout=30.0,
    )


@pytest.mark.asyncio
async def test_ollama_generate_timeout_raises_typed_error() -> None:
    adapter = OllamaHttpAdapter()
    await adapter.load(_ollama_descriptor())
    _install_timeout_transport(adapter)

    with pytest.raises(GenerationTimeoutError) as ei:
        await adapter.generate([ChatMessage(role="user", content="judge this")], GenerationParams())

    assert ei.value.backend == "ollama_http"
    assert ei.value.model == "gemma4:26b"
    assert ei.value.error_detail()["type"] == "generation_timeout"


@pytest.mark.asyncio
async def test_ollama_stream_timeout_raises_typed_error() -> None:
    adapter = OllamaHttpAdapter()
    await adapter.load(_ollama_descriptor())
    _install_timeout_transport(adapter)

    with pytest.raises(GenerationTimeoutError):
        async for _ in adapter.stream(
            [ChatMessage(role="user", content="judge this")], GenerationParams()
        ):
            pass
