"""Generation-timeout mapping for chat routes and HTTP-backed adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult, GenerationTimeoutError
from inference_engine.adapters.ollama_http import OllamaHttpAdapter
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
