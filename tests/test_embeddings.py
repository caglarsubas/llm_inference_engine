"""Embeddings — schema, adapter contract, route 200/501, span attrs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference_engine.adapters import (
    EmbeddingResult,
    EmbeddingsNotSupportedError,
    GenerationParams,
    InferenceAdapter,
    StreamChunk,
)
from inference_engine.adapters.base import GenerationResult
from inference_engine.cancellation import Cancellation
from inference_engine.main import app
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import EmbeddingRequest


# ---------------------------------------------------------------------------
# Schema acceptance
# ---------------------------------------------------------------------------


def test_request_accepts_single_string_input() -> None:
    req = EmbeddingRequest(model="x", input="hello")
    assert req.input == "hello"


def test_request_accepts_list_of_strings() -> None:
    req = EmbeddingRequest(model="x", input=["a", "b", "c"])
    assert req.input == ["a", "b", "c"]


def test_request_defaults_to_float_encoding() -> None:
    req = EmbeddingRequest(model="x", input="hi")
    assert req.encoding_format == "float"


# ---------------------------------------------------------------------------
# Adapter contract — default is "not supported", overrideable per-backend
# ---------------------------------------------------------------------------


class _NoEmbedAdapter(InferenceAdapter):
    backend_name = "no-embed"

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return None

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
        yield StreamChunk(text="", finish_reason="stop")


@pytest.mark.asyncio
async def test_default_embed_raises_not_supported() -> None:
    adapter = _NoEmbedAdapter()
    with pytest.raises(EmbeddingsNotSupportedError):
        await adapter.embed(["hi"])


# ---------------------------------------------------------------------------
# Route — uses TestClient with a stubbed manager
# ---------------------------------------------------------------------------


class _StubEmbedAdapter(InferenceAdapter):
    backend_name = "stub-embed"

    def __init__(self, vectors: list[list[float]] | None = None, fail: bool = False) -> None:
        self._vectors = vectors or [[0.1, 0.2, 0.3]]
        self._fail = fail
        self._descriptor: ModelDescriptor | None = None

    @property
    def is_loaded(self) -> bool:
        return self._descriptor is not None

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return self._descriptor

    async def load(self, descriptor: ModelDescriptor) -> None:
        self._descriptor = descriptor

    async def unload(self) -> None:
        self._descriptor = None

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> GenerationResult:
        return GenerationResult(text="", finish_reason="stop", prompt_tokens=0, completion_tokens=0)

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="", finish_reason="stop")

    async def embed(self, inputs: list[str]) -> EmbeddingResult:
        if self._fail:
            raise EmbeddingsNotSupportedError(self.backend_name)
        # Repeat the canned vector once per input so batch ordering is
        # observable without making this fixture too clever.
        out = [list(self._vectors[0]) for _ in inputs]
        return EmbeddingResult(embeddings=out, prompt_tokens=len(inputs) * 3)


def _stub_descriptor(name: str = "stub-embed:1") -> ModelDescriptor:
    return ModelDescriptor(
        name=name.split(":")[0],
        tag=name.split(":")[1] if ":" in name else "1",
        namespace="ns",
        registry="reg",
        model_path=Path(f"/tmp/{name}"),
        format="gguf",
        size_bytes=1024,
    )


@pytest.fixture
def patched_manager(monkeypatch):
    """Replace ModelManager.get with a stub that returns whatever adapter the
    test installs via ``patched_manager.set_adapter(...)``."""

    class _Box:
        adapter: InferenceAdapter | None = None

    box = _Box()

    async def _fake_get(model_id: str):
        if box.adapter is None:
            from inference_engine.manager import ModelNotFoundError  # noqa: PLC0415

            raise ModelNotFoundError(model_id)
        desc = _stub_descriptor(model_id)
        return box.adapter, desc

    from inference_engine.api.state import app_state  # noqa: PLC0415

    monkeypatch.setattr(app_state.manager, "get", _fake_get)

    def _set(adapter: InferenceAdapter) -> None:
        box.adapter = adapter

    box.set_adapter = _set  # type: ignore[attr-defined]
    return box


def test_route_returns_vectors_for_single_input(patched_manager) -> None:
    patched_manager.set_adapter(_StubEmbedAdapter(vectors=[[0.5, 0.5, 0.5]]))
    client = TestClient(app)
    r = client.post("/v1/embeddings", json={"model": "stub-embed:1", "input": "hello"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "stub-embed:1"
    assert len(body["data"]) == 1
    assert body["data"][0]["index"] == 0
    assert body["data"][0]["embedding"] == [0.5, 0.5, 0.5]
    assert body["usage"]["prompt_tokens"] == 3
    assert body["usage"]["total_tokens"] == 3
    assert body["usage"]["completion_tokens"] == 0


def test_route_returns_one_vector_per_batch_input(patched_manager) -> None:
    patched_manager.set_adapter(_StubEmbedAdapter(vectors=[[1.0, 2.0]]))
    client = TestClient(app)
    r = client.post("/v1/embeddings", json={"model": "stub-embed:1", "input": ["a", "b", "c"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    assert all(d["embedding"] == [1.0, 2.0] for d in body["data"])


def test_route_rejects_empty_list(patched_manager) -> None:
    patched_manager.set_adapter(_StubEmbedAdapter())
    client = TestClient(app)
    r = client.post("/v1/embeddings", json={"model": "stub-embed:1", "input": []})
    assert r.status_code == 400
    assert "at least one" in r.json()["detail"]


def test_route_returns_404_for_unknown_model(patched_manager) -> None:
    # patched_manager intentionally has no adapter set → ModelNotFoundError.
    client = TestClient(app)
    r = client.post("/v1/embeddings", json={"model": "ghost:1", "input": "hi"})
    assert r.status_code == 404
    assert "model not found" in r.json()["detail"]


def test_route_returns_501_when_backend_lacks_embeddings(patched_manager) -> None:
    patched_manager.set_adapter(_StubEmbedAdapter(fail=True))
    client = TestClient(app)
    r = client.post("/v1/embeddings", json={"model": "stub-embed:1", "input": "hi"})
    assert r.status_code == 501
    assert "embeddings not supported" in r.json()["detail"]


def test_route_emits_embeddings_run_span_with_dimensions(patched_manager, _session_exporter) -> None:
    """The span carries gen_ai.* + embedding.* attrs Prometa needs to slice
    embeddings traffic the same way it slices chat."""
    _session_exporter.clear()
    patched_manager.set_adapter(_StubEmbedAdapter(vectors=[[0.0, 1.0, 2.0, 3.0]]))
    client = TestClient(app)
    r = client.post("/v1/embeddings", json={"model": "stub-embed:1", "input": ["a", "b"]})
    assert r.status_code == 200

    spans = _session_exporter.get_finished_spans()
    embed_spans = [s for s in spans if s.name == "embeddings.run"]
    assert len(embed_spans) == 1
    s = embed_spans[0]
    assert s.attributes["gen_ai.system"] == "stub-embed"
    assert s.attributes["gen_ai.request.model"] == "stub-embed:1"
    assert s.attributes["embedding.batch_size"] == 2
    assert s.attributes["embedding.dimensions"] == 4
    assert s.attributes["gen_ai.usage.input_tokens"] == 6  # 2 inputs × 3 tokens
