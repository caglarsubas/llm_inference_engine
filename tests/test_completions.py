"""``/v1/completions`` — schema, adapter contract, route, span shape."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference_engine.adapters import (
    GenerationParams,
    InferenceAdapter,
    StreamChunk,
)
from inference_engine.adapters.base import GenerationResult
from inference_engine.cancellation import Cancellation
from inference_engine.main import app
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import CompletionRequest


# ---------------------------------------------------------------------------
# Schema acceptance
# ---------------------------------------------------------------------------


def test_request_accepts_string_prompt() -> None:
    req = CompletionRequest(model="x", prompt="hello")
    assert req.prompt == "hello"


def test_request_accepts_list_prompt() -> None:
    req = CompletionRequest(model="x", prompt=["a", "b"])
    assert req.prompt == ["a", "b"]


def test_request_default_max_tokens_is_128() -> None:
    req = CompletionRequest(model="x", prompt="hi")
    assert req.max_tokens == 128


# ---------------------------------------------------------------------------
# Adapter contract — default complete() routes through generate()
# ---------------------------------------------------------------------------


class _GenerateOnlyAdapter(InferenceAdapter):
    """Adapter that only implements generate(); complete() should fall back to it."""

    backend_name = "gen-only"

    def __init__(self) -> None:
        self.received_messages: list = []

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
        msgs = list(messages)
        self.received_messages = msgs
        # Echo the user content so the test can verify the prompt round-tripped.
        text = msgs[0].content if msgs else ""
        return GenerationResult(
            text=f"echo: {text}", finish_reason="stop", prompt_tokens=5, completion_tokens=3
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="", finish_reason="stop")


@pytest.mark.asyncio
async def test_default_complete_falls_back_to_generate() -> None:
    """Adapters that don't override complete() get the default ABC fallback."""
    adapter = _GenerateOnlyAdapter()
    result = await adapter.complete("raw prompt", GenerationParams())
    assert result.text == "echo: raw prompt"
    # The default fallback wraps the prompt in a single user-role message.
    assert len(adapter.received_messages) == 1
    assert adapter.received_messages[0].role == "user"
    assert adapter.received_messages[0].content == "raw prompt"


# ---------------------------------------------------------------------------
# Route — single + multi-prompt + error paths
# ---------------------------------------------------------------------------


class _StubCompletionAdapter(InferenceAdapter):
    backend_name = "stub-cmpl"

    def __init__(self) -> None:
        self.complete_calls: list[str] = []
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

    async def complete(
        self, prompt: str, params: GenerationParams, cancel: Cancellation | None = None
    ) -> GenerationResult:
        self.complete_calls.append(prompt)
        return GenerationResult(
            text=f"-> {prompt}", finish_reason="stop", prompt_tokens=4, completion_tokens=2
        )


def _stub_descriptor(name: str = "stub-cmpl:1") -> ModelDescriptor:
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
    class _Box:
        adapter: InferenceAdapter | None = None

    box = _Box()

    async def _fake_get(model_id: str):
        if box.adapter is None:
            from inference_engine.manager import ModelNotFoundError  # noqa: PLC0415

            raise ModelNotFoundError(model_id)
        return box.adapter, _stub_descriptor(model_id)

    from inference_engine.api.state import app_state  # noqa: PLC0415

    monkeypatch.setattr(app_state.manager, "get", _fake_get)

    def _set(adapter: InferenceAdapter) -> None:
        box.adapter = adapter

    box.set_adapter = _set  # type: ignore[attr-defined]
    return box


def test_route_single_prompt_returns_one_choice(patched_manager) -> None:
    patched_manager.set_adapter(_StubCompletionAdapter())
    client = TestClient(app)
    r = client.post("/v1/completions", json={"model": "stub-cmpl:1", "prompt": "hello"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "text_completion"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["text"] == "-> hello"
    assert body["choices"][0]["index"] == 0
    assert body["usage"]["prompt_tokens"] == 4
    assert body["usage"]["completion_tokens"] == 2


def test_route_multi_prompt_returns_one_choice_per_prompt(patched_manager) -> None:
    adapter = _StubCompletionAdapter()
    patched_manager.set_adapter(adapter)
    client = TestClient(app)
    r = client.post("/v1/completions", json={"model": "stub-cmpl:1", "prompt": ["a", "b", "c"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [c["index"] for c in body["choices"]] == [0, 1, 2]
    assert [c["text"] for c in body["choices"]] == ["-> a", "-> b", "-> c"]
    # Each prompt produced a separate adapter.complete call.
    assert adapter.complete_calls == ["a", "b", "c"]
    # Usage aggregates per-prompt token counts.
    assert body["usage"]["prompt_tokens"] == 4 * 3
    assert body["usage"]["completion_tokens"] == 2 * 3


def test_route_rejects_empty_list(patched_manager) -> None:
    patched_manager.set_adapter(_StubCompletionAdapter())
    client = TestClient(app)
    r = client.post("/v1/completions", json={"model": "stub-cmpl:1", "prompt": []})
    assert r.status_code == 400


def test_route_404_for_unknown_model(patched_manager) -> None:
    client = TestClient(app)
    r = client.post("/v1/completions", json={"model": "ghost:1", "prompt": "hi"})
    assert r.status_code == 404


def test_route_emits_completions_run_span(patched_manager, _session_exporter) -> None:
    _session_exporter.clear()
    patched_manager.set_adapter(_StubCompletionAdapter())
    client = TestClient(app)
    r = client.post("/v1/completions", json={"model": "stub-cmpl:1", "prompt": ["a", "b"]})
    assert r.status_code == 200

    spans = _session_exporter.get_finished_spans()
    cmpl_spans = [s for s in spans if s.name == "completions.run"]
    assert len(cmpl_spans) == 1
    s = cmpl_spans[0]
    assert s.attributes["gen_ai.system"] == "stub-cmpl"
    assert s.attributes["gen_ai.request.model"] == "stub-cmpl:1"
    assert s.attributes["completion.batch_size"] == 2
    assert s.attributes["gen_ai.usage.input_tokens"] == 8
    assert s.attributes["gen_ai.usage.output_tokens"] == 4
