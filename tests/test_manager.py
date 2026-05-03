"""ModelManager — LRU + memory-budget eviction."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.manager import ModelManager, ModelNotFoundError
from inference_engine.registry import ModelDescriptor


@dataclass
class _FakeRegistry:
    """In-memory registry stand-in keyed by qualified_name."""

    descriptors: dict[str, ModelDescriptor]

    def get(self, name: str) -> ModelDescriptor | None:
        return self.descriptors.get(name)

    def list_models(self) -> list[ModelDescriptor]:
        return list(self.descriptors.values())


class FakeAdapter(InferenceAdapter):
    backend_name = "fake"
    instance_count = 0
    load_calls: list[str] = []
    unload_calls: list[str] = []

    def __init__(self) -> None:
        FakeAdapter.instance_count += 1
        self._descriptor: ModelDescriptor | None = None

    @property
    def is_loaded(self) -> bool:
        return self._descriptor is not None

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return self._descriptor

    async def load(self, descriptor: ModelDescriptor) -> None:
        FakeAdapter.load_calls.append(descriptor.qualified_name)
        self._descriptor = descriptor

    async def unload(self) -> None:
        if self._descriptor is not None:
            FakeAdapter.unload_calls.append(self._descriptor.qualified_name)
        self._descriptor = None

    async def generate(
        self, messages: Iterable, params: GenerationParams
    ) -> GenerationResult:
        return GenerationResult(text="ok", finish_reason="stop", prompt_tokens=1, completion_tokens=1)

    async def stream(
        self, messages: Iterable, params: GenerationParams
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="ok", finish_reason="stop")


@pytest.fixture(autouse=True)
def _reset_fake_adapter() -> None:
    FakeAdapter.instance_count = 0
    FakeAdapter.load_calls = []
    FakeAdapter.unload_calls = []


def _desc(name: str, tag: str, size_bytes: int, fmt: str = "gguf") -> ModelDescriptor:
    return ModelDescriptor(
        name=name,
        tag=tag,
        namespace="library",
        registry="registry.ollama.ai",
        model_path=Path(f"/tmp/{name}-{tag}"),
        format=fmt,
        size_bytes=size_bytes,
    )


def _registry(*descs: ModelDescriptor) -> _FakeRegistry:
    return _FakeRegistry({d.qualified_name: d for d in descs})


@pytest.mark.asyncio
async def test_first_load_caches_model() -> None:
    a = _desc("a", "1", 10)
    mgr = ModelManager(
        registry=_registry(a),
        adapter_factory=lambda _desc: FakeAdapter(),
        memory_budget_bytes=100,
    )

    adapter, desc = await mgr.get("a:1")
    assert desc is a
    assert adapter.is_loaded
    assert mgr.loaded_models() == ["a:1"]
    assert mgr.loaded_bytes == 10
    assert FakeAdapter.load_calls == ["a:1"]


@pytest.mark.asyncio
async def test_repeat_get_returns_cached_adapter() -> None:
    a = _desc("a", "1", 10)
    mgr = ModelManager(_registry(a), lambda _d: FakeAdapter(), memory_budget_bytes=100)

    adapter1, _ = await mgr.get("a:1")
    adapter2, _ = await mgr.get("a:1")

    assert adapter1 is adapter2
    assert FakeAdapter.load_calls == ["a:1"]
    assert FakeAdapter.unload_calls == []


@pytest.mark.asyncio
async def test_unknown_model_raises() -> None:
    mgr = ModelManager(_registry(), lambda _d: FakeAdapter(), memory_budget_bytes=100)
    with pytest.raises(ModelNotFoundError):
        await mgr.get("ghost:1")


@pytest.mark.asyncio
async def test_lru_eviction_when_over_budget() -> None:
    a = _desc("a", "1", 40)
    b = _desc("b", "1", 40)
    c = _desc("c", "1", 40)
    mgr = ModelManager(_registry(a, b, c), lambda _d: FakeAdapter(), memory_budget_bytes=100)

    await mgr.get("a:1")
    await mgr.get("b:1")
    assert mgr.loaded_models() == ["a:1", "b:1"]
    assert mgr.loaded_bytes == 80

    # Loading c (40) would push us to 120 > 100. LRU is "a" → evict it.
    await mgr.get("c:1")

    assert mgr.loaded_models() == ["b:1", "c:1"]
    assert mgr.loaded_bytes == 80
    assert FakeAdapter.unload_calls == ["a:1"]


@pytest.mark.asyncio
async def test_get_touches_lru() -> None:
    a = _desc("a", "1", 40)
    b = _desc("b", "1", 40)
    c = _desc("c", "1", 40)
    mgr = ModelManager(_registry(a, b, c), lambda _d: FakeAdapter(), memory_budget_bytes=100)

    await mgr.get("a:1")
    await mgr.get("b:1")
    # touch a — now b is LRU
    await mgr.get("a:1")
    await mgr.get("c:1")

    assert mgr.loaded_models() == ["a:1", "c:1"]
    assert FakeAdapter.unload_calls == ["b:1"]


@pytest.mark.asyncio
async def test_shutdown_unloads_everything() -> None:
    a = _desc("a", "1", 10)
    b = _desc("b", "1", 10)
    mgr = ModelManager(_registry(a, b), lambda _d: FakeAdapter(), memory_budget_bytes=100)

    await mgr.get("a:1")
    await mgr.get("b:1")
    await mgr.shutdown()

    assert mgr.loaded_models() == []
    assert mgr.loaded_bytes == 0
    assert sorted(FakeAdapter.unload_calls) == ["a:1", "b:1"]
