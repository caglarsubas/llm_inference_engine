"""Dynamic batching — adapter capability fallback + cross-request coalescer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest

from inference_engine.adapters import (
    EmbeddingResult,
    GenerationParams,
    InferenceAdapter,
    StreamChunk,
)
from inference_engine.adapters.base import GenerationResult
from inference_engine.api._batcher import EmbedCoalescer
from inference_engine.cancellation import Cancellation
from inference_engine.config import settings
from inference_engine.registry import ModelDescriptor


# ---------------------------------------------------------------------------
# Adapter — capability detection + fallback
# ---------------------------------------------------------------------------


class _StubLlama:
    """Minimal Llama stand-in with a configurable batched-vs-serial response."""

    def __init__(self) -> None:
        self.calls: list[str | list[str]] = []
        # When True, batched calls (input is a list) raise the
        # llama_decode RuntimeError. Single-input calls always succeed.
        self.batch_fails: bool = False
        # Vector returned on every call.
        self.vector: list[float] = [0.1, 0.2, 0.3]

    def create_embedding(self, *, input):  # noqa: A002 — match llama_cpp arg name
        self.calls.append(input)
        if isinstance(input, list):
            if self.batch_fails:
                raise RuntimeError("llama_decode returned -1")
            return {
                "data": [
                    {"index": i, "embedding": list(self.vector)} for i in range(len(input))
                ],
                "usage": {"prompt_tokens": len(input) * 3},
            }
        return {
            "data": [{"index": 0, "embedding": list(self.vector)}],
            "usage": {"prompt_tokens": 3},
        }

    def set_cache(self, _cache) -> None:  # noqa: D401 — interface stub
        return None


@pytest.fixture
def adapter_with_stub(monkeypatch):
    """Build a real LlamaCppAdapter wired to a stub Llama (no GGUF load)."""
    import llama_cpp  # noqa: PLC0415

    instances: list[_StubLlama] = []

    def _factory(**_kwargs):
        s = _StubLlama()
        instances.append(s)
        return s

    monkeypatch.setattr(llama_cpp, "Llama", _factory)
    # Skip the LlamaRAMCache install in tests — it's not relevant here and
    # the stub doesn't implement set_cache fully.
    monkeypatch.setattr(settings, "prefix_cache_bytes", 0)

    from inference_engine.adapters.llama_cpp import LlamaCppAdapter  # noqa: PLC0415

    adapter = LlamaCppAdapter()
    return adapter, instances


def _desc() -> ModelDescriptor:
    return ModelDescriptor(
        name="m", tag="1", namespace="ns", registry="reg",
        model_path=Path("/tmp/m.gguf"), format="gguf", size_bytes=1024,
    )


@pytest.mark.asyncio
async def test_first_batch_call_succeeds_and_caches_capability(adapter_with_stub) -> None:
    adapter, _ = adapter_with_stub
    await adapter.load(_desc())

    result = await adapter.embed(["a", "b", "c"])
    assert len(result.embeddings) == 3
    assert adapter.supports_batched_embed is True
    assert adapter.last_embed_action == "batch"


@pytest.mark.asyncio
async def test_batch_failure_falls_back_to_serial(adapter_with_stub) -> None:
    adapter, instances = adapter_with_stub
    await adapter.load(_desc())
    instances[-1].batch_fails = True

    result = await adapter.embed(["a", "b", "c"])

    assert len(result.embeddings) == 3
    assert adapter.supports_batched_embed is False
    assert adapter.last_embed_action == "fallback"
    # First call was batch (the failed try); subsequent 3 were single-input.
    calls = instances[-1].calls
    assert isinstance(calls[0], list)  # the failed batch attempt
    assert all(isinstance(c, str) for c in calls[1:])


@pytest.mark.asyncio
async def test_subsequent_calls_skip_batch_after_fallback(adapter_with_stub) -> None:
    """Once we know the model fails on batch, never retry the batch path."""
    adapter, instances = adapter_with_stub
    await adapter.load(_desc())
    instances[-1].batch_fails = True

    await adapter.embed(["a", "b"])  # probes + falls back
    instances[-1].calls.clear()
    await adapter.embed(["x", "y", "z"])

    # No batch attempts in the second call.
    assert all(isinstance(c, str) for c in instances[-1].calls)
    assert adapter.last_embed_action == "serial"


@pytest.mark.asyncio
async def test_single_input_always_uses_serial_path(adapter_with_stub) -> None:
    """Single-input calls don't probe the batch path."""
    adapter, instances = adapter_with_stub
    await adapter.load(_desc())
    await adapter.embed(["only"])

    assert adapter.supports_batched_embed is None  # never probed
    assert adapter.last_embed_action == "serial"
    assert all(isinstance(c, str) for c in instances[-1].calls)


@pytest.mark.asyncio
async def test_unrelated_runtimeerror_not_swallowed(adapter_with_stub) -> None:
    """We catch only the llama_decode case; other RuntimeErrors must surface."""
    adapter, instances = adapter_with_stub
    await adapter.load(_desc())

    # Make the batch call raise a different RuntimeError.
    real_create = instances[-1].create_embedding

    def _fail_other(*, input):  # noqa: A002
        if isinstance(input, list):
            raise RuntimeError("something else entirely")
        return real_create(input=input)

    instances[-1].create_embedding = _fail_other  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="something else"):
        await adapter.embed(["a", "b"])


# ---------------------------------------------------------------------------
# EmbedCoalescer — coalescing semantics
# ---------------------------------------------------------------------------


class _RecordingAdapter(InferenceAdapter):
    """Adapter that records every call to embed(). For coalescer tests."""

    backend_name = "recording"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.last_embed_action: str = "batch"

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

    async def embed(self, inputs: list[str]) -> EmbeddingResult:
        self.calls.append(list(inputs))
        # One vector per input, distinct so per-request slicing is verifiable.
        embeddings = [[float(hash(s) % 1000) / 1000.0] for s in inputs]
        return EmbeddingResult(embeddings=embeddings, prompt_tokens=len(inputs) * 3)


@pytest.mark.asyncio
async def test_concurrent_submits_coalesce_into_single_call(monkeypatch) -> None:
    """Three concurrent submits within the wait window become one adapter.embed call."""
    monkeypatch.setattr(settings, "batch_enabled", True)
    monkeypatch.setattr(settings, "batch_max_wait_ms", 50.0)
    monkeypatch.setattr(settings, "batch_max_size", 100)

    adapter = _RecordingAdapter()
    coalescer = EmbedCoalescer()

    outcomes = await asyncio.gather(
        coalescer.submit(adapter, ["a"]),
        coalescer.submit(adapter, ["b"]),
        coalescer.submit(adapter, ["c"]),
    )

    assert len(adapter.calls) == 1
    assert adapter.calls[0] == ["a", "b", "c"]
    assert {o.coalesced_with for o in outcomes} == {3}
    assert {o.batch_id for o in outcomes} == {0}
    assert all(len(o.embeddings) == 1 for o in outcomes)
    # prompt_tokens split proportionally — each had 1/3 of inputs, total 9 → 3.
    assert {o.prompt_tokens for o in outcomes} == {3}


@pytest.mark.asyncio
async def test_size_threshold_flushes_immediately(monkeypatch) -> None:
    """Hitting BATCH_MAX_SIZE doesn't wait the full window."""
    monkeypatch.setattr(settings, "batch_enabled", True)
    monkeypatch.setattr(settings, "batch_max_wait_ms", 5000.0)  # generous
    monkeypatch.setattr(settings, "batch_max_size", 2)

    adapter = _RecordingAdapter()
    coalescer = EmbedCoalescer()

    import time as _t

    t0 = _t.perf_counter()
    outcomes = await asyncio.gather(
        coalescer.submit(adapter, ["a"]),
        coalescer.submit(adapter, ["b"]),
    )
    elapsed_ms = (_t.perf_counter() - t0) * 1000

    # Should NOT have waited 5 seconds.
    assert elapsed_ms < 500, f"size threshold didn't trigger early flush: {elapsed_ms:.0f}ms"
    assert {o.coalesced_with for o in outcomes} == {2}


@pytest.mark.asyncio
async def test_solo_request_flushes_after_wait_window(monkeypatch) -> None:
    """A single request with no siblings still completes after the window."""
    monkeypatch.setattr(settings, "batch_enabled", True)
    monkeypatch.setattr(settings, "batch_max_wait_ms", 20.0)
    monkeypatch.setattr(settings, "batch_max_size", 100)

    adapter = _RecordingAdapter()
    coalescer = EmbedCoalescer()

    outcome = await coalescer.submit(adapter, ["alone"])

    assert outcome.coalesced_with == 1
    assert outcome.total_inputs == 1
    assert len(outcome.embeddings) == 1


@pytest.mark.asyncio
async def test_disabled_batch_passes_straight_through(monkeypatch) -> None:
    monkeypatch.setattr(settings, "batch_enabled", False)

    adapter = _RecordingAdapter()
    coalescer = EmbedCoalescer()

    outcomes = await asyncio.gather(
        coalescer.submit(adapter, ["a"]),
        coalescer.submit(adapter, ["b"]),
    )

    # No coalescing — each submit hits adapter.embed independently.
    assert len(adapter.calls) == 2
    assert {o.batch_id for o in outcomes} == {-1}
    assert {o.coalesced_with for o in outcomes} == {1}


@pytest.mark.asyncio
async def test_per_request_slicing_preserves_order(monkeypatch) -> None:
    """When two requests submit different-sized chunks, each gets back exactly its own slice."""
    monkeypatch.setattr(settings, "batch_enabled", True)
    monkeypatch.setattr(settings, "batch_max_wait_ms", 50.0)
    monkeypatch.setattr(settings, "batch_max_size", 100)

    adapter = _RecordingAdapter()
    coalescer = EmbedCoalescer()

    inputs_a = ["alpha-1", "alpha-2"]
    inputs_b = ["beta-1", "beta-2", "beta-3"]

    out_a, out_b = await asyncio.gather(
        coalescer.submit(adapter, inputs_a),
        coalescer.submit(adapter, inputs_b),
    )

    assert len(out_a.embeddings) == 2
    assert len(out_b.embeddings) == 3
    assert out_a.total_inputs == 5
    assert out_b.total_inputs == 5
    # The full underlying call processed all 5 in one shot.
    assert adapter.calls[0] == inputs_a + inputs_b


@pytest.mark.asyncio
async def test_separate_adapters_get_separate_queues(monkeypatch) -> None:
    """Requests for different adapters never wait on each other."""
    monkeypatch.setattr(settings, "batch_enabled", True)
    monkeypatch.setattr(settings, "batch_max_wait_ms", 50.0)
    monkeypatch.setattr(settings, "batch_max_size", 100)

    adapter_a = _RecordingAdapter()
    adapter_b = _RecordingAdapter()
    coalescer = EmbedCoalescer()

    await asyncio.gather(
        coalescer.submit(adapter_a, ["a"]),
        coalescer.submit(adapter_b, ["b"]),
    )

    assert adapter_a.calls == [["a"]]
    assert adapter_b.calls == [["b"]]
