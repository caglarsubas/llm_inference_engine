"""LlamaCppAdapter.prefix_cache install + accessors.

These don't load a real model — they exercise the config-gated cache-install
branch of ``load()`` against a stubbed ``llama_cpp.Llama`` so we can verify
``set_cache`` is wired correctly without paying the GGUF-load cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inference_engine.adapters.llama_cpp import LlamaCppAdapter
from inference_engine.config import settings
from inference_engine.registry import ModelDescriptor


class _StubLlama:
    """Stand-in for llama_cpp.Llama. Records set_cache calls."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.cache: Any = None

    def set_cache(self, cache: Any) -> None:
        self.cache = cache


@pytest.fixture
def patch_llama(monkeypatch):
    """Replace llama_cpp.Llama with the stub. Yields the stub class so tests can
    inspect last-instance state."""
    import llama_cpp  # noqa: PLC0415

    instances: list[_StubLlama] = []

    def _factory(**kwargs):
        s = _StubLlama(**kwargs)
        instances.append(s)
        return s

    monkeypatch.setattr(llama_cpp, "Llama", _factory)
    return instances


def _desc() -> ModelDescriptor:
    return ModelDescriptor(
        name="x", tag="1", namespace="ns", registry="reg",
        model_path=Path("/tmp/x.gguf"), format="gguf", size_bytes=1024,
    )


@pytest.mark.asyncio
async def test_cache_installed_when_enabled(patch_llama, monkeypatch) -> None:
    monkeypatch.setattr(settings, "prefix_cache_bytes", 256 * 1024 * 1024)

    adapter = LlamaCppAdapter()
    await adapter.load(_desc())

    # set_cache was called on the underlying Llama instance.
    assert len(patch_llama) == 1
    stub_llm = patch_llama[0]
    assert stub_llm.cache is not None

    # Adapter exposes the introspection we'll wire to spans/metrics.
    assert adapter.prefix_cache_enabled is True
    assert adapter.prefix_cache_capacity_bytes == 256 * 1024 * 1024
    assert adapter.prefix_cache_size_bytes == 0  # nothing cached yet


@pytest.mark.asyncio
async def test_cache_skipped_when_disabled(patch_llama, monkeypatch) -> None:
    monkeypatch.setattr(settings, "prefix_cache_bytes", 0)

    adapter = LlamaCppAdapter()
    await adapter.load(_desc())

    stub_llm = patch_llama[0]
    assert stub_llm.cache is None
    assert adapter.prefix_cache_enabled is False
    assert adapter.prefix_cache_capacity_bytes == 0
    assert adapter.prefix_cache_size_bytes == 0


@pytest.mark.asyncio
async def test_unload_drops_cache_reference(patch_llama, monkeypatch) -> None:
    """After unload, the adapter shouldn't hold the cache (or the model)."""
    monkeypatch.setattr(settings, "prefix_cache_bytes", 64 * 1024 * 1024)

    adapter = LlamaCppAdapter()
    await adapter.load(_desc())
    assert adapter.prefix_cache_enabled

    await adapter.unload()
    assert not adapter.is_loaded
    assert not adapter.prefix_cache_enabled
    assert adapter.prefix_cache_capacity_bytes == 0
    assert adapter.prefix_cache_size_bytes == 0
    # Per-call introspection collapses to disabled on unload.
    assert adapter.prefix_cache_last_action == "disabled"
    assert adapter.prefix_cache_last_overlap_tokens == 0
    assert adapter.prefix_cache_last_prompt_tokens == 0


# ---------------------------------------------------------------------------
# _TrackedLlamaRAMCache — hit/miss bookkeeping via direct subclass tests.
# We exercise the real subclass against synthetic LlamaState-like objects so
# we don't need to load a real model.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tracked_cache_records_hit_with_prefix_length(monkeypatch) -> None:
    """Hit: __getitem__ returns a state; subclass captures state.n_tokens."""
    from inference_engine.adapters.llama_cpp import _tracked_cache  # noqa: PLC0415

    cache = _tracked_cache(64 * 1024 * 1024)
    cache.begin_call()
    assert cache.last_action == "unconsulted"

    # Inject a fake state directly into the OrderedDict the parent uses.
    fake_state = type("S", (), {"n_tokens": 17, "llama_state_size": 1000})()
    cache.cache_state[(1, 2, 3)] = fake_state

    # Lookup with a key that has the cached prefix at its start.
    result = cache[(1, 2, 3, 4, 5)]

    assert result is fake_state
    assert cache.last_action == "hit"
    assert cache.last_hit_prefix_len == 17  # taken from state.n_tokens
    assert cache.hit_count == 1
    assert cache.miss_count == 0


@pytest.mark.asyncio
async def test_tracked_cache_records_miss_on_no_prefix_match(monkeypatch) -> None:
    """Miss: __getitem__ raises KeyError; counter increments, action='miss'."""
    from inference_engine.adapters.llama_cpp import _tracked_cache  # noqa: PLC0415

    cache = _tracked_cache(64 * 1024 * 1024)
    cache.begin_call()

    with pytest.raises(KeyError):
        _ = cache[(99, 100, 101)]  # nothing cached

    assert cache.last_action == "miss"
    assert cache.last_hit_prefix_len == 0
    assert cache.hit_count == 0
    assert cache.miss_count == 1


@pytest.mark.asyncio
async def test_begin_call_resets_per_call_state(monkeypatch) -> None:
    """A fresh begin_call wipes last_action so a no-lookup call doesn't
    inherit the previous call's hit. This is the failure mode that motivates
    the explicit reset."""
    from inference_engine.adapters.llama_cpp import _tracked_cache  # noqa: PLC0415

    cache = _tracked_cache(64 * 1024 * 1024)

    # Simulate a prior hit.
    fake_state = type("S", (), {"n_tokens": 42, "llama_state_size": 1})()
    cache.cache_state[(1, 2)] = fake_state
    _ = cache[(1, 2, 3)]
    assert cache.last_action == "hit"
    assert cache.last_hit_prefix_len == 42

    # Next call: no cache lookup happens (e.g. within-conversation continuation).
    cache.begin_call()
    assert cache.last_action == "unconsulted"
    assert cache.last_hit_prefix_len == 0
    # Aggregate counters survive — they're cross-call.
    assert cache.hit_count == 1


@pytest.mark.asyncio
async def test_adapter_exposes_tracked_introspection_when_enabled(patch_llama, monkeypatch) -> None:
    """End-to-end through the adapter: properties wire through to the tracked cache."""
    monkeypatch.setattr(settings, "prefix_cache_bytes", 64 * 1024 * 1024)

    adapter = LlamaCppAdapter()
    await adapter.load(_desc())

    # Initial state — no calls made.
    assert adapter.prefix_cache_enabled is True
    assert adapter.prefix_cache_last_action == "none"
    assert adapter.prefix_cache_last_overlap_tokens == 0

    # Manually trip a hit on the underlying cache and re-read.
    fake_state = type("S", (), {"n_tokens": 88, "llama_state_size": 1})()
    adapter._cache.cache_state[(7, 8)] = fake_state
    _ = adapter._cache[(7, 8, 9, 10)]

    assert adapter.prefix_cache_last_action == "hit"
    assert adapter.prefix_cache_last_overlap_tokens == 88


@pytest.mark.asyncio
async def test_adapter_reports_disabled_when_cache_off(patch_llama, monkeypatch) -> None:
    """With PREFIX_CACHE_BYTES=0 the cache is never installed; properties
    should report 'disabled' rather than raising or silently lying."""
    monkeypatch.setattr(settings, "prefix_cache_bytes", 0)

    adapter = LlamaCppAdapter()
    await adapter.load(_desc())

    assert adapter.prefix_cache_enabled is False
    assert adapter.prefix_cache_last_action == "disabled"
    assert adapter.prefix_cache_last_overlap_tokens == 0
    assert adapter.prefix_cache_capacity_bytes == 0
