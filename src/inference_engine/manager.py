"""ModelManager — multi-model hot-keep with LRU eviction.

Caches adapters and routes requests by model id. The weight budget applies to
GGUF and MLX weights loaded by this process. An upstream adapter holds an HTTP
client; its server independently manages weight residency.

Lock model
----------

Two locks, sized to the actual contention they protect against:

* ``self._meta_lock`` — fine-grained, held only for **microsecond** windows
  when reading/mutating ``self._loaded`` (the cache + LRU order). Cache hits
  acquire-and-release this lock once per ``get()``; they never block on cold
  loads.

* ``self._key_locks[key]`` — one lock per model id. Held for the whole load
  of *that* model, so concurrent ``get(X)`` calls dedupe (the second sees a
  cache hit when it eventually acquires the lock). Different-model loads can
  fire in parallel because they hold *different* locks.

The simpler "single global asyncio.Lock around the whole get()" design we
started with serialised every cache hit behind any in-flight cold load — fine
in development, but the dominant tail-latency source under real concurrency.

Memory-budget caveat
--------------------

When two cold loads race, both adapters may briefly coexist before the
second's eviction step runs, so total resident bytes can momentarily
overshoot ``memory_budget_bytes``. That's an acceptable softness for a soft
budget; the next ``get()`` reconciles state.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable

from .adapters import InferenceAdapter
from .observability import get_logger
from .registry import ModelDescriptor, OllamaRegistry

log = get_logger("manager")
_MANAGED_WEIGHT_FORMATS = ("gguf", "mlx")


class ModelNotFoundError(KeyError):
    """Raised when a model id can't be resolved against the registry."""


class ModelManager:
    def __init__(
        self,
        registry: OllamaRegistry,
        adapter_factory: Callable[[ModelDescriptor], InferenceAdapter],
        memory_budget_bytes: int,
        resolver: Callable[[str], ModelDescriptor | None] | None = None,
    ) -> None:
        self._registry = registry
        # Probe-aware resolution lives in CompositeRegistry; we accept it as
        # a callable so the manager stays decoupled from the probe details
        # and the registry-source tree.  ``None`` falls back to plain
        # ``registry.get(model_id)`` so single-source tests keep working.
        self._resolver = resolver or registry.get
        self._adapter_factory = adapter_factory
        self._budget = memory_budget_bytes
        self._loaded: OrderedDict[str, tuple[InferenceAdapter, ModelDescriptor]] = OrderedDict()
        self._meta_lock = asyncio.Lock()
        self._key_locks: dict[str, asyncio.Lock] = {}

    # -----------------------------------------------------------------
    # introspection
    # -----------------------------------------------------------------

    @property
    def loaded_bytes(self) -> int:
        """On-disk weight sizes for cached native models, used as a memory estimate."""
        return sum(
            desc.size_bytes for _, desc in self._loaded.values()
            if desc.format in _MANAGED_WEIGHT_FORMATS
        )

    @property
    def memory_budget_bytes(self) -> int:
        return self._budget

    def loaded_models(self) -> list[str]:
        """Models whose adapters hold native weights, in LRU order."""
        return [
            name for name, (_, desc) in self._loaded.items()
            if desc.format in _MANAGED_WEIGHT_FORMATS
        ]

    def loaded_summary(self) -> list[dict]:
        return [
            {"model": name, "size_bytes": desc.size_bytes, "backend": adapter.backend_name}
            for name, (adapter, desc) in self._loaded.items()
            if desc.format in _MANAGED_WEIGHT_FORMATS
        ]

    def cached_upstream_clients(self) -> list[dict]:
        return [
            {"model": name, "backend": adapter.backend_name}
            for name, (adapter, desc) in self._loaded.items()
            if desc.format not in _MANAGED_WEIGHT_FORMATS
        ]

    def iter_loaded(self) -> list[tuple[str, InferenceAdapter, ModelDescriptor]]:
        """Snapshot the (name, adapter, descriptor) of every cached adapter.

        Returns a list (not an iterator) so callers don't pin lock state during
        iteration. Order is LRU: oldest-touched first.
        """
        return [(name, adapter, desc) for name, (adapter, desc) in self._loaded.items()]

    def resolve(self, model_id: str) -> ModelDescriptor | None:
        """The descriptor ``model_id`` would load, without loading anything."""
        return self._resolver(model_id)

    # -----------------------------------------------------------------
    # core path
    # -----------------------------------------------------------------

    async def get(self, model_id: str) -> tuple[InferenceAdapter, ModelDescriptor]:
        """Return a ready-to-use adapter for `model_id`, loading + evicting as needed.

        Cache hits hold ``_meta_lock`` for one cheap dict lookup and an
        ``OrderedDict.move_to_end()``. Cold loads hold the **per-key** lock
        for the duration of ``adapter.load()`` so concurrent callers for the
        same model dedupe, while concurrent callers for different models
        proceed in parallel.
        """
        descriptor = self._resolver(model_id)
        if descriptor is None:
            raise ModelNotFoundError(model_id)
        key = descriptor.qualified_name

        # Fast path: cache hit. Brief meta-lock for OrderedDict mutation.
        async with self._meta_lock:
            cached = self._loaded.get(key)
            if cached is not None:
                self._loaded.move_to_end(key)
                return cached[0], descriptor
            key_lock = self._key_locks.setdefault(key, asyncio.Lock())

        # Slow path: acquire the per-key lock. Concurrent get(key) calls
        # serialise here so we only ever load this model once.
        async with key_lock:
            # Re-check under meta-lock — a coroutine ahead of us may have
            # finished loading while we were waiting on key_lock.
            async with self._meta_lock:
                cached = self._loaded.get(key)
                if cached is not None:
                    self._loaded.move_to_end(key)
                    return cached[0], descriptor

            log.info(
                "manager.load",
                model=key,
                format=descriptor.format,
                size_bytes=descriptor.size_bytes,
                loaded_bytes_before=self.loaded_bytes,
                budget=self._budget,
            )
            adapter = self._adapter_factory(descriptor)
            await adapter.load(descriptor)

            async with self._meta_lock:
                if descriptor.format in _MANAGED_WEIGHT_FORMATS:
                    await self._evict_until_fits(descriptor.size_bytes)
                self._loaded[key] = (adapter, descriptor)

            log.info(
                "manager.loaded",
                model=key,
                backend=adapter.backend_name,
                loaded_models=self.loaded_models(),
                loaded_bytes=self.loaded_bytes,
            )
            return adapter, descriptor

    async def _evict_until_fits(self, incoming_bytes: int) -> None:
        """Evict LRU entries until the incoming model fits.

        Caller MUST hold ``self._meta_lock``. ``adapter.unload()`` is awaited
        with the lock held — this is fine because unload is a simple reference
        drop on llama-cpp-python and a tiny weight-array detach on MLX, both
        sub-millisecond. Holding the lock across unload keeps the cache view
        consistent for any concurrent observers.
        """
        bytes_to_free = self.loaded_bytes + incoming_bytes - self._budget
        for evicted_key in self.loaded_models():
            if bytes_to_free <= 0:
                break
            evicted_adapter, evicted_desc = self._loaded.pop(evicted_key)
            log.info(
                "manager.evict",
                model=evicted_key,
                size_bytes=evicted_desc.size_bytes,
                reason="lru_over_budget",
            )
            await evicted_adapter.unload()
            bytes_to_free -= evicted_desc.size_bytes

    async def shutdown(self) -> None:
        async with self._meta_lock:
            for key, (adapter, _) in list(self._loaded.items()):
                log.info("manager.shutdown.unload", model=key)
                await adapter.unload()
            self._loaded.clear()
