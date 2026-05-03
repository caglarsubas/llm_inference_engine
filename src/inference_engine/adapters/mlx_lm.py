"""MLX-LM adapter — Apple Silicon native inference + multi-slot LRU prompt cache.

Uses the upstream ``mlx-lm`` package, which loads HuggingFace-style
(safetensors + tokenizer + config.json) MLX-converted models. Models live in
the directory pointed to by ``descriptor.model_path``.

Why MLX as a second adapter:
- Native to Apple Silicon, runs directly on the unified-memory Metal stack.
- Tracks newer architectures (mistral3, gemma4, qwen3.6, nemotron3) that the
  bundled llama-cpp-python release doesn't yet support.
- Keeps the adapter contract honest — the same routes serve both backends.

Prompt cache (multi-slot, token-indexed)
----------------------------------------

The adapter holds up to ``MLX_PREFIX_CACHE_MAX_SLOTS`` independent
``CacheSlot`` instances per loaded model. Each slot is one MLX prompt cache
plus the token sequence it represents.

On every call we tokenize the incoming prompt and **scan all slots** for the
one with the longest matching token-prefix:

* full overlap   → reuse the slot verbatim (``action="full"``)
* partial overlap → ``trim_prompt_cache`` the slot to the divergence point
  (``action="trimmed"``)
* zero overlap on every slot → evict LRU if at capacity, allocate a fresh
  slot via ``make_prompt_cache`` (``action="miss"``)

Setting ``MLX_PREFIX_CACHE_MAX_SLOTS=1`` reproduces the original single-slot
behaviour exactly. The default of 4 is a pragmatic middle for agent traffic
where you might have a few stable prefixes in rotation (different agents,
different conversations, judge vs. candidate prompts) without overcommitting
GPU memory.

Trim is destructive — when a slot is partially reused, the divergent suffix
is dropped. If a later request wants the original full prefix back, it'll
re-prefill from the trim point. Cloning the cache to preserve both versions
would require deep-copying GPU arrays per call; not worth the cost.

Token-precise reuse counts are exposed on every call via ``prefix_cache_last_*``
properties so spans/metrics can report exact hit rate per call.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..cancellation import Cancellation
from ..config import settings
from ..observability import get_logger
from ..registry import ModelDescriptor
from ..schemas import ChatMessage
from .base import GenerationParams, GenerationResult, InferenceAdapter, StreamChunk

log = get_logger("adapter.mlx")


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


@dataclass
class CacheSlot:
    """One independent MLX prompt cache and the token sequence it represents."""

    cache: Any
    tokens: list[int] = field(default_factory=list)
    last_used: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_used = time.monotonic()


class MLXAdapter(InferenceAdapter):
    backend_name = "mlx"

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._descriptor: ModelDescriptor | None = None
        self._lock = asyncio.Lock()

        # Multi-slot prompt cache. Slots are owned per-adapter; they go away
        # automatically on unload (which the ModelManager triggers on eviction).
        self._slots: list[CacheSlot] = []

        # Per-call observability — reset by every _resolve_cache().
        self._last_prompt_tokens: int = 0
        self._last_overlap_tokens: int = 0
        self._last_action: str = "none"  # miss | full | trimmed | disabled | none

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return self._descriptor

    # ------------------------------------------------------------------
    # Prompt-cache introspection (token-precise, vs. llama.cpp's bytes)
    # ------------------------------------------------------------------

    @property
    def prefix_cache_enabled(self) -> bool:
        return settings.mlx_prefix_cache_enabled and self._model is not None

    @property
    def prefix_cache_slots_used(self) -> int:
        return len(self._slots)

    @property
    def prefix_cache_slots_max(self) -> int:
        return settings.mlx_prefix_cache_max_slots

    @property
    def prefix_cache_tokens(self) -> int:
        """Sum of tokens across every slot."""
        return sum(len(s.tokens) for s in self._slots)

    @property
    def prefix_cache_last_overlap_tokens(self) -> int:
        return self._last_overlap_tokens

    @property
    def prefix_cache_last_prompt_tokens(self) -> int:
        return self._last_prompt_tokens

    @property
    def prefix_cache_last_action(self) -> str:
        return self._last_action

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self, descriptor: ModelDescriptor) -> None:
        if descriptor.format != "mlx":
            raise ValueError(f"MLXAdapter only handles mlx, got {descriptor.format!r}")
        if self._descriptor and self._descriptor.model_path == descriptor.model_path:
            return

        from mlx_lm import load as mlx_load  # noqa: PLC0415

        log.info(
            "loading_model",
            model=descriptor.qualified_name,
            model_path=str(descriptor.model_path),
            size_bytes=descriptor.size_bytes,
            prefix_cache_enabled=settings.mlx_prefix_cache_enabled,
            prefix_cache_max_slots=settings.mlx_prefix_cache_max_slots,
        )

        await self.unload()
        self._model, self._tokenizer = await asyncio.to_thread(
            mlx_load, str(descriptor.model_path)
        )
        self._descriptor = descriptor
        self._reset_cache()
        log.info("model_loaded", model=descriptor.qualified_name)

    async def unload(self) -> None:
        if self._model is not None:
            log.info(
                "unloading_model",
                model=self._descriptor.qualified_name if self._descriptor else None,
            )
            self._model = None
            self._tokenizer = None
            self._descriptor = None
            self._reset_cache()

    def _reset_cache(self) -> None:
        self._slots = []
        self._last_prompt_tokens = 0
        self._last_overlap_tokens = 0
        self._last_action = "none"

    # ------------------------------------------------------------------
    # Prompt formatting + tokenization
    # ------------------------------------------------------------------

    def _format_prompt(self, messages: Iterable[ChatMessage]) -> str:
        chat = [{"role": m.role, "content": m.content} for m in messages]
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"

    def _tokenize(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text))

    # ------------------------------------------------------------------
    # Cache resolution — multi-slot
    # ------------------------------------------------------------------

    def _lookup_best_slot(self, prompt_tokens: list[int]) -> tuple[CacheSlot | None, int]:
        """Pick the slot with the longest matching token-prefix.

        Linear over slots — fine for the small ``max_slots`` numbers we
        target. Tie-break on most-recently-used.
        """
        best: CacheSlot | None = None
        best_overlap = 0
        best_last_used = -1.0
        for slot in self._slots:
            overlap = _common_prefix_len(prompt_tokens, slot.tokens)
            if overlap > best_overlap or (
                overlap == best_overlap and overlap > 0 and slot.last_used > best_last_used
            ):
                best = slot
                best_overlap = overlap
                best_last_used = slot.last_used
        return best, best_overlap

    def _evict_lru_until_room(self) -> None:
        """Drop the oldest slot until len(slots) < max_slots."""
        while len(self._slots) >= settings.mlx_prefix_cache_max_slots:
            idx = min(range(len(self._slots)), key=lambda i: self._slots[i].last_used)
            evicted = self._slots.pop(idx)
            log.info(
                "prefix_cache.evict",
                tokens=len(evicted.tokens),
                age_seconds=round(time.monotonic() - evicted.last_used, 2),
            )

    def _new_slot(self) -> CacheSlot:
        from mlx_lm.models.cache import make_prompt_cache  # noqa: PLC0415

        slot = CacheSlot(cache=make_prompt_cache(self._model), tokens=[])
        self._slots.append(slot)
        return slot

    def _resolve_cache(self, prompt_tokens: list[int]) -> tuple[Any, CacheSlot | None]:
        """Pick / build / trim the cache slot for this call.

        Returns ``(mlx_cache_object, owning_slot)``. When disabled, returns
        ``(None, None)`` and mlx-lm allocates an internal per-call cache that's
        discarded after the call. Sets ``_last_*`` introspection counters.

        Resolution policy for partial overlap:

        * Full overlap on a slot → reuse verbatim.
        * Capacity available → allocate a new slot; preserve the candidate
          best slot for future calls. The new slot pays full prefill cost,
          but the workload (alternating agents, multi-tenant) gets warm
          steady-state on every prefix instead of thrashing the single slot.
        * At capacity → trim the best slot to the divergence point. We lose
          the trimmed-off suffix but keep the overlap-prefill saving.

        The ``can_trim`` capability check stays as a fallback when the model
        architecture (e.g. some rotating/quantized caches) refuses to trim.
        """
        self._last_prompt_tokens = len(prompt_tokens)

        if not settings.mlx_prefix_cache_enabled:
            self._last_overlap_tokens = 0
            self._last_action = "disabled"
            return None, None

        from mlx_lm.models.cache import (  # noqa: PLC0415
            can_trim_prompt_cache,
            trim_prompt_cache,
        )

        best, overlap = self._lookup_best_slot(prompt_tokens)
        at_capacity = len(self._slots) >= settings.mlx_prefix_cache_max_slots

        # No useful slot — allocate a fresh one (evict LRU if at capacity).
        if best is None or overlap == 0:
            self._evict_lru_until_room()
            slot = self._new_slot()
            slot.touch()
            self._last_overlap_tokens = 0
            self._last_action = "miss"
            return slot.cache, slot

        # Best slot already holds the exact prefix we want — full reuse.
        if overlap == len(best.tokens):
            best.touch()
            self._last_overlap_tokens = overlap
            self._last_action = "full"
            return best.cache, best

        # Partial overlap. With capacity, preserve the best slot and start
        # fresh — the right call for alternating-agent workloads.
        if not at_capacity:
            slot = self._new_slot()
            slot.touch()
            self._last_overlap_tokens = 0
            self._last_action = "miss"
            return slot.cache, slot

        # At capacity — trimming the best slot is the right eviction strategy.
        trim_n = len(best.tokens) - overlap
        if can_trim_prompt_cache(best.cache):
            trim_prompt_cache(best.cache, trim_n)
            best.tokens = best.tokens[:overlap]
            best.touch()
            self._last_overlap_tokens = overlap
            self._last_action = "trimmed"
            return best.cache, best

        # Architecture refuses trim — drop the slot and rebuild from scratch.
        self._slots.remove(best)
        self._evict_lru_until_room()
        slot = self._new_slot()
        slot.touch()
        self._last_overlap_tokens = 0
        self._last_action = "miss"
        return slot.cache, slot

    # ------------------------------------------------------------------
    # Sampler
    # ------------------------------------------------------------------

    @staticmethod
    def _make_sampler(params: GenerationParams) -> Any | None:
        try:
            from mlx_lm.sample_utils import make_sampler  # type: ignore  # noqa: PLC0415
        except ImportError:
            return None
        return make_sampler(temp=params.temperature, top_p=params.top_p, top_k=params.top_k)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: Iterable[ChatMessage],
        params: GenerationParams,
        cancel: Cancellation | None = None,  # noqa: ARG002 (mlx_lm.generate has no cancel hook)
    ) -> GenerationResult:
        if not self.is_loaded:
            raise RuntimeError("model not loaded")

        from mlx_lm import generate as mlx_generate  # noqa: PLC0415

        prompt = self._format_prompt(messages)
        prompt_tokens = self._tokenize(prompt)
        sampler = self._make_sampler(params)

        async with self._lock:
            cache, slot = self._resolve_cache(prompt_tokens)

            def _run() -> str:
                kwargs: dict = {
                    "model": self._model,
                    "tokenizer": self._tokenizer,
                    "prompt": prompt,
                    "max_tokens": params.max_tokens,
                    "verbose": False,
                }
                if sampler is not None:
                    kwargs["sampler"] = sampler
                if cache is not None:
                    kwargs["prompt_cache"] = cache
                return mlx_generate(**kwargs)

            text: str = await asyncio.to_thread(_run)

            if slot is not None:
                generated = self._tokenize(text) if text else []
                slot.tokens = prompt_tokens + generated
                slot.touch()

        return GenerationResult(
            text=text,
            finish_reason="stop",
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(self._tokenize(text)) if text else 0,
        )

    async def complete(
        self,
        prompt: str,
        params: GenerationParams,
        cancel: Cancellation | None = None,  # noqa: ARG002 — mlx_lm.generate has no cancel hook
    ) -> GenerationResult:
        """Raw text completion via ``mlx_lm.generate`` — no chat template.

        The chat path runs ``apply_chat_template`` on inbound messages; this
        path skips that, sending the prompt verbatim to the model. Goes through
        the same multi-slot prompt cache as ``generate()``.
        """
        if not self.is_loaded:
            raise RuntimeError("model not loaded")

        from mlx_lm import generate as mlx_generate  # noqa: PLC0415

        prompt_tokens = self._tokenize(prompt)
        sampler = self._make_sampler(params)

        async with self._lock:
            cache, slot = self._resolve_cache(prompt_tokens)

            def _run() -> str:
                kwargs: dict = {
                    "model": self._model,
                    "tokenizer": self._tokenizer,
                    "prompt": prompt,
                    "max_tokens": params.max_tokens,
                    "verbose": False,
                }
                if sampler is not None:
                    kwargs["sampler"] = sampler
                if cache is not None:
                    kwargs["prompt_cache"] = cache
                return mlx_generate(**kwargs)

            text: str = await asyncio.to_thread(_run)

            if slot is not None:
                generated = self._tokenize(text) if text else []
                slot.tokens = prompt_tokens + generated
                slot.touch()

        return GenerationResult(
            text=text,
            finish_reason="stop",
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(self._tokenize(text)) if text else 0,
            tool_calls=None,
        )

    async def stream(
        self,
        messages: Iterable[ChatMessage],
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> AsyncIterator[StreamChunk]:
        if not self.is_loaded:
            raise RuntimeError("model not loaded")

        from mlx_lm import stream_generate as mlx_stream  # noqa: PLC0415

        prompt = self._format_prompt(messages)
        prompt_tokens = self._tokenize(prompt)
        sampler = self._make_sampler(params)

        queue: asyncio.Queue[StreamChunk | tuple[str, list[int]] | Exception] = asyncio.Queue(maxsize=64)
        loop = asyncio.get_running_loop()

        async with self._lock:
            cache, slot = self._resolve_cache(prompt_tokens)

            def _producer() -> None:
                try:
                    kwargs: dict = {
                        "model": self._model,
                        "tokenizer": self._tokenizer,
                        "prompt": prompt,
                        "max_tokens": params.max_tokens,
                    }
                    if sampler is not None:
                        kwargs["sampler"] = sampler
                    if cache is not None:
                        kwargs["prompt_cache"] = cache

                    generated_token_ids: list[int] = []
                    for response in mlx_stream(**kwargs):
                        if cancel is not None and bool(cancel):
                            break
                        token_id = getattr(response, "token", None)
                        if token_id is not None:
                            generated_token_ids.append(int(token_id))
                        text = getattr(response, "text", None)
                        if text is None and isinstance(response, str):
                            text = response  # older mlx-lm API
                        if text:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(StreamChunk(text=text)), loop
                            ).result()
                    asyncio.run_coroutine_threadsafe(
                        queue.put(StreamChunk(text="", finish_reason="stop")), loop
                    ).result()
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("__done__", prompt_tokens + generated_token_ids)), loop
                    ).result()
                except Exception as exc:  # noqa: BLE001
                    asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()

            producer_future = asyncio.ensure_future(asyncio.to_thread(_producer))
            try:
                while True:
                    item = await queue.get()
                    if isinstance(item, Exception):
                        raise item
                    if isinstance(item, tuple) and item and item[0] == "__done__":
                        if slot is not None:
                            slot.tokens = list(item[1])
                            slot.touch()
                        return
                    yield item  # type: ignore[misc]
            finally:
                if not producer_future.done():
                    await producer_future
