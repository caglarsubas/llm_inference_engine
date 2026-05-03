"""Dynamic batching — request coalescing for ``/v1/embeddings``.

Concurrent ``/v1/embeddings`` calls hitting the same adapter within a small
wait window get merged into one underlying ``adapter.embed()`` call. Each
caller awaits its own future and gets back its slice of the batched result.

Why this matters
----------------

The original document's "dynamic batching" item from Phase 3 is large enough
to be its own multi-round project for chat completions (continuous batching
for autoregressive decode means reimplementing the inference loop). But for
**embeddings**, batching is cheap to deliver and high-value:

* On embedding-native GGUFs (bge / nomic / e5), one batched ``llama_decode``
  call with N inputs is materially faster than N serial calls.
* On RAG indexing workloads, requests arrive in volume — there's real
  opportunity to coalesce.
* The adapter already detects batched-embed support automatically (round 16)
  and falls back to serial for chat-model misuse, so the coalescer never has
  to know which it's getting.

Limits
------

* **Single-process only.** If uvicorn runs with multiple workers, each worker
  has its own coalescer — no cross-worker batching. The default deployment
  is a single async process, so this is fine for now.
* **Embeddings only.** Chat coalescing without true continuous batching at
  the model level just adds latency. Documented in the README.
* **Wait window cost.** Solo traffic pays up to ``BATCH_MAX_WAIT_MS`` extra
  latency. Default 10 ms is small enough to be invisible on the M5 Max but
  big enough to coalesce burst traffic.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from ..adapters import EmbeddingResult, InferenceAdapter
from ..config import settings
from ..observability import get_logger

log = get_logger("batcher")


@dataclass
class _BatchOutcome:
    """What each waiting request reads back from a flushed batch."""

    embeddings: list[list[float]]  # this caller's slice
    prompt_tokens: int             # this caller's contribution (proportional split)
    batch_id: int                  # join key — identifies the underlying call
    coalesced_with: int            # how many requests merged into this batch
    total_inputs: int              # total inputs the batched call processed
    wait_ms: float                 # how long this caller waited before flush
    adapter_action: str            # adapter.last_embed_action ("batch" | "serial" | "fallback")


@dataclass
class _Pending:
    inputs: list[str]
    submitted_at: float
    future: asyncio.Future = field(default_factory=asyncio.Future)


class _AdapterQueue:
    """One coalescing queue per loaded adapter id.

    Separate queues per adapter so requests for different models never wait
    on each other. Keyed by ``id(adapter)`` so adapter unload + reload (which
    creates a new instance) gets a fresh queue automatically.
    """

    def __init__(self, adapter: InferenceAdapter) -> None:
        self._adapter = adapter
        self._lock = asyncio.Lock()
        self._pending: list[_Pending] = []
        self._flush_scheduled: bool = False
        self._next_batch_id: int = 0

    async def submit(self, inputs: list[str]) -> _BatchOutcome:
        pending = _Pending(inputs=list(inputs), submitted_at=time.perf_counter())
        async with self._lock:
            self._pending.append(pending)
            total = sum(len(p.inputs) for p in self._pending)
            if total >= settings.batch_max_size:
                # Always fire an immediate flush when the size threshold hits,
                # regardless of whether a wait task is already scheduled. The
                # wait task, if any, will harmlessly find an empty queue when
                # its timer expires.
                asyncio.get_running_loop().create_task(self._flush_now())
            elif not self._flush_scheduled:
                self._flush_scheduled = True
                # Wait window — if more requests come in before the timer
                # fires, they piggyback on this same scheduled flush.
                asyncio.get_running_loop().create_task(self._flush_after_wait())
        return await pending.future

    async def _flush_after_wait(self) -> None:
        await asyncio.sleep(settings.batch_max_wait_ms / 1000.0)
        await self._flush_now()

    async def _flush_now(self) -> None:
        async with self._lock:
            queue = self._pending
            self._pending = []
            self._flush_scheduled = False
            batch_id = self._next_batch_id
            self._next_batch_id += 1

        if not queue:
            return

        # Concatenate all inputs; remember each caller's slice boundaries so
        # we can hand each future its own piece of the result.
        all_inputs: list[str] = []
        boundaries: list[tuple[int, int]] = []  # [(start, end), ...]
        for p in queue:
            start = len(all_inputs)
            all_inputs.extend(p.inputs)
            boundaries.append((start, len(all_inputs)))

        flush_at = time.perf_counter()

        try:
            result: EmbeddingResult = await self._adapter.embed(all_inputs)
        except Exception as exc:  # noqa: BLE001 — fan out to every waiter
            for p in queue:
                if not p.future.done():
                    p.future.set_exception(exc)
            return

        adapter_action = getattr(self._adapter, "last_embed_action", "unknown")
        coalesced = len(queue)
        total_inputs = len(all_inputs)
        # llama-cpp-python returns total tokens for the whole batch; split it
        # proportionally by per-caller input count so each caller gets a fair
        # share for usage accounting.
        for (start, end), p in zip(boundaries, queue, strict=True):
            share = (end - start) / total_inputs if total_inputs else 0.0
            outcome = _BatchOutcome(
                embeddings=result.embeddings[start:end],
                prompt_tokens=int(round(result.prompt_tokens * share)),
                batch_id=batch_id,
                coalesced_with=coalesced,
                total_inputs=total_inputs,
                wait_ms=(flush_at - p.submitted_at) * 1000.0,
                adapter_action=adapter_action,
            )
            if not p.future.done():
                p.future.set_result(outcome)

        if coalesced > 1:
            log.info(
                "batch.flushed",
                batch_id=batch_id,
                coalesced=coalesced,
                total_inputs=total_inputs,
                adapter_action=adapter_action,
            )


class EmbedCoalescer:
    """Process-global registry of per-adapter queues."""

    def __init__(self) -> None:
        self._queues: dict[int, _AdapterQueue] = {}
        self._lock = asyncio.Lock()

    async def submit(
        self, adapter: InferenceAdapter, inputs: list[str]
    ) -> _BatchOutcome:
        if not settings.batch_enabled:
            # Pass-through path. Still emit a uniform _BatchOutcome so callers
            # don't branch on whether batching was on.
            result = await adapter.embed(inputs)
            adapter_action = getattr(adapter, "last_embed_action", "unknown")
            return _BatchOutcome(
                embeddings=result.embeddings,
                prompt_tokens=result.prompt_tokens,
                batch_id=-1,
                coalesced_with=1,
                total_inputs=len(inputs),
                wait_ms=0.0,
                adapter_action=adapter_action,
            )

        key = id(adapter)
        async with self._lock:
            q = self._queues.get(key)
            if q is None:
                q = _AdapterQueue(adapter)
                self._queues[key] = q
        return await q.submit(inputs)
