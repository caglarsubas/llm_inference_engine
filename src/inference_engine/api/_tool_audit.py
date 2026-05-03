"""Tool-call audit — emit ``gen_ai.tool_*`` span events + execution timing.

Two event shapes:

* ``gen_ai.tool_result`` — fired for every ``role="tool"`` message in the
  inbound request. The agent ran the tool externally and is now passing the
  result back to the model; we capture (tool_call_id, content).

* ``gen_ai.tool_call`` — fired for every ``tool_calls`` entry the model
  emitted in its response. Captures (id, function.name, function.arguments).

Both event shapes truncate the variable-length payload (``content`` /
``arguments``) to ``settings.tool_audit_max_payload_chars`` so spans don't
balloon when an agent passes a 2 MB JSON blob through. Truncation is signalled
explicitly via a ``*_truncated=True`` flag so downstream consumers don't
mistake a clipped string for the original.

Disable the whole thing with ``TOOL_AUDIT_ENABLED=false``.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

from ..config import settings
from ..observability import Span
from ..schemas import ChatMessage


# ---------------------------------------------------------------------------
# Tool execution timing — correlate gen_ai.tool_call (turn N) with
# gen_ai.tool_result (turn N+1) on the same call_id, surface the wall-clock
# gap as ``tool.execution_ms`` on the result event.
# ---------------------------------------------------------------------------


class ToolCallTimingStore:
    """In-memory ``call_id → emit_timestamp`` store, TTL- and LRU-bounded.

    The store is process-global. With a single uvicorn worker (the default
    deployment shape on this engine) all tool_call emissions and tool_result
    receipts hit the same instance, so the join works. With multiple workers,
    each worker has its own store and a tool_result that arrives at a
    different worker than the one that emitted the call sees no timing — fine
    for now, documented as a deployment caveat.

    Two bounds:

    * **TTL** — entries older than ``ttl_seconds`` are swept on every record()
      so a tool that's never resolved doesn't pin memory forever.
    * **LRU max_entries** — hard cap on the dict size to prevent unbounded
      growth from a runaway agent that opens calls but never closes them.

    Both bounds are configurable via ``TOOL_TIMING_TTL_SECONDS`` and
    ``TOOL_TIMING_MAX_ENTRIES``.
    """

    def __init__(self, ttl_seconds: float, max_entries: int) -> None:
        self._entries: OrderedDict[str, float] = OrderedDict()
        self._ttl = ttl_seconds
        self._max = max_entries

    def record(self, call_id: str) -> None:
        """Stamp this call_id with the current monotonic time."""
        if not call_id:
            return
        now = time.monotonic()
        # Sweep expired entries before inserting — keeps the dict small even
        # under bursty traffic where consume() rarely runs.
        self._evict_expired(now)
        # Re-record overwrites any prior timestamp for the same id (shouldn't
        # happen in practice — call ids are unique — but keeps semantics
        # idempotent).
        self._entries[call_id] = now
        self._entries.move_to_end(call_id)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def consume(self, call_id: str) -> float | None:
        """Pop the entry and return ``elapsed_seconds`` since record(), or None."""
        if not call_id:
            return None
        ts = self._entries.pop(call_id, None)
        if ts is None:
            return None
        return time.monotonic() - ts

    def __len__(self) -> int:
        return len(self._entries)

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self._ttl
        # Iterate over a snapshot so we can mutate during the walk.
        for k in [k for k, ts in self._entries.items() if ts < cutoff]:
            self._entries.pop(k, None)


# Process-global instance. Constructed once in AppState; tests use a fresh
# instance per run via the module-level setter below.
_timing_store: ToolCallTimingStore | None = None


def get_timing_store() -> ToolCallTimingStore:
    """Return the process-global timing store, lazy-initialising on first use."""
    global _timing_store
    if _timing_store is None:
        _timing_store = ToolCallTimingStore(
            ttl_seconds=settings.tool_timing_ttl_seconds,
            max_entries=settings.tool_timing_max_entries,
        )
    return _timing_store


def set_timing_store(store: ToolCallTimingStore | None) -> None:
    """Override the global store. Used by tests to reset state cleanly."""
    global _timing_store
    _timing_store = store


def _truncate(value: str | None, *, cap: int) -> tuple[str, bool]:
    if value is None:
        return "", False
    if len(value) <= cap:
        return value, False
    return value[:cap], True


def emit_tool_results(span: Span, messages: list[ChatMessage]) -> int:
    """Emit one ``gen_ai.tool_result`` event per inbound tool message.

    Returns the count emitted; 0 if audit is disabled or no tool messages
    are present. The chat span gets the count bound to it so downstream
    aggregations can compute "tools per turn" without scanning events.

    If a matching ``gen_ai.tool_call`` for the same ``call_id`` was previously
    recorded in the timing store, the elapsed wall-clock is added to this
    event as ``tool.execution_ms`` — that's the agent-side latency between
    "model decided to call this tool" and "agent fed back the result".
    """
    if not settings.tool_audit_enabled:
        return 0

    cap = settings.tool_audit_max_payload_chars
    timing = get_timing_store()
    count = 0
    for m in messages:
        if m.role != "tool":
            continue
        content, truncated = _truncate(m.content, cap=cap)
        call_id = m.tool_call_id or ""
        attrs: dict[str, Any] = {
            "gen_ai.tool.call.id": call_id,
            "gen_ai.tool.result.content": content,
            "gen_ai.tool.result.content_truncated": truncated,
        }
        if m.name:
            attrs["gen_ai.tool.name"] = m.name

        # Correlate against the prior tool_call emission.
        elapsed = timing.consume(call_id)
        if elapsed is not None:
            attrs["tool.execution_ms"] = round(elapsed * 1000.0, 2)

        span.event("gen_ai.tool_result", **attrs)
        count += 1
    return count


class ToolCallReassembler:
    """Merge streamed tool-call deltas into complete OpenAI tool-call dicts.

    Streaming tool calls arrive as a sequence of per-chunk fragments keyed by
    ``index``. Each fragment may carry any subset of ``id``, ``type``,
    ``function.name``, or a partial slice of ``function.arguments`` — clients
    concatenate ``arguments`` per index until the stream finishes. We do the
    same here so we can emit one ``gen_ai.tool_call`` event per *completed*
    call at stream end.

    The output of ``assembled()`` is shaped identically to the blocking-mode
    response.tool_calls so ``emit_tool_calls`` can consume it unchanged.
    """

    def __init__(self) -> None:
        self._by_index: dict[int, dict] = {}

    def feed(self, deltas: list[dict] | None) -> None:
        if not deltas:
            return
        for d in deltas:
            idx = int(d.get("index", 0))
            bucket = self._by_index.setdefault(
                idx,
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if d.get("id"):
                bucket["id"] = d["id"]
            if d.get("type"):
                bucket["type"] = d["type"]
            fn = d.get("function") or {}
            if fn.get("name"):
                bucket["function"]["name"] = fn["name"]
            if "arguments" in fn and fn["arguments"] is not None:
                # Arguments arrive as fragments — accumulate.
                bucket["function"]["arguments"] += fn["arguments"]

    def has_calls(self) -> bool:
        return bool(self._by_index)

    def assembled(self) -> list[dict]:
        """Return completed tool calls ordered by their stream index."""
        return [self._by_index[k] for k in sorted(self._by_index.keys())]


def emit_tool_calls(span: Span, tool_calls: list[dict] | None) -> int:
    """Emit one ``gen_ai.tool_call`` event per tool the model invoked.

    ``tool_calls`` is the raw OpenAI shape returned by the backend
    (``[{"id", "type", "function": {"name", "arguments"}}, ...]``).

    Side effect: each call_id is recorded in the process-global timing store
    so a future ``gen_ai.tool_result`` event for the same id can compute
    ``tool.execution_ms`` (round 21).
    """
    if not settings.tool_audit_enabled or not tool_calls:
        return 0

    cap = settings.tool_audit_max_payload_chars
    timing = get_timing_store()
    count = 0
    for tc in tool_calls:
        function = tc.get("function") or {}
        args, truncated = _truncate(function.get("arguments"), cap=cap)
        call_id = tc.get("id") or ""
        span.event(
            "gen_ai.tool_call",
            **{
                "gen_ai.tool.call.id": call_id,
                "gen_ai.tool.name": function.get("name") or "",
                "gen_ai.tool.call.arguments": args,
                "gen_ai.tool.call.arguments_truncated": truncated,
            },
        )
        # Stamp the timing store after emission so the test fixtures can
        # observe the event without racing the record path.
        if call_id:
            timing.record(call_id)
        count += 1
    return count
