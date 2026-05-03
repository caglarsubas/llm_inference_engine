"""Tool execution timing — store mechanics + cross-event correlation."""

from __future__ import annotations

import time

import pytest

from inference_engine.api._tool_audit import (
    ToolCallTimingStore,
    emit_tool_calls,
    emit_tool_results,
    get_timing_store,
    set_timing_store,
)
from inference_engine.config import settings
from inference_engine.observability import span
from inference_engine.schemas import ChatMessage


# ---------------------------------------------------------------------------
# Store unit tests
# ---------------------------------------------------------------------------


def test_record_then_consume_returns_elapsed_seconds() -> None:
    s = ToolCallTimingStore(ttl_seconds=60.0, max_entries=100)
    s.record("call_1")
    time.sleep(0.005)
    elapsed = s.consume("call_1")
    assert elapsed is not None
    assert elapsed >= 0.005


def test_consume_unknown_returns_none() -> None:
    s = ToolCallTimingStore(ttl_seconds=60.0, max_entries=100)
    assert s.consume("ghost") is None


def test_consume_pops_entry() -> None:
    s = ToolCallTimingStore(ttl_seconds=60.0, max_entries=100)
    s.record("once")
    assert s.consume("once") is not None
    # Second consume sees nothing — entry was popped.
    assert s.consume("once") is None


def test_ttl_eviction_drops_stale_entries() -> None:
    s = ToolCallTimingStore(ttl_seconds=0.05, max_entries=100)
    s.record("a")
    time.sleep(0.1)
    # Recording another id triggers the sweep; "a" is now expired.
    s.record("b")
    assert s.consume("a") is None
    assert s.consume("b") is not None


def test_lru_eviction_caps_at_max_entries() -> None:
    s = ToolCallTimingStore(ttl_seconds=60.0, max_entries=3)
    for i in range(5):
        s.record(f"c{i}")
    assert len(s) == 3
    # Oldest two are gone.
    assert s.consume("c0") is None
    assert s.consume("c1") is None
    # Newer three retained.
    assert s.consume("c4") is not None


def test_empty_call_id_is_ignored() -> None:
    """Defensive: don't record/consume blank ids — that's noise from malformed events."""
    s = ToolCallTimingStore(ttl_seconds=60.0, max_entries=100)
    s.record("")
    assert len(s) == 0
    assert s.consume("") is None


# ---------------------------------------------------------------------------
# emit_* hooks — cross-event correlation
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_store(monkeypatch):
    """Install a fresh, generously-sized store; clean up after each test."""
    s = ToolCallTimingStore(ttl_seconds=60.0, max_entries=100)
    set_timing_store(s)
    monkeypatch.setattr(settings, "tool_audit_enabled", True)
    yield s
    set_timing_store(None)


def test_emit_tool_calls_records_each_call_id(fresh_store) -> None:
    tool_calls = [
        {"id": "call_a", "function": {"name": "f1", "arguments": "{}"}},
        {"id": "call_b", "function": {"name": "f2", "arguments": "{}"}},
    ]
    with span("test") as s:
        emit_tool_calls(s, tool_calls)

    # The store should now hold both ids.
    assert fresh_store.consume("call_a") is not None
    assert fresh_store.consume("call_b") is not None


def test_emit_tool_results_correlates_with_prior_call(fresh_store, _session_exporter) -> None:
    """The wall-clock between tool_call emission and tool_result receipt
    surfaces as ``tool.execution_ms`` on the result event."""
    _session_exporter.clear()

    tool_calls = [{"id": "call_xyz", "function": {"name": "get_weather", "arguments": "{}"}}]
    with span("turn_n") as s:
        emit_tool_calls(s, tool_calls)

    # Simulate the agent doing real work between turns.
    time.sleep(0.02)

    messages = [
        ChatMessage(role="user", content="..."),
        ChatMessage(role="tool", tool_call_id="call_xyz", content="Cloudy, 12C"),
    ]
    with span("turn_n_plus_1") as s:
        emit_tool_results(s, messages)

    # Find the tool_result event and check the new attribute.
    spans = _session_exporter.get_finished_spans()
    result_events = [
        e for sp in spans for e in sp.events if e.name == "gen_ai.tool_result"
    ]
    assert len(result_events) == 1
    e = result_events[0]
    assert "tool.execution_ms" in e.attributes
    assert e.attributes["tool.execution_ms"] >= 20.0  # we slept 20ms between record/consume


def test_emit_tool_results_without_prior_call_omits_timing(fresh_store, _session_exporter) -> None:
    """A tool_result with no matching prior tool_call has no execution_ms field
    (we don't fabricate a value when we don't have one)."""
    _session_exporter.clear()

    messages = [ChatMessage(role="tool", tool_call_id="orphan", content="result")]
    with span("orphan_turn") as s:
        emit_tool_results(s, messages)

    result_events = [
        e
        for sp in _session_exporter.get_finished_spans()
        for e in sp.events
        if e.name == "gen_ai.tool_result"
    ]
    assert len(result_events) == 1
    assert "tool.execution_ms" not in result_events[0].attributes


def test_consume_only_fires_once_per_call_id(fresh_store, _session_exporter) -> None:
    """If the agent inexplicably sends the same tool_result twice, the second
    consume returns None — no fake repeat-timing on the second event."""
    _session_exporter.clear()

    tool_calls = [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]
    with span("call_turn") as s:
        emit_tool_calls(s, tool_calls)

    msg = ChatMessage(role="tool", tool_call_id="c1", content="r")
    with span("first_result") as s:
        emit_tool_results(s, [msg])
    with span("second_result") as s:
        emit_tool_results(s, [msg])

    result_events = [
        e
        for sp in _session_exporter.get_finished_spans()
        for e in sp.events
        if e.name == "gen_ai.tool_result"
    ]
    assert len(result_events) == 2
    assert "tool.execution_ms" in result_events[0].attributes
    assert "tool.execution_ms" not in result_events[1].attributes


def test_get_timing_store_returns_singleton() -> None:
    """The global accessor returns the same instance across calls."""
    set_timing_store(None)  # reset so we test lazy init
    s1 = get_timing_store()
    s2 = get_timing_store()
    assert s1 is s2
    set_timing_store(None)
