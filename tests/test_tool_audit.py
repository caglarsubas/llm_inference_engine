"""Tool-call audit — schema acceptance, event emission, truncation, disable gate."""

from __future__ import annotations

import pytest

from inference_engine.api._tool_audit import emit_tool_calls, emit_tool_results
from inference_engine.config import settings
from inference_engine.observability import span
from inference_engine.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
)


# ---------------------------------------------------------------------------
# Schema acceptance — the OpenAI tool-calling shape parses cleanly.
# ---------------------------------------------------------------------------


def test_assistant_message_with_tool_calls_parses() -> None:
    msg = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call_abc",
                function=ToolCallFunction(name="get_weather", arguments='{"city": "SF"}'),
            )
        ],
    )
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].function.name == "get_weather"
    assert msg.content is None  # null content allowed when tool_calls present


def test_tool_message_with_id_parses() -> None:
    msg = ChatMessage(role="tool", content="Cloudy, 12C", tool_call_id="call_abc")
    assert msg.tool_call_id == "call_abc"


def test_request_accepts_tools_and_tool_choice() -> None:
    req = ChatCompletionRequest(
        model="x",
        messages=[ChatMessage(role="user", content="hi")],
        tools=[
            ToolDefinition(
                function={
                    "name": "get_weather",
                    "description": "Returns current weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            )
        ],
        tool_choice="auto",
    )
    assert req.tools is not None
    assert req.tools[0].function["name"] == "get_weather"
    assert req.tool_choice == "auto"


# ---------------------------------------------------------------------------
# Event emission — verified through OTel's in-memory exporter (shared via
# conftest.py).
# ---------------------------------------------------------------------------


@pytest.fixture
def exporter(_session_exporter):
    _session_exporter.clear()
    return _session_exporter


def test_tool_results_emit_one_event_per_tool_message(exporter) -> None:
    messages = [
        ChatMessage(role="user", content="check the weather"),
        ChatMessage(role="tool", content="Cloudy, 12C", tool_call_id="call_1"),
        ChatMessage(role="tool", content="Rainy, 8C", tool_call_id="call_2"),
    ]

    with span("chat.generate") as s:
        n = emit_tool_results(s, messages)
    assert n == 2

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    events = finished[0].events
    tool_events = [e for e in events if e.name == "gen_ai.tool_result"]
    assert len(tool_events) == 2

    ids = {e.attributes["gen_ai.tool.call.id"] for e in tool_events}
    assert ids == {"call_1", "call_2"}


def test_tool_calls_emit_one_event_per_tool_call(exporter) -> None:
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"SF"}'}},
        {"id": "call_2", "type": "function", "function": {"name": "get_time", "arguments": "{}"}},
    ]

    with span("chat.generate") as s:
        n = emit_tool_calls(s, tool_calls)
    assert n == 2

    events = exporter.get_finished_spans()[0].events
    tool_events = [e for e in events if e.name == "gen_ai.tool_call"]
    assert len(tool_events) == 2

    by_name = {e.attributes["gen_ai.tool.name"]: e for e in tool_events}
    assert by_name["get_weather"].attributes["gen_ai.tool.call.arguments"] == '{"city":"SF"}'
    assert by_name["get_weather"].attributes["gen_ai.tool.call.arguments_truncated"] is False


def test_arguments_truncated_when_over_cap(exporter, monkeypatch) -> None:
    monkeypatch.setattr(settings, "tool_audit_max_payload_chars", 16)

    big_args = "x" * 1000
    tool_calls = [{"id": "call_1", "function": {"name": "f", "arguments": big_args}}]

    with span("chat.generate") as s:
        emit_tool_calls(s, tool_calls)

    e = next(
        ev for ev in exporter.get_finished_spans()[0].events if ev.name == "gen_ai.tool_call"
    )
    assert e.attributes["gen_ai.tool.call.arguments_truncated"] is True
    assert len(e.attributes["gen_ai.tool.call.arguments"]) == 16


def test_disabled_audit_emits_nothing(exporter, monkeypatch) -> None:
    monkeypatch.setattr(settings, "tool_audit_enabled", False)

    with span("chat.generate") as s:
        n_results = emit_tool_results(
            s, [ChatMessage(role="tool", content="x", tool_call_id="c")]
        )
        n_calls = emit_tool_calls(
            s, [{"id": "c", "function": {"name": "f", "arguments": "{}"}}]
        )

    assert n_results == 0
    assert n_calls == 0
    events = exporter.get_finished_spans()[0].events
    assert not any(e.name.startswith("gen_ai.tool_") for e in events)


def test_no_tool_messages_no_events(exporter) -> None:
    """A regular chat (no tools) shouldn't emit any tool events."""
    with span("chat.generate") as s:
        emit_tool_results(s, [ChatMessage(role="user", content="hi")])
        emit_tool_calls(s, None)

    events = exporter.get_finished_spans()[0].events
    assert not any(e.name.startswith("gen_ai.tool_") for e in events)


def test_tool_result_includes_optional_name_attribute(exporter) -> None:
    messages = [
        ChatMessage(
            role="tool", content="result", tool_call_id="call_x", name="weather_tool"
        )
    ]
    with span("chat.generate") as s:
        emit_tool_results(s, messages)

    e = next(
        ev for ev in exporter.get_finished_spans()[0].events if ev.name == "gen_ai.tool_result"
    )
    assert e.attributes["gen_ai.tool.name"] == "weather_tool"
