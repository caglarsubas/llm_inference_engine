"""Regression tests for LlamaCppAdapter._to_llama_messages tool-call rendering.

Background
----------
OpenAI spec: ``tool_calls[].function.arguments`` is a JSON-encoded **string**.
Many HuggingFace-style GGUF chat templates (Nemotron-Nano, Qwen-coder,
GLM-4-tool, …) iterate that field with Jinja filters like ``arguments | items``
which only work on a **mapping**. ``transformers.apply_chat_template`` papers
this over by JSON-decoding the string before render; llama-cpp-python's Jinja
path does not, so turn 2 of any tool-calling conversation crashed with
``TypeError: Can only get item pairs from a mapping.``

These tests pin the shim that mirrors HF's behavior at the adapter boundary.
"""

from __future__ import annotations

from inference_engine.adapters.llama_cpp import LlamaCppAdapter
from inference_engine.schemas import ChatMessage, ToolCall, ToolCallFunction


def _assistant_with_tool_call(arguments: str) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call_1",
                function=ToolCallFunction(name="list_files", arguments=arguments),
            )
        ],
    )


def test_arguments_json_string_is_parsed_to_dict() -> None:
    """Templates iterating ``arguments | items`` need a mapping, not a string."""
    msgs = LlamaCppAdapter._to_llama_messages(
        [_assistant_with_tool_call('{"path": "/tmp", "limit": 10}')]
    )

    args = msgs[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict)
    assert args == {"path": "/tmp", "limit": 10}


def test_empty_arguments_string_becomes_empty_dict() -> None:
    """``"{}"`` and ``""`` both represent zero-argument calls — render as ``{}``
    so ``arguments | items`` yields an empty iterator instead of choking on a
    string."""
    for empty in ("", "{}"):
        msgs = LlamaCppAdapter._to_llama_messages([_assistant_with_tool_call(empty)])
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == {}


def test_non_json_arguments_string_falls_back_to_raw() -> None:
    """OpenAI-strict templates that index ``arguments`` as a string keep working
    when the model emits free-form text instead of JSON (rare, but valid)."""
    msgs = LlamaCppAdapter._to_llama_messages(
        [_assistant_with_tool_call("not-json-at-all")]
    )
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == "not-json-at-all"


def test_arguments_holding_json_array_is_passed_through() -> None:
    """A few tools take a positional list. JSON arrays decode to ``list``;
    templates that expect that shape get it. Tests the lenient JSON path
    independently of the dict case."""
    msgs = LlamaCppAdapter._to_llama_messages(
        [_assistant_with_tool_call('["a", "b", "c"]')]
    )
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == ["a", "b", "c"]


def test_multi_turn_round_trip_preserves_other_fields() -> None:
    """Full Nemotron-style turn-2 payload: user → assistant w/ tool_calls →
    tool result. We're only fixing arguments shape; everything else
    (role, content, tool_call_id, ids) must round-trip verbatim."""
    history = [
        ChatMessage(role="user", content="list files please"),
        _assistant_with_tool_call("{}"),
        ChatMessage(role="tool", content="[]", tool_call_id="call_1"),
    ]
    msgs = LlamaCppAdapter._to_llama_messages(history)

    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[0]["content"] == "list files please"
    # Assistant turn: arguments became {} dict, ids preserved.
    assistant = msgs[1]
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["type"] == "function"
    assert assistant["tool_calls"][0]["function"]["name"] == "list_files"
    assert assistant["tool_calls"][0]["function"]["arguments"] == {}
    # Tool result: tool_call_id preserved.
    assert msgs[2]["content"] == "[]"
    assert msgs[2]["tool_call_id"] == "call_1"


def test_no_tool_calls_messages_unchanged() -> None:
    """Plain user/assistant chat — no tool_calls field on the rendered dicts."""
    msgs = LlamaCppAdapter._to_llama_messages(
        [
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]
    )
    assert "tool_calls" not in msgs[0]
    assert "tool_calls" not in msgs[1]
    assert msgs[0]["content"] == "hi"
    assert msgs[1]["content"] == "hello"
