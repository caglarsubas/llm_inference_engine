"""Tests for the vendor-XML normalization layer.

The payload strings are kept *verbatim* from the original bug report
(DeclarAI / Nemotron-3-nano) so a regression that puts the leak back will
fail this file even if it passes synthetic well-formed XML.
"""

from __future__ import annotations

import json

from inference_engine.response_normalize import (
    StreamDelta,
    StreamNormalizer,
    infer_model_capabilities,
    normalize_assistant_text,
)


# ---------------------------------------------------------------------------
# Blocking normalizer — the real bug-report payload
# ---------------------------------------------------------------------------


# Verbatim from the DeclarAI bug report. Note the *orphan* </think> close tag
# (no opening tag) and the Nemotron NIM `<function=NAME>` form. This is what
# llama-cpp-python returns when no Nemotron grammar is loaded.
NEMOTRON_BUG_REPORT_PAYLOAD = (
    "We need to respond with feature engineering suggestions ... \n"
    "</think>\n"
    "<tool_call>\n"
    "<function=get_data_dictionary>\n"
    "</function>\n"
    "</tool_call>"
)


def test_nemotron_bug_report_payload_is_repaired() -> None:
    """The exact payload from the bug report must come out clean."""
    out = normalize_assistant_text(NEMOTRON_BUG_REPORT_PAYLOAD, tools_requested=True)

    # 1. content is null (or whitespace-only -> None) on a tool-calling turn.
    assert out.content is None
    # 2. reasoning is captured into its own channel.
    assert out.reasoning_content is not None
    assert "feature engineering suggestions" in out.reasoning_content
    # 3. tool call is structured.
    assert out.tool_calls is not None and len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_data_dictionary"
    assert tc["function"]["arguments"] == "{}"
    assert tc["id"].startswith("call_")
    # 4. finish_reason is flipped to tool_calls.
    assert out.finish_reason == "tool_calls"


def test_nemotron_function_with_parameters_preserves_types() -> None:
    """`<parameter=key>value</parameter>` extracts and JSON-coerces values."""
    raw = (
        "</think>\n"
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>San Francisco</parameter>\n"
        "<parameter=days>3</parameter>\n"
        "<parameter=imperial>true</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    out = normalize_assistant_text(raw, tools_requested=True)
    assert out.tool_calls is not None
    args = json.loads(out.tool_calls[0]["function"]["arguments"])
    assert args["city"] == "San Francisco"  # plain string survives
    assert args["days"] == 3                # JSON-coerced to int
    assert args["imperial"] is True         # JSON-coerced to bool


def test_orphan_close_think_only() -> None:
    """`prose </think>` with no opening tag splits into reasoning + content."""
    raw = "Let me consider... </think>\nThe answer is 42."
    out = normalize_assistant_text(raw)
    assert out.reasoning_content == "Let me consider..."
    assert out.content == "The answer is 42."
    assert out.tool_calls is None
    assert out.finish_reason == "stop"


def test_orphan_open_think_only() -> None:
    """`<think> prose` with no closing tag captures everything as reasoning."""
    raw = "Final answer: 42\n<think>actually let me reconsider"
    out = normalize_assistant_text(raw)
    assert out.reasoning_content == "actually let me reconsider"
    assert out.content == "Final answer: 42"


def test_well_formed_think_pair_still_works() -> None:
    raw = "<think>cogito ergo sum</think>I think therefore I am."
    out = normalize_assistant_text(raw)
    assert out.reasoning_content == "cogito ergo sum"
    assert out.content == "I think therefore I am."


def test_pre_tool_call_text_is_reclassified_as_reasoning() -> None:
    """Anything the model says BEFORE a tool call is reasoning, by definition."""
    raw = (
        "I'll need to check the data dictionary.\n"
        "<tool_call>\n"
        "<function=get_data_dictionary>\n"
        "</function>\n"
        "</tool_call>"
    )
    out = normalize_assistant_text(raw, tools_requested=True)
    assert out.content is None
    assert out.reasoning_content == "I'll need to check the data dictionary."
    assert out.tool_calls is not None and len(out.tool_calls) == 1


def test_existing_tool_calls_are_not_re_parsed() -> None:
    """Pre-parsed tool_calls keep their ids; only <think> is stripped."""
    raw = "<think>reasoning here</think>"
    existing = [
        {
            "id": "call_originally_assigned",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }
    ]
    out = normalize_assistant_text(raw, existing_tool_calls=existing, tools_requested=True)
    assert out.tool_calls == existing
    assert out.reasoning_content == "reasoning here"
    assert out.finish_reason == "tool_calls"


def test_json_payload_tool_call_format() -> None:
    """Some templates emit `<tool_call>[{...}]</tool_call>` as JSON."""
    raw = (
        '<tool_call>[{"name": "get_weather", "arguments": {"city": "SF"}}]</tool_call>'
    )
    out = normalize_assistant_text(raw, tools_requested=True)
    assert out.tool_calls is not None and len(out.tool_calls) == 1
    args = json.loads(out.tool_calls[0]["function"]["arguments"])
    assert args == {"city": "SF"}


def test_no_markup_returns_unchanged() -> None:
    out = normalize_assistant_text("hello world", tools_requested=False)
    assert out.content == "hello world"
    assert out.reasoning_content is None
    assert out.tool_calls is None


def test_empty_string() -> None:
    out = normalize_assistant_text("", tools_requested=False)
    assert out.content is None
    assert out.reasoning_content is None
    assert out.tool_calls is None


# ---------------------------------------------------------------------------
# Capability heuristic
# ---------------------------------------------------------------------------


def test_capabilities_nemotron_reasoning() -> None:
    caps = infer_model_capabilities("nemotron-3-nano:30b", backend="llama_cpp", fmt="gguf")
    assert caps["reasoning"] is True
    assert caps["tool_calling_mode"] == "native"


def test_capabilities_qwen3_base_is_NOT_reasoning() -> None:
    """The base qwen3 chat models don't think; only -thinking / qwq do."""
    caps = infer_model_capabilities("qwen3:8b", backend="llama_cpp", fmt="gguf")
    assert caps["reasoning"] is False


def test_capabilities_qwen3_thinking_is_reasoning() -> None:
    caps = infer_model_capabilities("qwen3-thinking:8b", backend="llama_cpp", fmt="gguf")
    assert caps["reasoning"] is True


def test_capabilities_qwq_is_reasoning() -> None:
    caps = infer_model_capabilities("qwq:32b", backend="llama_cpp", fmt="gguf")
    assert caps["reasoning"] is True


def test_capabilities_mlx_is_unsupported_tools() -> None:
    """MLX-LM has no tool-calling plumbing today — be honest about it."""
    caps = infer_model_capabilities("llama-3.2-1b-instruct-4bit:mlx", backend="mlx", fmt="mlx")
    assert caps["tool_calling_mode"] == "unsupported"


def test_capabilities_plain_llama_is_not_reasoning() -> None:
    caps = infer_model_capabilities("llama3.2:3b", backend="llama_cpp", fmt="gguf")
    assert caps["reasoning"] is False
    assert caps["tool_calling_mode"] == "native"


# ---------------------------------------------------------------------------
# StreamNormalizer
# ---------------------------------------------------------------------------


def _collect(deltas: list[StreamDelta]) -> dict:
    """Aggregate StreamDelta frames into terminal content/reasoning/tool_calls."""
    out: dict = {"content": "", "reasoning": "", "tool_calls": []}
    for d in deltas:
        if d.content:
            out["content"] += d.content
        if d.reasoning_content:
            out["reasoning"] += d.reasoning_content
        if d.tool_call:
            out["tool_calls"].append(d.tool_call)
    return out


def test_stream_plain_text_passes_through() -> None:
    n = StreamNormalizer()
    deltas: list[StreamDelta] = []
    for tok in ["Hello", " world", "!"]:
        deltas.extend(n.feed(tok))
    deltas.extend(n.flush())
    agg = _collect(deltas)
    assert agg["content"] == "Hello world!"
    assert agg["reasoning"] == ""
    assert agg["tool_calls"] == []


def test_stream_nemotron_bug_report_payload_token_split() -> None:
    """Feed the bug-report payload one character at a time and verify clean split.

    Because the Nemotron chat template silently pre-emits ``<think>`` before
    generation starts, we tell the normalizer ``expects_reasoning_prelude=True``.
    """
    n = StreamNormalizer(tools_requested=True, expects_reasoning_prelude=True)
    deltas: list[StreamDelta] = []
    for ch in NEMOTRON_BUG_REPORT_PAYLOAD:
        deltas.extend(n.feed(ch))
    deltas.extend(n.flush())

    agg = _collect(deltas)
    # No raw vendor markup ever reached the content channel.
    assert "<tool_call>" not in agg["content"]
    assert "</think>" not in agg["content"]
    assert "<function=" not in agg["content"]
    # The prelude is captured as reasoning.
    assert "feature engineering" in agg["reasoning"]
    # The tool call is structured and complete.
    assert len(agg["tool_calls"]) == 1
    assert agg["tool_calls"][0]["function"]["name"] == "get_data_dictionary"
    assert agg["tool_calls"][0]["function"]["arguments"] == "{}"
    assert n.has_tool_calls()


def test_stream_nemotron_bug_report_payload_single_chunk() -> None:
    """Same payload in one shot — verifies the buffer doesn't get stuck."""
    n = StreamNormalizer(tools_requested=True, expects_reasoning_prelude=True)
    deltas = n.feed(NEMOTRON_BUG_REPORT_PAYLOAD)
    deltas.extend(n.flush())
    agg = _collect(deltas)
    assert "feature engineering" in agg["reasoning"]
    assert len(agg["tool_calls"]) == 1
    assert agg["tool_calls"][0]["function"]["name"] == "get_data_dictionary"


def test_stream_holdback_prevents_tag_leak_across_chunks() -> None:
    """A naive emitter would push `<think` before `>` arrives — we don't."""
    n = StreamNormalizer(tools_requested=False, expects_reasoning_prelude=False)
    # Hello there + <think> split across chunks at the tag boundary.
    pieces = ["Hello there", "<thi", "nk>cogito", " ergo", " sum</thi", "nk>done"]
    deltas: list[StreamDelta] = []
    for p in pieces:
        deltas.extend(n.feed(p))
    deltas.extend(n.flush())
    agg = _collect(deltas)
    assert agg["content"] == "Hello theredone"
    assert "cogito ergo sum" in agg["reasoning"]


def test_stream_tool_call_close_tag_split_across_chunks() -> None:
    """`</tool_call>` straddling a chunk boundary must still parse cleanly."""
    n = StreamNormalizer(tools_requested=True, expects_reasoning_prelude=True)
    pieces = [
        "thinking...\n</think>\n<tool_call>\n<function=foo>\n<parameter=x>1</parameter>\n</func",
        "tion>\n</tool_",
        "call>",
    ]
    deltas: list[StreamDelta] = []
    for p in pieces:
        deltas.extend(n.feed(p))
    deltas.extend(n.flush())
    agg = _collect(deltas)
    assert len(agg["tool_calls"]) == 1
    assert agg["tool_calls"][0]["function"]["name"] == "foo"
    args = json.loads(agg["tool_calls"][0]["function"]["arguments"])
    assert args == {"x": 1}


def test_stream_pure_text_with_expects_reasoning_flag_still_works() -> None:
    """A reasoning-flagged model that happens to skip <think> still flushes cleanly."""
    n = StreamNormalizer(tools_requested=False, expects_reasoning_prelude=True)
    # Model emits straight content with no `</think>` ever — flush moves it
    # through the reasoning channel (which is the conservative interpretation
    # — the model template pre-emitted `<think>`, so absent a close tag,
    # everything is reasoning). Client UIs can render reasoning as
    # collapsed-by-default text.
    deltas = n.feed("just a plain answer")
    deltas.extend(n.flush())
    agg = _collect(deltas)
    assert agg["reasoning"] == "just a plain answer"
    assert agg["content"] == ""


def test_stream_multiple_tool_calls() -> None:
    raw = (
        "<tool_call><function=a></function></tool_call>"
        "<tool_call><function=b></function></tool_call>"
    )
    n = StreamNormalizer(tools_requested=True, expects_reasoning_prelude=False)
    deltas = n.feed(raw)
    deltas.extend(n.flush())
    agg = _collect(deltas)
    assert [tc["function"]["name"] for tc in agg["tool_calls"]] == ["a", "b"]


def test_stream_flush_drains_residue() -> None:
    """A short trailing token shorter than the holdback must still flush."""
    n = StreamNormalizer()
    deltas = n.feed("ok")  # below holdback threshold — won't emit yet
    assert _collect(deltas)["content"] == ""
    deltas.extend(n.flush())
    assert _collect(deltas)["content"] == "ok"
