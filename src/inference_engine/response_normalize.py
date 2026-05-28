"""Normalize raw model text into OpenAI-compatible chat completion fields.

Some backends (notably ``llama-cpp-python`` without Nemotron-specific grammars,
and any Ollama-HTTP build that pre-dates the model's tool-format parser)
return chain-of-thought and tool invocations as plain ``content`` instead of
structured ``tool_calls`` / ``reasoning_content``. This module repairs that
at the engine boundary so clients never have to parse vendor XML.

Two entry points:

* :func:`normalize_assistant_text` — blocking-path repair on a fully-realised
  assistant turn. Splits leaked ``<think>``/``<tool_call>``/``<TOOLCALL>``
  markup into structured fields.
* :class:`StreamNormalizer` — streaming-path state machine that converts the
  same vendor XML to OpenAI SSE deltas (``content`` / ``reasoning_content`` /
  ``tool_calls``) on the fly, with no client-side reassembly.

Both honour two invariants:

1. **Anything before a parsed tool call is reasoning**, by definition: the
   user-facing answer doesn't come until the agent feeds the tool result
   back. Putting that prelude in ``content`` would render the model's
   chain-of-thought to the user, which is exactly the leak we're fixing.
2. **Orphan close tags** (``</think>`` with no open) are treated as if the
   chat template silently pre-emitted ``<think>`` at generation start.
   Nemotron/DeepSeek-R1 Jinja templates routinely do this — the bug report
   payload (``"…suggestions … \\n</think>\\n<tool_call>…"``) is exactly that
   shape.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass

# Chain-of-thought wrappers seen in the wild (Nemotron, DeepSeek-R1, Qwen3-Thinking, QwQ).
_THINK_TAGS = ("think", "thinking", "redacted_thinking", "reasoning")
_THINK_OPEN_RE = re.compile(rf"<(?:{'|'.join(_THINK_TAGS)})>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(rf"</(?:{'|'.join(_THINK_TAGS)})>", re.IGNORECASE)
_THINK_PAIR_RE = re.compile(
    rf"<(?:{'|'.join(_THINK_TAGS)})>\s*(.*?)\s*</(?:{'|'.join(_THINK_TAGS)})>",
    re.DOTALL | re.IGNORECASE,
)

# Tool-call container blocks — Nemotron NIM (`<tool_call>`) + llama.cpp test
# vectors (`<TOOLCALL>`) + the Hermes/Qwen variants (`<tool_call>` again).
_TOOL_TAGS = ("tool_call", "TOOLCALL")
_TOOL_BLOCK_RE = re.compile(
    rf"<(?:{'|'.join(_TOOL_TAGS)})>\s*(.*?)\s*</(?:{'|'.join(_TOOL_TAGS)})>",
    re.DOTALL,
)

# Nemotron NIM function declaration: `<function=name>` or `<function name="x">`.
_FUNCTION_RE = re.compile(
    r"<function=([^\s>/]+)\s*/?>|<function\s+name=[\"']([^\"']+)[\"'][^>]*>",
    re.IGNORECASE,
)

# Nemotron NIM parameter declarations. Two flavours coexist in the wild:
#   <parameter=key>value</parameter>     ← the canonical NIM spec
#   <parameter name="key">value</...>    ← Hermes-style fallback some templates use
_PARAMETER_RE = re.compile(
    r"<parameter=([^\s>]+)\s*>\s*(.*?)\s*</parameter>"
    r"|<parameter\s+name=[\"']([^\"']+)[\"'][^>]*>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class NormalizedAssistant:
    """Parsed assistant turn after stripping vendor-specific markup."""

    content: str | None
    reasoning_content: str | None
    tool_calls: list[dict] | None
    finish_reason: str


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _coerce_arg_value(raw: str) -> object:
    """Best-effort: parse a Nemotron parameter value as JSON, fall back to string."""
    s = raw.strip()
    if not s:
        return ""
    # Bare JSON literals first — covers numbers, booleans, null, objects, arrays.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Otherwise keep as-is; tool execution layer can coerce further.
    return raw


def _parse_function_block(inner: str) -> dict | None:
    """Parse a Nemotron-style ``<function=NAME> [<parameter=…>…]* </function>``."""
    fm = _FUNCTION_RE.search(inner)
    if not fm:
        return None
    name = (fm.group(1) or fm.group(2) or "").strip()
    if not name:
        return None
    args: dict = {}
    for pm in _PARAMETER_RE.finditer(inner):
        key = (pm.group(1) or pm.group(3) or "").strip()
        val = (pm.group(2) or pm.group(4) or "")
        if key:
            args[key] = _coerce_arg_value(val)
    return {
        "id": _new_call_id(),
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _parse_tool_inner(inner: str) -> list[dict]:
    """Parse the contents of a ``<tool_call>…</tool_call>`` block.

    Two payload shapes are accepted:

    * **JSON** — ``[{"name": "x", "arguments": {...}}, ...]`` or a single
      dict. This is what llama.cpp's own Nemotron grammar emits, and what
      DeepSeek-R1/Hermes-style templates emit.
    * **XML** — ``<function=name>[<parameter=key>val</parameter>]*</function>``.
      The NIM-canonical Nemotron format that leaks through llama-cpp-python
      when no grammar is loaded.
    """
    inner = inner.strip()
    if not inner:
        return []

    if inner.startswith(("[", "{")):
        try:
            payload = json.loads(inner)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            items = payload if isinstance(payload, list) else [payload]
            out: list[dict] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or (item.get("function") or {}).get("name")
                if not name:
                    continue
                raw_args = item.get("arguments")
                if raw_args is None:
                    raw_args = (item.get("function") or {}).get("arguments", {})
                args_str = raw_args if isinstance(raw_args, str) else json.dumps(
                    raw_args, ensure_ascii=False
                )
                out.append(
                    {
                        "id": _new_call_id(),
                        "type": "function",
                        "function": {"name": str(name), "arguments": args_str},
                    }
                )
            if out:
                return out

    one = _parse_function_block(inner)
    return [one] if one else []


def _split_reasoning(text: str) -> tuple[str, str | None]:
    """Pull all reasoning out of ``text``; return ``(body, reasoning_or_None)``.

    Handles three shapes:

    * Well-formed ``<think>…</think>`` pairs (any number).
    * **Orphan close** — ``</think>`` with no opening tag. The chat template
      pre-emitted ``<think>`` invisibly; everything before the close is
      reasoning, everything after is body.
    * **Orphan open** — ``<think>`` with no close. Treat everything from the
      open onward as reasoning (the model ran out of tokens before closing
      its thought block, but the prelude is still reasoning).
    """
    remainder = text
    parts: list[str] = []

    # 1. Paired blocks — repeated to catch nested-but-not-overlapping pairs.
    while True:
        m = _THINK_PAIR_RE.search(remainder)
        if not m:
            break
        parts.append(m.group(1).strip())
        remainder = (remainder[: m.start()] + remainder[m.end() :])

    # 2. Orphan close — assume an implicit `<think>` at start of remainder.
    close = _THINK_CLOSE_RE.search(remainder)
    open_match = _THINK_OPEN_RE.search(remainder)
    if close and (not open_match or close.start() < open_match.start()):
        prelude = remainder[: close.start()].strip()
        if prelude:
            parts.append(prelude)
        remainder = remainder[close.end() :]

    # 3. Orphan open — capture from open to end-of-text as reasoning.
    open_match = _THINK_OPEN_RE.search(remainder)
    if open_match:
        tail = remainder[open_match.end() :].strip()
        if tail:
            parts.append(tail)
        remainder = remainder[: open_match.start()]

    reasoning = "\n\n".join(p for p in parts if p) or None
    return remainder.strip(), reasoning


def _extract_tool_calls(text: str) -> tuple[str, str | None, list[dict]]:
    """Strip tool-call XML blocks.

    Returns ``(content_after_last_call, pre_call_text_or_None, tool_calls)``.
    The pre-call text is everything before the first ``<tool_call>`` and is
    reasoning by definition — any words the model spoke before invoking a tool
    are part of its private deliberation, not the user-facing answer.
    """
    matches = list(_TOOL_BLOCK_RE.finditer(text))
    if not matches:
        return text, None, []

    pre = text[: matches[0].start()].strip() or None
    calls: list[dict] = []
    for m in matches:
        inner = m.group(1)
        calls.extend(_parse_tool_inner(inner))

    if not calls:
        return text, None, []

    # Content after the *last* close tag is the assistant's post-call narration —
    # rare for tool-calling turns, but preserved when present.
    tail = text[matches[-1].end() :].strip()
    return tail, pre, calls


def _strip_residue(text: str) -> str:
    """Remove orphan vendor tags that survived paired-block extraction."""
    text = _THINK_OPEN_RE.sub("", text)
    text = _THINK_CLOSE_RE.sub("", text)
    text = re.sub(r"</?tool_call>|</?TOOLCALL>", "", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_assistant_text(
    text: str,
    *,
    existing_tool_calls: list[dict] | None = None,
    finish_reason: str = "stop",
    tools_requested: bool = False,
) -> NormalizedAssistant:
    """Parse leaked reasoning + tool-call XML from raw assistant ``content``.

    When ``existing_tool_calls`` is already populated by the backend, we only
    strip ``<think>`` blocks from ``content`` and leave the structured tool
    calls untouched (re-parsing would issue fresh ``call_id``s that wouldn't
    match the agent's subsequent ``tool_call_id`` replies).
    """
    raw = text or ""
    body, reasoning = _split_reasoning(raw)
    reasoning_parts: list[str] = [reasoning] if reasoning else []

    tool_calls: list[dict] = list(existing_tool_calls) if existing_tool_calls else []
    if not tool_calls and tools_requested:
        body, pre_call_text, parsed = _extract_tool_calls(body)
        if pre_call_text:
            # Pre-tool-call prose is reasoning by definition.
            reasoning_parts.append(pre_call_text)
        tool_calls.extend(parsed)

    body = _strip_residue(body)

    reasoning_content = "\n\n".join(p for p in reasoning_parts if p) or None
    content = body or None

    if tool_calls:
        # OpenAI convention: ``content`` is null on tool-calling turns. Anything
        # left in body was already moved to reasoning above, but keep this guard
        # so adapters that hand us pre-parsed tool_calls + a stray newline don't
        # leak whitespace through.
        if content and not content.strip():
            content = None
        finish = "tool_calls"
    else:
        finish = finish_reason if finish_reason in ("stop", "length", "tool_calls") else "stop"

    return NormalizedAssistant(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls or None,
        finish_reason=finish,
    )


# ---------------------------------------------------------------------------
# Streaming normalizer — converts a sequence of raw text chunks into the right
# mix of OpenAI ``content`` / ``reasoning_content`` / ``tool_calls`` deltas.
# ---------------------------------------------------------------------------

# Longest tag prefix we might need to hold back across a chunk boundary so a
# token split like ``"<tool"`` + ``"_call>"`` doesn't accidentally emit the
# prefix as plain content before we realise it's a vendor tag.
_HOLDBACK = 24


@dataclass
class StreamDelta:
    """One frame of normalized output. Any subset of fields may be populated."""

    content: str | None = None
    reasoning_content: str | None = None
    tool_call: dict | None = None  # one fully-assembled OpenAI tool_call dict


class StreamNormalizer:
    """Stateful XML→OpenAI translator for a single streamed assistant turn.

    Mode of operation::

        norm = StreamNormalizer(
            tools_requested=bool(params.tools),
            expects_reasoning_prelude=caps["reasoning"],
        )
        async for chunk in adapter.stream(...):
            for delta in norm.feed(chunk.text):
                ...  # emit as SSE
        for delta in norm.flush():
            ...

    ``expects_reasoning_prelude=True`` tells the normalizer the chat template
    pre-emitted ``<think>`` (Nemotron / DeepSeek-R1 templates do this), so the
    stream starts in reasoning state instead of content state. This avoids the
    "first N tokens already emitted as content" problem that no SSE retract
    mechanism can fix.
    """

    # State machine
    _S_CONTENT = "content"
    _S_REASONING = "reasoning"
    _S_TOOL_CALL = "tool_call"
    _S_TOOL_PRELUDE = "tool_prelude"  # post-tool-call, treat words before next tool as reasoning

    def __init__(
        self,
        *,
        tools_requested: bool = False,
        expects_reasoning_prelude: bool = False,
    ) -> None:
        self._tools_requested = tools_requested
        self._buf = ""
        self._tool_inner = ""  # accumulates raw text inside <tool_call>...</tool_call>
        self._state = self._S_REASONING if expects_reasoning_prelude else self._S_CONTENT
        # Track whether we've ever transitioned to "real content" after a tool
        # call. Words emitted between the assistant's start and the first
        # tool_call get re-classified to reasoning_content on flush if the
        # turn ends up being a tool-calling turn.
        self._content_before_tool = ""
        self._saw_tool_call = False

    # -- public API ----------------------------------------------------------

    def feed(self, text: str) -> list[StreamDelta]:
        if not text:
            return []
        self._buf += text
        return self._drain(final=False)

    def flush(self) -> list[StreamDelta]:
        """Drain whatever's left at end-of-stream."""
        out = self._drain(final=True)
        # If a tool call was emitted earlier in the stream, anything we'd
        # buffered as plain "content" before it should retroactively be
        # reasoning. We can't retract already-sent SSE frames, but we DO
        # avoid emitting any residual content that was held back.
        return out

    def has_tool_calls(self) -> bool:
        return self._saw_tool_call

    # -- internal ------------------------------------------------------------

    def _drain(self, *, final: bool) -> list[StreamDelta]:
        out: list[StreamDelta] = []

        while True:
            if self._state == self._S_CONTENT:
                made_progress = self._drain_content(out, final=final)
            elif self._state == self._S_REASONING:
                made_progress = self._drain_reasoning(out, final=final)
            elif self._state == self._S_TOOL_CALL:
                made_progress = self._drain_tool_call(out, final=final)
            elif self._state == self._S_TOOL_PRELUDE:
                made_progress = self._drain_tool_prelude(out, final=final)
            else:  # pragma: no cover — defensive
                break
            if not made_progress:
                break

        return out

    def _emit_content_text(self, out: list[StreamDelta], text: str) -> None:
        if not text:
            return
        if not self._saw_tool_call:
            self._content_before_tool += text
        out.append(StreamDelta(content=text))

    def _emit_reasoning_text(self, out: list[StreamDelta], text: str) -> None:
        if not text:
            return
        out.append(StreamDelta(reasoning_content=text))

    def _find_next_marker(self, text: str) -> tuple[int, str] | None:
        """Return ``(offset, kind)`` for the next interesting tag, or None.

        ``kind`` ∈ {``"think_open"``, ``"think_close"``, ``"tool_open"``}.
        Tool markers are only considered when tools were requested.
        """
        candidates: list[tuple[int, str]] = []
        m = _THINK_OPEN_RE.search(text)
        if m:
            candidates.append((m.start(), "think_open"))
        m = _THINK_CLOSE_RE.search(text)
        if m:
            candidates.append((m.start(), "think_close"))
        if self._tools_requested:
            m = re.search(r"<tool_call>|<TOOLCALL>", text)
            if m:
                candidates.append((m.start(), "tool_open"))
        if not candidates:
            return None
        return min(candidates, key=lambda c: c[0])

    def _safe_emit_text(self, out: list[StreamDelta], emit: "callable") -> bool:
        """Emit as much of ``self._buf`` as is safe (no partial tag prefix at end)."""
        if not self._buf:
            return False
        marker = self._find_next_marker(self._buf)
        if marker is None:
            # No tag in sight; emit everything except the last _HOLDBACK chars
            # (which could be the start of a tag arriving on the next chunk).
            if len(self._buf) <= _HOLDBACK:
                return False
            slice_ = self._buf[:-_HOLDBACK]
            self._buf = self._buf[-_HOLDBACK:]
            emit(out, slice_)
            return False  # already drained; nothing more to do this round
        offset, _ = marker
        if offset > 0:
            slice_ = self._buf[:offset]
            self._buf = self._buf[offset:]
            emit(out, slice_)
            return True
        return True  # marker is at position 0 — the state branch will handle it

    def _drain_content(self, out: list[StreamDelta], *, final: bool) -> bool:
        # If a tool call is already known to have happened, this branch is unreachable
        # (we move to TOOL_PRELUDE / REASONING). Plain text emission path.
        marker = self._find_next_marker(self._buf)
        if marker is None:
            if final:
                self._emit_content_text(out, self._buf)
                self._buf = ""
            else:
                self._safe_emit_text(out, self._emit_content_text)
            return False
        offset, kind = marker
        if offset > 0:
            self._emit_content_text(out, self._buf[:offset])
            self._buf = self._buf[offset:]
        return self._consume_marker(kind)

    def _drain_reasoning(self, out: list[StreamDelta], *, final: bool) -> bool:
        # In reasoning state, only `</think>` and `<tool_call>` close it.
        candidates: list[tuple[int, str]] = []
        m = _THINK_CLOSE_RE.search(self._buf)
        if m:
            candidates.append((m.start(), "think_close"))
        if self._tools_requested:
            m = re.search(r"<tool_call>|<TOOLCALL>", self._buf)
            if m:
                candidates.append((m.start(), "tool_open"))
        if not candidates:
            if final:
                self._emit_reasoning_text(out, self._buf)
                self._buf = ""
            else:
                # Hold back potential tag prefix.
                if len(self._buf) > _HOLDBACK:
                    slice_ = self._buf[:-_HOLDBACK]
                    self._buf = self._buf[-_HOLDBACK:]
                    self._emit_reasoning_text(out, slice_)
            return False
        offset, kind = min(candidates, key=lambda c: c[0])
        if offset > 0:
            self._emit_reasoning_text(out, self._buf[:offset])
            self._buf = self._buf[offset:]
        if kind == "think_close":
            close = _THINK_CLOSE_RE.match(self._buf)
            if close:
                self._buf = self._buf[close.end() :]
                self._state = self._S_TOOL_PRELUDE if self._saw_tool_call else self._S_CONTENT
                # If we expected reasoning prelude and tools were requested,
                # treat the post-close text as the user-facing content unless
                # another tool_call arrives.
                return True
        elif kind == "tool_open":
            return self._consume_marker(kind)
        return False

    def _drain_tool_prelude(self, out: list[StreamDelta], *, final: bool) -> bool:
        """After a tool_call closed, any subsequent text up to the next tool_call
        is itself reasoning (the model is deliberating about whether to call
        another tool). Once we hit ``finish_reason`` without another tool call,
        on flush we'll emit it as content because by then we know the model
        is actually answering."""
        if not self._tools_requested:
            self._state = self._S_CONTENT
            return True
        m = re.search(r"<tool_call>|<TOOLCALL>", self._buf)
        if m is None:
            if final:
                # No further tool call — treat the prelude as final content.
                self._emit_content_text(out, self._buf)
                self._buf = ""
            else:
                if len(self._buf) > _HOLDBACK:
                    slice_ = self._buf[:-_HOLDBACK]
                    self._buf = self._buf[-_HOLDBACK:]
                    self._emit_reasoning_text(out, slice_)
            return False
        if m.start() > 0:
            self._emit_reasoning_text(out, self._buf[: m.start()])
            self._buf = self._buf[m.start() :]
        return self._consume_marker("tool_open")

    def _drain_tool_call(self, out: list[StreamDelta], *, final: bool) -> bool:
        # Accumulate raw text until we see </tool_call> or </TOOLCALL>.
        close = re.search(r"</tool_call>|</TOOLCALL>", self._buf)
        if close is None:
            if final:
                # Stream ended mid-tool-call. Best-effort: try to parse what we
                # have; if it doesn't yield a function, drop on the floor.
                self._tool_inner += self._buf
                self._buf = ""
                self._finalize_tool_call(out, partial=True)
            else:
                # Hold back potential close-tag prefix.
                if len(self._buf) > _HOLDBACK:
                    self._tool_inner += self._buf[:-_HOLDBACK]
                    self._buf = self._buf[-_HOLDBACK:]
            return False
        self._tool_inner += self._buf[: close.start()]
        self._buf = self._buf[close.end() :]
        self._finalize_tool_call(out, partial=False)
        return True

    def _finalize_tool_call(self, out: list[StreamDelta], *, partial: bool) -> None:
        calls = _parse_tool_inner(self._tool_inner)
        self._tool_inner = ""
        for call in calls:
            out.append(StreamDelta(tool_call=call))
            self._saw_tool_call = True
        # After a tool_call, return to a state where subsequent prose is treated
        # as reasoning (until the next tool call) and then content.
        self._state = self._S_TOOL_PRELUDE if not partial else self._S_CONTENT

    def _consume_marker(self, kind: str) -> bool:
        """Consume the marker at ``self._buf[0]`` and transition state."""
        if kind == "think_open":
            m = _THINK_OPEN_RE.match(self._buf)
            if m:
                self._buf = self._buf[m.end() :]
                self._state = self._S_REASONING
                return True
        elif kind == "think_close":
            m = _THINK_CLOSE_RE.match(self._buf)
            if m:
                # Orphan close in content state — treat what we just emitted
                # (we can't take it back) as content; continue in content
                # state after stripping the tag.
                self._buf = self._buf[m.end() :]
                return True
        elif kind == "tool_open":
            m = re.match(r"<tool_call>|<TOOLCALL>", self._buf)
            if m:
                self._buf = self._buf[m.end() :]
                self._state = self._S_TOOL_CALL
                return True
        return False


# ---------------------------------------------------------------------------
# Capability heuristic — informs UI badges and the StreamNormalizer's
# ``expects_reasoning_prelude`` knob. After engine-side normalization, every
# reachable model delivers OpenAI-shaped tool_calls; the ``tool_calling_mode``
# field reflects deliverability, not the underlying model's native format.
# ---------------------------------------------------------------------------

# Substring → reasoning-family hit. Narrower than "qwen3" alone because the
# base Qwen3 chat models don't think; only the *-thinking / QwQ variants do.
_REASONING_MARKERS = (
    "nemotron",
    "deepseek-r1",
    "qwen3-thinking",
    "qwen3.6-thinking",
    "qwen-thinking",
    "qwq",
    "glm-z1",
    "phi-4-reasoning",
    "phi-4-mini-reasoning",
    "o1-",
    "o3-",
    "thinking",
)


def infer_model_capabilities(model_id: str, *, backend: str, fmt: str) -> dict:
    """Heuristic capability hints surfaced on ``/v1/models``.

    Returns OpenAI-shaped fields that DeclarAI (and other clients) can use to
    pick a model, render badges, and skip tools where unsupported. Two flags:

    * ``reasoning`` — model is in a known thinking family. The streaming
      normalizer starts in reasoning state for these so the chat-template
      pre-emitted ``<think>`` doesn't leak.
    * ``tool_calling_mode`` — what the engine *delivers* to the client.
      Always ``"native"`` for chat-capable adapters (we normalize vendor XML
      ourselves); ``"unsupported"`` for adapters with no tool plumbing at all
      (e.g. embedding-only GGUFs surfaced as chat by mistake).
    """
    mid = model_id.lower()
    reasoning = any(k in mid for k in _REASONING_MARKERS)

    # MLX-LM has no tool-calling path today; everything else can deliver native
    # tool_calls (via grammar, vLLM/Ollama native, or our normalize layer).
    if backend in ("llama_cpp", "vllm", "ollama_http"):
        tool_mode = "native"
    else:
        tool_mode = "unsupported"

    return {
        "reasoning": reasoning,
        "thinking": reasoning,
        "thinking_level": "med" if reasoning else None,
        "tool_calling_mode": tool_mode,
    }
