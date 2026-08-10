"""Route wiring for the guardrail call-out.

Every helper here starts by reading ``app_state.guardrail``. When it is
``None`` — the default, and the state of any deployment that left
``GUARDRAIL_ENABLED`` off — the helper returns the caller's own value before
building a payload, so the route behaves exactly as it did before this seam
existed.

Payload kinds per call site, since the contract leaves the choice to the
implementer and a transform can only be applied back onto the shape it was
sent as:

* chat ``tool_result``   — ``text``, one evaluation per inbound ``role="tool"``
  message, over the same messages ``_tool_audit.emit_tool_results`` walks. The
  contract's ``tool`` object needs a name and an operation; OpenAI's tool
  message carries the name directly or by ``tool_call_id`` reference to the
  preceding assistant ``tool_calls``, and has no separate operation, so the
  resolved name serves as both. ``mcpServer`` is null: the engine is not an
  MCP client.
* chat ``llm_input``     — ``json``, the dumped message list
* chat ``llm_output``    — ``text``, the assistant content and, as a second
  evaluation, the reasoning content. Both are model output the caller
  receives, so both are guarded; they are separate evaluations because they
  are separate channels and a transform must land back on the right one.
  Streaming guards a rolling window of each.
* completions inputs and outputs, embeddings inputs — ``json``, the list of
  strings, so one evaluation covers a batched request

A non-``allow`` verdict raises a typed, payload-free ``HTTPException``; a
guardrail fault never escapes as an unhandled exception.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from ..auth import Identity
from ..guardrail import (
    STAGE_LLM_INPUT,
    STAGE_LLM_OUTPUT,
    STAGE_TOOL_RESULT,
    VERDICT_ALLOW,
    VERDICT_DENY,
    VERDICT_TRANSFORM,
    VERDICT_UNAVAILABLE,
    GuardrailOutcome,
    client_fault_outcome,
)
from ..observability import get_logger, span
from ..schemas import ChatMessage, chat_content_text
from .state import app_state


log = get_logger("api.guardrail")

# Sizes both streaming window mechanisms: how much of the buffer ``feed``
# refuses to release, and how much already-released text is carried into the
# next window. Both are documented on ``StreamOutputGuard``.
STREAM_HOLDBACK_CHARS = 256

_UNAVAILABLE_RETRY_AFTER_SECONDS = 2

_BLOCKED_MESSAGES = {
    "guardrail_blocked": "request blocked by a guardrail policy",
    "guardrail_unavailable": "guardrail evaluation is unavailable",
}


@dataclass(frozen=True)
class StreamGuardStep:
    """Text cleared for the wire, plus the terminal error when it is not."""

    released: str = ""
    blocked: dict | None = None


def new_request_id(request: Any = None) -> str:
    """The correlation id a guardrail evaluation is tagged with.

    THE ATTRIBUTE NAME IS LOAD-BEARING. ``main.py``'s outermost middleware
    stores the minted id as ``request.state.engine_request_id`` and returns it
    as the ``x-request-id`` response header. It was called
    ``request.state.request_id`` until #93 split the server-owned id from the
    tenant-runtime identities, and this reader kept naming the old attribute —
    so this returned a *fresh* uuid on every guarded request, and the id the
    guardrail service and the span carried was one nobody else ever held.
    Reading the attribute the middleware actually writes is what fixes that.

    OPEN QUESTION, deliberately not settled here: whether the right value is
    the engine id or the tenant runtime's. The span key is
    ``prometa.runtime.request_id`` and #93 captures
    ``x-orchestra-runtime-request-id`` into ``request.state.runtime_request_id``
    for exactly that identity, which argues for the runtime's; the engine id is
    what the caller gets back and what the ledger uses. This function returns
    the engine id — the value this code has always *meant* to return — and no
    claim is made here about which one the kernel's ``runtime.guard.<stage>``
    events join against, because that has not been checked end to end.

    The fallback stays for callers that reach the guardrail without a request
    object at all (the direct-call paths in tests): an id nobody can join on is
    still better than an empty one, which would join on everything.
    """
    state = getattr(request, "state", None)
    request_id = getattr(state, "engine_request_id", "") if state is not None else ""
    return request_id or f"req_{uuid.uuid4().hex}"


def _span_attrs(outcome: GuardrailOutcome, identity: Identity, request_id: str) -> dict:
    client = app_state.guardrail
    attrs: dict = {
        # The kernel binds this exact key on its ``runtime.guard.<stage>``
        # events, so the key has to match. Whether the VALUE lines up is the
        # open question in :func:`new_request_id`.
        "prometa.runtime.request_id": request_id,
        "prometa.guardrail.stage": outcome.stage,
        "prometa.guardrail.verdict": outcome.verdict,
        "prometa.guardrail.profile": getattr(getattr(client, "config", None), "profile", ""),
        "prometa.guardrail.reason_code": outcome.reason_code,
        "prometa.guardrail.evaluated": outcome.evaluated_count,
        "prometa.guardrail.findings": outcome.findings_count,
        "prometa.guardrail.unknown_fields_dropped": outcome.unknown_fields_dropped,
        "prometa.guardrail.request_fields_dropped_by_server": (
            outcome.request_fields_dropped_by_server
        ),
        "prometa.guardrail.fail_open": outcome.failed_open,
        "prometa.tenant": identity.tenant,
        "prometa.key_id": identity.key_id,
    }
    if outcome.detector_digest is not None:
        attrs["prometa.guardrail.detector_digest"] = outcome.detector_digest
    if outcome.latency_ms is not None:
        attrs["prometa.guardrail.latency_ms"] = outcome.latency_ms
    if identity.org_id is not None:
        attrs["prometa.org_id"] = identity.org_id
    return attrs


def _http_error(outcome: GuardrailOutcome) -> HTTPException:
    if outcome.verdict == VERDICT_UNAVAILABLE:
        code, status = "guardrail_unavailable", 503
    else:
        code, status = "guardrail_blocked", 403

    detail = {
        "message": _BLOCKED_MESSAGES[code],
        "type": code,
        "code": code,
        "param": None,
        "stage": outcome.stage,
        "reason_code": outcome.reason_code,
    }
    headers = None
    if status == 503:
        detail["retry_after_seconds"] = _UNAVAILABLE_RETRY_AFTER_SECONDS
        headers = {"Retry-After": str(_UNAVAILABLE_RETRY_AFTER_SECONDS)}
    return HTTPException(status_code=status, detail=detail, headers=headers)


async def _outcome(
    stage: str,
    *,
    identity: Identity,
    request_id: str,
    payload: Any,
    tool: dict | None = None,
) -> GuardrailOutcome | None:
    """Evaluate and record. Returns ``None`` only when no client is configured."""
    client = app_state.guardrail
    if client is None:
        return None
    try:
        outcome = await client.evaluate(
            stage=stage,
            payload=payload,
            tenant=identity.tenant,
            request_id=request_id,
            org_id=identity.org_id,
            tool=tool,
        )
    except Exception as exc:  # noqa: BLE001 - a guardrail fault is not a 500
        log.warning("guardrail.client_failed", stage=stage, error_type=type(exc).__name__)
        outcome = client_fault_outcome(stage, client)
    with span("guardrail.evaluate", **_span_attrs(outcome, identity, request_id)):
        pass
    return outcome


def _resolve(outcome: GuardrailOutcome | None) -> Any:
    """Return the transformed payload, or raise. ``None`` means keep the original."""
    if outcome is None or outcome.verdict == VERDICT_ALLOW:
        return None
    if outcome.verdict == VERDICT_TRANSFORM:
        return outcome.transformed
    raise _http_error(outcome)


def _denied(outcome: GuardrailOutcome, reason_code: str) -> HTTPException:
    """Refuse a transform this engine cannot apply back onto its own request.

    A transformation policy must never release the original payload when the
    transformed value is unusable, so an inapplicable transform becomes a deny.
    """
    log.warning("guardrail.transform_unusable", stage=outcome.stage, reason_code=reason_code)
    return _http_error(
        GuardrailOutcome(stage=outcome.stage, verdict=VERDICT_DENY, reason_code=reason_code)
    )


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return list(value)


def _tool_names(messages: list[ChatMessage]) -> dict[str, str]:
    """Map ``tool_call_id`` to the function name the assistant asked for."""
    names: dict[str, str] = {}
    for message in messages:
        for call in message.tool_calls or ():
            if call.id and call.function.name:
                names[call.id] = call.function.name
    return names


async def guard_tool_results(
    *,
    identity: Identity,
    request_id: str,
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    """Evaluate every inbound ``role="tool"`` message at the ``tool_result`` stage.

    Data coming back from a tool is the stage this contract exists for: the
    model reads it as if it were instruction. It runs ahead of the
    ``llm_input`` evaluation so a transform lands on the message list that
    evaluation then sees.
    """
    if app_state.guardrail is None:
        return messages
    names = _tool_names(messages)
    guarded = list(messages)
    for index, message in enumerate(messages):
        if message.role != "tool":
            continue
        # OpenAI's tool message has no operation of its own, so the resolved
        # function name stands in for both identifiers the contract requires.
        name = message.name or names.get(message.tool_call_id or "") or "unknown"
        outcome = await _outcome(
            STAGE_TOOL_RESULT,
            identity=identity,
            request_id=request_id,
            payload=chat_content_text(message.content) if message.content is not None else "",
            tool={
                "name": name,
                "operation": name,
                "mcpServer": None,
                "riskLevel": None,
                "sideEffects": None,
                "requiredGuardrails": [],
            },
        )
        transformed = _resolve(outcome)
        if transformed is None:
            continue
        if not isinstance(transformed, str):
            raise _denied(outcome, "guardrail_transform_invalid")
        guarded[index] = message.model_copy(update={"content": transformed})
    return guarded


async def guard_chat_messages(
    *,
    identity: Identity,
    request_id: str,
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    outcome = await _outcome(
        STAGE_LLM_INPUT,
        identity=identity,
        request_id=request_id,
        payload=[message.model_dump(exclude_none=True) for message in messages],
    )
    transformed = _resolve(outcome)
    if transformed is None:
        return messages
    if not isinstance(transformed, list):
        raise _denied(outcome, "guardrail_transform_invalid")
    try:
        return [ChatMessage.model_validate(item) for item in transformed]
    except ValidationError as exc:
        raise _denied(outcome, "guardrail_transform_invalid") from exc


async def guard_string_inputs(
    *,
    identity: Identity,
    request_id: str,
    inputs: list[str],
) -> list[str]:
    outcome = await _outcome(
        STAGE_LLM_INPUT, identity=identity, request_id=request_id, payload=list(inputs)
    )
    transformed = _resolve(outcome)
    if transformed is None:
        return inputs
    replaced = _string_list(transformed)
    if replaced is None or len(replaced) != len(inputs):
        raise _denied(outcome, "guardrail_transform_invalid")
    return replaced


async def guard_string_outputs(
    *,
    identity: Identity,
    request_id: str,
    outputs: list[str],
) -> list[str]:
    outcome = await _outcome(
        STAGE_LLM_OUTPUT, identity=identity, request_id=request_id, payload=list(outputs)
    )
    transformed = _resolve(outcome)
    if transformed is None:
        return outputs
    replaced = _string_list(transformed)
    if replaced is None or len(replaced) != len(outputs):
        raise _denied(outcome, "guardrail_transform_invalid")
    return replaced


async def guard_completion_text(
    *,
    identity: Identity,
    request_id: str,
    text: str,
) -> str:
    outcome = await _outcome(
        STAGE_LLM_OUTPUT, identity=identity, request_id=request_id, payload=text
    )
    transformed = _resolve(outcome)
    if transformed is None:
        return text
    if not isinstance(transformed, str):
        raise _denied(outcome, "guardrail_transform_invalid")
    return transformed


class StreamOutputGuard:
    """Rolling-window ``llm_output`` guard for one SSE text channel.

    The channel is split into ``_released_tail`` (already on the wire, retained
    only as window context) and ``_pending`` (buffered, not yet on the wire).
    Every window is ``_released_tail + _pending``, and two separate mechanisms
    are sized by ``holdback_chars``:

    *Holdback.* ``feed`` never releases the last ``holdback`` characters of
    ``_pending``. A character therefore reaches the caller only after a window
    that also carried the ``holdback`` characters following it — or, at the
    terminal ``flush``, after the stream ended and there are none. A detector
    that needs the tail of a pattern to recognise its head sees the pattern
    whole while all of it is still holdable. Streaming cannot retract, so
    releasing the head of a secret before the tail arrives would be a partial
    leak no later deny can undo.

    *Overlap.* The last ``holdback`` released characters stay in
    ``_released_tail`` and open the next window. Consecutive windows share that
    much text, so a pattern no longer than ``holdback`` that straddles a
    release boundary is inside the later window in full. The overlap is kept
    across a full release — a ``flush`` or a transform — which is what stops
    either from opening an unwatched seam.

    Evaluation is deferred until ``_pending`` can release at least ``holdback``
    characters, so round trips are one per holdback of output rather than one
    per adapter chunk.

    Text already released cannot be recalled, which is inherent to streaming: a
    mid-stream deny stops the stream and terminates it with an error frame, it
    does not retract what the client already read.
    """

    def __init__(
        self,
        *,
        identity: Identity,
        request_id: str,
        holdback_chars: int = STREAM_HOLDBACK_CHARS,
    ) -> None:
        self._identity = identity
        self._request_id = request_id
        self._holdback = holdback_chars
        self._pending = ""
        self._released_tail = ""

    async def feed(self, text: str) -> StreamGuardStep:
        self._pending += text
        if len(self._pending) < self._holdback * 2:
            return StreamGuardStep()
        return await self._evaluate(keep=self._holdback)

    async def flush(self) -> StreamGuardStep:
        if not self._pending:
            return StreamGuardStep()
        return await self._evaluate(keep=0)

    def _advance(self, released: str) -> None:
        self._released_tail = (self._released_tail + released)[-self._holdback :]

    async def _evaluate(self, *, keep: int) -> StreamGuardStep:
        window = self._released_tail + self._pending
        outcome = await _outcome(
            STAGE_LLM_OUTPUT,
            identity=self._identity,
            request_id=self._request_id,
            payload=window,
        )
        if outcome is None or outcome.verdict == VERDICT_ALLOW:
            cut = len(self._pending) - keep
            released, self._pending = self._pending[:cut], self._pending[cut:]
            self._advance(released)
            return StreamGuardStep(released=released)
        if outcome.verdict == VERDICT_TRANSFORM:
            replacement = self._transformed_tail(outcome.transformed)
            if replacement is not None:
                self._pending = ""
                self._advance(replacement)
                return StreamGuardStep(released=replacement)
            outcome = GuardrailOutcome(
                stage=outcome.stage,
                verdict=VERDICT_DENY,
                reason_code="guardrail_transform_invalid",
            )
        return StreamGuardStep(blocked=_http_error(outcome).detail)

    def _transformed_tail(self, transformed: Any) -> str | None:
        """The part of a transformed window this guard is still able to apply.

        Every window after the first opens with text the caller has already
        read, so a transform that rewrote that prefix is asking for an edit the
        wire cannot carry; it is refused rather than applied to the rest.
        """
        if not isinstance(transformed, str) or not transformed.startswith(self._released_tail):
            return None
        return transformed[len(self._released_tail) :]


def stream_output_guard(*, identity: Identity, request_id: str) -> StreamOutputGuard | None:
    if app_state.guardrail is None:
        return None
    return StreamOutputGuard(identity=identity, request_id=request_id)


__all__ = [
    "STREAM_HOLDBACK_CHARS",
    "StreamGuardStep",
    "StreamOutputGuard",
    "guard_chat_messages",
    "guard_completion_text",
    "guard_string_inputs",
    "guard_string_outputs",
    "guard_tool_results",
    "new_request_id",
    "stream_output_guard",
]
