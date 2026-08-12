"""``/v1/responses`` — the stateless subset of the OpenAI Responses API.

This endpoint owns no inference logic. It translates a Responses request onto
``ChatCompletionRequest``, hands it to the chat handler, and translates the
result back. That is deliberate: the priced request spine — budget
reservation, guardrails, routing, fallback, usage accounting, reservation
settlement — is ~1700 lines in ``api/chat.py`` and must have exactly one
implementation. A second copy here would drift, and the halves that drift
first are the ones nobody notices: billing and safety.

Two consequences worth stating, because they are easy to get wrong later:

* ``/v1/responses`` must appear in ``main._USAGE_LEDGER_PATHS``. The ledger
  middleware keys on ``request.url.path``, so delegating to the chat handler
  does *not* inherit chat's billing — the path never changes. A route missing
  from that set serves traffic and invoices nothing.
* The stateful half of the API is refused, not ignored. ``store`` defaults to
  true on OpenAI's own service; a caller who sends it and gets a 200 back is
  entitled to believe the turn was persisted.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth import Identity, require_identity
from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ResponsesIncompleteDetails,
    ResponsesOutputMessage,
    ResponsesOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)
from .chat import chat_completions

router = APIRouter()

# Responses spells the system role "developer"; the chat models only know
# "system". Everything else maps one to one.
_ROLE_MAP = {"developer": "system", "system": "system", "user": "user", "assistant": "assistant"}

# ``finish_reason`` values that mean the model stopped short of a complete
# answer. Responses reports these as status="incomplete" plus a reason rather
# than as a successful completion.
_INCOMPLETE_REASONS = {"length": "max_output_tokens"}


def _unsupported(message: str, param: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "message": message,
            "type": "unsupported_parameter",
            "code": "unsupported_parameter",
            "param": param,
        },
    )


def _reject_stateful(req: ResponsesRequest) -> None:
    """Refuse the parts of the API a stateless engine cannot honour."""
    if req.store:
        raise _unsupported(
            "store=true is not supported: this engine does not persist "
            "responses. Omit store, and continue a conversation by echoing "
            "prior turns into 'input'.",
            "store",
        )
    if req.previous_response_id is not None:
        raise _unsupported(
            "previous_response_id is not supported: this engine does not "
            "persist responses, so there is no prior response to continue "
            "from. Echo the prior turns into 'input' instead.",
            "previous_response_id",
        )
    if req.background:
        raise _unsupported(
            "background=true is not supported: this engine has no run store "
            "to poll. Use stream=false and await the response.",
            "background",
        )


def _part_text(content: str | list) -> str:
    """Flatten Responses content parts into the plain text chat expects."""
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content)


def _to_chat_messages(req: ResponsesRequest) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    if req.instructions:
        messages.append(ChatMessage(role="system", content=req.instructions))
    if isinstance(req.input, str):
        messages.append(ChatMessage(role="user", content=req.input))
        return messages
    for item in req.input:
        messages.append(
            ChatMessage(role=_ROLE_MAP[item.role], content=_part_text(item.content))
        )
    return messages


def _to_chat_request(req: ResponsesRequest) -> ChatCompletionRequest:
    """Build the chat request, leaving unset knobs to the chat defaults.

    Responses has no defaults of its own for temperature/top_p, so passing
    ``None`` through would override chat's documented defaults with nulls.
    Only fields the caller actually set are forwarded.
    """
    fields: dict = {
        "model": req.model,
        "messages": _to_chat_messages(req),
        "stream": False,
        "user": req.user,
    }
    if req.max_output_tokens is not None:
        fields["max_tokens"] = req.max_output_tokens
    if req.temperature is not None:
        fields["temperature"] = req.temperature
    if req.top_p is not None:
        fields["top_p"] = req.top_p
    if req.text and req.text.format and req.text.format.type == "json_object":
        fields["response_format"] = {"type": "json_object"}
    if req.metadata is not None:
        fields["metadata"] = req.metadata
    return ChatCompletionRequest(**fields)


def _to_responses(req: ResponsesRequest, chat: ChatCompletionResponse) -> ResponsesResponse:
    choice = chat.choices[0] if chat.choices else None
    text = ""
    finish_reason = None
    if choice is not None:
        finish_reason = choice.finish_reason
        content = choice.message.content if choice.message else None
        # Multimodal content parts never come back from a text completion, but
        # the field is typed to allow them, so flatten defensively.
        text = content if isinstance(content, str) else ("" if content is None else str(content))

    incomplete = _INCOMPLETE_REASONS.get(finish_reason or "")
    message = ResponsesOutputMessage(
        id=f"msg_{uuid.uuid4().hex}",
        status="incomplete" if incomplete else "completed",
        content=[ResponsesOutputText(text=text)],
    )
    return ResponsesResponse(
        id=f"resp_{uuid.uuid4().hex}",
        created_at=chat.created or int(time.time()),
        model=chat.model,
        status="incomplete" if incomplete else "completed",
        output=[message],
        usage=ResponsesUsage(
            input_tokens=chat.usage.prompt_tokens,
            output_tokens=chat.usage.completion_tokens,
            total_tokens=chat.usage.total_tokens,
        ),
        incomplete_details=ResponsesIncompleteDetails(reason=incomplete) if incomplete else None,
        instructions=req.instructions,
        metadata=req.metadata,
        request_key_source=chat.request_key_source,
        fallback_from_model=chat.fallback_from_model,
        fallback_from_backend=chat.fallback_from_backend,
        fallback_reason=chat.fallback_reason,
        fallback_error_type=chat.fallback_error_type,
    )


@router.post("/v1/responses", response_model=ResponsesResponse)
async def create_response(
    req: ResponsesRequest,
    request: Request,
    identity: Identity = Depends(require_identity),
) -> ResponsesResponse:
    _reject_stateful(req)
    if req.stream:
        # Responses streams a typed event sequence (response.created,
        # response.output_text.delta, response.completed), not chat's chunk
        # shape, so it is not a passthrough of the chat stream. Refused until
        # that translation exists rather than served in the wrong wire format.
        raise _unsupported(
            "stream=true is not yet supported on /v1/responses. Use "
            "/v1/chat/completions for streaming, or omit stream.",
            "stream",
        )
    chat = await chat_completions(_to_chat_request(req), request, identity)
    return _to_responses(req, chat)
