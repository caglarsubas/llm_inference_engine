"""Contract tests for ``/v1/responses`` — the stateless Responses subset.

The translation layer is tested directly (pure functions in and out) and the
refusals through the handler, so the suite does not need a live model.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from inference_engine.api import responses as responses_api
from inference_engine.main import _USAGE_LEDGER_PATHS, app
from inference_engine.schemas import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    ResponsesRequest,
    Usage,
)


def _chat_response(text: str = "hi", finish: str = "stop") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-1",
        created=1_700_000_000,
        model="test-model",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason=finish,
            )
        ],
        usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
    )


# --------------------------------------------------------------------------
# Billing wiring. The ledger middleware keys on request.url.path, so a priced
# route absent from this set serves traffic and invoices nothing. Delegating
# to the chat handler does not inherit chat's entry — the path never changes.
# --------------------------------------------------------------------------


def test_responses_is_metered():
    assert "/v1/responses" in _USAGE_LEDGER_PATHS


def test_route_is_registered():
    assert "/v1/responses" in {r.path for r in app.routes if hasattr(r, "path")}


# --------------------------------------------------------------------------
# Stateful features are refused, not silently ignored.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("store", True),
        ("previous_response_id", "resp_abc"),
        ("background", True),
    ],
)
def test_stateful_features_are_refused_by_name(field, value):
    req = ResponsesRequest(model="m", input="hello", **{field: value})
    with pytest.raises(HTTPException) as exc:
        responses_api._reject_stateful(req)
    assert exc.value.status_code == 400
    assert exc.value.detail["type"] == "unsupported_parameter"
    # The caller has to be told which parameter was refused, or they cannot
    # act on it.
    assert exc.value.detail["param"] == field


def test_store_false_is_accepted():
    # The default. Refusing it would reject every well-formed request.
    responses_api._reject_stateful(ResponsesRequest(model="m", input="hi"))


# --------------------------------------------------------------------------
# Request translation onto the chat models.
# --------------------------------------------------------------------------


def test_string_input_becomes_a_user_turn():
    chat = responses_api._to_chat_request(ResponsesRequest(model="m", input="hello"))
    assert [(m.role, m.content) for m in chat.messages] == [("user", "hello")]
    assert chat.stream is False


def test_instructions_lead_as_a_system_turn():
    chat = responses_api._to_chat_request(
        ResponsesRequest(model="m", input="hello", instructions="be terse")
    )
    assert [(m.role, m.content) for m in chat.messages] == [
        ("system", "be terse"),
        ("user", "hello"),
    ]


def test_developer_role_maps_to_system():
    chat = responses_api._to_chat_request(
        ResponsesRequest(
            model="m",
            input=[{"role": "developer", "content": "policy"}, {"role": "user", "content": "q"}],
        )
    )
    assert [m.role for m in chat.messages] == ["system", "user"]


def test_content_parts_are_flattened():
    chat = responses_api._to_chat_request(
        ResponsesRequest(
            model="m",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "one "},
                        {"type": "input_text", "text": "two"},
                    ],
                }
            ],
        )
    )
    assert chat.messages[0].content == "one two"


def test_prior_assistant_turn_can_be_echoed_back():
    # The stateless replacement for previous_response_id, so the refusal above
    # leaves the caller a way to continue a conversation.
    chat = responses_api._to_chat_request(
        ResponsesRequest(
            model="m",
            input=[
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": [{"type": "output_text", "text": "a1"}]},
                {"role": "user", "content": "q2"},
            ],
        )
    )
    assert [(m.role, m.content) for m in chat.messages] == [
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
    ]


def test_unset_sampling_knobs_keep_chat_defaults():
    # Forwarding None would overwrite the documented chat defaults with nulls.
    chat = responses_api._to_chat_request(ResponsesRequest(model="m", input="hi"))
    assert chat.temperature == 0.7
    assert chat.top_p == 0.95


def test_set_sampling_knobs_are_forwarded():
    chat = responses_api._to_chat_request(
        ResponsesRequest(model="m", input="hi", temperature=0.1, top_p=0.5, max_output_tokens=64)
    )
    assert (chat.temperature, chat.top_p, chat.max_tokens) == (0.1, 0.5, 64)


# --------------------------------------------------------------------------
# Response translation back out.
# --------------------------------------------------------------------------


def test_output_carries_the_assistant_text():
    req = ResponsesRequest(model="m", input="hi")
    out = responses_api._to_responses(req, _chat_response("hello there"))
    assert out.object == "response"
    assert out.status == "completed"
    assert out.output[0].role == "assistant"
    assert out.output[0].content[0].type == "output_text"
    assert out.output[0].content[0].text == "hello there"
    assert out.id.startswith("resp_")
    assert out.output[0].id.startswith("msg_")


def test_usage_is_renamed_not_dropped():
    out = responses_api._to_responses(
        ResponsesRequest(model="m", input="hi"), _chat_response()
    )
    assert out.usage.input_tokens == 7
    assert out.usage.output_tokens == 3
    assert out.usage.total_tokens == 10


def test_truncation_reports_incomplete_with_a_reason():
    out = responses_api._to_responses(
        ResponsesRequest(model="m", input="hi"), _chat_response(finish="length")
    )
    assert out.status == "incomplete"
    assert out.incomplete_details.reason == "max_output_tokens"
    assert out.output[0].status == "incomplete"


def test_fallback_provenance_survives_the_translation():
    chat = _chat_response()
    chat.fallback_from_model = "big-model"
    chat.fallback_reason = "backend_unavailable"
    chat.request_key_source = "routed"
    out = responses_api._to_responses(ResponsesRequest(model="m", input="hi"), chat)
    assert out.fallback_from_model == "big-model"
    assert out.fallback_reason == "backend_unavailable"
    assert out.request_key_source == "routed"


def test_empty_choices_do_not_crash_the_translation():
    chat = _chat_response()
    chat.choices = []
    out = responses_api._to_responses(ResponsesRequest(model="m", input="hi"), chat)
    assert out.output[0].content[0].text == ""
