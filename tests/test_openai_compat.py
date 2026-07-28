"""OpenAI wire-compatibility surface.

Covers the client-facing contract added for standardization: the streaming
usage trailer, the ``error`` envelope, ``max_completion_tokens``, Structured
Outputs, cached-token accounting, and the tokenizer probes.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable

import pytest

from inference_engine.adapters import (
    GenerationParams,
    InferenceAdapter,
    StreamChunk,
    TokenizationNotSupportedError,
)
from inference_engine.adapters.base import GenerationResult
from inference_engine.api.chat import _params_from_request, _stream_response
from inference_engine.api.errors import build_error_object, error_type_for_status
from inference_engine.auth import Identity
from inference_engine.cancellation import Cancellation
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import ChatCompletionRequest, ChatMessage


class _FakeAdapter(InferenceAdapter):
    """Adapter that reports token counts on a terminal, text-free chunk."""

    backend_name = "fake"

    def __init__(
        self,
        *,
        text: str = "hello world",
        prompt_tokens: int = 11,
        completion_tokens: int = 3,
        cached_tokens: int = 0,
        supports_structured_outputs: bool = False,
    ) -> None:
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cached_tokens = cached_tokens
        self.supports_structured_outputs = supports_structured_outputs
        self.seen_params: list[GenerationParams] = []
        self.generate_calls = 0

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return None

    @property
    def last_cached_prompt_tokens(self) -> int:
        return self.cached_tokens

    async def load(self, descriptor: ModelDescriptor) -> None: ...
    async def unload(self) -> None: ...

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> GenerationResult:
        self.seen_params.append(params)
        self.generate_calls += 1
        return GenerationResult(
            text=self.text,
            finish_reason="stop",
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cached_tokens=self.cached_tokens,
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        self.seen_params.append(params)
        for word in self.text.split():
            yield StreamChunk(text=word + " ")
        yield StreamChunk(text="", finish_reason="stop")
        yield StreamChunk(
            text="",
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )


class _NoopRequest:
    async def is_disconnected(self) -> bool:
        return False


async def _collect(adapter: _FakeAdapter, *, include_usage: bool) -> list[dict]:
    out = []
    async for event in _stream_response(
        adapter=adapter,
        model_name="fake:1",
        messages=[ChatMessage(role="user", content="hi")],
        params=GenerationParams(),
        identity=Identity(tenant="dev", key_id="sk-x"),
        request=_NoopRequest(),
        include_usage=include_usage,
    ):
        out.append(event)
    return out


def _payloads(events: list[dict]) -> list[dict]:
    return [
        json.loads(e["data"])
        for e in events
        if e.get("data") and e["data"] != "[DONE]" and e.get("event") != "error"
    ]


# --- streaming usage ---------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_emits_usage_trailer_when_requested() -> None:
    adapter = _FakeAdapter(prompt_tokens=11, completion_tokens=2)
    events = await _collect(adapter, include_usage=True)

    assert events[-1] == {"data": "[DONE]"}
    trailer = json.loads(events[-2]["data"])
    # OpenAI's shape: empty choices, usage populated.
    assert trailer["choices"] == []
    assert trailer["usage"]["prompt_tokens"] == 11
    assert trailer["usage"]["completion_tokens"] == 2
    assert trailer["usage"]["total_tokens"] == 13


@pytest.mark.asyncio
async def test_streaming_omits_usage_trailer_by_default() -> None:
    adapter = _FakeAdapter()
    events = await _collect(adapter, include_usage=False)

    assert events[-1] == {"data": "[DONE]"}
    # Every remaining frame must be a normal choices-bearing chunk.
    for payload in _payloads(events):
        assert payload["choices"], "no usage-only frame should be emitted"
        assert payload["usage"] is None


@pytest.mark.asyncio
async def test_streaming_usage_trailer_reports_cached_tokens() -> None:
    adapter = _FakeAdapter(prompt_tokens=100, completion_tokens=4, cached_tokens=64)
    events = await _collect(adapter, include_usage=True)

    trailer = json.loads(events[-2]["data"])
    assert trailer["usage"]["prompt_tokens_details"]["cached_tokens"] == 64


def test_stream_options_parsed_from_request() -> None:
    req = ChatCompletionRequest(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
        stream_options={"include_usage": True},
    )
    assert req.stream_options is not None
    assert req.stream_options.include_usage is True


# --- max_completion_tokens ---------------------------------------------------


def test_max_completion_tokens_overrides_max_tokens() -> None:
    req = ChatCompletionRequest(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        max_completion_tokens=64,
    )
    assert req.max_tokens == 64
    assert _params_from_request(req).max_tokens == 64


def test_legacy_max_tokens_still_honoured() -> None:
    """The orchestra-python-sdk model gateway sends max_tokens, not the new name."""
    req = ChatCompletionRequest(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=128,
    )
    assert _params_from_request(req).max_tokens == 128


# --- structured outputs ------------------------------------------------------

_SDK_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "score": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_json_schema_response_format_reaches_generation_params() -> None:
    """This is the exact body orchestra-python-sdk's model gateway sends."""
    req = ChatCompletionRequest(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "orchestra_runtime_output",
                "strict": True,
                "schema": _SDK_SCHEMA,
            },
        },
    )
    params = _params_from_request(req)
    assert params.json_mode is True
    assert params.json_schema == _SDK_SCHEMA
    assert params.json_schema_name == "orchestra_runtime_output"
    assert params.json_schema_strict is True


def test_plain_json_object_mode_sets_no_schema() -> None:
    req = ChatCompletionRequest(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        response_format={"type": "json_object"},
    )
    params = _params_from_request(req)
    assert params.json_mode is True
    assert params.json_schema is None


def test_json_schema_type_requires_payload() -> None:
    with pytest.raises(ValueError, match="json_schema is required"):
        ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
            response_format={"type": "json_schema"},
        )


def test_llama_cpp_maps_schema_to_grammar_response_format() -> None:
    from inference_engine.adapters.llama_cpp import LlamaCppAdapter

    kwargs = LlamaCppAdapter._completion_kwargs(
        GenerationParams(json_mode=True, json_schema=_SDK_SCHEMA)
    )
    # llama-cpp-python compiles response_format["schema"] into GBNF.
    assert kwargs["response_format"] == {"type": "json_object", "schema": _SDK_SCHEMA}


def test_vllm_forwards_native_json_schema_shape() -> None:
    from inference_engine.adapters.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter()
    adapter._model_id = "upstream/model"
    kwargs = adapter._completion_kwargs(
        GenerationParams(
            json_mode=True,
            json_schema=_SDK_SCHEMA,
            json_schema_name="orchestra_runtime_output",
            json_schema_strict=True,
        )
    )
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "orchestra_runtime_output",
            "schema": _SDK_SCHEMA,
            "strict": True,
        },
    }


# --- sampling params ---------------------------------------------------------


def test_sampling_params_reach_generation_params() -> None:
    req = ChatCompletionRequest(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        frequency_penalty=0.5,
        presence_penalty=-0.25,
        repetition_penalty=1.1,
        logit_bias={"128": -100.0},
        logprobs=True,
        top_logprobs=3,
        parallel_tool_calls=False,
    )
    params = _params_from_request(req)
    assert params.frequency_penalty == 0.5
    assert params.presence_penalty == -0.25
    assert params.repetition_penalty == 1.1
    assert params.logit_bias == {"128": -100.0}
    assert params.logprobs is True
    assert params.top_logprobs == 3
    assert params.parallel_tool_calls is False


def test_top_logprobs_requires_logprobs() -> None:
    with pytest.raises(ValueError, match="top_logprobs requires logprobs"):
        ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="hi")],
            top_logprobs=3,
        )


def test_llama_cpp_logprobs_defaults_top_n_to_one() -> None:
    """Without top_logprobs, llama-cpp-python would silently disable logprobs."""
    from inference_engine.adapters.llama_cpp import LlamaCppAdapter

    kwargs = LlamaCppAdapter._sampling_kwargs(GenerationParams(logprobs=True))
    assert kwargs["logprobs"] is True
    assert kwargs["top_logprobs"] == 1


def test_llama_cpp_raw_completion_uses_integer_logprobs() -> None:
    """create_completion takes a count, not a bool + separate top_logprobs."""
    from inference_engine.adapters.llama_cpp import LlamaCppAdapter

    kwargs = LlamaCppAdapter._sampling_kwargs(
        GenerationParams(logprobs=True, top_logprobs=4), chat=False
    )
    assert kwargs["logprobs"] == 4
    assert "top_logprobs" not in kwargs


def test_llama_cpp_maps_repetition_penalty_to_repeat_penalty() -> None:
    from inference_engine.adapters.llama_cpp import LlamaCppAdapter

    kwargs = LlamaCppAdapter._sampling_kwargs(GenerationParams(repetition_penalty=1.15))
    assert kwargs["repeat_penalty"] == 1.15
    assert "repetition_penalty" not in kwargs


def test_llama_cpp_omits_unset_sampling_knobs() -> None:
    """Unset knobs must not be sent — llama.cpp's own defaults differ from ours."""
    from inference_engine.adapters.llama_cpp import LlamaCppAdapter

    assert LlamaCppAdapter._sampling_kwargs(GenerationParams()) == {}


# --- error envelope ----------------------------------------------------------


def test_error_object_preserves_typed_detail_fields() -> None:
    detail = {
        "message": "too long",
        "type": "context_length_exceeded",
        "code": "context_length_exceeded",
        "param": "messages",
        "context_window": 8192,
    }
    error = build_error_object(detail, 400)
    assert error["type"] == "context_length_exceeded"
    assert error["context_window"] == 8192


def test_error_object_derives_type_from_status_for_plain_strings() -> None:
    error = build_error_object("nope", 429)
    assert error == {
        "message": "nope",
        "type": "rate_limit_error",
        "code": "rate_limit_error",
        "param": None,
    }


def test_error_type_falls_back_by_status_class() -> None:
    assert error_type_for_status(599) == "api_error"
    assert error_type_for_status(499) == "invalid_request_error"


# --- tokenizer surface -------------------------------------------------------


def test_proxy_adapters_report_tokenization_unsupported() -> None:
    from inference_engine.adapters.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter()
    with pytest.raises(TokenizationNotSupportedError):
        adapter.tokenize("hello")
    with pytest.raises(TokenizationNotSupportedError):
        adapter.detokenize([1, 2, 3])
    with pytest.raises(TokenizationNotSupportedError):
        adapter.format_chat_prompt([ChatMessage(role="user", content="hi")])


def test_last_cached_prompt_tokens_is_zero_without_a_prefix_cache() -> None:
    from inference_engine.adapters.vllm_adapter import VLLMAdapter

    assert VLLMAdapter().last_cached_prompt_tokens == 0


def test_cached_tokens_never_exceed_prompt_tokens() -> None:
    """llama.cpp cache entries include generated tokens, so the raw value can
    overshoot the current prompt. Usage must stay internally coherent."""
    from inference_engine.api.chat import _usage_from

    usage = _usage_from(39, 2, 43)
    assert usage.prompt_tokens_details.cached_tokens == 39


def test_cached_tokens_omitted_when_nothing_was_cached() -> None:
    from inference_engine.api.chat import _usage_from

    assert _usage_from(10, 2, 0).prompt_tokens_details is None


def test_user_field_is_stamped_on_spans() -> None:
    """Accepting `user` without recording it anywhere would make it a no-op."""
    from inference_engine.api.chat import _end_user_attrs

    assert _end_user_attrs("customer-42") == {"user.id": "customer-42"}
    assert _end_user_attrs(None) == {}
    assert _end_user_attrs("") == {}


def test_user_span_attribute_is_bounded() -> None:
    from inference_engine.api.chat import _end_user_attrs

    assert len(_end_user_attrs("u" * 5000)["user.id"]) == 128
