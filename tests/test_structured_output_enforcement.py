"""Validate-and-retry for backends with no grammar-constrained decoding.

Backends that enforce a schema in the sampler are trusted and skipped; only
MLX-shaped adapters (``supports_structured_outputs = False``) go through the
retry loop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

import pytest
from fastapi import HTTPException

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.api.chat import _enforce_structured_output
from inference_engine.auth import Identity
from inference_engine.cancellation import Cancellation
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import ChatMessage

_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class _ScriptedAdapter(InferenceAdapter):
    """Returns each scripted reply in turn, recording what it was asked."""

    backend_name = "scripted"

    def __init__(self, replies: list[str], *, supports_structured_outputs: bool = False) -> None:
        self.replies = list(replies)
        self.supports_structured_outputs = supports_structured_outputs
        self.prompts: list[list[ChatMessage]] = []

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return None

    async def load(self, descriptor: ModelDescriptor) -> None: ...
    async def unload(self) -> None: ...

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> GenerationResult:
        self.prompts.append(list(messages))
        text = self.replies.pop(0)
        return GenerationResult(
            text=text, finish_reason="stop", prompt_tokens=5, completion_tokens=7
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="", finish_reason="stop")


def _params() -> GenerationParams:
    return GenerationParams(
        json_mode=True, json_schema=_SCHEMA, json_schema_name="out", json_schema_strict=True
    )


def _result(text: str) -> GenerationResult:
    return GenerationResult(
        text=text, finish_reason="stop", prompt_tokens=5, completion_tokens=7
    )


async def _run(adapter: _ScriptedAdapter, first_text: str, params=None):
    return await _enforce_structured_output(
        _result(first_text),
        adapter,
        "fake:1",
        [ChatMessage(role="user", content="hi")],
        params or _params(),
        Identity(tenant="dev", key_id="sk-x"),
    )


@pytest.mark.asyncio
async def test_valid_document_passes_through_without_a_retry() -> None:
    adapter = _ScriptedAdapter([])
    result = await _run(adapter, '{"answer": "yes"}')
    assert result.text == '{"answer": "yes"}'
    assert adapter.prompts == [], "no retry should have been issued"


@pytest.mark.asyncio
async def test_invalid_document_is_retried_once_and_accepted() -> None:
    adapter = _ScriptedAdapter(['{"answer": "now valid"}'])
    result = await _run(adapter, "sorry, I cannot")
    assert result.text == '{"answer": "now valid"}'
    assert len(adapter.prompts) == 1


@pytest.mark.asyncio
async def test_retry_prompt_carries_the_validation_error_and_schema() -> None:
    adapter = _ScriptedAdapter(['{"answer": "ok"}'])
    await _run(adapter, '{"wrong": 1}')

    repair = adapter.prompts[0][-1]
    assert repair.role == "user"
    assert "missing required property 'answer'" in repair.content
    assert "out" in repair.content
    # The failed attempt is replayed so the model can see what it did.
    assert adapter.prompts[0][-2].role == "assistant"
    assert adapter.prompts[0][-2].content == '{"wrong": 1}'


@pytest.mark.asyncio
async def test_usage_covers_both_attempts() -> None:
    adapter = _ScriptedAdapter(['{"answer": "ok"}'])
    result = await _run(adapter, "nope")
    # The caller paid for the failed generation too.
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 14


@pytest.mark.asyncio
async def test_second_failure_raises_a_typed_502() -> None:
    adapter = _ScriptedAdapter(["still not json"])
    with pytest.raises(HTTPException) as excinfo:
        await _run(adapter, "not json")
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail["type"] == "structured_output_invalid"
    assert excinfo.value.detail["param"] == "response_format"


@pytest.mark.asyncio
async def test_grammar_capable_backend_is_not_revalidated() -> None:
    """A false negative in our schema subset must not break a working backend."""
    adapter = _ScriptedAdapter([], supports_structured_outputs=True)
    result = await _run(adapter, "this would never validate")
    assert result.text == "this would never validate"
    assert adapter.prompts == []


@pytest.mark.asyncio
async def test_no_schema_means_no_enforcement() -> None:
    adapter = _ScriptedAdapter([])
    result = await _run(adapter, "free text", params=GenerationParams())
    assert result.text == "free text"
    assert adapter.prompts == []


@pytest.mark.asyncio
async def test_tool_call_turns_skip_validation() -> None:
    """A tool call is a valid outcome — there is no document to check."""
    adapter = _ScriptedAdapter([])
    result = GenerationResult(
        text="",
        finish_reason="tool_calls",
        prompt_tokens=1,
        completion_tokens=1,
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
    )
    out = await _enforce_structured_output(
        result,
        adapter,
        "fake:1",
        [ChatMessage(role="user", content="hi")],
        _params(),
        Identity(tenant="dev", key_id="sk-x"),
    )
    assert out is result
    assert adapter.prompts == []
