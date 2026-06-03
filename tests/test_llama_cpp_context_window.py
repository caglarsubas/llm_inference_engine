"""Regression tests for the n_ctx-truncation fix.

Covers three engine-side behaviours added in response to the
``nemotron-context-window-8k-answer-truncation`` report:

1. ``LlamaCppAdapter._effective_n_ctx`` — the configured context *ceiling* is
   clamped per-model to the GGUF's trained context, so long-context reasoning
   models get the full window while short-context models don't over-allocate.
2. ``LlamaCppAdapter._as_context_error`` — llama.cpp's opaque overflow
   ``ValueError`` is translated into a typed ``ContextLengthExceededError``.
3. The chat + completions routes map that typed error to a deterministic
   ``400 context_length_exceeded`` (blocking) / terminal SSE ``error`` event
   (streaming) instead of a generic 500.

None of these need the native ``llama_cpp`` module — the heavy import is lazy,
and the logic under test is pure Python.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

import pytest
from fastapi import HTTPException

from inference_engine.adapters import (
    ContextLengthExceededError,
    GenerationParams,
    InferenceAdapter,
    StreamChunk,
)
from inference_engine.adapters.base import GenerationResult
from inference_engine.adapters.llama_cpp import LlamaCppAdapter
from inference_engine.api.chat import _blocking_response, _stream_response
from inference_engine.auth import Identity
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import ChatMessage


# --------------------------------------------------------------------------- #
# 1. Effective-n_ctx clamping
# --------------------------------------------------------------------------- #


def test_effective_n_ctx_clamps_ceiling_to_trained_context() -> None:
    # Model trained at 8192, ceiling 32768 → keep the smaller trained window.
    assert LlamaCppAdapter._effective_n_ctx(32768, 8192) == 8192


def test_effective_n_ctx_uses_ceiling_when_model_supports_more() -> None:
    # Long-context model (n_ctx_train 131072) under a 32768 ceiling → 32768.
    assert LlamaCppAdapter._effective_n_ctx(32768, 131072) == 32768


def test_effective_n_ctx_unknown_train_falls_back_to_requested() -> None:
    # 0 = probe couldn't read metadata → don't clamp, honour the config.
    assert LlamaCppAdapter._effective_n_ctx(32768, 0) == 32768


def test_effective_n_ctx_floor_guards_bogus_tiny_train() -> None:
    # A nonsense tiny trained context can't collapse the window below the floor.
    assert LlamaCppAdapter._effective_n_ctx(32768, 16) >= 512


def test_effective_n_ctx_equal_values_are_stable() -> None:
    assert LlamaCppAdapter._effective_n_ctx(8192, 8192) == 8192


# --------------------------------------------------------------------------- #
# 2. Overflow-error translation
# --------------------------------------------------------------------------- #


def test_as_context_error_parses_llamacpp_overflow() -> None:
    exc = ValueError("Requested tokens (9001) exceed context window of 8192")
    err = LlamaCppAdapter._as_context_error(exc, backend="llama_cpp")
    assert isinstance(err, ContextLengthExceededError)
    assert err.requested_tokens == 9001
    assert err.context_window == 8192
    assert err.backend == "llama_cpp"


def test_as_context_error_is_case_and_spacing_insensitive() -> None:
    exc = ValueError("requested token (33000) EXCEED  context  window  of 32768")
    err = LlamaCppAdapter._as_context_error(exc, backend="llama_cpp")
    assert isinstance(err, ContextLengthExceededError)
    assert (err.requested_tokens, err.context_window) == (33000, 32768)


def test_as_context_error_ignores_unrelated_value_errors() -> None:
    assert LlamaCppAdapter._as_context_error(ValueError("bad json"), backend="x") is None


def test_as_context_error_ignores_non_value_errors() -> None:
    assert LlamaCppAdapter._as_context_error(RuntimeError("nope"), backend="x") is None


def test_context_error_detail_shape_is_openai_typed() -> None:
    err = ContextLengthExceededError(requested_tokens=9001, context_window=8192)
    detail = err.error_detail()
    assert detail["type"] == "context_length_exceeded"
    assert detail["code"] == "context_length_exceeded"
    assert detail["requested_tokens"] == 9001
    assert detail["context_window"] == 8192
    assert "8192" in detail["message"]


def test_context_error_default_message_without_counts() -> None:
    err = ContextLengthExceededError()
    detail = err.error_detail()
    assert detail["type"] == "context_length_exceeded"
    assert "requested_tokens" not in detail
    assert "context_window" not in detail


# --------------------------------------------------------------------------- #
# 3. Route mapping
# --------------------------------------------------------------------------- #


class _OverflowAdapter(InferenceAdapter):
    """Adapter whose generation always overflows the context window."""

    backend_name = "fake-overflow"

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return None

    async def load(self, descriptor: ModelDescriptor) -> None: ...
    async def unload(self) -> None: ...

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel=None
    ) -> GenerationResult:
        raise ContextLengthExceededError(
            requested_tokens=9001, context_window=8192, backend=self.backend_name
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel=None
    ) -> AsyncIterator[StreamChunk]:
        raise ContextLengthExceededError(
            requested_tokens=9001, context_window=8192, backend=self.backend_name
        )
        yield  # pragma: no cover — unreachable, marks this an async generator


@pytest.mark.asyncio
async def test_blocking_overflow_maps_to_typed_400() -> None:
    adapter = _OverflowAdapter()
    identity = Identity(tenant="dev", key_id="sk-x")
    with pytest.raises(HTTPException) as ei:
        await _blocking_response(
            adapter,
            "nemotron-3-nano:30b",
            [ChatMessage(role="user", content="be exhaustive")],
            GenerationParams(),
            identity,
        )
    assert ei.value.status_code == 400
    assert ei.value.detail["type"] == "context_length_exceeded"
    assert ei.value.detail["context_window"] == 8192


@pytest.mark.asyncio
async def test_streaming_overflow_emits_terminal_error_event() -> None:
    adapter = _OverflowAdapter()
    identity = Identity(tenant="dev", key_id="sk-x")

    class _Req:
        async def is_disconnected(self) -> bool:
            return False

    events = [
        chunk
        async for chunk in _stream_response(
            adapter,
            "nemotron-3-nano:30b",
            [ChatMessage(role="user", content="be exhaustive")],
            GenerationParams(),
            identity,
            _Req(),
        )
    ]
    # Role delta is sent first, then a typed error event closes the stream.
    error_events = [e for e in events if e.get("event") == "error"]
    assert len(error_events) == 1
    assert "context_length_exceeded" in error_events[0]["data"]
    # The normal "[DONE]" trailer must NOT follow an error close.
    assert not any(e.get("data") == "[DONE]" for e in events)
