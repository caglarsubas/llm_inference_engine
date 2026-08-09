"""Per-request billing ledger — ``prometa.model-usage.v2``.

Exercised through the real ASGI app so the middleware, the ContextVar
accumulator, and the SSE generator hand-off are all in the path. The governed
harness is reused from ``test_model_routing_api`` so records carry real signed
policy attribution rather than a hand-built stub.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import threading
from collections.abc import AsyncIterator, Iterable
from dataclasses import fields
from pathlib import Path

import httpx
import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from inference_engine import auth as auth_mod
from inference_engine import otel as otel_mod
from inference_engine import usage_ledger
from inference_engine.adapters import (
    GenerationParams,
    StreamChunk,
    UpstreamGenerationError,
)
from inference_engine.api import _usage
from inference_engine.api.state import app_state
from inference_engine.cancellation import Cancellation
from inference_engine.config import settings
from inference_engine.evals import PolicyRegistry
from inference_engine.main import app, model_usage_ledger
from inference_engine.model_routing_runtime import (
    ModelRoutingRateLimiter,
    usage_cost_micros,
)
from inference_engine.usage_ledger import UsageLedger

from .test_model_routing_api import (
    _active_policy,
    _install_models,
    _pricing,
    _replace_reasoning_limits,
    _RoutingAdapter,
    _runtime_state,
)

CHAT = "/v1/chat/completions"
_MESSAGES = [{"role": "user", "content": "hi"}]


class _UsageStreamAdapter(_RoutingAdapter):
    """Streams a terminal usage frame, the way vLLM and OpenRouter do."""

    async def stream(
        self,
        messages: Iterable,
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> AsyncIterator[StreamChunk]:
        del messages, params, cancel
        self.stream_calls += 1
        self._raise_if_failed()
        yield StreamChunk(text=self.text)
        yield StreamChunk(
            text="",
            finish_reason="stop",
            prompt_tokens=11,
            completion_tokens=5,
        )


class _MidStreamFailureAdapter(_RoutingAdapter):
    """Fails after the first chunk, which is past the fallback cut-off."""

    async def stream(
        self,
        messages: Iterable,
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> AsyncIterator[StreamChunk]:
        del messages, params, cancel
        self.stream_calls += 1
        yield StreamChunk(text="partial")
        raise UpstreamGenerationError(
            error_type="upstream_error",
            upstream_status_code=502,
            backend=self.backend_name,
        )


class _CrashingAdapter(_RoutingAdapter):
    """Raises something no route handler expects, so it escapes to Starlette."""

    async def generate(
        self,
        messages: Iterable,
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ):
        del messages, params, cancel
        self.generate_calls += 1
        raise ZeroDivisionError("boom")


class _UpstreamFailureAdapter(_RoutingAdapter):
    async def generate(
        self,
        messages: Iterable,
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ):
        del messages, params, cancel
        self.generate_calls += 1
        raise UpstreamGenerationError(
            error_type="upstream_error",
            upstream_status_code=502,
            backend=self.backend_name,
        )


@pytest.fixture(autouse=True)
def _ledger_runtime(monkeypatch):
    previous_runtime = app_state.model_routing_runtime
    previous_limiter = app_state.model_routing_rate_limiter
    previous_policy_registry = app_state.policy_registry
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "model_routing_expected_org_id", "org-golden")
    monkeypatch.setattr(settings, "model_routing_input_token_reserve", 0)
    monkeypatch.setattr(settings, "openrouter_fallback_enabled", True)
    monkeypatch.setattr(settings, "openrouter_fallback_model", "escape:openrouter")
    monkeypatch.setattr(settings, "usage_ledger_enabled", True)
    monkeypatch.setattr(settings, "usage_ledger_max_buffer", 64)
    app_state.model_routing_runtime = _runtime_state()
    app_state.model_routing_rate_limiter = ModelRoutingRateLimiter()
    app_state.policy_registry = PolicyRegistry([])
    auth_mod._reset_for_tests()
    usage_ledger._reset_for_tests()
    yield
    usage_ledger._reset_for_tests()
    auth_mod._reset_for_tests()
    app_state.model_routing_runtime = previous_runtime
    app_state.model_routing_rate_limiter = previous_limiter
    app_state.policy_registry = previous_policy_registry


def _drain() -> list[dict]:
    ledger = usage_ledger.usage_ledger
    batch = ledger.peek()
    ledger.commit(len(batch))
    return batch


def _one() -> dict:
    [record] = _drain()
    return record


# --- master switch -----------------------------------------------------------


def test_ledger_is_off_by_default_and_emits_nothing(monkeypatch) -> None:
    monkeypatch.setattr(settings, "usage_ledger_enabled", False)
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text
    assert "x-orchestra-usage-record-id" not in response.headers
    assert usage_ledger.begin(engine_request_id="req_x", route=CHAT) is None
    assert _drain() == []


# --- blocking chat -----------------------------------------------------------


def test_blocking_chat_emits_one_fully_attributed_record(monkeypatch) -> None:
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text
    record = _one()
    assert record["schema"] == usage_ledger.SCHEMA
    assert record["engine_request_id"] == response.headers["x-request-id"]
    assert record["usage_record_id"] == response.headers["x-orchestra-usage-record-id"]
    assert re.fullmatch(r"req_[0-9a-f]{32}", record["engine_request_id"])
    assert re.fullmatch(r"usage_[0-9a-f]{32}", record["usage_record_id"])
    assert record["runtime_request_id"] is None
    assert record["model_invocation_id"] is None
    assert record["model_attempt_id"] is None
    assert record["route"] == CHAT
    assert record["operation"] == "chat"
    assert record["stream"] is False
    assert record["tenant"] == "anonymous"
    assert record["key_id"] == "anon"
    assert record["org_id"] == "org-golden"
    assert record["policy_id"] == "routing-golden-v1"
    assert record["route_id"] == "reasoning"
    assert record["pricing_digest"] == "sha256:api-pricing"
    assert record["policy_digest"].startswith("sha256:")
    assert record["requested_model"] == "reasoning"
    assert record["resolved_model"] == "qwen3:32b"
    assert record["backend"] == "routing-test"
    assert record["request_key_source"] == "local-inference"
    assert record["input_tokens"] == 7
    assert record["output_tokens"] == 3
    assert record["cached_tokens"] == 0
    # 7 in at 10_000 micros/M and 3 out at 20_000 micros/M, each component
    # rounded up the way the pre-flight cost ceiling does.
    assert record["cost_micros"] == 2
    assert record["finish_reason"] == "stop"
    assert record["outcome"] == "ok"
    assert record["http_status"] == 200
    assert record["error_type"] is None
    assert record["denial_code"] is None
    assert record["fallback"] is False
    assert record["duration_ms"] > 0
    assert record["input_token_upper_bound"] > 0
    assert record["output_token_budget"] == 512


def test_record_contains_only_declared_metadata_fields(monkeypatch) -> None:
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("SENTINEL-COMPLETION-4c1b")})

    response = TestClient(app).post(
        CHAT,
        json={
            "model": "reasoning",
            "messages": [{"role": "user", "content": "SENTINEL-PROMPT-9f3a"}],
        },
    )

    assert response.status_code == 200, response.text
    record = _one()
    assert set(record) == usage_ledger.SCHEMA_FIELDS
    encoded = json.dumps(record)
    assert "SENTINEL-PROMPT-9f3a" not in encoded
    assert "SENTINEL-COMPLETION-4c1b" not in encoded


def test_same_runtime_request_keeps_distinct_invocation_and_attempt_identities(
    monkeypatch,
) -> None:
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})
    client = TestClient(app)

    first = client.post(
        CHAT,
        json={"model": "reasoning", "messages": _MESSAGES},
        headers={
            "x-orchestra-runtime-request-id": "runtime-constant",
            "x-orchestra-model-invocation-id": "invocation-1",
            "x-orchestra-model-attempt-id": "attempt-1",
        },
    )
    second = client.post(
        CHAT,
        json={"model": "reasoning", "messages": _MESSAGES},
        headers={
            "x-orchestra-runtime-request-id": "runtime-constant",
            "x-orchestra-model-invocation-id": "invocation-2",
            "x-orchestra-model-attempt-id": "attempt-2",
        },
    )

    assert first.status_code == second.status_code == 200
    first_record, second_record = _drain()
    assert [first_record["runtime_request_id"], second_record["runtime_request_id"]] == [
        "runtime-constant",
        "runtime-constant",
    ]
    assert [first_record["model_invocation_id"], second_record["model_invocation_id"]] == [
        "invocation-1",
        "invocation-2",
    ]
    assert [first_record["model_attempt_id"], second_record["model_attempt_id"]] == [
        "attempt-1",
        "attempt-2",
    ]
    assert first_record["usage_record_id"] != second_record["usage_record_id"]
    assert first_record["engine_request_id"] != second_record["engine_request_id"]
    assert first.headers["x-orchestra-usage-record-id"] == first_record["usage_record_id"]
    assert second.headers["x-orchestra-usage-record-id"] == second_record["usage_record_id"]


def test_blocking_fallback_records_the_model_that_served(monkeypatch) -> None:
    _install_models(
        monkeypatch,
        {
            "qwen3:32b": _RoutingAdapter("primary", fail=True, backend="local"),
            "llama3.3:70b:openrouter": _RoutingAdapter("signed", backend="openrouter"),
        },
    )

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text
    record = _one()
    assert record["resolved_model"] == "llama3.3:70b:openrouter"
    assert record["backend"] == "openrouter"
    assert record["fallback"] is True
    assert record["fallback_from_model"] == "qwen3:32b"
    assert record["fallback_from_backend"] == "local"
    assert record["fallback_reason"] == "generation_timeout"
    assert record["outcome"] == "ok"


# --- streaming chat ----------------------------------------------------------


def test_streaming_chat_is_flushed_by_the_generator_not_the_middleware(monkeypatch) -> None:
    _install_models(monkeypatch, {"qwen3:32b": _UsageStreamAdapter("streamed")})

    with TestClient(app).stream(
        "POST",
        CHAT,
        json={"model": "reasoning", "messages": _MESSAGES, "stream": True},
    ) as response:
        assert response.status_code == 200
        usage_record_id = response.headers["x-orchestra-usage-record-id"]
        body = "".join(response.iter_text())

    assert "[DONE]" in body
    record = _one()
    assert record["usage_record_id"] == usage_record_id
    assert record["stream"] is True
    # http_status can only have come from the middleware and the token counts
    # only from the generator, so seeing both proves the deferred hand-off.
    assert record["http_status"] == 200
    assert record["input_tokens"] == 11
    assert record["output_tokens"] == 5
    assert record["cost_micros"] == 2
    assert record["ttft_ms"] is not None
    assert record["ttft_ms"] >= 0
    assert record["finish_reason"] == "stop"
    assert record["outcome"] == "ok"


@pytest.mark.asyncio
async def test_a_stream_whose_body_never_runs_is_still_emitted() -> None:
    """The client vanishes between response construction and first iteration."""
    started = False

    async def _body():
        nonlocal started
        started = True
        yield b"data: [DONE]\n\n"

    async def _call_next(_request):
        usage_ledger.mark_billable()
        usage_ledger.bind(stream=True, tenant="acme", resolved_model="qwen3:32b")
        return StreamingResponse(_body(), media_type="text/event-stream")

    scope = {
        "type": "http",
        "asgi": {"spec_version": "2.4"},
        "method": "POST",
        "path": CHAT,
        "headers": [],
    }
    request = Request(scope)
    request.state.engine_request_id = "req_00000000000000000000000000000000"
    request.state.runtime_request_id = "runtime-vanished"
    request.state.model_invocation_id = "invocation-vanished"
    request.state.model_attempt_id = "attempt-vanished"

    response = await model_usage_ledger(request, _call_next)

    async def _send(message):
        raise OSError("client vanished")

    async def _receive():
        return {"type": "http.disconnect"}

    with pytest.raises(ClientDisconnect):
        await response(scope, _receive, _send)

    # The SSE generator never ran, so the flush it owns never happened — and
    # the record is on the wire anyway.
    assert started is False
    record = _one()
    assert record["engine_request_id"] == "req_00000000000000000000000000000000"
    assert record["runtime_request_id"] == "runtime-vanished"
    assert record["model_invocation_id"] == "invocation-vanished"
    assert record["model_attempt_id"] == "attempt-vanished"
    assert record["stream"] is True
    assert record["http_status"] == 200
    assert record["outcome"] == "ok"


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason=(
        "A streamed response that falls back deadlocks the in-process test "
        "harness on 3.11 — TestClient's blocking portal deadlocks on both "
        "versions, ASGITransport only on 3.11, and both reproduce on an "
        "unmodified main with none of the ledger in the path. The same request "
        "completes in 0.01s over uvicorn, so the limitation is the harness, "
        "not the app."
    ),
)
@pytest.mark.asyncio
async def test_streaming_fallback_emits_one_record_for_the_serving_model(monkeypatch) -> None:
    # ASGITransport rather than TestClient: TestClient's blocking portal
    # deadlocks on a streamed response that falls back.
    _install_models(
        monkeypatch,
        {
            "qwen3:32b": _RoutingAdapter("primary", fail=True, backend="local"),
            "llama3.3:70b:openrouter": _UsageStreamAdapter("signed", backend="openrouter"),
        },
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://engine",
    ) as client:
        async with client.stream(
            "POST",
            CHAT,
            json={"model": "reasoning", "messages": _MESSAGES, "stream": True},
        ) as response:
            assert response.status_code == 200
            body = "".join([chunk async for chunk in response.aiter_text()])

    assert "signed" in body
    record = _one()
    assert record["resolved_model"] == "llama3.3:70b:openrouter"
    assert record["fallback"] is True
    assert record["fallback_from_model"] == "qwen3:32b"
    assert record["fallback_reason"] == "generation_timeout"
    assert record["input_tokens"] == 11
    assert record["outcome"] == "ok"


def test_streaming_error_frame_is_recorded_as_error_despite_http_200(monkeypatch) -> None:
    _install_models(monkeypatch, {"qwen3:32b": _MidStreamFailureAdapter("partial")})

    with TestClient(app).stream(
        "POST",
        CHAT,
        json={"model": "reasoning", "messages": _MESSAGES, "stream": True},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "upstream_error" in body
    record = _one()
    assert record["http_status"] == 200
    assert record["outcome"] == "error"
    assert record["finish_reason"] == "error"


# --- denials and failures ----------------------------------------------------


def test_bounds_denial_is_recorded_as_denied_without_token_fields(monkeypatch) -> None:
    calls = _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("unused")})

    response = TestClient(app).post(
        CHAT,
        json={"model": "reasoning", "max_tokens": 4_097, "messages": _MESSAGES},
    )

    assert response.status_code == 400
    assert calls == []
    record = _one()
    assert record["outcome"] == "denied"
    assert record["denial_code"] == "output_token_limit_exceeded"
    assert record["http_status"] == 400
    assert record["requested_model"] == "reasoning"
    assert record["resolved_model"] is None
    assert record["input_tokens"] is None
    assert record["output_tokens"] is None
    assert record["cost_micros"] is None
    assert response.headers["x-orchestra-usage-record-id"] == record["usage_record_id"]


def test_org_binding_denial_is_recorded_as_denied(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    auth_mod._set_keys_for_tests([("sk-wrong", "tenant-wrong", "org-other")])
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("unused")})

    response = TestClient(app).post(
        CHAT,
        json={"model": "reasoning", "messages": _MESSAGES},
        headers={"Authorization": "Bearer sk-wrong"},
    )

    assert response.status_code == 403
    record = _one()
    assert record["outcome"] == "denied"
    assert record["denial_code"] == "org_identity_mismatch"
    assert record["tenant"] == "tenant-wrong"
    assert record["resolved_model"] is None


def test_rate_limit_denial_is_recorded_as_denied(monkeypatch) -> None:
    app_state.model_routing_runtime = _runtime_state(
        _replace_reasoning_limits(_active_policy(), max_requests_per_minute=1)
    )
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})
    client = TestClient(app)

    assert client.post(CHAT, json={"model": "reasoning", "messages": _MESSAGES}).status_code == 200
    limited = client.post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert limited.status_code == 429
    served, denied = _drain()
    assert served["outcome"] == "ok"
    assert denied["outcome"] == "denied"
    assert denied["denial_code"] == "rate_limit_exceeded"
    assert denied["http_status"] == 429


def test_backend_timeout_is_recorded_as_timeout(monkeypatch) -> None:
    _install_models(
        monkeypatch,
        {
            "qwen3:32b": _RoutingAdapter("primary", fail=True),
            "llama3.3:70b:openrouter": _RoutingAdapter("signed", fail=True),
        },
    )

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 504
    record = _one()
    assert record["outcome"] == "timeout"
    assert record["error_type"] == "generation_timeout"
    assert record["resolved_model"] == "llama3.3:70b:openrouter"


def test_backend_error_is_recorded_as_error(monkeypatch) -> None:
    _install_models(
        monkeypatch,
        {
            "qwen3:32b": _UpstreamFailureAdapter("primary"),
            "llama3.3:70b:openrouter": _UpstreamFailureAdapter("signed"),
        },
    )

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 502
    record = _one()
    assert record["outcome"] == "error"
    assert record["error_type"] == "upstream_error"
    assert response.headers["x-orchestra-usage-record-id"] == record["usage_record_id"]


def test_a_rejected_structured_output_still_bills_both_attempts(monkeypatch) -> None:
    adapter = _RoutingAdapter("not json")
    _install_models(monkeypatch, {"qwen3:32b": adapter})

    response = TestClient(app).post(
        CHAT,
        json={
            "model": "reasoning",
            "messages": _MESSAGES,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "reply",
                    "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
                },
            },
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"]["type"] == "structured_output_invalid"
    assert adapter.generate_calls == 2
    record = _one()
    assert record["outcome"] == "error"
    assert record["error_type"] == "structured_output_invalid"
    assert record["input_tokens"] == 14
    assert record["output_tokens"] == 6


def test_unknown_model_records_the_route_unavailable_error_type(monkeypatch) -> None:
    _install_models(monkeypatch, {})

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 503
    record = _one()
    assert record["error_type"] == "model_route_unavailable"
    assert record["outcome"] == "error"
    assert record["resolved_model"] is None


# --- the other priced routes -------------------------------------------------


def test_completions_emit_one_record(monkeypatch) -> None:
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})

    response = TestClient(app).post("/v1/completions", json={"model": "reasoning", "prompt": "raw"})

    assert response.status_code == 200, response.text
    record = _one()
    assert record["route"] == "/v1/completions"
    assert record["operation"] == "text_completion"
    assert record["stream"] is False
    assert record["resolved_model"] == "qwen3:32b"
    assert record["input_tokens"] == 4
    assert record["output_tokens"] == 2
    assert record["outcome"] == "ok"


def test_embeddings_emit_one_record_with_no_output_tokens(monkeypatch) -> None:
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})

    response = TestClient(app).post(
        "/v1/embeddings",
        json={"model": "reasoning", "input": ["first", "second"]},
    )

    assert response.status_code == 200, response.text
    record = _one()
    assert record["route"] == "/v1/embeddings"
    assert record["operation"] == "embeddings"
    assert record["input_tokens"] == 6
    assert record["output_tokens"] == 0
    assert record["output_token_budget"] == 0
    assert record["outcome"] == "ok"


def test_unpriced_routes_produce_no_record(monkeypatch) -> None:
    _install_models(monkeypatch, {"reasoning": _RoutingAdapter("unused")})

    response = TestClient(app).post(
        "/v1/rerank",
        json={"model": "reasoning", "query": "q", "documents": ["d"]},
    )

    assert response.status_code == 503
    assert _drain() == []


# --- attribution gate --------------------------------------------------------


def test_an_unauthenticated_request_produces_no_record(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    auth_mod._set_keys_for_tests([("sk-real", "tenant-real", "org-golden")])
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("unused")})

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 401
    assert response.headers["x-request-id"].startswith("req_")
    assert "x-orchestra-usage-record-id" not in response.headers
    assert _drain() == []


def test_a_body_the_schema_rejects_produces_no_record(monkeypatch) -> None:
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("unused")})

    response = TestClient(app).post(CHAT, json={"model": "reasoning"})

    assert response.status_code == 422
    assert response.headers["x-request-id"].startswith("req_")
    assert "x-orchestra-usage-record-id" not in response.headers
    assert _drain() == []


def test_an_empty_prompt_list_is_rejected_before_the_seam_and_records_nothing(
    monkeypatch,
) -> None:
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("unused")})

    response = TestClient(app).post("/v1/completions", json={"model": "reasoning", "prompt": []})

    assert response.status_code == 400
    assert "x-orchestra-usage-record-id" not in response.headers
    assert _drain() == []


def test_an_empty_input_list_is_rejected_before_the_seam_and_records_nothing(
    monkeypatch,
) -> None:
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("unused")})

    response = TestClient(app).post("/v1/embeddings", json={"model": "reasoning", "input": []})

    assert response.status_code == 400
    assert "x-orchestra-usage-record-id" not in response.headers
    assert _drain() == []


def test_an_unhandled_pre_bind_failure_has_no_usage_record_header(monkeypatch) -> None:
    async def _explode_before_bind():
        raise RuntimeError("dependency crashed before bind_request")

    app.dependency_overrides[auth_mod.require_identity] = _explode_before_bind
    try:
        response = TestClient(app, raise_server_exceptions=False).post(
            CHAT,
            json={"model": "reasoning", "messages": _MESSAGES},
        )
    finally:
        app.dependency_overrides.pop(auth_mod.require_identity, None)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_server_error"
    assert response.headers["x-request-id"].startswith("req_")
    assert "x-orchestra-usage-record-id" not in response.headers
    assert _drain() == []


def test_a_failure_binding_attribution_degrades_the_record_instead_of_dropping_it(
    monkeypatch,
) -> None:
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})
    real_bind = usage_ledger.bind

    def exploding_bind(**fields) -> None:
        if "tenant" in fields:
            raise RuntimeError("a bug in the attribution mapping")
        real_bind(**fields)

    monkeypatch.setattr(usage_ledger, "bind", exploding_bind)

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text
    record = _one()
    assert record["tenant"] is None
    assert record["operation"] is None
    assert record["requested_model"] is None
    assert record["resolved_model"] == "qwen3:32b"
    assert record["input_tokens"] == 7
    assert record["output_tokens"] == 3
    assert record["http_status"] == 200
    assert record["outcome"] == "ok"


def test_unbillable_traffic_cannot_evict_a_billed_record(monkeypatch) -> None:
    monkeypatch.setattr(settings, "usage_ledger_max_buffer", 2)
    usage_ledger._reset_for_tests()
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})
    client = TestClient(app)

    # Fill every slot first, so the 422s arrive against a full buffer: if one
    # of them submitted, it would either evict a billed record or be counted as
    # a refusal. Neither happens, because it never reaches the ledger at all.
    for index in range(2):
        assert (
            client.post(
                CHAT,
                json={"model": "reasoning", "messages": _MESSAGES},
                headers={"x-orchestra-runtime-request-id": f"billed-{index}"},
            ).status_code
            == 200
        )
    for _ in range(10):
        assert client.post(CHAT, json={"model": "reasoning"}).status_code == 422

    records = _drain()
    assert [item["runtime_request_id"] for item in records] == ["billed-0", "billed-1"]
    assert [item["outcome"] for item in records] == ["ok", "ok"]
    assert [item["tenant"] for item in records] == ["anonymous", "anonymous"]
    snapshot = usage_ledger.usage_ledger.snapshot()
    assert snapshot["emitted_total"] == 2
    assert snapshot["dropped_total"] == 0


def test_a_full_buffer_refuses_the_arriving_record_not_the_buffered_ones(monkeypatch) -> None:
    monkeypatch.setattr(settings, "usage_ledger_max_buffer", 2)
    usage_ledger._reset_for_tests()
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})
    client = TestClient(app)

    for index in range(5):
        assert (
            client.post(
                CHAT,
                json={"model": "reasoning", "messages": _MESSAGES},
                headers={"x-orchestra-runtime-request-id": f"billed-{index}"},
            ).status_code
            == 200
        )

    # The two oldest invoice lines are the ones the drain task still owes, so
    # they are the two that survive three billable arrivals hitting a full
    # buffer.
    assert [item["runtime_request_id"] for item in _drain()] == ["billed-0", "billed-1"]
    assert usage_ledger.usage_ledger.snapshot()["dropped_total"] == 3


def test_a_caller_supplied_model_name_is_bounded_in_the_record(monkeypatch) -> None:
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("unused")})

    response = TestClient(app).post(
        CHAT,
        json={"model": "z" * 5_000, "messages": _MESSAGES},
    )

    assert response.status_code >= 400
    record = _one()
    assert record["requested_model"] == "z" * 200


def test_an_unhandled_exception_records_the_status_the_client_got(monkeypatch) -> None:
    _install_models(monkeypatch, {"qwen3:32b": _CrashingAdapter("unused")})

    response = TestClient(app, raise_server_exceptions=False).post(
        CHAT,
        json={"model": "reasoning", "messages": _MESSAGES},
    )

    assert response.status_code == 500
    record = _one()
    assert record["http_status"] == 500
    assert record["outcome"] == "error"
    assert response.json()["error"]["code"] == "internal_server_error"
    assert response.headers["x-request-id"] == record["engine_request_id"]
    assert response.headers["x-orchestra-usage-record-id"] == record["usage_record_id"]


# --- buffer, sink, and blast radius -----------------------------------------


def test_buffer_keeps_what_it_accepted_and_counts_what_it_refused() -> None:
    ledger = UsageLedger(2)

    for index in range(5):
        ledger.submit({"usage_record_id": f"usage-{index}"})

    snapshot = ledger.snapshot()
    assert snapshot["emitted_total"] == 2
    assert snapshot["dropped_total"] == 3
    assert snapshot["buffered"] == 2
    # Accepted records are owed to an invoice, so overflow refuses the arriving
    # record instead of evicting them.
    assert [item["usage_record_id"] for item in ledger.peek()] == ["usage-0", "usage-1"]


def test_buffer_overflow_is_logged_once_per_episode(monkeypatch) -> None:
    warnings: list[tuple[str, dict]] = []

    class _Recorder:
        def warning(self, event: str, **fields) -> None:
            warnings.append((event, fields))

    monkeypatch.setattr(usage_ledger, "log", _Recorder())
    ledger = UsageLedger(1)

    ledger.submit({"usage_record_id": "usage-0"})
    ledger.submit({"usage_record_id": "usage-1"})
    ledger.submit({"usage_record_id": "usage-2"})

    assert [event for event, _ in warnings] == ["usage_ledger.buffer_full"]
    assert warnings[0][1] == {"max_buffer": 1, "dropped_total": 1}

    ledger.commit(1)
    ledger.submit({"usage_record_id": "usage-3"})
    ledger.submit({"usage_record_id": "usage-4"})

    assert [event for event, _ in warnings] == ["usage_ledger.buffer_full"] * 2
    assert ledger.snapshot()["dropped_total"] == 3


def test_the_emitted_payload_is_exactly_the_declared_schema() -> None:
    assert set(usage_ledger._render(_record_stub())) == usage_ledger.SCHEMA_FIELDS


def test_versioned_contract_fixture_matches_the_emitter_and_wire_identity_contract() -> None:
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "contracts"
        / "prometa-model-usage-v2.schema.json"
    )
    raw = contract_path.read_bytes()
    contract = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == (
        "845f830df424f1626717e60a5dbd05e01187f84e2e96223527cceda521f3d55a"
    )
    assert contract["properties"]["schema"]["const"] == usage_ledger.SCHEMA
    assert contract["properties"]["event"]["const"] == usage_ledger.EVENT
    assert contract["required"] == list(usage_ledger._SCHEMA_FIELD_ORDER)
    assert set(contract["properties"]) == usage_ledger.SCHEMA_FIELDS
    assert "request_id" not in contract["properties"]
    assert contract["x-prometa-identity-order"] == [
        "usage_record_id",
        "engine_request_id",
        "runtime_request_id",
        "model_invocation_id",
        "model_attempt_id",
    ]
    assert contract["x-prometa-header-mapping"] == {
        "request": {
            "x-orchestra-runtime-request-id": "runtime_request_id",
            "x-orchestra-model-invocation-id": "model_invocation_id",
            "x-orchestra-model-attempt-id": "model_attempt_id",
        },
        "response": {
            "x-request-id": "engine_request_id",
            "x-orchestra-usage-record-id": "usage_record_id",
        },
        "inbound-x-request-id-alias": None,
    }
    assert contract["x-prometa-delivery"] == {
        "mode": "best-effort-buffered",
        "dedupeField": "usage_record_id",
        "redeliveryWindow": {
            "condition": "drain-cancelled-after-sink-acceptance-uncertain",
            "retainsDedupeValue": True,
        },
    }
    external_pattern = contract["$defs"]["externalIdentity"]["pattern"]
    for value in ["runtime-1", "null-runtime", "none-1", "nil.value", "undefined/1"]:
        assert re.fullmatch(external_pattern, value)
    for value in [
        "null",
        "NULL",
        "NuLl",
        "none",
        "NoNe",
        "nil",
        "NIL",
        "undefined",
        "UnDeFiNeD",
    ]:
        assert re.fullmatch(external_pattern, value) is None
    assert contract["properties"]["fallback_from_model"]["type"] == [
        "string",
        "null",
    ]
    invocation_rule, attempt_rule = contract["allOf"]
    assert invocation_rule["if"]["properties"] == {
        "model_invocation_id": {"type": "string"}
    }
    assert invocation_rule["then"]["properties"] == {
        "runtime_request_id": {"type": "string"}
    }
    assert attempt_rule["if"]["properties"] == {
        "model_attempt_id": {"type": "string"}
    }
    assert attempt_rule["then"]["properties"] == {
        "runtime_request_id": {"type": "string"},
        "model_invocation_id": {"type": "string"},
    }


def test_a_schema_key_with_no_field_behind_it_is_refused_at_import() -> None:
    record_fields = frozenset(field.name for field in fields(usage_ledger.UsageRecord))

    with pytest.raises(RuntimeError, match="no accumulator attribute"):
        usage_ledger._check_schema_coverage(
            schema_fields=usage_ledger.SCHEMA_FIELDS | {"seat_count"},
            record_fields=record_fields,
        )
    with pytest.raises(RuntimeError, match="neither emitted nor declared"):
        usage_ledger._check_schema_coverage(
            schema_fields=usage_ledger.SCHEMA_FIELDS,
            record_fields=record_fields | {"prompt_text"},
        )
    with pytest.raises(RuntimeError, match="not all present"):
        usage_ledger._check_schema_coverage(
            schema_fields=usage_ledger.SCHEMA_FIELDS,
            record_fields=record_fields - {"billable"},
        )


def test_sink_renders_one_json_line_per_record(capsys) -> None:
    usage_ledger._sink([usage_ledger._render(_record_stub())])

    line = capsys.readouterr().out.strip()
    assert set(json.loads(line)) == usage_ledger.SCHEMA_FIELDS
    assert json.loads(line)["event"] == usage_ledger.EVENT


def _record_stub() -> usage_ledger.UsageRecord:
    return usage_ledger.UsageRecord(
        engine_request_id="req_00000000000000000000000000000000",
        route=CHAT,
        started_at=0.0,
        usage_record_id="usage_00000000000000000000000000000000",
    )


@pytest.mark.asyncio
async def test_a_failing_sink_is_counted_and_does_not_raise(monkeypatch) -> None:
    def _boom(batch):
        raise OSError("stdout is gone")

    monkeypatch.setattr(usage_ledger, "_sink", _boom)
    usage_ledger.usage_ledger.submit({"usage_record_id": "usage-1"})

    await usage_ledger.drain_once(usage_ledger.usage_ledger)

    assert usage_ledger.usage_ledger.snapshot()["sink_failures_total"] == 1


@pytest.mark.asyncio
async def test_the_drain_loop_survives_a_sink_failure_and_stops_on_cancel(monkeypatch) -> None:
    shipped: list[dict] = []
    # threading, not asyncio, events: ``_sink`` runs on a worker thread.
    failed_once = threading.Event()
    shipped_after_failure = threading.Event()

    def _sink(batch: list[dict]) -> None:
        if not failed_once.is_set():
            failed_once.set()
            raise OSError("stdout is gone")
        shipped.extend(batch)
        shipped_after_failure.set()

    monkeypatch.setattr(usage_ledger, "_sink", _sink)
    ledger = UsageLedger(16)
    task = asyncio.create_task(usage_ledger.run_usage_ledger_drain(ledger, interval_seconds=0.001))
    try:
        ledger.submit({"usage_record_id": "usage-1"})
        assert await asyncio.to_thread(failed_once.wait, 5.0)
        ledger.submit({"usage_record_id": "usage-2"})
        assert await asyncio.to_thread(shipped_after_failure.wait, 5.0)
    finally:
        task.cancel()
        # The lifespan teardown relies on cancellation propagating out cleanly.
        with pytest.raises(asyncio.CancelledError):
            await task

    assert [item["usage_record_id"] for item in shipped] == ["usage-2"]
    assert ledger.snapshot()["sink_failures_total"] == 1


@pytest.mark.asyncio
async def test_a_cancel_mid_sink_keeps_the_batch_for_the_next_drain(monkeypatch) -> None:
    """The shutdown race: cancellation lands after the read, before the write."""
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    first_delivery: list[dict] = []

    def _stalled_sink(batch: list[dict]) -> None:
        first_delivery.extend(batch)
        entered.set()
        release.wait(5.0)
        completed.set()

    monkeypatch.setattr(usage_ledger, "_sink", _stalled_sink)
    ledger = UsageLedger(4)
    ledger.submit({"usage_record_id": "usage-1"})

    task = asyncio.create_task(usage_ledger.drain_once(ledger))
    assert await asyncio.to_thread(entered.wait, 5.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await asyncio.to_thread(completed.wait, 5.0)

    # The batch is still owed, so it is still buffered — and it is not a sink
    # failure, which would have retired it.
    assert ledger.snapshot() == {
        "emitted_total": 1,
        "dropped_total": 0,
        "sink_failures_total": 0,
        "buffered": 1,
    }

    shipped: list[dict] = []
    monkeypatch.setattr(usage_ledger, "_sink", shipped.extend)
    await usage_ledger.drain_once(ledger)

    assert [item["usage_record_id"] for item in shipped] == ["usage-1"]
    assert first_delivery[0]["usage_record_id"] == shipped[0]["usage_record_id"]
    assert ledger.snapshot()["buffered"] == 0


@pytest.mark.asyncio
async def test_a_shipped_batch_is_retired_and_not_reshipped(monkeypatch) -> None:
    shipped: list[dict] = []
    monkeypatch.setattr(usage_ledger, "_sink", shipped.extend)
    ledger = UsageLedger(4)
    ledger.submit({"usage_record_id": "usage-1"})

    await usage_ledger.drain_once(ledger)
    ledger.submit({"usage_record_id": "usage-2"})
    await usage_ledger.drain_once(ledger)

    assert [item["usage_record_id"] for item in shipped] == ["usage-1", "usage-2"]
    assert ledger.snapshot()["buffered"] == 0


def test_a_failing_flush_never_fails_the_request(monkeypatch) -> None:
    def _boom(record):
        raise RuntimeError("ledger is broken")

    monkeypatch.setattr(usage_ledger.usage_ledger, "submit", _boom)
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "primary"


def test_a_failing_binder_never_fails_the_request(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("binder is broken")

    monkeypatch.setattr(usage_ledger, "bind", _boom)
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text


def test_binding_outside_a_request_scope_is_a_noop() -> None:
    usage_ledger.bind(resolved_model="qwen3:32b")
    usage_ledger.capture_trace_context()
    _usage.bind_usage(input_tokens=1, output_tokens=1)
    _usage.bind_error_type("model_not_found")
    usage_ledger.flush(usage_ledger.current())

    assert usage_ledger.current() is None
    assert _drain() == []


# --- pricing -----------------------------------------------------------------


def test_usage_cost_micros_reports_unknown_rather_than_free() -> None:
    pricing = _pricing()

    assert usage_cost_micros(None, model="qwen3:32b", input_tokens=1, output_tokens=1) is None
    assert usage_cost_micros(pricing, model=None, input_tokens=1, output_tokens=1) is None
    assert usage_cost_micros(pricing, model="not-in-catalog", input_tokens=1, output_tokens=1) is None
    # A single token of a priced model rounds up to a whole micro rather than
    # down to nothing.
    assert usage_cost_micros(pricing, model="qwen3:32b", input_tokens=1, output_tokens=0) == 1
    assert usage_cost_micros(pricing, model="qwen3:32b", input_tokens=0, output_tokens=0) == 0


def test_cost_micros_is_none_when_the_served_model_is_unpriced(monkeypatch) -> None:
    app_state.model_routing_runtime = _runtime_state(
        _replace_reasoning_limits(_active_policy(), max_cost_micros_per_request=None)
    )
    app_state.model_routing_pricing = None
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text
    record = _one()
    assert record["input_tokens"] == 7
    assert record["cost_micros"] is None


# --- trace correlation -------------------------------------------------------


def test_trace_ids_match_the_serving_span(monkeypatch, _session_exporter) -> None:
    _session_exporter.clear()
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text
    record = _one()
    [generation] = [
        item for item in _session_exporter.get_finished_spans() if item.name == "chat.generate"
    ]
    assert record["trace_id"] == format(generation.context.trace_id, "032x")
    assert record["span_id"] == format(generation.context.span_id, "016x")
    assert len(record["trace_id"]) == 32
    assert len(record["span_id"]) == 16


def test_trace_ids_are_absent_when_tracing_is_off(monkeypatch) -> None:
    monkeypatch.setattr(otel_mod, "_tracer", None)
    _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("primary")})

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text
    record = _one()
    assert record["trace_id"] is None
    assert record["span_id"] is None
