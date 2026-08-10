"""Retries, the signed-policy budget reservation, and the usage ledger — together.

These three now sit on the same request, so the interaction is the thing worth
testing: a retried attempt must reach the *same* deployment before candidate
selection is allowed to change the model, and it must not turn one request into
two budget reservations or two invoice lines.

Driven through the real ASGI app so the enforcement, the reservation settle in
``chat_completions``' ``finally``, and the ledger middleware are all in the path.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from inference_engine import auth as auth_mod
from inference_engine import usage_ledger
from inference_engine.api import _model_routing
from inference_engine.adapters.vllm_adapter import VLLMAdapter
from inference_engine.adapters._upstream_retry import retry_counters
from inference_engine.api.state import app_state
from inference_engine.config import settings
from inference_engine.evals import PolicyRegistry
from inference_engine.main import app
from inference_engine.model_routing_runtime import ModelRoutingRateLimiter
from inference_engine.registry import ModelDescriptor
from inference_engine.registry.breaker import get_upstream_breaker

from .test_model_routing_api import (
    _RoutingAdapter,
    _install_models,
    _runtime_state,
)

CHAT = "/v1/chat/completions"
_MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _governed_ledger_runtime(monkeypatch):
    previous_runtime = app_state.model_routing_runtime
    previous_limiter = app_state.model_routing_rate_limiter
    previous_policy_registry = app_state.policy_registry
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "model_routing_expected_org_id", "org-golden")
    monkeypatch.setattr(settings, "model_routing_input_token_reserve", 0)
    monkeypatch.setattr(settings, "openrouter_fallback_enabled", True)
    monkeypatch.setattr(settings, "usage_ledger_enabled", True)
    monkeypatch.setattr(settings, "usage_ledger_max_buffer", 64)
    monkeypatch.setattr(settings, "upstream_retry_enabled", True)
    monkeypatch.setattr(settings, "upstream_retry_max_attempts", 2)
    monkeypatch.setattr(settings, "upstream_retry_base_delay_seconds", 0.001)
    monkeypatch.setattr(settings, "upstream_retry_max_delay_seconds", 0.002)
    app_state.model_routing_runtime = _runtime_state()
    app_state.model_routing_rate_limiter = ModelRoutingRateLimiter()
    app_state.policy_registry = PolicyRegistry([])
    auth_mod._reset_for_tests()
    usage_ledger._reset_for_tests()
    get_upstream_breaker().reset()
    retry_counters.reset()
    yield
    get_upstream_breaker().reset()
    retry_counters.reset()
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


def _count_reservation_calls(monkeypatch) -> dict[str, int]:
    counts = {"enforced": 0, "settled": 0}
    real_enforce = _model_routing.enforce_model_routing_request
    real_settle = _model_routing.settle_reservation

    def enforce(*args, **kwargs):
        counts["enforced"] += 1
        return real_enforce(*args, **kwargs)

    async def settle(decision):
        counts["settled"] += 1
        return await real_settle(decision)

    monkeypatch.setattr(_model_routing, "enforce_model_routing_request", enforce)
    monkeypatch.setattr(_model_routing, "settle_reservation", settle)
    return counts


def _vllm_adapter_over(handler) -> VLLMAdapter:
    """A real vLLM adapter whose transport is ``handler``, loaded and ready."""
    adapter = VLLMAdapter()
    descriptor = ModelDescriptor(
        name="qwen3",
        tag="32b",
        namespace="vllm",
        registry="local",
        model_path=Path("vllm://vllm:8000/qwen3-32b"),
        format="vllm",
        params={"model_id": "qwen3-32b"},
        size_bytes=0,
        endpoint="http://vllm:8000",
    )
    # ``load`` is async but does no IO beyond constructing the client, so the
    # TestClient's own loop is not needed to prime it.
    import asyncio  # noqa: PLC0415 — local to the helper

    asyncio.run(adapter.load(descriptor))
    adapter._client = httpx.AsyncClient(  # noqa: SLF001 — test scaffolding
        base_url="http://vllm:8000",
        transport=httpx.MockTransport(handler),
        timeout=30.0,
    )
    return adapter


def _chat_body(content: str) -> dict:
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": "qwen3-32b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


def test_a_transient_503_retries_the_same_model_instead_of_changing_it(
    monkeypatch,
) -> None:
    """The headline behaviour: a dropped packet must not swap the caller's model."""
    upstream_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(1)
        if len(upstream_calls) == 1:
            return httpx.Response(503, json={"error": "no capacity"})
        return httpx.Response(200, json=_chat_body("primary-recovered"))

    primary = _vllm_adapter_over(handler)
    signed_fallback = _RoutingAdapter("signed-fallback", backend="openrouter")
    acquired = _install_models(
        monkeypatch,
        {"qwen3:32b": primary, "llama3.3:70b:openrouter": signed_fallback},
    )

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "qwen3:32b"
    assert body["choices"][0]["message"]["content"] == "primary-recovered"
    assert body["fallback_from_model"] is None
    # The second candidate was never even acquired.
    assert acquired == ["qwen3:32b"]
    assert signed_fallback.generate_calls == 0
    assert len(upstream_calls) == 2


def test_a_retried_request_reserves_and_settles_the_budget_exactly_once(
    monkeypatch,
) -> None:
    counts = _count_reservation_calls(monkeypatch)
    upstream_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(1)
        if len(upstream_calls) == 1:
            return httpx.Response(503, json={"error": "no capacity"})
        return httpx.Response(200, json=_chat_body("primary-recovered"))

    _install_models(monkeypatch, {"qwen3:32b": _vllm_adapter_over(handler)})

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text
    assert len(upstream_calls) == 2
    # Enforcement runs once per request, above the adapter, so the retry cannot
    # place a second reservation against the tenant's budget.
    assert counts == {"enforced": 1, "settled": 1}


def test_a_retried_request_emits_exactly_one_ledger_record(monkeypatch) -> None:
    upstream_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(1)
        if len(upstream_calls) == 1:
            return httpx.Response(503, json={"error": "no capacity"})
        return httpx.Response(200, json=_chat_body("primary-recovered"))

    _install_models(monkeypatch, {"qwen3:32b": _vllm_adapter_over(handler)})

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text
    assert len(upstream_calls) == 2
    [record] = _drain()
    assert record["resolved_model"] == "qwen3:32b"
    assert record["fallback"] is False
    assert record["outcome"] == "ok"
    # One record billing the tokens of the attempt that actually produced the
    # answer — the failed attempt generated none.
    assert record["input_tokens"] == 7
    assert record["output_tokens"] == 3
    assert record["usage_record_id"] == response.headers["x-orchestra-usage-record-id"]


def test_fallback_still_happens_when_the_retry_does_not_recover(monkeypatch) -> None:
    """Retrying first does not remove fallback — it only reorders it."""
    upstream_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(1)
        return httpx.Response(503, json={"error": "still down"})

    primary = _vllm_adapter_over(handler)
    signed_fallback = _RoutingAdapter("signed-fallback", backend="openrouter")
    _install_models(
        monkeypatch,
        {"qwen3:32b": primary, "llama3.3:70b:openrouter": signed_fallback},
    )

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "llama3.3:70b:openrouter"
    assert body["fallback_from_model"] == "qwen3:32b"
    # Two attempts on the primary, then one on the signed fallback.
    assert len(upstream_calls) == 2
    assert signed_fallback.generate_calls == 1
    [record] = _drain()
    assert record["fallback"] is True
    assert record["resolved_model"] == "llama3.3:70b:openrouter"


def test_the_chat_span_carries_the_deployment_and_its_breaker_state(
    monkeypatch,
    _session_exporter,
) -> None:
    _session_exporter.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_body("ok"))

    _install_models(monkeypatch, {"qwen3:32b": _vllm_adapter_over(handler)})

    response = TestClient(app).post(CHAT, json={"model": "reasoning", "messages": _MESSAGES})

    assert response.status_code == 200, response.text
    [generation] = [
        item for item in _session_exporter.get_finished_spans() if item.name == "chat.generate"
    ]
    assert generation.attributes["llm.upstream.deployment"] == (
        "vllm|http://vllm:8000|qwen3-32b"
    )
    assert generation.attributes["llm.upstream.breaker.state"] == "closed"
