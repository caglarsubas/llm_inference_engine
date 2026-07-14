from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import pytest
from fastapi.testclient import TestClient

from inference_engine import auth as auth_mod
from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult, GenerationTimeoutError
from inference_engine.api import _model_routing
from inference_engine.api.chat import _stream_response
from inference_engine.api.state import app_state
from inference_engine.auth import Identity
from inference_engine.cancellation import Cancellation
from inference_engine.config import settings
from inference_engine.evals import PolicyRegistry
from inference_engine.main import app
from inference_engine.manager import ModelNotFoundError
from inference_engine.model_routing import (
    ActivatedModelRoutingPolicy,
    ModelRoutingPolicyEnvelope,
    ModelRoutingTrustStore,
    verify_model_routing_policy,
)
from inference_engine.model_routing_runtime import (
    LoadedModelRoutingPricingCatalog,
    ModelRoutingEnforcementError,
    ModelRoutingModelPrice,
    ModelRoutingPricingCatalog,
    ModelRoutingRateLimiter,
    ModelRoutingRuntimeState,
    build_model_routing_runtime_state,
)
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import ChatMessage


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "model-routing-policy-v1.json"


def _active_policy() -> ActivatedModelRoutingPolicy:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    envelope = ModelRoutingPolicyEnvelope.model_validate(fixture["policy"], strict=True)
    trust = ModelRoutingTrustStore.model_validate(
        {
            "trustVersion": 1,
            "entries": [fixture["trust"]],
            "revokedKeyIds": [],
            "revokedJtis": [],
        },
        strict=True,
    )
    verified = verify_model_routing_policy(
        envelope,
        trust,
        now=datetime(2026, 7, 13, 0, 10, tzinfo=UTC),
        expected_environment="staging",
        expected_org_id="org-golden",
    )
    claims = verified.claims.model_copy(
        update={
            "expires_at": "2099-07-13T01:00:00.000Z",
            "offline_lease_expires_at": "2099-07-13T00:30:00.000Z",
        }
    )
    return ActivatedModelRoutingPolicy(
        verified=replace(verified, claims=claims),
        source="candidate",
    )


def _pricing() -> LoadedModelRoutingPricingCatalog:
    prices = [
        ModelRoutingModelPrice(
            model="qwen3:32b",
            input_cost_micros_per_million_tokens=10_000,
            output_cost_micros_per_million_tokens=20_000,
        ),
        ModelRoutingModelPrice(
            model="llama3.3:70b:openrouter",
            input_cost_micros_per_million_tokens=30_000,
            output_cost_micros_per_million_tokens=40_000,
        ),
    ]
    catalog = ModelRoutingPricingCatalog(pricing_version=1, models=prices)
    return LoadedModelRoutingPricingCatalog(
        catalog=catalog,
        digest="sha256:api-pricing",
        by_model=MappingProxyType({price.model: price for price in prices}),
    )


def _runtime_state(
    active: ActivatedModelRoutingPolicy | None = None,
) -> ModelRoutingRuntimeState:
    return build_model_routing_runtime_state(
        active or _active_policy(),
        _pricing(),
        auth_enabled=False,
        expected_org_id="org-golden",
    )


def _replace_reasoning_limits(active: ActivatedModelRoutingPolicy, **updates):
    claims = active.verified.claims
    route = claims.routes[0]
    limits = route.limits.model_copy(update=updates)
    next_route = route.model_copy(update={"limits": limits})
    next_claims = claims.model_copy(update={"routes": [next_route, *claims.routes[1:]]})
    return replace(active, verified=replace(active.verified, claims=next_claims))


def _descriptor(model_id: str) -> ModelDescriptor:
    name, tag = model_id.rsplit(":", 1)
    return ModelDescriptor(
        name=name,
        tag=tag,
        namespace="test",
        registry="test",
        model_path=Path(f"/tmp/{model_id}"),
        format="gguf",
        size_bytes=1,
    )


class _RoutingAdapter(InferenceAdapter):
    backend_name = "routing-test"

    def __init__(self, text: str, *, fail: bool = False, backend: str = "routing-test"):
        self.text = text
        self.fail = fail
        self.backend_name = backend
        self.generate_calls = 0
        self.complete_calls = 0
        self.stream_calls = 0

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return None

    async def load(self, descriptor: ModelDescriptor) -> None:
        return None

    async def unload(self) -> None:
        return None

    def _raise_if_failed(self) -> None:
        if self.fail:
            raise GenerationTimeoutError(
                timeout_seconds=1,
                backend=self.backend_name,
            )

    async def generate(
        self,
        messages: Iterable,
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> GenerationResult:
        del messages, params, cancel
        self.generate_calls += 1
        self._raise_if_failed()
        return GenerationResult(
            text=self.text,
            finish_reason="stop",
            prompt_tokens=7,
            completion_tokens=3,
        )

    async def complete(
        self,
        prompt: str,
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> GenerationResult:
        del params, cancel
        self.complete_calls += 1
        self._raise_if_failed()
        return GenerationResult(
            text=f"{self.text}:{prompt}",
            finish_reason="stop",
            prompt_tokens=4,
            completion_tokens=2,
        )

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
        yield StreamChunk(text="", finish_reason="stop")


@pytest.fixture(autouse=True)
def _governed_runtime(monkeypatch):
    previous_runtime = app_state.model_routing_runtime
    previous_limiter = app_state.model_routing_rate_limiter
    previous_policy_registry = app_state.policy_registry
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "model_routing_expected_org_id", "org-golden")
    monkeypatch.setattr(settings, "model_routing_input_token_reserve", 0)
    monkeypatch.setattr(settings, "openrouter_fallback_enabled", True)
    monkeypatch.setattr(settings, "openrouter_fallback_model", "escape:openrouter")
    app_state.model_routing_runtime = _runtime_state()
    app_state.model_routing_rate_limiter = ModelRoutingRateLimiter()
    app_state.policy_registry = PolicyRegistry([])
    auth_mod._reset_for_tests()
    yield
    auth_mod._reset_for_tests()
    app_state.model_routing_runtime = previous_runtime
    app_state.model_routing_rate_limiter = previous_limiter
    app_state.policy_registry = previous_policy_registry


def _install_models(monkeypatch, models: dict[str, _RoutingAdapter]):
    calls: list[str] = []

    async def _get(model_id: str):
        calls.append(model_id)
        adapter = models.get(model_id)
        if adapter is None:
            raise ModelNotFoundError(model_id)
        return adapter, _descriptor(model_id)

    monkeypatch.setattr(app_state.manager, "get", _get)
    return calls


def test_chat_alias_routes_to_signed_primary_and_stamps_evidence(
    monkeypatch,
    _session_exporter,
) -> None:
    _session_exporter.clear()
    primary = _RoutingAdapter("primary")
    calls = _install_models(monkeypatch, {"qwen3:32b": primary})

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "reasoning", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "qwen3:32b"
    assert response.json()["choices"][0]["message"]["content"] == "primary"
    assert calls == ["qwen3:32b"]
    [generation] = [
        item for item in _session_exporter.get_finished_spans() if item.name == "chat.generate"
    ]
    assert generation.attributes["model_routing.policy.id"] == "routing-golden-v1"
    assert generation.attributes["model_routing.route.id"] == "reasoning"
    assert generation.attributes["model_routing.route.selected_model"] == "qwen3:32b"
    assert generation.attributes["model_routing.policy.release_id"] == ("release-golden-model-v1")


def test_blocking_chat_uses_ordered_signed_fallback(monkeypatch) -> None:
    primary = _RoutingAdapter("primary", fail=True, backend="local")
    fallback = _RoutingAdapter("signed-fallback", backend="openrouter")
    calls = _install_models(
        monkeypatch,
        {
            "qwen3:32b": primary,
            "llama3.3:70b:openrouter": fallback,
        },
    )

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "reasoning", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "llama3.3:70b:openrouter"
    assert body["fallback_from_model"] == "qwen3:32b"
    assert body["fallback_reason"] == "generation_timeout"
    assert calls == ["qwen3:32b", "llama3.3:70b:openrouter"]


def test_model_acquire_failure_uses_next_signed_candidate(monkeypatch) -> None:
    fallback = _RoutingAdapter("acquire-fallback")
    calls: list[str] = []

    async def _get(model_id: str):
        calls.append(model_id)
        if model_id == "qwen3:32b":
            raise RuntimeError("local model load failed")
        if model_id == "llama3.3:70b:openrouter":
            return fallback, _descriptor(model_id)
        raise ModelNotFoundError(model_id)

    monkeypatch.setattr(app_state.manager, "get", _get)
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "reasoning", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["fallback_reason"] == "model_acquire_error"
    assert response.json()["fallback_error_type"] == "RuntimeError"
    assert calls == ["qwen3:32b", "llama3.3:70b:openrouter"]


def test_fallback_acquire_failure_continues_through_signed_order(monkeypatch) -> None:
    active = _active_policy()
    claims = active.verified.claims
    route = claims.routes[0]
    limits = route.limits.model_copy(update={"max_cost_micros_per_request": None})
    route = route.model_copy(
        update={
            "fallback_models": ["load-fails:test", "final:test"],
            "limits": limits,
        }
    )
    claims = claims.model_copy(update={"routes": [route, *claims.routes[1:]]})
    active = replace(active, verified=replace(active.verified, claims=claims))
    app_state.model_routing_runtime = _runtime_state(active)

    primary = _RoutingAdapter("primary", fail=True)
    final = _RoutingAdapter("final")
    calls: list[str] = []

    async def _get(model_id: str):
        calls.append(model_id)
        if model_id == "qwen3:32b":
            return primary, _descriptor(model_id)
        if model_id == "load-fails:test":
            raise RuntimeError("fallback load failed")
        if model_id == "final:test":
            return final, _descriptor(model_id)
        raise ModelNotFoundError(model_id)

    monkeypatch.setattr(app_state.manager, "get", _get)
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "reasoning", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "final:test"
    assert response.json()["fallback_from_model"] == "load-fails:test"
    assert response.json()["fallback_reason"] == "model_acquire_error"
    assert response.json()["fallback_error_type"] == "RuntimeError"
    assert calls == ["qwen3:32b", "load-fails:test", "final:test"]


def test_governed_failure_never_escapes_to_global_fallback(monkeypatch) -> None:
    calls = _install_models(
        monkeypatch,
        {
            "qwen3:32b": _RoutingAdapter("primary", fail=True),
            "llama3.3:70b:openrouter": _RoutingAdapter("signed", fail=True),
            "escape:openrouter": _RoutingAdapter("unsigned escape"),
        },
    )

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "reasoning", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 504
    assert calls == ["qwen3:32b", "llama3.3:70b:openrouter"]


@pytest.mark.asyncio
async def test_streaming_chat_falls_back_before_first_chunk(monkeypatch) -> None:
    primary = _RoutingAdapter("primary", fail=True)
    fallback = _RoutingAdapter("streamed-signed-fallback")
    calls = _install_models(
        monkeypatch,
        {
            "qwen3:32b": primary,
            "llama3.3:70b:openrouter": fallback,
        },
    )

    identity = Identity(tenant="runtime", key_id="sk-test", org_id="org-golden")
    decision = await _model_routing.enforce_generation_request(
        identity=identity,
        requested_model="reasoning",
        input_token_upper_bound=10,
        output_token_budget=512,
    )
    active = await _model_routing.resolve_initial_candidate(
        requested_model="reasoning",
        decision=decision,
        identity=identity,
    )

    class _Request:
        async def is_disconnected(self) -> bool:
            return False

    events = [
        event
        async for event in _stream_response(
            active.adapter,
            active.model_name,
            [ChatMessage(role="user", content="hi")],
            GenerationParams(),
            identity,
            _Request(),
            fallback_info=active.fallback_info,
            routing_decision=decision,
            candidate_index=active.candidate_index,
        )
    ]

    encoded = json.dumps(events)
    assert "streamed-signed-fallback" in encoded
    assert "llama3.3:70b:openrouter" in encoded
    assert events[-1]["data"] == "[DONE]"
    assert calls == ["qwen3:32b", "llama3.3:70b:openrouter"]


def test_legacy_completions_uses_same_alias_and_fallback(monkeypatch) -> None:
    primary = _RoutingAdapter("primary", fail=True)
    fallback = _RoutingAdapter("completion-fallback")
    calls = _install_models(
        monkeypatch,
        {
            "qwen3:32b": primary,
            "llama3.3:70b:openrouter": fallback,
        },
    )

    response = TestClient(app).post(
        "/v1/completions",
        json={"model": "reasoning", "prompt": "raw"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "llama3.3:70b:openrouter"
    assert response.json()["choices"][0]["text"] == "completion-fallback:raw"
    assert calls == ["qwen3:32b", "llama3.3:70b:openrouter"]


def test_output_and_cost_denials_happen_before_model_lookup(monkeypatch) -> None:
    calls = _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("unused")})
    client = TestClient(app)

    output_denial = client.post(
        "/v1/chat/completions",
        json={
            "model": "reasoning",
            "max_tokens": 4_097,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert output_denial.status_code == 400
    assert output_denial.json()["detail"]["type"] == "output_token_limit_exceeded"

    active = _replace_reasoning_limits(
        _active_policy(),
        max_cost_micros_per_request=1,
    )
    app_state.model_routing_runtime = _runtime_state(active)
    cost_denial = client.post(
        "/v1/chat/completions",
        json={"model": "reasoning", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert cost_denial.status_code == 400
    assert cost_denial.json()["detail"]["type"] == "cost_limit_exceeded"
    assert calls == []


def test_denial_span_carries_policy_identity_without_request_payload(
    monkeypatch,
    _session_exporter,
) -> None:
    _session_exporter.clear()
    calls = _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("unused")})

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "reasoning",
            "max_tokens": 4_097,
            "messages": [{"role": "user", "content": "private payload"}],
        },
    )

    assert response.status_code == 400
    [denial] = [
        item
        for item in _session_exporter.get_finished_spans()
        if item.name == "model.routing.decision"
    ]
    assert denial.attributes["model_routing.decision"] == "deny"
    assert denial.attributes["model_routing.denial.code"] == "output_token_limit_exceeded"
    assert denial.attributes["model_routing.policy.id"] == "routing-golden-v1"
    assert "private payload" not in json.dumps(dict(denial.attributes))
    assert calls == []


def test_governed_chat_blocks_unrouted_auto_eval(monkeypatch) -> None:
    primary = _RoutingAdapter("unused")
    calls = _install_models(monkeypatch, {"qwen3:32b": primary})

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "reasoning",
            "messages": [{"role": "user", "content": "hi"}],
            "auto_eval": {"rubrics": ["safety"], "mode": "background"},
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["type"] == ("model_routing_workload_not_integrated")
    assert response.json()["detail"]["workload"] == "chat.auto_eval"
    assert calls == ["qwen3:32b"]
    assert primary.generate_calls == 0


@pytest.mark.parametrize(
    ("path", "payload", "workload"),
    [
        (
            "/v1/embeddings",
            {"model": "reasoning", "input": "embed this"},
            "embeddings.run",
        ),
        (
            "/v1/rerank",
            {"model": "reasoning", "query": "q", "documents": ["d"]},
            "rerank.run",
        ),
        (
            "/v1/evals/run",
            {"rubric": "safety", "prompt": "p", "response": "r"},
            "eval.run",
        ),
    ],
)
def test_unintegrated_model_workloads_fail_closed_without_lookup(
    monkeypatch,
    path: str,
    payload: dict,
    workload: str,
) -> None:
    calls = _install_models(monkeypatch, {"reasoning": _RoutingAdapter("unused")})

    response = TestClient(app).post(path, json=payload)

    assert response.status_code == 503
    assert response.json()["detail"]["type"] == ("model_routing_workload_not_integrated")
    assert response.json()["detail"]["workload"] == workload
    assert calls == []


def test_remote_image_is_denied_when_bounded_input_cannot_be_estimated(monkeypatch) -> None:
    calls = _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("unused")})

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "reasoning",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect"},
                        {"type": "image_url", "image_url": {"url": "https://example.test/x.png"}},
                    ],
                }
            ],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "input_token_estimate_unavailable"
    assert calls == []


def test_rate_limit_isolated_by_authenticated_tenant_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    auth_mod._set_keys_for_tests(
        [
            ("sk-tenant-one", "tenant-one", "org-golden"),
            ("sk-tenant-two", "tenant-two", "org-golden"),
        ]
    )
    active = _replace_reasoning_limits(
        _active_policy(),
        max_requests_per_minute=1,
    )
    app_state.model_routing_runtime = _runtime_state(active)
    adapter = _RoutingAdapter("ok")
    _install_models(monkeypatch, {"qwen3:32b": adapter})
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={"model": "reasoning", "messages": [{"role": "user", "content": "one"}]},
        headers={"Authorization": "Bearer sk-tenant-one"},
    )
    second = client.post(
        "/v1/chat/completions",
        json={"model": "reasoning", "messages": [{"role": "user", "content": "two"}]},
        headers={"Authorization": "Bearer sk-tenant-one"},
    )
    other_tenant = client.post(
        "/v1/chat/completions",
        json={"model": "reasoning", "messages": [{"role": "user", "content": "three"}]},
        headers={"Authorization": "Bearer sk-tenant-two"},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert other_tenant.status_code == 200
    assert second.headers["Retry-After"]
    assert second.json()["detail"]["type"] == "rate_limit_exceeded"
    assert adapter.generate_calls == 2


def test_shared_rate_limit_outage_denies_before_model_acquire(monkeypatch) -> None:
    class UnavailableSharedLimiter:
        scope = "deployment-shared"

        def consume(self, **kwargs) -> None:
            raise ModelRoutingEnforcementError(
                "rate_limit_backend_unavailable",
                policy_id=kwargs["policy_id"],
                route_id=kwargs["route_id"],
                retry_after_seconds=1,
            )

    active = _replace_reasoning_limits(
        _active_policy(),
        max_requests_per_minute=1,
    )
    app_state.model_routing_runtime = _runtime_state(active)
    app_state.model_routing_rate_limiter = UnavailableSharedLimiter()
    adapter = _RoutingAdapter("must-not-run")
    calls = _install_models(monkeypatch, {"qwen3:32b": adapter})

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "reasoning", "messages": [{"role": "user", "content": "one"}]},
    )

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert response.json()["detail"]["type"] == "rate_limit_backend_unavailable"
    assert calls == []
    assert adapter.generate_calls == 0
