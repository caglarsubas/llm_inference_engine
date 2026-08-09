from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import pytest
from fastapi.testclient import TestClient

from inference_engine import auth as auth_mod
from inference_engine.adapters import (
    EmbeddingResult,
    EmbeddingsNotSupportedError,
    GenerationParams,
    InferenceAdapter,
    StreamChunk,
)
from inference_engine.adapters.base import GenerationResult, GenerationTimeoutError
from inference_engine.api import _model_routing
from inference_engine.api.chat import _stream_response, chat_completions
from inference_engine.api.state import app_state
from inference_engine.auth import Identity
from inference_engine.cancellation import Cancellation
from inference_engine.config import CERTIFIED_MODEL_WORKLOAD_SURFACE, settings
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
from inference_engine.scheduler import TenantScheduler
from inference_engine.schemas import ChatCompletionRequest, ChatMessage


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

    def __init__(
        self,
        text: str,
        *,
        fail: bool = False,
        backend: str = "routing-test",
        embed_supported: bool = True,
        embedding_value: float = 1.0,
    ):
        self.text = text
        self.fail = fail
        self.backend_name = backend
        self.embed_supported = embed_supported
        self.embedding_value = embedding_value
        self.generate_calls = 0
        self.complete_calls = 0
        self.stream_calls = 0
        self.embed_calls = 0

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
        yield StreamChunk(text="", finish_reason="stop", prompt_tokens=7, completion_tokens=3)

    async def embed(self, inputs: list[str]) -> EmbeddingResult:
        self.embed_calls += 1
        self._raise_if_failed()
        if not self.embed_supported:
            raise EmbeddingsNotSupportedError(self.backend_name)
        return EmbeddingResult(
            embeddings=[[self.embedding_value, self.embedding_value + 1] for _ in inputs],
            prompt_tokens=len(inputs) * 3,
        )


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
    assert generation.attributes["prometa.artifact.type"] == "model-routing-policy"
    assert generation.attributes["prometa.artifact.digest"] == (
        generation.attributes["model_routing.policy.digest"]
    )
    assert generation.attributes["prometa.policy.digest"] == (
        generation.attributes["model_routing.policy.digest"]
    )
    assert generation.attributes["prometa.release.id"] == "release-golden-model-v1"
    assert generation.attributes["prometa.deployment.id"] == "model-plane-golden-v1"


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


def test_embeddings_use_signed_alias_limits_and_policy_evidence(
    monkeypatch,
    _session_exporter,
) -> None:
    _session_exporter.clear()
    primary = _RoutingAdapter("primary", embedding_value=2.0)
    calls = _install_models(monkeypatch, {"qwen3:32b": primary})

    response = TestClient(app).post(
        "/v1/embeddings",
        json={"model": "reasoning", "input": ["first", "second"]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "qwen3:32b"
    assert body["data"][0]["embedding"] == [2.0, 3.0]
    assert body["usage"] == {
        "prompt_tokens": 6,
        "completion_tokens": 0,
        "total_tokens": 6,
        # Embeddings have no prefix-cache accounting to report.
        "prompt_tokens_details": None,
    }
    assert body["fallback_from_model"] is None
    assert calls == ["qwen3:32b"]
    assert primary.embed_calls == 1

    [embedding] = [
        item for item in _session_exporter.get_finished_spans() if item.name == "embeddings.run"
    ]
    assert embedding.attributes["model_routing.policy.id"] == "routing-golden-v1"
    assert embedding.attributes["model_routing.route.id"] == "reasoning"
    assert embedding.attributes["model_routing.route.selected_model"] == "qwen3:32b"
    assert embedding.attributes["model_routing.output_token_budget"] == 0
    assert embedding.attributes["model_routing.estimated_max_cost_micros"] > 0
    assert embedding.attributes["prometa.org_id"] == "org-golden"


def test_embeddings_try_only_ordered_signed_fallbacks(monkeypatch) -> None:
    primary = _RoutingAdapter("primary", embed_supported=False, backend="local")
    fallback = _RoutingAdapter(
        "signed-fallback",
        backend="openrouter",
        embedding_value=4.0,
    )
    unsigned = _RoutingAdapter("unsigned", embedding_value=9.0)
    calls = _install_models(
        monkeypatch,
        {
            "qwen3:32b": primary,
            "llama3.3:70b:openrouter": fallback,
            "escape:openrouter": unsigned,
        },
    )

    response = TestClient(app).post(
        "/v1/embeddings",
        json={"model": "reasoning", "input": "embed this"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "llama3.3:70b:openrouter"
    assert body["data"][0]["embedding"] == [4.0, 5.0]
    assert body["fallback_from_model"] == "qwen3:32b"
    assert body["fallback_reason"] == "capability_not_supported"
    assert body["fallback_error_type"] == "EmbeddingsNotSupportedError"
    assert calls == ["qwen3:32b", "llama3.3:70b:openrouter"]
    assert unsigned.embed_calls == 0


def test_embeddings_do_not_escape_signed_candidates(monkeypatch) -> None:
    primary = _RoutingAdapter("primary", embed_supported=False)
    signed_fallback = _RoutingAdapter("signed", embed_supported=False)
    unsigned = _RoutingAdapter("unsigned", embedding_value=9.0)
    calls = _install_models(
        monkeypatch,
        {
            "qwen3:32b": primary,
            "llama3.3:70b:openrouter": signed_fallback,
            "escape:openrouter": unsigned,
        },
    )

    response = TestClient(app).post(
        "/v1/embeddings",
        json={"model": "reasoning", "input": "embed this"},
    )

    assert response.status_code == 501
    assert "embeddings not supported" in response.json()["detail"]
    assert calls == ["qwen3:32b", "llama3.3:70b:openrouter"]
    assert unsigned.embed_calls == 0


def test_embedding_bounds_deny_before_model_lookup(monkeypatch) -> None:
    active = _replace_reasoning_limits(
        _active_policy(),
        max_input_tokens=1,
        max_cost_micros_per_request=None,
    )
    app_state.model_routing_runtime = _runtime_state(active)
    calls = _install_models(monkeypatch, {"qwen3:32b": _RoutingAdapter("unused")})

    response = TestClient(app).post(
        "/v1/embeddings",
        json={"model": "reasoning", "input": "private input"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "input_token_limit_exceeded"
    assert calls == []


def test_embeddings_enforce_authenticated_org_binding(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    auth_mod._set_keys_for_tests(
        [
            ("sk-embed-good", "tenant-good", "org-golden"),
            ("sk-embed-wrong", "tenant-wrong", "org-other"),
        ]
    )
    adapter = _RoutingAdapter("primary", embedding_value=6.0)
    calls = _install_models(monkeypatch, {"qwen3:32b": adapter})
    client = TestClient(app)

    denied = client.post(
        "/v1/embeddings",
        json={"model": "reasoning", "input": "private"},
        headers={"Authorization": "Bearer sk-embed-wrong"},
    )
    accepted = client.post(
        "/v1/embeddings",
        json={"model": "reasoning", "input": "private"},
        headers={"Authorization": "Bearer sk-embed-good"},
    )

    assert denied.status_code == 403
    assert denied.json()["detail"]["type"] == "org_identity_mismatch"
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["data"][0]["embedding"] == [6.0, 7.0]
    assert calls == ["qwen3:32b"]


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
    assert denial.attributes["prometa.artifact.type"] == "model-routing-policy"
    assert denial.attributes["prometa.artifact.digest"] == (
        denial.attributes["model_routing.policy.digest"]
    )
    assert denial.attributes["prometa.release.id"] == "release-golden-model-v1"
    assert denial.attributes["prometa.deployment.id"] == "model-plane-golden-v1"
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


def test_certified_surface_denies_unintegrated_workload_without_active_policy(
    monkeypatch,
) -> None:
    app_state.model_routing_runtime = ModelRoutingRuntimeState()
    monkeypatch.setattr(
        settings,
        "model_plane_workload_surface",
        CERTIFIED_MODEL_WORKLOAD_SURFACE,
    )
    calls = _install_models(monkeypatch, {"reasoning": _RoutingAdapter("unused")})

    response = TestClient(app).post(
        "/v1/rerank",
        json={"model": "reasoning", "query": "q", "documents": ["d"]},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["type"] == "model_plane_workload_not_certified"
    assert detail["workload"] == "rerank.run"
    assert detail["workload_surface"] == CERTIFIED_MODEL_WORKLOAD_SURFACE
    assert "policy_id" not in detail
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


def _tpm_window(tenant: str = "anonymous"):
    return app_state.model_routing_rate_limiter._buckets[
        (
            "tpm",
            app_state.model_routing_runtime.policy.digest,
            "reasoning",
            "org-golden",
            tenant,
        )
    ]


def test_token_rate_limit_denies_before_model_acquire(monkeypatch) -> None:
    app_state.model_routing_runtime = _runtime_state(
        _replace_reasoning_limits(_active_policy(), max_tokens_per_minute=1)
    )
    adapter = _RoutingAdapter("must-not-run")
    calls = _install_models(monkeypatch, {"qwen3:32b": adapter})

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "reasoning", "messages": [{"role": "user", "content": "one"}]},
    )

    assert response.status_code == 429
    assert response.json()["detail"]["type"] == "token_rate_limit_exceeded"
    assert response.headers["Retry-After"] == "60"
    assert response.headers["x-ratelimit-limit-tokens"] == "1"
    assert response.headers["x-ratelimit-remaining-tokens"] == "0"
    assert response.headers["x-ratelimit-reset-tokens"] == "60s"
    assert "x-ratelimit-limit-requests" not in response.headers
    assert calls == []
    assert adapter.generate_calls == 0


def test_window_spend_budget_denies_before_model_acquire(monkeypatch) -> None:
    app_state.model_routing_runtime = _runtime_state(
        _replace_reasoning_limits(
            _active_policy(),
            max_cost_micros_per_window=1,
            budget_window_seconds=60,
        )
    )
    adapter = _RoutingAdapter("must-not-run")
    calls = _install_models(monkeypatch, {"qwen3:32b": adapter})

    response = TestClient(app).post(
        "/v1/completions",
        json={"model": "reasoning", "prompt": "one"},
    )

    assert response.status_code == 429
    assert response.json()["detail"]["type"] == "budget_exceeded"
    assert response.headers["Retry-After"] == "60"
    # 119 of the route's 120 requests a minute are still available, so none of
    # the request-window headers may claim otherwise, and no token window was
    # signed at all.
    assert "x-ratelimit-limit-tokens" not in response.headers
    assert "x-ratelimit-remaining-tokens" not in response.headers
    assert "x-ratelimit-limit-requests" not in response.headers
    assert "x-ratelimit-remaining-requests" not in response.headers
    assert "x-ratelimit-reset-requests" not in response.headers
    assert calls == []
    assert adapter.complete_calls == 0


def test_a_request_denial_still_carries_the_request_window_headers(monkeypatch) -> None:
    app_state.model_routing_runtime = _runtime_state(
        _replace_reasoning_limits(_active_policy(), max_requests_per_minute=1)
    )
    adapter = _RoutingAdapter("ok")
    _install_models(monkeypatch, {"qwen3:32b": adapter})
    client = TestClient(app)
    body = {"model": "reasoning", "messages": [{"role": "user", "content": "one"}]}

    assert client.post("/v1/chat/completions", json=body).status_code == 200
    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 429
    assert response.json()["detail"]["type"] == "rate_limit_exceeded"
    assert response.headers["x-ratelimit-limit-requests"] == "1"
    assert response.headers["x-ratelimit-remaining-requests"] == "0"
    assert response.headers["x-ratelimit-reset-requests"] == response.headers["Retry-After"] + "s"
    assert "x-ratelimit-limit-tokens" not in response.headers


def test_settled_usage_and_not_the_admission_reserve_fills_the_token_window(
    monkeypatch,
) -> None:
    app_state.model_routing_runtime = _runtime_state(
        _replace_reasoning_limits(_active_policy(), max_tokens_per_minute=1_000)
    )
    adapter = _RoutingAdapter("ok")
    _install_models(monkeypatch, {"qwen3:32b": adapter})
    client = TestClient(app)

    for _ in range(20):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "reasoning", "messages": [{"role": "user", "content": "one"}]},
        )
        assert response.status_code == 200, response.text

    # The conservative reserve is well over 40 tokens a call, so 20 of them fit
    # only because each settles down to the 7 + 3 it really used.
    assert adapter.generate_calls == 20
    assert _tpm_window().total == 200


def test_a_streamed_request_settles_its_reservation_when_the_body_ends(
    monkeypatch,
) -> None:
    app_state.model_routing_runtime = _runtime_state(
        _replace_reasoning_limits(_active_policy(), max_tokens_per_minute=1_000)
    )
    adapter = _RoutingAdapter("streamed")
    _install_models(monkeypatch, {"qwen3:32b": adapter})

    with TestClient(app).stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "reasoning",
            "stream": True,
            "messages": [{"role": "user", "content": "one"}],
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "streamed" in body
    assert _tpm_window().total == 10


def test_a_request_that_never_reached_a_model_returns_its_reservation(
    monkeypatch,
) -> None:
    app_state.model_routing_runtime = _runtime_state(
        _replace_reasoning_limits(_active_policy(), max_tokens_per_minute=1_000)
    )
    adapter = _RoutingAdapter("ok")
    models: dict[str, _RoutingAdapter] = {}
    _install_models(monkeypatch, models)
    client = TestClient(app)
    body = {"model": "reasoning", "messages": [{"role": "user", "content": "one"}]}

    for _ in range(10):
        unavailable = client.post("/v1/chat/completions", json=body)
        assert unavailable.status_code == 503
        assert unavailable.json()["detail"]["type"] == "model_route_unavailable"
    assert _tpm_window().total == 0

    # Ten leaked reservations would have closed the window well before here.
    models["qwen3:32b"] = adapter
    served = client.post("/v1/chat/completions", json=body)
    assert served.status_code == 200, served.text


class _AbandonableAdapter(_RoutingAdapter):
    """Streams indefinitely, reporting token counts only on a terminal frame.

    On a cancel it stops without that frame, which is what a backend does when
    the generation it was relaying is aborted part-way.
    """

    async def stream(
        self,
        messages: Iterable,
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> AsyncIterator[StreamChunk]:
        del messages, params
        self.stream_calls += 1
        for index in range(100):
            if cancel is not None and bool(cancel):
                return
            await asyncio.sleep(0.02)
            yield StreamChunk(text=f"{self.text}{index} ")
        yield StreamChunk(text="", finish_reason="stop", prompt_tokens=7, completion_tokens=3)


async def _reserved_stream(adapter, request, tenant: str = "runtime"):
    identity = Identity(tenant=tenant, key_id="sk-test", org_id="org-golden")
    decision = await _model_routing.enforce_generation_request(
        identity=identity,
        requested_model="reasoning",
        input_token_upper_bound=10,
        output_token_budget=512,
    )
    assert decision is not None
    assert decision.reserved_tokens == 522
    return decision, _stream_response(
        adapter,
        "qwen3:32b",
        [ChatMessage(role="user", content="hi")],
        GenerationParams(),
        identity,
        request,
        routing_decision=decision,
        candidate_index=0,
    )


@pytest.mark.asyncio
async def test_a_stream_abandoned_mid_flight_keeps_its_whole_reservation(
    monkeypatch,
) -> None:
    app_state.model_routing_runtime = _runtime_state(
        _replace_reasoning_limits(_active_policy(), max_tokens_per_minute=10_000)
    )
    adapter = _AbandonableAdapter("tok")
    _install_models(monkeypatch, {"qwen3:32b": adapter})

    class _Connected:
        async def is_disconnected(self) -> bool:
            return False

    decision, events = await _reserved_stream(adapter, _Connected())
    assert await anext(events) is not None
    assert await anext(events) is not None
    await events.aclose()

    assert decision.reservation is not None
    assert decision.reservation.settled
    # The terminal usage frame never arrived, so settling to the counts on hand
    # would have released the full 522-token reserve for a stream that really
    # generated.
    assert _tpm_window("runtime").total == 522


@pytest.mark.asyncio
async def test_a_stream_the_client_dropped_keeps_its_whole_reservation(
    monkeypatch,
) -> None:
    app_state.model_routing_runtime = _runtime_state(
        _replace_reasoning_limits(_active_policy(), max_tokens_per_minute=10_000)
    )
    adapter = _AbandonableAdapter("tok")
    _install_models(monkeypatch, {"qwen3:32b": adapter})

    class _Drops:
        _start = 0.0

        async def is_disconnected(self) -> bool:
            if self._start == 0.0:
                self._start = asyncio.get_running_loop().time()
            return asyncio.get_running_loop().time() - self._start >= 0.15

    decision, events = await _reserved_stream(adapter, _Drops())
    emitted = [event async for event in events]

    assert len(emitted) > 1
    assert decision.reservation is not None
    assert decision.reservation.settled
    # The adapter stopped on the cancel flag, so this generator's own loop ended
    # normally and only the usage frame is missing — still an abandoned
    # generation, and the one shape a real client disconnect produces.
    assert _tpm_window("runtime").total == 522


@pytest.mark.asyncio
async def test_a_stream_that_reports_usage_still_settles_down_to_it(
    monkeypatch,
) -> None:
    app_state.model_routing_runtime = _runtime_state(
        _replace_reasoning_limits(_active_policy(), max_tokens_per_minute=10_000)
    )
    adapter = _RoutingAdapter("done")
    _install_models(monkeypatch, {"qwen3:32b": adapter})

    class _Connected:
        async def is_disconnected(self) -> bool:
            return False

    decision, events = await _reserved_stream(adapter, _Connected())
    emitted = [event async for event in events]

    assert emitted[-1]["data"] == "[DONE]"
    assert decision.reservation is not None
    assert not decision.reservation.retained
    assert _tpm_window("runtime").total == 10


class _EarlyCountAdapter(_RoutingAdapter):
    """Reports a prompt-token count on a frame that is not the terminal one."""

    async def stream(
        self,
        messages: Iterable,
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> AsyncIterator[StreamChunk]:
        del messages, params
        self.stream_calls += 1
        yield StreamChunk(text=f"{self.text} ", prompt_tokens=9)
        for index in range(100):
            if cancel is not None and bool(cancel):
                return
            await asyncio.sleep(0.02)
            yield StreamChunk(text=f"{self.text}{index} ")
        yield StreamChunk(text="", finish_reason="stop", prompt_tokens=9, completion_tokens=120)


@pytest.mark.asyncio
async def test_a_stream_abandoned_after_a_partial_count_keeps_its_whole_reservation(
    monkeypatch,
) -> None:
    app_state.model_routing_runtime = _runtime_state(
        _replace_reasoning_limits(_active_policy(), max_tokens_per_minute=10_000)
    )
    adapter = _EarlyCountAdapter("tok")
    _install_models(monkeypatch, {"qwen3:32b": adapter})

    class _Connected:
        async def is_disconnected(self) -> bool:
            return False

    decision, events = await _reserved_stream(adapter, _Connected())
    assert await anext(events) is not None
    assert await anext(events) is not None
    await events.aclose()

    assert decision.reservation is not None
    assert decision.reservation.settled
    # A count reported before the adapter finished covers part of a generation
    # that went on without it, so 522 rather than the 9 that were on hand.
    assert _tpm_window("runtime").total == 522


@pytest.mark.asyncio
async def test_a_stream_whose_body_is_never_read_keeps_its_whole_reservation(
    monkeypatch,
) -> None:
    app_state.model_routing_runtime = _runtime_state(
        _replace_reasoning_limits(_active_policy(), max_tokens_per_minute=10_000)
    )
    adapter = _RoutingAdapter("unread")
    _install_models(monkeypatch, {"qwen3:32b": adapter})
    # A body that is never read never releases its scheduler slot either, and
    # that slot is per model: a throwaway scheduler keeps it out of every test
    # that runs after this one.
    monkeypatch.setattr(app_state, "scheduler", TenantScheduler())

    class _Connected:
        async def is_disconnected(self) -> bool:
            return False

    response = await chat_completions(
        ChatCompletionRequest(
            model="reasoning",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
            max_tokens=512,
        ),
        _Connected(),
        Identity(tenant="runtime", key_id="sk-test", org_id="org-golden"),
    )

    # Closing a generator that never started skips its body, so this is the
    # response Starlette drops when the client is gone before the first byte.
    await response.body_iterator.aclose()

    # 138 estimated input tokens plus the 512-token output budget: the whole
    # admission reserve, held until the window slides past it.
    assert adapter.stream_calls == 0
    assert _tpm_window("runtime").total == 650
