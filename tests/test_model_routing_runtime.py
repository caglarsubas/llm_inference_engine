from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import pytest

from inference_engine.auth import Identity
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
    ModelRoutingRuntimeConfigError,
    ModelRoutingRuntimeState,
    build_model_routing_runtime_state,
    enforce_model_routing_request,
    load_model_routing_pricing_catalog,
    model_routing_span_attrs,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "model-routing-policy-v1.json"
NOW = datetime(2026, 7, 13, 0, 10, tzinfo=UTC)


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
        now=NOW,
        expected_environment="staging",
        expected_org_id="org-golden",
    )
    return ActivatedModelRoutingPolicy(verified=verified, source="candidate")


def _pricing() -> LoadedModelRoutingPricingCatalog:
    prices = [
        ModelRoutingModelPrice(
            model="qwen3:32b",
            input_cost_micros_per_million_tokens=1_000_000,
            output_cost_micros_per_million_tokens=2_000_000,
        ),
        ModelRoutingModelPrice(
            model="llama3.3:70b:openrouter",
            input_cost_micros_per_million_tokens=3_000_000,
            output_cost_micros_per_million_tokens=4_000_000,
        ),
    ]
    catalog = ModelRoutingPricingCatalog(pricing_version=1, models=prices)
    return LoadedModelRoutingPricingCatalog(
        catalog=catalog,
        digest="sha256:pricing",
        by_model=MappingProxyType({price.model: price for price in prices}),
    )


def _state(
    active: ActivatedModelRoutingPolicy | None = None,
    pricing: LoadedModelRoutingPricingCatalog | None = None,
) -> ModelRoutingRuntimeState:
    return ModelRoutingRuntimeState(
        policy=active if active is not None else _active_policy(),
        pricing=pricing if pricing is not None else _pricing(),
    )


def _identity(tenant: str = "runtime") -> Identity:
    return Identity(tenant=tenant, key_id="sk-test", org_id="org-golden")


def _enforce(
    *,
    state: ModelRoutingRuntimeState | None = None,
    requested_model: str = "reasoning",
    input_tokens: int | None = 100,
    output_tokens: int = 10,
    limiter: ModelRoutingRateLimiter | None = None,
    identity: Identity | None = None,
    now: datetime = NOW,
):
    return enforce_model_routing_request(
        state or _state(),
        identity=identity or _identity(),
        requested_model=requested_model,
        input_token_upper_bound=input_tokens,
        output_token_budget=output_tokens,
        rate_limiter=limiter or ModelRoutingRateLimiter(),
        now=now,
    )


def _replace_reasoning_route(active: ActivatedModelRoutingPolicy, **changes):
    claims = active.verified.claims
    route = claims.routes[0].model_copy(update=changes)
    next_claims = claims.model_copy(update={"routes": [route, *claims.routes[1:]]})
    return replace(active, verified=replace(active.verified, claims=next_claims))


def test_pricing_catalog_loads_strictly_and_has_stable_digest(tmp_path: Path) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "pricingVersion": 1,
                "models": [
                    {
                        "model": "qwen3:32b",
                        "inputCostMicrosPerMillionTokens": 0,
                        "outputCostMicrosPerMillionTokens": 100_000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_model_routing_pricing_catalog(path)

    assert loaded is not None
    assert loaded.by_model["qwen3:32b"].output_cost_micros_per_million_tokens == 100_000
    assert loaded.digest.startswith("sha256:")
    assert load_model_routing_pricing_catalog(tmp_path / "missing.json") is None


@pytest.mark.parametrize(
    "payload",
    [
        {"pricingVersion": 1, "models": []},
        {
            "pricingVersion": 1,
            "models": [
                {
                    "model": "x",
                    "inputCostMicrosPerMillionTokens": 0,
                    "outputCostMicrosPerMillionTokens": 0,
                },
                {
                    "model": "x",
                    "inputCostMicrosPerMillionTokens": 0,
                    "outputCostMicrosPerMillionTokens": 0,
                },
            ],
        },
        {
            "pricingVersion": 1,
            "models": [
                {
                    "model": "x",
                    "inputCostMicrosPerMillionTokens": -1,
                    "outputCostMicrosPerMillionTokens": 0,
                }
            ],
        },
    ],
)
def test_pricing_catalog_rejects_unsafe_shapes(tmp_path: Path, payload: dict) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelRoutingRuntimeConfigError):
        load_model_routing_pricing_catalog(path)


def test_runtime_state_requires_deployment_org_when_auth_is_disabled() -> None:
    with pytest.raises(ModelRoutingRuntimeConfigError, match="org_binding_required"):
        build_model_routing_runtime_state(
            _active_policy(),
            _pricing(),
            auth_enabled=False,
            expected_org_id=None,
        )


def test_runtime_state_requires_prices_for_every_costed_candidate() -> None:
    with pytest.raises(ModelRoutingRuntimeConfigError, match="pricing_catalog_required"):
        build_model_routing_runtime_state(
            _active_policy(),
            None,
            auth_enabled=True,
            expected_org_id=None,
        )

    incomplete = _pricing()
    incomplete = replace(
        incomplete,
        by_model=MappingProxyType({"qwen3:32b": incomplete.by_model["qwen3:32b"]}),
    )
    with pytest.raises(ModelRoutingRuntimeConfigError, match="pricing_model_missing"):
        build_model_routing_runtime_state(
            _active_policy(),
            incomplete,
            auth_enabled=True,
            expected_org_id=None,
        )


def test_exact_alias_and_wildcard_resolve_to_signed_candidates() -> None:
    exact = _enforce()
    wildcard = _enforce(requested_model="unlisted", output_tokens=100)

    assert exact is not None
    assert exact.route.route_id == "reasoning"
    assert exact.candidate_models == (
        "qwen3:32b",
        "llama3.3:70b:openrouter",
    )
    assert wildcard is not None
    assert wildcard.route.route_id == "default"
    assert wildcard.candidate_models == ("llama3.2:3b",)


def test_request_without_exact_or_wildcard_route_is_denied() -> None:
    active = _active_policy()
    claims = active.verified.claims.model_copy(
        update={"routes": [active.verified.claims.routes[0]]}
    )
    active = replace(active, verified=replace(active.verified, claims=claims))

    with pytest.raises(ModelRoutingEnforcementError, match="route_not_allowed"):
        _enforce(state=_state(active=active), requested_model="unlisted")


def test_missing_or_wrong_org_identity_is_denied() -> None:
    with pytest.raises(ModelRoutingEnforcementError, match="org_identity_missing"):
        _enforce(identity=Identity(tenant="runtime", key_id="sk-test"))
    with pytest.raises(ModelRoutingEnforcementError, match="org_identity_mismatch"):
        _enforce(identity=Identity(tenant="runtime", key_id="sk-test", org_id="other"))


def test_invalid_alias_and_negative_bounds_cannot_reach_wildcard_route() -> None:
    with pytest.raises(ModelRoutingEnforcementError, match="invalid_requested_model"):
        _enforce(requested_model=" ")
    with pytest.raises(ModelRoutingEnforcementError, match="invalid_request_bounds"):
        _enforce(output_tokens=-1)
    with pytest.raises(ModelRoutingEnforcementError, match="invalid_request_bounds"):
        _enforce(input_tokens=-1)


def test_input_and_output_limits_are_enforced_before_dispatch() -> None:
    with pytest.raises(ModelRoutingEnforcementError, match="input_token_limit_exceeded"):
        _enforce(input_tokens=32_769)
    with pytest.raises(ModelRoutingEnforcementError, match="output_token_limit_exceeded"):
        _enforce(output_tokens=4_097)
    with pytest.raises(ModelRoutingEnforcementError, match="input_token_estimate_unavailable"):
        _enforce(input_tokens=None)


def test_cost_limit_is_worst_case_across_primary_and_all_fallbacks() -> None:
    decision = _enforce(input_tokens=100, output_tokens=10)
    assert decision is not None
    assert decision.estimated_max_cost_micros == 460

    active = _active_policy()
    limits = active.verified.claims.routes[0].limits.model_copy(
        update={"max_cost_micros_per_request": 459}
    )
    active = _replace_reasoning_route(active, limits=limits)
    with pytest.raises(ModelRoutingEnforcementError, match="cost_limit_exceeded"):
        _enforce(state=_state(active=active), input_tokens=100, output_tokens=10)


def test_rate_limit_is_per_policy_route_org_and_tenant() -> None:
    active = _active_policy()
    limits = active.verified.claims.routes[0].limits.model_copy(
        update={"max_requests_per_minute": 2}
    )
    state = _state(active=_replace_reasoning_route(active, limits=limits))
    clock = [10.0]
    limiter = ModelRoutingRateLimiter(clock=lambda: clock[0])

    _enforce(state=state, limiter=limiter)
    _enforce(state=state, limiter=limiter)
    with pytest.raises(ModelRoutingEnforcementError) as raised:
        _enforce(state=state, limiter=limiter)
    assert raised.value.code == "rate_limit_exceeded"
    assert raised.value.retry_after_seconds == 60

    _enforce(state=state, limiter=limiter, identity=_identity("other-tenant"))
    clock[0] = 70.1
    _enforce(state=state, limiter=limiter)


def test_request_time_freshness_rejects_offline_lease_and_expiry() -> None:
    with pytest.raises(ModelRoutingEnforcementError, match="policy_offline_lease_expired"):
        _enforce(now=datetime(2026, 7, 13, 0, 31, tzinfo=UTC))
    with pytest.raises(ModelRoutingEnforcementError, match="policy_expired"):
        _enforce(now=datetime(2026, 7, 13, 1, 1, tzinfo=UTC))


def test_policy_evidence_attributes_are_complete_and_payload_free() -> None:
    decision = _enforce()
    attrs = model_routing_span_attrs(
        decision,
        candidate_model="qwen3:32b",
        candidate_index=0,
    )

    assert attrs["model_routing.policy.id"] == "routing-golden-v1"
    assert attrs["model_routing.policy.revision"] == 1
    assert attrs["model_routing.policy.release_id"] == "release-golden-model-v1"
    assert attrs["model_routing.policy.deployment_id"] == "model-plane-golden-v1"
    assert attrs["model_routing.policy.org_id"] == "org-golden"
    assert attrs["model_routing.policy.environment"] == "staging"
    assert attrs["prometa.artifact.type"] == "model-routing-policy"
    assert attrs["prometa.artifact.digest"] == attrs["model_routing.policy.digest"]
    assert attrs["prometa.policy.digest"] == attrs["model_routing.policy.digest"]
    assert attrs["prometa.release.id"] == "release-golden-model-v1"
    assert attrs["prometa.deployment.id"] == "model-plane-golden-v1"
    assert attrs["prometa.environment"] == "staging"
    assert attrs["model_routing.route.id"] == "reasoning"
    assert attrs["model_routing.route.selected_model"] == "qwen3:32b"
    assert attrs["model_routing.pricing.digest"] == "sha256:pricing"
    assert "signed_payload" not in attrs


def test_policy_evidence_reports_deployment_shared_rate_limit_scope() -> None:
    class SharedScopeLimiter(ModelRoutingRateLimiter):
        scope = "deployment-shared"

    decision = _enforce(limiter=SharedScopeLimiter())
    attrs = model_routing_span_attrs(decision)

    assert attrs["model_routing.rate_limit.scope"] == "deployment-shared"
